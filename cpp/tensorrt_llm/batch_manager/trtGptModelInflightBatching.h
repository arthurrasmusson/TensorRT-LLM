/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include "requestScheduler.h"
#include "sequenceSlotManager.h"
#include "tensorrt_llm/batch_manager/BatchManager.h"
#include "tensorrt_llm/batch_manager/cacheTransceiver.h"
#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/kvCacheUtils.h"
#include "tensorrt_llm/common/mpiUtils.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/runtime/iGptDecoderBatched.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/rawEngine.h"
#include "tensorrt_llm/runtime/worldConfig.h"
#include "trtGptModel.h"

#include <NvInferRuntime.h>

namespace tensorrt_llm::runtime
{
class TllmRuntime;
class IGptDecoderBatched;
class AllReduceBuffers;
class NcclCommunicator;
} // namespace tensorrt_llm::runtime

namespace tensorrt_llm::batch_manager
{

namespace kv_cache_manager
{
class KVCacheManager;
}

namespace rnn_state_manager
{
class RnnStateManager;
}
class SequenceSlotManager;
class DecoderStepAsyncSend;
class DecoderSlotAsyncSend;
class DecoderBuffers;
class SlotDecoderBuffers;
class LlmRequest;
class RuntimeBuffers;
class BasePeftCacheManager;

class TrtGptModelInflightBatching : public TrtGptModel
{
    using KVCacheManager = kv_cache_manager::KVCacheManager;
    using KvCacheType = kv_cache_manager::CacheType;
    using KvCacheConfig = kv_cache_manager::KvCacheConfig;
    using RnnStateManager = rnn_state_manager::RnnStateManager;
    using LlmRequestPtr = std::shared_ptr<batch_manager::LlmRequest>;

public:
    class IterationStatsIFB
    {
    public:
        explicit IterationStatsIFB(SizeType32 microBatchId)
            : microBatchId{microBatchId}
        {
        }

        SizeType32 microBatchId;
        SizeType32 numCtxRequests{};
        SizeType32 numGenRequests{};
        SizeType32 numCtxTokens{};
        float avgNumDecodedTokensPerIter{};
        ReqIdsSet scheduledRequests;
        ReqIdsSet pausedRequests;
    };

    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using TokenIdType = tensorrt_llm::runtime::TokenIdType;
    using BufferManager = tensorrt_llm::runtime::BufferManager;
    using PeftTable = PeftCacheManager::PeftTable;
    using TensorMap = runtime::StringPtrMap<runtime::ITensor>;
    using TensorPtr = runtime::ITensor::SharedPtr;
    using TokenPtr = std::unique_ptr<runtime::decoder_batch::Token const>;

    TrtGptModelInflightBatching(std::shared_ptr<nvinfer1::ILogger> logger, runtime::ModelConfig const& modelConfig,
        runtime::WorldConfig const& worldConfig, runtime::RawEngine const& rawEngine, bool ctxGenFusion,
        TrtGptModelOptionalParams const& optionalParams = TrtGptModelOptionalParams());

    ~TrtGptModelInflightBatching() override;

    void terminateRequest(std::shared_ptr<LlmRequest> const& llmRequest, bool pause = false) override;

    /// @brief Function that waits for the decoding of requests in flight.
    ///        When the requests have finished or using speculative decoding, the state of requests
    ///        will become REQUEST_STATE_GENERATION_COMPLETE. Else, it will be set to
    ///        REQUEST_STATE_GENERATION_IN_PROGRESS.
    void forwardSync() override;

    /// @brief Function that tries to advance the active requests.
    ///        Depending on resources available, it's possible that not all requests will get advanced.
    ///        Requests that may be in state REQUEST_STATE_CONTEXT_INIT become REQUEST_STATE_GENERATION_IN_PROGRESS or
    ///        REQUEST_STATE_GENERATION_TO_COMPLETE.
    /// @param activeRequests The list of request to try to advance.
    void forwardAsync(RequestList const& activeRequests) override;

    void updatePeftCache(std::shared_ptr<LlmRequest> const& llmRequest) override;

    [[nodiscard]] IterationStatsIFB getLastIterationStats() const
    {
        return mLastIterationStatsIFB;
    }

    [[nodiscard]] TrtGptModelType getModelType() const override
    {
        return mCtxGenFusion ? TrtGptModelType::InflightFusedBatching : TrtGptModelType::InflightBatching;
    };

    [[nodiscard]] runtime::BufferManager const& getBufferManager() const override;
    [[nodiscard]] runtime::BufferManager::CudaStreamPtr getRuntimeStreamPtr() const override;

    void getCurrentIterationStats(executor::IterationStats& stats) const override;
    void getCurrentRequestStats(executor::RequestStatsPerIteration& stats) const override;
    [[nodiscard]] executor::DebugTensorsPerIteration getCurrentDebugTensors() const override;

    [[nodiscard]] executor::IterationType getIterCounter() const noexcept override
    {
        return mIterCounter;
    }

    [[nodiscard]] static bool optionalParamsAreValid(
        runtime::ModelConfig const& modelConfig, TrtGptModelOptionalParams const& optionalParams);
    [[nodiscard]] static TrtGptModelOptionalParams fixOptionalParams(
        runtime::ModelConfig const& modelConfig, TrtGptModelOptionalParams const& optionalParams);
    void prepareDistGenInitRequests(RequestList const& activeRequests);

    RequestVector scheduleDistGenInitRequests(RequestList const& activeRequests);

private:
    [[nodiscard]] SizeType32 getContextBufferId() const
    {
        return mMicroBatchId;
    }

    [[nodiscard]] SizeType32 getGenerationBufferId() const
    {
        return mNumMicroBatches + mMicroBatchId;
    }

    [[nodiscard]] SizeType32 getFusedBufferId() const
    {
        return mMicroBatchId;
    }

    //! @brief Store full kv cache blocks contributed by req.
    //! These blocks become reusable from next step.
    void storeContextBlocks(std::shared_ptr<LlmRequest> const& req);

    //! @brief Set LayerProfiler to collect performance per layer.
    void setLayerProfiler() override;

    //! @brief Print profile information per layer.
    std::string getLayerProfileInfo() const override;

    void executeContext(SizeType32 runtimeContextId);
    void executeBatch(ScheduledRequests const& scheduledRequests);
    void executeStep(
        RequestVector const& contextRequests, RequestVector const& generationRequests, SizeType32 bufferId);

    void debugIOTensors(RequestVector const& contextRequests, RequestVector const& generationRequests,
        TensorMap const& inputMap, TensorMap const& outputMap);

    void createRuntimeContexts();
    void createDecoder(std::optional<executor::DecodingMode> const& decodingModeOpt);
    void createBuffers(executor::DecodingConfig const& decodingConfig,
        executor::ExtendedRuntimePerfKnobConfig const& extendedRuntimePerfKnobConfig);
    std::shared_ptr<KVCacheManager> createKvCacheManager(
        KvCacheConfig const& kvCacheConfig, KvCacheType kvCacheType = KvCacheType::kSELF);
    void createRnnStateManager();
    void createCustomAllReduceWorkspace();

    /// @brief Verify draft token length and beam width of all active requests.
    ///        May change operating beam width if all requests agree on same beam width.
    void verifyRequests(RequestList const& activeRequests);

    /// @brief Change the operating beam width.
    ///        Only possible if no requests are currently in-flight.
    /// @param beamWidth New operating beam width. Must be smaller than initial maxBeamWidth.
    void changeBeamWidth(SizeType32 beamWidth);

    void assignReqSeqSlots(ScheduledRequests const& scheduledRequests);

    /// @details Should be called after setting up the current batch in executeBatch to get the correct number of
    /// context tokens.
    IterationStatsIFB fillIterationStats(
        ScheduledRequests const& scheduledRequests, RequestVector const& requestsToPause);

    /// @brief Function that sets up the TensorRT execution context that is going to be used for execution. If multiple
    /// TensorRT optimization profiles are built in the engine, it selects the corresponding context that is going to be
    /// used, and prepares the input and output tensors so that both buffers and the context is ready for the execution.
    /// @return The TensorRT execution context index that has been setup.
    void setupContext(
        RequestVector const& contextRequests, RequestVector const& generationRequests, SizeType32 bufferId);

    void setupDecoderStep(RequestVector const& contextRequests);
    TokenPtr decoderStepAsync(ScheduledRequests const& scheduledRequests);
    std::vector<std::unique_ptr<DecoderStepAsyncSend>> decoderSync(
        ScheduledRequests const& scheduledRequests, TokenPtr const& decoderToken);
    /// @brief It gathers the logits if they need to be returned, calls getDecoderSlotHostOutputs,
    /// and overwrites the llmRequest tokens buffer.
    /// Called either on request finishing, or at every step when doing beam search and streaming.
    void postProcessRequest(LlmRequest& llmReq, SizeType32 bid, std::vector<SizeType32> const& numDroppedTokens);
    /// @brief Calls gatherTree (via finalize) and transmits the reveived data across ranks if PP>1
    void getDecoderSlotHostOutputs(
        SizeType32 seqSlot, bool returnLogProbs, runtime::SamplingConfig const& samplingConfig, bool streaming);
    void rewindKVCacheBlocks(SizeType32 numSequences);
    void setupSpeculativeDecodingModule(executor::DecodingConfig const& decodingConfig);

    std::vector<bool> computeActiveVec(ScheduledRequests const& scheduledRequests);

    /// @brief Copies the content of the cache indirection outputs to the cache indirection inputs.
    /// @param[in] scheduledRequests The requests to copy the cache indirections for.
    /// @param[in] genBufferId The id of the generation buffers for those requests.
    void copyCacheIndirectionFromOutputsToInputs(ScheduledRequests const& scheduledRequests, SizeType32 genBufferId);

    [[nodiscard]] runtime::ModelConfig const& getModelConfig() const override
    {
        return mModelConfig;
    }

    [[nodiscard]] runtime::WorldConfig const& getWorldConfig() const override
    {
        return mWorldConfig;
    }

    [[nodiscard]] nvinfer1::DataType getLogitDataType() const override;

    void reshapeKvTensors(KVCacheManager const& kvCacheManager);

protected:
    std::shared_ptr<KVCacheManager> getKVCacheManager() override
    {
        return mKvCacheManager;
    }

    [[nodiscard]] std::shared_ptr<KVCacheManager const> getKVCacheManager() const override
    {
        return mKvCacheManager;
    }

    std::shared_ptr<KVCacheManager> getCrossKVCacheManager()
    {
        return mCrossKvCacheManager;
    }

    [[nodiscard]] std::shared_ptr<KVCacheManager const> getCrossKVCacheManager() const
    {
        return mCrossKvCacheManager;
    }

    [[nodiscard]] std::shared_ptr<BasePeftCacheManager> getPeftCacheManager() override
    {
        return mPeftCacheManager;
    }

    [[nodiscard]] std::shared_ptr<BasePeftCacheManager const> getPeftCacheManager() const override
    {
        return mPeftCacheManager;
    }

    void setLogitsPostProcessorBatched(std::optional<LogitsPostProcessorBatched> logitsPostProcessorBatched) override
    {
        mLogitsPostProcessorBatched = logitsPostProcessorBatched;
    }

    void setReplicateLogitsPostProcessor(bool replicateLogitsPostProcessor) override
    {
        mReplicateLogitsPostProcessor = replicateLogitsPostProcessor;
    }

    void initDataTransceiver(KVCacheManager* cacheManager);

    void checkCacheTranferStatus(bool blocking = false);

private:
    /******************** Configs ********************/
    // Parameters of the model (TRT engine)
    runtime::ModelConfig mModelConfig;
    // Parameters of the execution environment
    runtime::WorldConfig mWorldConfig;
    // Device ID of this instance
    int mDevice{-1};
    // Config for (speculative) decoding
    executor::DecodingConfig mDecodingConfig;
    // Performance knobs for the engine.
    executor::ExtendedRuntimePerfKnobConfig mExtendedRuntimePerfKnobConfig;
    // Config for debugging output
    std::optional<executor::DebugConfig> mDebugConfig;

    /******************** Components ********************/
    std::shared_ptr<nvinfer1::ILogger> mLogger;
    // Runner for the TRT engine. The engine produces logits.
    std::shared_ptr<runtime::TllmRuntime> mRuntime;
    // Decoder that generates new tokens from the logits.
    std::shared_ptr<runtime::IGptDecoderBatched> mDecoder;
    // Synchronization handles for decoder
    std::vector<TokenPtr> mDecoderWaitEvents;

    // Manager that maps requests to slots
    std::shared_ptr<SequenceSlotManager> mSeqSlotManager;
    // Scheduler that selects which requests to run in each iteration
    std::shared_ptr<batch_scheduler::RequestScheduler> mRequestScheduler;
    // KV cache manager for attention layers (optional)
    std::shared_ptr<KVCacheManager> mKvCacheManager;
    // KV cache manager for cross attention in enc-dec models (optional)
    std::shared_ptr<KVCacheManager> mCrossKvCacheManager = nullptr;
    // RNN state manager for recurrent layers (optional)
    std::shared_ptr<RnnStateManager> mRnnStateManager;
    // PEFT cache manager for LoRA tasks (optional)
    std::shared_ptr<BasePeftCacheManager> mPeftCacheManager;
    // BufferManager using a separate stream for async copy operations.
    runtime::BufferManager mCopyBufferManager;

    /******************** Logits Post-Processor ********************/
    std::optional<LogitsPostProcessorBatched> mLogitsPostProcessorBatched;
    bool mReplicateLogitsPostProcessor{true};
    // Set if any request invoked a logits processor in current step
    bool mLogitsPostProcessorIsApplied{false};

    constexpr bool broadcastPostDecoder()
    {
        return mWorldConfig.isTensorParallel() && !mReplicateLogitsPostProcessor && mLogitsPostProcessorIsApplied;
    }

    /******************** Pipeline parallelism ********************/
    std::shared_ptr<tensorrt_llm::mpi::MpiComm> mMpiCommPipelinePara;
    std::vector<std::unique_ptr<DecoderStepAsyncSend>> mDecStepAsyncSndHdls;
    std::vector<std::unique_ptr<DecoderSlotAsyncSend>> mDecSlotAsyncSndHdls;
    std::unique_ptr<std::thread> mMpiWaitThread;

    /******************** Tensor parallelism ********************/
    std::shared_ptr<tensorrt_llm::mpi::MpiComm> mMpiCommTensorPara;
    std::shared_ptr<runtime::AllReduceBuffers> mAllReduceBuffers;

    /******************** Runtime parameters ********************/
    // Flag to select fused or unfused context+generation execution
    bool mCtxGenFusion;
    // ID of current micro batch, changes after each iteration
    SizeType32 mMicroBatchId{0};
    // Number of micro batches. Multiple batches are used for overlapping setup and execution,
    // and in pipeline parallelism.
    SizeType32 mNumMicroBatches;
    // Number of buffers to be added to mBuffers.
    SizeType32 mNumBuffers;
    // Current operating beam width. Can be changed with changeBeamWidth function.
    SizeType32 mOperatingBeamWidth;

    /******************** Buffers ********************/
    // Buffers for each micro batch. Unfused path (mCtxGenFusion==false) uses two times the buffers.
    std::vector<std::shared_ptr<RuntimeBuffers>> mBuffers;
    // Global buffer to interface with decoder. Slots in this buffer are selected by mSeqSlotManager.
    std::shared_ptr<DecoderBuffers> mDecoderBuffers;
    // Decoder input for each micro batch
    std::vector<std::shared_ptr<runtime::decoder_batch::Input>> mDecodingInputs;
    std::shared_ptr<runtime::decoder_batch::Output> mDecodingOutput;
    // Buffers for each slot in the decoder
    std::vector<std::shared_ptr<SlotDecoderBuffers>> mSlotDecoderBuffers;
    // PEFT table for each micro batch
    std::vector<PeftTable> mPeftTables;

    /******************** Book keeping ********************/
    // List of requests in each micro batch
    std::vector<ScheduledRequests> mMicroBatchScheduledRequests;
    // Set of in-flight requests of *all* micro batches
    ReqIdsSet mInflightReqIds;
    // Requests that the scheduler selected to be paused
    ReqIdsSet mReqIdsToPause;
    // Stats collected in last iteration
    IterationStatsIFB mLastIterationStatsIFB{-1};
    // Iteration counter used to distinguish debug output
    executor::IterationType mIterCounter{0};
    // Debug tensors of last itreation
    TensorMap mLastIterationDebugTensors;

    /******************** Cache transceiver ********************/
    std::unique_ptr<DataResponder> mDataResponder;
    std::unique_ptr<DataRequester> mDataRequester;
    std::map<LlmRequest*, std::future<void>> mResponderFutures;
};

} // namespace tensorrt_llm::batch_manager
