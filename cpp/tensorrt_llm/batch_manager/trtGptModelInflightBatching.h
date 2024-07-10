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
#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/common/mpiUtils.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/runtime/iGptDecoderBatch.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/rawEngine.h"
#include "tensorrt_llm/runtime/worldConfig.h"
#include "trtGptModel.h"

#include <NvInferRuntime.h>

namespace tensorrt_llm::runtime
{
class TllmRuntime;
class IGptDecoderBatch;
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

public:
    struct IterationStatsIFB
    {
        SizeType32 numScheduledRequests;
        SizeType32 numCtxRequests;
        SizeType32 numGenRequests;
        SizeType32 numCtxTokens;
        SizeType32 microBatchId;
        SizeType32 numPausedRequests;
        float avgNumDecodedTokensPerIter;
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

    /// @brief Function that tries to advance the active requests
    ///        Depending on resources available, it's possible that not all requests will get advanced
    ///        Requests that may be in state REQUEST_STATE_CONTEXT_INIT become REQUEST_STATE_GENERATION_IN_PROGRESS or
    ///        REQUEST_STATE_GENERATION_TO_COMPLETE
    /// @param activeRequests The list of request to try to advance
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

    //! @brief Set LayerProfiler to collect performance per layer.
    void setLayerProfiler() override;

    //! @brief Print profile information per layer.
    std::string getLayerProfileInfo() const override;

    void executeContext(SizeType32 runtimeContextId);
    void executeBatch(ScheduledRequests const& scheduledRequests);
    void executeStep(
        RequestVector const& contextRequests, RequestVector const& generationRequests, SizeType32 bufferId);

    void createRuntimeContexts();
    void createDecoder(std::optional<executor::DecodingMode> const& decodingModeOpt);
    void createBuffers(executor::DecodingConfig const& decodingConfig);
    std::shared_ptr<KVCacheManager> createKvCacheManager(
        KvCacheConfig const& kvCacheConfig, KvCacheType kvCacheType = KvCacheType::kSELF);
    void createRnnStateManager();
    void createCustomAllReduceWorkspace();

    void verifyRequests(ScheduledRequests const& scheduledRequests);

    void assignReqSeqSlots(ScheduledRequests const& scheduledRequests);

    /// @brief Function that sets up the TensorRT execution context that is going to be used for execution. If multiple
    /// TensorRT optimization profiles are built in the engine, it selects the corresponding context that is going to be
    /// used, and prepares the input and output tensors so that both buffers and the context is ready for the execution.
    /// @return The TensorRT execution context index that has been setup.
    void setupContext(
        RequestVector const& contextRequests, RequestVector const& generationRequests, SizeType32 bufferId);

    void setupDecoderStep(RequestVector const& contextRequests);
    TokenPtr decoderStepAsync(ScheduledRequests const& scheduledRequests);
    std::unique_ptr<DecoderStepAsyncSend> decoderSync(
        ScheduledRequests const& scheduledRequests, TokenPtr const& decoderToken);
    void postProcessRequest(LlmRequest& llmReq, SizeType32 bid);
    void getDecoderSlotHostOutputs(
        SizeType32 seqSlot, bool returnLogProbs, runtime::SamplingConfig const& samplingConfig);
    void rewindKVCacheBlocks(SizeType32 numSequences);
    void setupSpecualtiveDecodingModule(executor::DecodingConfig const& decodingConfig);

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

private:
    runtime::ModelConfig mModelConfig;
    runtime::WorldConfig mWorldConfig;
    int mDevice{-1};
    std::shared_ptr<tensorrt_llm::mpi::MpiComm> mMpiCommPipelinePara;

    std::shared_ptr<runtime::AllReduceBuffers> mAllReduceBuffers;

    std::shared_ptr<nvinfer1::ILogger> mLogger;
    std::shared_ptr<runtime::TllmRuntime> mRuntime;
    std::shared_ptr<runtime::IGptDecoderBatch> mDecoder;

    SizeType32 mMicroBatchId;
    bool mCtxGenFusion;

    std::vector<std::shared_ptr<RuntimeBuffers>> mBuffers;

    SizeType32 mNumMicroBatches;
    SizeType32 mNumBuffers;

    std::vector<ScheduledRequests> mMicroBatchScheduledRequests;
    std::vector<TokenPtr> mDecoderWaitEvents;
    ReqIdsSet mInflightReqIds;
    ReqIdsSet mReqIdsToPause;

    std::shared_ptr<SequenceSlotManager> mSeqSlotManager;
    std::shared_ptr<batch_scheduler::RequestScheduler> mRequestScheduler;
    std::shared_ptr<KVCacheManager> mKvCacheManager;
    std::shared_ptr<KVCacheManager> mCrossKvCacheManager = nullptr;
    std::shared_ptr<RnnStateManager> mRnnStateManager;
    std::shared_ptr<BasePeftCacheManager> mPeftCacheManager;

    std::shared_ptr<DecoderBuffers> mDecoderBuffers;
    std::shared_ptr<runtime::decoder_batch::Input> mDecodingInput;
    std::shared_ptr<runtime::decoder_batch::Output> mDecodingOutput;
    std::vector<std::shared_ptr<SlotDecoderBuffers>> mSlotDecoderBuffers;

    runtime::BufferManager mCopyBufferManager;

    static IterationStatsIFB fillIterationStats(
        ScheduledRequests const& scheduledRequests, SizeType32 microBatchId, RequestVector const& requestsToPause);
    IterationStatsIFB mLastIterationStatsIFB;

    std::vector<PeftTable> mPeftTables;
    std::unique_ptr<DecoderStepAsyncSend> mDecStepAsyncSndHdl;
    std::vector<std::unique_ptr<DecoderSlotAsyncSend>> mDecSlotAsyncSndHdls;
    std::unique_ptr<std::thread> mMpiWaitThread;

    std::optional<LogitsPostProcessorBatched> mLogitsPostProcessorBatched;
};

} // namespace tensorrt_llm::batch_manager
