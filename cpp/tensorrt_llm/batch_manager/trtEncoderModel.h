/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include "tensorrt_llm/batch_manager/capacityScheduler.h"
#include "tensorrt_llm/batch_manager/microBatchScheduler.h"
#include "tensorrt_llm/common/mpiUtils.h"
#include "tensorrt_llm/runtime/iGptDecoderBatched.h"
#include "tensorrt_llm/runtime/rawEngine.h"
#include "trtGptModel.h"

#include <NvInferRuntime.h>

namespace tensorrt_llm::runtime
{
class TllmRuntime;
class NcclCommunicator;
} // namespace tensorrt_llm::runtime

namespace tensorrt_llm::batch_manager
{

class EncoderBuffers;

class TrtEncoderModel : public TrtGptModel
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using TokenIdType = tensorrt_llm::runtime::TokenIdType;
    using BufferManager = tensorrt_llm::runtime::BufferManager;
    using TensorMap = runtime::StringPtrMap<runtime::ITensor>;
    using TensorPtr = runtime::ITensor::SharedPtr;
    using DecoderFinishedEventPtr = std::unique_ptr<runtime::decoder_batch::DecoderFinishedEvent const>;

    TrtEncoderModel(runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig,
        runtime::RawEngine const& rawEngine, std::shared_ptr<nvinfer1::ILogger> logger,
        TrtGptModelOptionalParams const& optionalParams);

    void terminateRequest(std::shared_ptr<LlmRequest> const& llmRequest, bool pause = false) override;
    void forward(RequestVector& activeRequests);

    void forwardSync() override;

    void forwardAsync(RequestList const& activeRequests) override;

    [[nodiscard]] runtime::BufferManager const& getBufferManager() const override;
    [[nodiscard]] runtime::BufferManager::CudaStreamPtr getRuntimeStreamPtr() const override;

    runtime::ModelConfig const& getModelConfig() const override
    {
        return mModelConfig;
    }

    runtime::WorldConfig const& getWorldConfig() const override
    {
        return mWorldConfig;
    }

    [[nodiscard]] SizeType32 getHiddenSize() const override
    {
        return mHiddenSize;
    }

    [[nodiscard]] SizeType32 getMaxInputLen() const override
    {
        return mMaxInputLen;
    }

    [[nodiscard]] SizeType32 getNumMicroBatches() const override
    {
        return mNumMicroBatches;
    }

    [[nodiscard]] nvinfer1::DataType getLogitDataType() const override
    {
        return getModelConfig().getDataType();
    }

    [[nodiscard]] TrtGptModelType getModelType() const override
    {
        throw std::runtime_error("TrtEncoderModel does not have model type."); // FIXME:
    }

    [[nodiscard]] executor::IterationType getIterCounter() const noexcept override
    {
        return mIterCounter;
    }

    void updatePeftCache(std::shared_ptr<LlmRequest> const& /*llmRequest*/) override
    {
        throw std::runtime_error("TrtEncoderModel does not have Peft Cache.");
    }

    void getCurrentIterationStats(executor::IterationStats& stats) const override;
    void getCurrentRequestStats(executor::RequestStatsPerIteration& stats) const override;
    [[nodiscard]] executor::DebugTensorsPerIteration getCurrentDebugTensors() const override;

    void setLayerProfiler() override;
    std::string getLayerProfileInfo() const override;

    void setLogitsPostProcessorBatched(std::optional<LogitsPostProcessorBatched> logitsPostProcessorBatched) override;
    void setReplicateLogitsPostProcessor(bool replicateLogitsPostProcessor) override;

    void resetIterationStats() override {}

protected:
    std::shared_ptr<kv_cache_manager::KVCacheManager> getKVCacheManager() override
    {
        throw std::runtime_error("TrtEncoderModel does not have KVCache.");
    }

    [[nodiscard]] std::shared_ptr<kv_cache_manager::KVCacheManager const> getKVCacheManager() const override
    {
        throw std::runtime_error("TrtEncoderModel does not have KVCache.");
    }

    [[nodiscard]] std::shared_ptr<BasePeftCacheManager> getPeftCacheManager() override
    {
        throw std::runtime_error("TrtEncoderModel does not use PEFT.");
    }

    [[nodiscard]] std::shared_ptr<BasePeftCacheManager const> getPeftCacheManager() const override
    {
        throw std::runtime_error("TrtEncoderModel does not use PEFT.");
    }

private:
    [[nodiscard]] SizeType32 getBufferId() const
    {
        return mMicroBatchId;
    }

    void createRuntimeContexts();
    void executeContext(SizeType32 runtimeContextId);
    void createBuffers();
    void executeBatch(RequestVector const& requestList);
    void executeBatch(ScheduledRequests const& scheduledRequests);
    void rearrangeOutputs(ScheduledRequests const& scheduledRequests);
    void createCustomAllReduceWorkspace();
    void fillEncoderOutputSync(RequestVector const& requestList, TensorMap outputTensors);

    runtime::ModelConfig const mModelConfig;
    runtime::WorldConfig const mWorldConfig;
    int mDevice{-1};
    std::shared_ptr<tensorrt_llm::mpi::MpiComm> mMpiCommPipelinePara;

    std::shared_ptr<nvinfer1::ILogger> mLogger;
    std::shared_ptr<runtime::TllmRuntime> mRuntime;

    SizeType32 mMicroBatchId;

    // TODO: Add runtime buffers for async PP
    std::vector<std::shared_ptr<EncoderBuffers>> mBuffers;

    SizeType32 mNumMicroBatches;
    SizeType32 mNumBuffers;

    std::vector<ScheduledRequests> mMicroBatchScheduledRequests;
    std::vector<DecoderFinishedEventPtr> mEncoderWaitEvents;
    ReqIdsSet mInflightReqIds;
    ReqIdsSet mReqIdsToPause;

    tensorrt_llm::batch_manager::CapacityScheduler mCapacityScheduler;
    tensorrt_llm::batch_manager::MicroBatchScheduler mMicroBatchScheduler;

    SizeType32 mHiddenSize;  // already divided by Tensor Parallelism
    SizeType32 mMaxInputLen; // WAR for max_input_len == max_seq_len at all circumstances

    runtime::BufferManager mCopyBufferManager;

    // Iteration counter used to distinguish debug output
    executor::IterationType mIterCounter{0};
};

} // namespace tensorrt_llm::batch_manager
