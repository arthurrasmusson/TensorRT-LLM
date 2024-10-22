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

#include "tensorrt_llm/batch_manager/capacityScheduler.h"
#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/microBatchScheduler.h"
#include "tensorrt_llm/runtime/generationInput.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/rawEngine.h"
#include "tensorrt_llm/runtime/worldConfig.h"
#include "trtGptModel.h"

#include <NvInferRuntime.h>

#include <tuple>

namespace tensorrt_llm::runtime
{
class GptSession;
}

namespace tensorrt_llm::batch_manager
{
class LlmRequest;

class [[deprecated("Use the InflightBatching model instead.")]] TrtGptModelV1 : public TrtGptModel
{
    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using TokenIdType = tensorrt_llm::runtime::TokenIdType;
    using TensorPtr = runtime::ITensor::SharedPtr;

public:
    struct IterationStatsV1
    {
        SizeType32 numScheduledRequests;
        SizeType32 numCtxTokensInBatch;
        SizeType32 numGenTokensInBatch;
        SizeType32 emptyGenSlots;
        ReqIdsSet scheduledRequests;
        ReqIdsSet pausedRequests;
    };

    TrtGptModelV1(std::shared_ptr<nvinfer1::ILogger> logger, runtime::ModelConfig const& modelConfig,
        runtime::WorldConfig const& worldConfig, runtime::RawEngine const& rawEngine,
        TrtGptModelOptionalParams const& optionalParams = TrtGptModelOptionalParams());

    // V1 model is stateless, so nothing to do here
    void terminateRequest(std::shared_ptr<LlmRequest> const& llmRequest, bool pause = false) override{};

    /// @brief This override is empty and solely exists to adhere to the interface
    void forwardSync() override;

    /// @brief Function that tries to advance the active requests
    ///        Depending on resources available, it's possible that not all requests will get advanced
    /// @param activeRequests The list of request to try to advance
    void forwardAsync(RequestList const& activeRequests) override;

    //! @brief Set LayerProfiler to collect performance per layer.
    void setLayerProfiler() override;

    //! @brief Print profile information per layer.
    std::string getLayerProfileInfo() const override;

    void updatePeftCache(std::shared_ptr<LlmRequest> const& llmRequest) override {}

    [[nodiscard]] runtime::ModelConfig const& getModelConfig() const override;

    [[nodiscard]] TrtGptModelType getModelType() const override
    {
        return TrtGptModelType::V1;
    };

    [[nodiscard]] SizeType32 getNumMicroBatches() const override;
    [[nodiscard]] runtime::WorldConfig const& getWorldConfig() const override;
    [[nodiscard]] IterationStatsV1 getLastIterationStats() const;
    [[nodiscard]] runtime::BufferManager const& getBufferManager() const override;
    [[nodiscard]] runtime::BufferManager::CudaStreamPtr getRuntimeStreamPtr() const override;
    [[nodiscard]] nvinfer1::DataType getLogitDataType() const override;

    [[nodiscard]] executor::IterationType getIterCounter() const noexcept override
    {
        return mIterCounter;
    }

    void getCurrentIterationStats(executor::IterationStats& stats) const override;
    void getCurrentRequestStats(executor::RequestStatsPerIteration& stats) const override;
    [[nodiscard]] executor::DebugTensorsPerIteration getCurrentDebugTensors() const override;

    void setLogitsPostProcessorBatched(std::optional<LogitsPostProcessorBatched> logitsPostProcessorBatched) override;
    void setReplicateLogitsPostProcessor(bool replicateLogitsPostProcessor) override;

    void resetIterationStats() override;

protected:
    [[nodiscard]] std::shared_ptr<kv_cache_manager::KVCacheManager> getKVCacheManager() override;
    [[nodiscard]] std::shared_ptr<kv_cache_manager::KVCacheManager const> getKVCacheManager() const override;

    [[nodiscard]] std::shared_ptr<BasePeftCacheManager> getPeftCacheManager() override
    {
        return mPeftCacheManager;
    }

    [[nodiscard]] std::shared_ptr<BasePeftCacheManager const> getPeftCacheManager() const override
    {
        return mPeftCacheManager;
    }

private:
    // callback stats
    static IterationStatsV1 fillIterationStats(RequestVector const& scheduledRequests, SizeType32 cappedMaxNewTokens,
        RequestVector const& requestsToTerminate);

    // Helper function to fill the generation table and batch sampling config from scheduled requests
    static std::tuple<runtime::GenerationInput, runtime::SamplingConfig> fillGenInputAndSamplingConfig(
        RequestVector const& scheduledRequests, runtime::BufferManager const& bufferManager,
        runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig, SizeType32 maxSeqLen,
        SizeType32 maxBatchSize, bool normalizeLogProbs);

    std::shared_ptr<runtime::GptSession> mSession;
    tensorrt_llm::batch_manager::CapacityScheduler mCapacityScheduler;
    tensorrt_llm::batch_manager::MicroBatchScheduler mMicroBatchScheduler;
    std::shared_ptr<BasePeftCacheManager> mPeftCacheManager;
    IterationStatsV1 mLastIterationStatsV1;
    // Iteration counter used to distinguish debug output
    executor::IterationType mIterCounter{0};
};

} // namespace tensorrt_llm::batch_manager
