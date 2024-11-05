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

#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/worldConfig.h"

#include <nlohmann/json.hpp>

namespace tensorrt_llm::executor
{

class Model
{
    using LlmRequestPtr = std::shared_ptr<batch_manager::LlmRequest>;

public:
    Model() = default;

    virtual ~Model() = default;

    /// @brief Function that marks a request Id as complete and cleans up associated state
    virtual void terminateRequest(LlmRequestPtr const& llmRequest, bool pause) = 0;

    void terminateRequest(LlmRequestPtr const& llmRequest)
    {
        terminateRequest(llmRequest, false);
    }

    /// @brief Function that synchronizes the decoder
    virtual void forwardSync() = 0;

    /// @brief Function that tries to advance the active requests
    ///        Depending on resources available, it's possible that not all requests will get advanced
    /// @param activeRequests The list of request to try to advance
    virtual void forwardAsync(batch_manager::RequestList const& activeRequests) = 0;

    /// @brief Override the runtime batch size for the model
    virtual void setRuntimeBatchSize(SizeType32 runtimeBatchSize)
    {
        // By default, we ignore the runtimeBatchSize unless the model actively supports it
    }

    /// @brieft Override the runtime max num tokens for the model
    virtual void setRuntimeMaxNumTokens(SizeType32 runtimeMaxNumTokens)
    {
        // By default, we ignore the runtimeMaxNumTokens unless the model actively supports it
    }

    virtual void updatePeftCache(LlmRequestPtr const& llmRequest) = 0;

    /// @brief Reset the iteration stats when there are no inflight requests
    virtual void resetIterationStats() = 0;

    [[nodiscard]] virtual SizeType32 getMaxNumSequences() const = 0;
    [[nodiscard]] virtual SizeType32 getMaxInputLen() const = 0;
    [[nodiscard]] virtual SizeType32 getHiddenSize() const = 0;
    [[nodiscard]] virtual SizeType32 getMaxSequenceLen() const = 0;
    [[nodiscard]] virtual SizeType32 getVocabSizePadded() const = 0;
    [[nodiscard]] virtual SizeType32 getMaxDraftLen() const = 0;
    [[nodiscard]] virtual SizeType32 getNumMicroBatches() const = 0;
    [[nodiscard]] virtual nvinfer1::DataType getLogitDataType() const = 0;
    [[nodiscard]] virtual runtime::WorldConfig const& getWorldConfig() const = 0;
    [[nodiscard]] virtual runtime::ModelConfig const& getModelConfig() const = 0;
    [[nodiscard]] virtual runtime::BufferManager const& getBufferManager() const = 0;
    [[nodiscard]] virtual runtime::BufferManager::CudaStreamPtr getRuntimeStreamPtr() const = 0;
    [[nodiscard]] virtual IterationType getIterCounter() const noexcept = 0;
    [[nodiscard]] virtual bool hasSpeculativeDecodingFastLogits() const noexcept = 0;

    /// @brief Function that provides per iteration stats specific to a certain model
    /// @param stats The json object to write stats to
    virtual void getCurrentIterationStats(IterationStats& stats) const = 0;

    /// @brief Function that provides per request stats specific to a certain model
    /// @param stats The request stats to be updated
    virtual void getCurrentRequestStats(RequestStatsPerIteration& stats) const = 0;

    [[nodiscard]] virtual DebugTensorsPerIteration getCurrentDebugTensors() const = 0;

    using LogitsPostProcessorBatched = std::function<void(std::vector<batch_manager::LlmRequest::RequestIdType> const&,
        std::vector<batch_manager::LlmRequest::TensorPtr>&,
        std::vector<std::reference_wrapper<batch_manager::LlmRequest::BeamTokens const>> const&,
        runtime::BufferManager::CudaStreamPtr const&,
        std::vector<std::optional<batch_manager::LlmRequest::RequestIdType>> const&)>;

    virtual void setLogitsPostProcessorBatched(std::optional<LogitsPostProcessorBatched> logitsPostProcessorBatched)
        = 0;
    virtual void setReplicateLogitsPostProcessor(bool replicateLogitsPostProcessor) = 0;

    //! \brief Get the batch size that can fill the kv cache to the maximum capacity give the sequence length
    //! \param seqLen The sequence length
    //! \return The batch size that can fill the kv cache to the maximum capacity. If unsuporrted, return 0.
    [[nodiscard]] virtual SizeType32 getMaxCapacityBatchSize(SizeType32 seqLen)
    {
        return 0;
    }
};

} // namespace tensorrt_llm::executor
