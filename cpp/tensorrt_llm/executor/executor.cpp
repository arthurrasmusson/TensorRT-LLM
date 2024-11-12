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

#include <tensorrt_llm/executor/executor.h>
#include <tensorrt_llm/executor/executorImpl.h>

namespace tensorrt_llm::executor
{

Executor::Executor(std::filesystem::path const& modelPath, ModelType modelType, ExecutorConfig const& executorConfig)
    : mImpl(std::make_unique<Executor::Impl>(modelPath, std::nullopt, modelType, executorConfig))
{
}

Executor::Executor(std::filesystem::path const& encoderModelPath, std::filesystem::path const& decoderModelPath,
    ModelType modelType, ExecutorConfig const& executorConfig)
    : mImpl(std::make_unique<Executor::Impl>(decoderModelPath, encoderModelPath, modelType, executorConfig))
{
}

Executor::Executor(BufferView const& engineBuffer, std::string const& jsonConfigStr, ModelType modelType,
    ExecutorConfig const& executorConfig, std::optional<std::map<std::string, Tensor>> const& managedWeights)
    : mImpl(std::make_unique<Executor::Impl>(
        engineBuffer, jsonConfigStr, std::nullopt, std::nullopt, modelType, executorConfig, managedWeights))
{
}

Executor::Executor(BufferView const& encoderEngineBuffer, std::string const& encoderJsonConfigStr,
    BufferView const& decoderEngineBuffer, std::string const& decoderJsonConfigStr, ModelType modelType,
    ExecutorConfig const& executorConfig)
    : mImpl(std::make_unique<Executor::Impl>(decoderEngineBuffer, decoderJsonConfigStr, encoderEngineBuffer,
        encoderJsonConfigStr, modelType, executorConfig, std::nullopt))
{
}

Executor::Executor(std::shared_ptr<Model> model, ExecutorConfig const& executorConfig)
    : mImpl(std::make_unique<Executor::Impl>(std::move(model), std::nullopt, executorConfig))
{
}

Executor::Executor(
    std::shared_ptr<Model> encoderModel, std::shared_ptr<Model> decoderModel, ExecutorConfig const& executorConfig)
    : mImpl(std::make_unique<Executor::Impl>(std::move(decoderModel), std::move(encoderModel), executorConfig))
{
}

Executor::~Executor() = default;

IdType Executor::enqueueRequest(Request const& llmRequest)
{
    return mImpl->enqueueRequest(llmRequest);
}

std::vector<IdType> Executor::enqueueRequests(std::vector<Request> const& llmRequests)
{
    return mImpl->enqueueRequests(llmRequests);
}

std::vector<Response> Executor::awaitResponses(std::optional<std::chrono::milliseconds> const& timeout)
{
    return mImpl->awaitResponses(timeout);
}

std::vector<Response> Executor::awaitResponses(
    IdType const& requestId, std::optional<std::chrono::milliseconds> const& timeout)
{
    return mImpl->awaitResponses(requestId, timeout);
}

std::vector<std::vector<Response>> Executor::awaitResponses(
    std::vector<IdType> const& requestIds, std::optional<std::chrono::milliseconds> const& timeout)
{
    return mImpl->awaitResponses(requestIds, timeout);
}

SizeType32 Executor::getNumResponsesReady(std::optional<IdType> const& requestId) const
{
    return mImpl->getNumResponsesReady(requestId);
}

void Executor::cancelRequest(IdType requestId)
{
    return mImpl->cancelRequest(requestId);
}

void Executor::shutdown()
{
    return mImpl->shutdown();
}

std::deque<IterationStats> Executor::getLatestIterationStats()
{
    return mImpl->getLatestIterationStats();
}

std::deque<RequestStatsPerIteration> Executor::getLatestRequestStats()
{
    return mImpl->getLatestRequestStats();
}

std::deque<DebugTensorsPerIteration> Executor::getLatestDebugTensors()
{
    return mImpl->getLatestDebugTensors();
}

bool Executor::canEnqueueRequests() const
{
    return mImpl->canEnqueueRequests();
}

bool Executor::isParticipant() const
{
    return mImpl->isParticipant();
}

std::optional<std::shared_ptr<KVCacheEventManager>> Executor::getKVCacheEventManager() const
{
    return mImpl->getKVCacheEventManager();
}

KVCacheEvent::KVCacheEvent(size_t eventId, KVCacheEventData data)
    : eventId{eventId}
    , data{std::move(data)}
{
}

} // namespace tensorrt_llm::executor
