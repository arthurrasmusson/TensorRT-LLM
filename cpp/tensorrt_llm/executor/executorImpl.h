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

#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/common/arrayView.h"
#include "tensorrt_llm/common/mpiUtils.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/intervalSet.h"
#include "tensorrt_llm/executor/model.h"
#include "tensorrt_llm/executor/orchestratorUtils.h"
#include "tensorrt_llm/executor/requestWithId.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/runtime/gptJsonConfig.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/rawEngine.h"
#include "tensorrt_llm/runtime/worldConfig.h"

#include <atomic>
#include <condition_variable>
#include <list>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace tensorrt_llm::executor
{

class RequestWithIdAsyncSend;
class CancelledRequestsAsyncSend;

std::vector<RequestWithId> requestWithIdRecv(std::shared_ptr<tensorrt_llm::mpi::MpiComm> commSession, int const peer);

class MpiMessageQueue
{
public:
    void push(MpiMessage&& message)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        mQueue.push(std::move(message));
        mCv.notify_one();
    }

    MpiMessage pop()
    {
        std::unique_lock<std::mutex> lock(mMutex);
        mCv.wait(lock, [this] { return !mQueue.empty(); });
        MpiMessage message = std::move(mQueue.front());
        mQueue.pop();
        return message;
    }

private:
    std::queue<MpiMessage> mQueue;
    std::mutex mMutex;
    std::condition_variable mCv;
};

class Executor::Impl

{
    using LlmRequestPtr = std::shared_ptr<batch_manager::LlmRequest>;
    using RequestList = std::list<LlmRequestPtr>;

public:
    Impl(std::filesystem::path const& modelPath, std::optional<std::filesystem::path> const& encoderModelPath,
        [[maybe_unused]] ModelType modelType, ExecutorConfig const& executorConfig);

    Impl(std::vector<uint8_t> const& engineBuffer, std::string const& jsonConfigStr,
        std::optional<std::vector<uint8_t>> const& encoderEngineBuffer,
        std::optional<std::string> const& encoderJsonConfigStr, [[maybe_unused]] ModelType modelType,
        ExecutorConfig const& executorConfig);

    Impl(std::shared_ptr<Model> model, std::optional<std::shared_ptr<Model>> encoderModel,
        ExecutorConfig const& executorConfig);

    ~Impl()
    {
        shutdown();
    }

    IdType enqueueRequest(Request const& request);

    std::vector<IdType> enqueueRequests(std::vector<Request> const& requests);

    std::vector<IdType> enqueueRequests(common::ArrayView<Request const> const& requests);

    std::vector<Response> awaitResponses(std::optional<std::chrono::milliseconds> const& timeout = std::nullopt);

    std::vector<Response> awaitResponses(
        IdType const& optId, std::optional<std::chrono::milliseconds> const& optTimeout = std::nullopt);

    std::vector<std::vector<Response>> awaitResponses(
        std::vector<IdType> const& optId, std::optional<std::chrono::milliseconds> const& optTimeout = std::nullopt);

    SizeType32 getNumResponsesReady(std::optional<IdType> const& optId = std::nullopt) const;

    void cancelRequest(IdType requestId);

    void shutdown();

    std::deque<IterationStats> getLatestIterationStats();

    std::deque<RequestStatsPerIteration> getLatestRequestStats();

    bool canEnqueueRequests() const;

private:
    using RtTensorPtr = runtime::ITensor::SharedPtr;
    using CudaStreamPtr = runtime::BufferManager::CudaStreamPtr;
    using LlmRequestLogitsPostProcessor
        = std::function<void(IdType, RtTensorPtr&, BeamTokens const&, CudaStreamPtr, std::optional<IdType>)>;

    void initialize(ExecutorConfig const& executorConfig);

    void loadModel(std::optional<std::filesystem::path> const& modelPath,
        std::optional<std::vector<uint8_t>> const& engineBuffer, runtime::GptJsonConfig const& jsonConfig,
        ExecutorConfig const& executorConfig, bool isEncoder);

    static std::shared_ptr<Model> createModel(runtime::RawEngine const& rawEngine,
        runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig,
        ExecutorConfig const& executorConfig);

    static std::shared_ptr<Model> createEncoderModel(runtime::RawEngine const& rawEngine,
        runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig,
        ExecutorConfig const& executorConfig);

    void setOrchLeaderComm(SizeType32 tp, SizeType32 pp, ParallelConfig const& parallelConfig);

    void initializeCommAndWorkers(SizeType32 tp, SizeType32 pp, ExecutorConfig const& executorConfig,
        std::optional<ModelType> modelType = std::nullopt,
        std::optional<std::filesystem::path> const& modelPath = std::nullopt);

    static void validateParallelConfig(ParallelConfig const& parallelConfig, std::optional<ModelType> modelType,
        std::optional<std::filesystem::path> const& modelPath);

    void initializeOrchestrator(SizeType32 tp, SizeType32 pp, ExecutorConfig const& executorConfig,
        ParallelConfig parallelConfig, ModelType modelType, std::filesystem::path const& modelPath);

    void initializeWorkers(SizeType32 tp, SizeType32 pp, ParallelConfig& parallelConfig);

    void initializeLogitsPostProcessorBatched(LogitsPostProcessorConfig const& logitsProcConfig);

    IdType generateReqId()
    {
        return (mLastReqId++ % UINT64_MAX);
    }

    std::vector<RequestWithId> getNewReqWithIds(
        SizeType32 numActiveRequests, std::optional<PriorityType> lowestPriorityActive);

    RequestList fetchNewRequests(SizeType32 numActiveRequests, std::optional<PriorityType> lowestPriorityActive);

    void forwardSync(RequestList& activeRequests);

    void forwardAsync(RequestList& activeRequests);

    void terminateActiveRequests(RequestList& activeRequests, std::string const& err);

    IterationStats getCurrentIterationStats(
        RequestList const& activeRequests, IterationType iterCounter, double iterLatencyMS);

    void appendCurrentIterStats(IterationStats&& currentIterStats);

    void updateIterationStats(RequestList const& activeRequests, IterationType iterCounter, double iterLatencyMS);

    RequestStatsPerIteration getCurrentRequestStats(
        RequestList const& activeRequests, IterationType iterCounter, RequestList const& finishedRequests);

    void updateRequestStats(
        RequestList const& activeRequests, IterationType iterCounter, RequestList const& finishedRequests);

    void terminateCancelledRequests(RequestList& activeRequests);

    void appendNewResponses(std::vector<Response>&& newResponses);

    /// @brief Populates new responses from active requests.
    ///        Active requests that have completed are erased from activeRequests
    ///        and returned for bookkeeping.
    /// @return A list of requests that have completed.
    RequestList populateNewResponses(RequestList& activeRequests);

    void executionLoop();

    void enqueueTerminateRequest();
    void enqueueNewResponses(std::vector<Response>&& newResponses);

    LlmRequestLogitsPostProcessor getLogitsPostProcessor(std::string const& name);

    void orchSendReqThread();
    void orchRecvThread(int32_t idTag, int32_t dataTag);
    void leaderRecvReqThread();
    void leaderSendThread(MpiMessageQueue& sendQueue, int32_t idTag, int32_t dataTag);

    void addTerminatedReqId(std::vector<Response> const& responses, IdType const& reqId);

    // The model to execute
    std::shared_ptr<Model> mModel = nullptr;
    std::shared_ptr<Model> mEncoderModel = nullptr;

    // The maximum number of activeRequests
    SizeType32 mMaxNumActiveRequests;

    // Thread the executes the main loop
    std::thread mExecutionThread;

    // Atomic that indicates threads should shutdown
    std::atomic<bool> mShutdown;

    // Atomic that indicates if shutdown method has been called
    std::atomic<bool> mShutdownCalled = false;

    // Queued requests
    std::mutex mQueuedReqMtx;
    std::condition_variable mQueuedReqCv;
    std::deque<RequestWithId> mQueuedRequests;
    std::optional<SizeType32> mMaxQueueSize;

    // Cancelled requests
    std::mutex mCancelReqMtx;
    std::unordered_set<IdType> mCancelledReqIds;

    // Ready responses
    std::unordered_map<IdType, std::vector<Response>> mResponses;
    mutable std::mutex mResponsesMtx;
    std::condition_variable mResponsesCv;

    // Since the request IDs are generated sequentially, IntervalSet is preferred over unordered_set for its efficient
    // memory usage to stores request ID intervals rather than individual request ID numbers.
    IntervalSet<IdType> mTerminatedReqIds;

    // Iteration stats
    IterationType mIterStatsMaxIterations;
    std::mutex mIterStatsMtx;
    std::deque<IterationStats> mIterationStats;

    // Request stats
    IterationType mRequestStatsMaxIterations;
    std::mutex mRequestStatsMtx;
    std::deque<RequestStatsPerIteration> mRequestStats;

    IdType mLastReqId = 1;

    static constexpr IdType mTerminateReqId = 0;

    BatchingType mBatchingType;
    bool mIsSchedulerGuaranteedNoEvict;
    bool mIsChunkedContext;

    CommunicationMode mCommMode;
    bool mIsWorker = false;
    bool mIsLeader = false;

    std::unordered_map<std::string, LogitsPostProcessor> mLogitsPostProcessorMap;
    std::optional<Model::LogitsPostProcessorBatched> mLogitsPostProcessorBatched;
    bool mReplicateLogitsPostProcessor;

    bool mIsOrchestrator = false;
    std::shared_ptr<tensorrt_llm::mpi::MpiComm> mOrchLeaderComm;

    std::thread mOrchSendReqThread;
    std::thread mOrchRecvThread;
    std::thread mLeaderRecvReqThread;
    std::thread mLeaderSendThread;

    int32_t mLeaderRank = -1;
    int32_t mOrchRank = 0;

    MpiMessageQueue mSendQueue;

    std::shared_ptr<tensorrt_llm::mpi::MpiComm> mCommTensorParallel;
    std::shared_ptr<tensorrt_llm::mpi::MpiComm> mCommPipelineParallel;
    std::unique_ptr<std::thread> mRequestWithIdWaitThread;
    std::unique_ptr<std::thread> mCancelledRequestsWaitThread;

    inline static std::string const kPROFILE_START_STOP_ENV_VAR_NAME = "TLLM_PROFILE_START_STOP";
    inline static std::string const kLEGACY_PROFILE_START_STOP_ENV_VAR_NAME = "TLLM_GPTM_PROFILE_START_STOP";
};

} // namespace tensorrt_llm::executor
