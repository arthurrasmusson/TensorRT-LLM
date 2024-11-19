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

#ifndef TOP_LEVEL_DIR
#error "Define TOP_LEVEL_DIR"
#endif
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "modelSpec.h"
#include "tensorrt_llm/batch_manager/trtGptModel.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/mpiUtils.h"
#include "tensorrt_llm/common/utils.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/requestWithId.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/executor/version.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"
#include "tensorrt_llm/runtime/gptJsonConfig.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/tllmLogger.h"
#include "tensorrt_llm/runtime/utils/numpyUtils.h"
#include "tests/utils/common.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <functional>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

namespace tr = tensorrt_llm::runtime;
namespace tb = tensorrt_llm::batch_manager;
namespace tc = tensorrt_llm::common;

using namespace tensorrt_llm::testing;
using namespace tensorrt_llm::executor;

namespace tensorrt_llm::testing::executor::disaggexecutor
{

constexpr int32_t kM_INSTANCE_ID_TAG{12024};
constexpr int32_t kM_CONTROLLER_ID_TAG{22024};
constexpr int32_t kM_INSTANCE_DATA_TAG{32024};
constexpr int32_t kM_CONTROLLER_DATA_TAG{42024};

enum class MessageID : uint64_t
{
    PENDING_CONTEXT_REQUEST = 1,
    PENDING_GENERATION_REQUEST = 2,
    CONTEXT_RESPONSE = 3,
    GENERATION_RESPONSE = 4,

    TERMINATION = 5,
};

struct RequestsData
{
    std::vector<RequestWithId> requests;
};

struct ResponseWithId
{

    tensorrt_llm::executor::Response response;
    IdType gid;

    ResponseWithId(tensorrt_llm::executor::Response&& response, IdType gid)
        : response(std::move(response))
        , gid(gid)
    {
    }

    ResponseWithId(tensorrt_llm::executor::Response const& response, IdType gid)
        : response(response)
        , gid(gid)
    {
    }

    ResponseWithId(ResponseWithId&& other) noexcept
        : response(std::move(other.response))
        , gid(other.gid)
    {
        other.gid = {};
    }

    ResponseWithId(ResponseWithId const& other) = default;

    ResponseWithId& operator=(ResponseWithId&& other) noexcept
    {
        if (this != &other)
        {
            response = std::move(other.response);
            gid = other.gid;
            other.gid = {};
        }
        return *this;
    }

    ResponseWithId& operator=(ResponseWithId const& other)
    {

        if (this != &other)
        {
            response = other.response;
            gid = other.gid;
        }
        return *this;
    }

    ~ResponseWithId() = default;
};

static std::vector<char> serializeResponseWithIds(std::vector<ResponseWithId> const& responseWithIds)
{
    size_t totalSize = 0;
    totalSize += sizeof(size_t);
    for (auto const& responseWithId : responseWithIds)
    {
        totalSize += su::serializedSize(responseWithId.gid);
        totalSize += su::serializedSize(responseWithId.response);
    }

    std::vector<char> buffer(totalSize);
    std::stringbuf strbuf{std::ios_base::out | std::ios_base::in};
    strbuf.pubsetbuf(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    std::ostream ostream{&strbuf};

    su::serialize(responseWithIds.size(), ostream);
    for (auto const& responseWithId : responseWithIds)
    {
        su::serialize(responseWithId.gid, ostream);
        su::serialize(responseWithId.response, ostream);
    }
    return buffer;
}

static std::vector<ResponseWithId> deserializeResponseWithIds(std::vector<char>& buffer)
{
    std::vector<ResponseWithId> responseWithIds;
    su::VectorWrapBuf<char> strbuf{buffer};
    std::istream istream{&strbuf};
    auto numReq = su::deserialize<std::int64_t>(istream);
    for (int64_t req = 0; req < numReq; ++req)
    {
        auto const id = su::deserialize<std::uint64_t>(istream);
        responseWithIds.emplace_back(ResponseWithId{Serialization::deserializeResponse(istream), id});
    }
    return responseWithIds;
}

struct ResponsesData
{
    std::vector<ResponseWithId> response;
};

using MessageData = std::variant<RequestsData, ResponsesData>;

struct Message
{

    MessageID id;

    MessageData data;
};

class MessageQueue
{
public:
    void push(Message&& message)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        mQueue.push(std::move(message));
        mCv.notify_one();
    }

    Message pop()
    {
        std::unique_lock<std::mutex> lock(mMutex);
        mCv.wait(lock, [this] { return !mQueue.empty(); });
        Message message = std::move(mQueue.front());
        mQueue.pop();
        return message;
    }

private:
    std::queue<Message> mQueue;
    std::mutex mMutex;
    std::condition_variable mCv;
};

class DisaggExecutorLeader
{
public:
    DisaggExecutorLeader(std::filesystem::path const& modelPath, ModelType modelType,
        ExecutorConfig const& executorConfig, bool isController, bool isContext, int numRequests,
        std::vector<int>& participatIds, std::vector<int> const& participantDeviceIdsThisInstance, int worldRank)
        : mIsContext(isContext)
        , mNumRequests(numRequests)
        , mWorldRanksInstances(participatIds)
        , mDeviceIdsThisInstance(participantDeviceIdsThisInstance)
        , mWorldRank(worldRank)
        , mShutdown(false)
        , mWorldComm(tensorrt_llm::mpi::MpiComm::world())

    {

#if ENABLE_MULTI_DEVICE

        auto world_size = mWorldComm.getSize();
        mRolesPerRank.resize(world_size);
        mIsLeaderInstance = false;
        if (!mWorldRanksInstances.empty())
        {
            mIsLeaderInstance = mWorldRank == mWorldRanksInstances.front();
        };

        // bool needExecutor = (!mIsController) || (mIsController && mIsLeaderInstance);
        bool needExecutor = (std::find(mWorldRanksInstances.begin(), mWorldRanksInstances.end(), worldRank)
            != mWorldRanksInstances.end());
        if (needExecutor)
        {
            ExecutorConfig executorConfigC = executorConfig;

            auto parallelConfig = executorConfigC.getParallelConfig().value_or(ParallelConfig{});
            std::vector<int> participantIds = mWorldRanksInstances;

            parallelConfig.setParticipantIds(participantIds);
            TLLM_CHECK(parallelConfig.getCommunicationMode() == tensorrt_llm::executor::CommunicationMode::kLEADER);
            parallelConfig.setCommunicationType(tensorrt_llm::executor::CommunicationType::kMPI);
            parallelConfig.setDeviceIds(mDeviceIdsThisInstance);
            executorConfigC.setParallelConfig(parallelConfig);

            mExecutor = std::make_unique<Executor>(modelPath, modelType, executorConfigC);
            // mIsLeaderInstance = (COMM_SESSION.getRank() == 0);
        }

        mIsController = false;
        uint32_t role = 0;
        if (mIsLeaderInstance)
        {
            role |= 0b001;
        }
        if (mIsContext)
        {
            role |= 0b010;
        }
        if (isController)
        {
            mIsController = true;
            role |= 0b100;
        }

        TLLM_CHECK(mWorldRanksInstances.size() == mDeviceIdsThisInstance.size());

        mWorldComm.allgather(&role, mRolesPerRank.data(), 1, tensorrt_llm::mpi::MpiType::kUINT32);

        generatedRoles();

        if (mIsController)
        {
            mControllerSendThread = std::thread(&DisaggExecutorLeader::ControllerSendThread, this);
            mControllerRecvThread = std::thread(&DisaggExecutorLeader::ControllerRecvThread, this);
        }
        if (mIsLeaderInstance)
        {
            mInstanceRecvThread = std::thread(&DisaggExecutorLeader::InstanceLeaderRecvThread, this);
            mInstanceSendThread = std::thread(&DisaggExecutorLeader::InstanceLeaderSendThread, this);
            mInstanceLoopThread = std::thread(&DisaggExecutorLeader::InstanceLeaderLoopThread, this);
        }
#else
        TLLM_THROW("DisaggExecutor only support being compiled with ENABLE_MULTI_DEVICE");

#endif
    }

    bool isController() const
    {
        return mIsController;
    }

    std::vector<IdType> enqueueRequests(std::vector<Request> const& llmRequests)

    {
        if (!mIsController)
            return {};

        std::vector<RequestWithId> requestWithIds;
        std::vector<IdType> reqIds;
        for (auto req : llmRequests)
        {
            IdType id = generatedControlId();
            reqIds.push_back(id);

            RequestWithId reqWithId{req, id};
            reqWithId.req.setRequestType(RequestType::REQUEST_TYPE_CONTEXT_ONLY);

            requestWithIds.push_back(std::move(reqWithId));

            mRequestMap.insert(std::make_pair(id, req));
        }

        Message message{MessageID::PENDING_CONTEXT_REQUEST, MessageData{RequestsData{requestWithIds}}};

        mControllerSendQueue.push(std::move(message));

        return reqIds;
    }

    std::vector<Response> awaitResponses(std::optional<std::chrono::milliseconds> const& timeout)
    {
        // wait for responseQueue , modify reqid-
        std::vector<Response> responses;
        std::unique_lock<std::mutex> lck(mResponsesMtx);
        auto pred = [&mShutdown = mShutdown, &resp = this->mResponses]() -> bool { return !resp.empty() || mShutdown; };
        auto storeResponses = [this, &resp = this->mResponses, &responses]()
        {
            for (auto it = resp.cbegin(); it != resp.cend();)
            {
                responses.insert(responses.end(), it->second.begin(), it->second.end());
                resp.erase(it++);
            }
        };

        if (timeout)
        {
            if (mResponsesCv.wait_for(lck, timeout.value(), pred))
            {
                storeResponses();
            }
        }
        else
        {
            mResponsesCv.wait(lck, pred);
            storeResponses();
        }
        return responses;
    }

    std::deque<RequestStatsPerIteration> getLatestRequestStats()
    {
        if (mExecutor && mExecutor->canEnqueueRequests())
        {
            return mExecutor->getLatestRequestStats();
        }
        return {};
    }

    bool isContextRank()
    {
        return mIsContext;
    }

    bool isGenerationRank()
    {
        return !mIsContext;
    }

    void shutDown()
    {
        if (mShutdown)
        {
            return;
        }

        if (mIsController)
        {
            std::call_once(mHasSendTerminFlag,
                [&]()
                {
                    MessageID terminationMessage = MessageID::TERMINATION;
                    for (auto&& leaderRanks : {mContextLeaderRanks, mGenerationLeaderRanks})
                    {
                        for (auto&& leaderRank : leaderRanks)
                        {

                            mWorldComm.send(&terminationMessage, 1, tensorrt_llm::mpi::MpiType::kUINT64, leaderRank,
                                kM_CONTROLLER_ID_TAG);
                        }
                    }

                    mWorldComm.send(&terminationMessage, 1, tensorrt_llm::mpi::MpiType::kUINT64, mControllerRank,
                        kM_INSTANCE_ID_TAG);
                });
            // end recv thread;
        }
        mShutdown = true;

        // end send thread
        if (mIsController)
        {
            mControllerSendQueue.push({MessageID::TERMINATION, {}});
        }
        mInstanceSendQueue.push({MessageID::TERMINATION, {}});
    }

    ~DisaggExecutorLeader()
    {

        if (mIsController)

            shutDown();

        if (mIsLeaderInstance)
        {
            if (mInstanceSendThread.joinable())
            {
                mInstanceSendThread.join();
            }
            if (mInstanceRecvThread.joinable())
            {
                mInstanceRecvThread.join();
            }
            if (mInstanceLoopThread.joinable())
            {
                mInstanceLoopThread.join();
            }
        }

        if (mIsController)
        {
            if (mControllerSendThread.joinable())
            {
                mControllerSendThread.join();
            }
            if (mControllerRecvThread.joinable())
            {
                mControllerRecvThread.join();
            }
        }

        if (!mIsController)
        {
            mExecutor->shutdown();
        }
        if (mIsController && mIsLeaderInstance)
        {
            mExecutor->shutdown();
        }

        shutDown();
    }

private:
    bool mIsContext;
    tensorrt_llm::mpi::MpiComm const& mWorldComm;
    std::unique_ptr<Executor> mExecutor;
    std::thread mInstanceSendThread;
    std::thread mInstanceRecvThread;
    std::thread mInstanceLoopThread;
    std::thread mControllerSendThread;
    std::thread mControllerRecvThread;
    int mNumRequests;
    std::map<std::uint64_t, Request> mRequestMap;
    std::map<IdType, DataTransceiverState> mGenIdToContextPhase;
    std::unordered_map<IdType, IdType> mInstanceIdToGlobalId;
    std::mutex mIdToGlbalMutex;

    std::vector<int> mWorldRanksInstances;

    int mWorldRank;
    int mControllerRank = 0;
    bool mIsController;
    bool mIsLeaderInstance;
    std::vector<uint32_t> mRolesPerRank;
    std::vector<int> mContextLeaderRanks;
    std::vector<int> mGenerationLeaderRanks;

    IdType mLastId = 1;
    MessageQueue mControllerSendQueue;
    MessageQueue mInstanceSendQueue;

    std::atomic<bool> mShutdown;

    // Ready responses
    std::unordered_map<IdType, std::vector<Response>> mResponses;
    mutable std::mutex mResponsesMtx;
    std::condition_variable mResponsesCv;

    std::vector<int> mDeviceIdsThisInstance;
    std::once_flag mHasSendTerminFlag;

    void appendNewResponses(std::vector<ResponseWithId>&& newResponses)
    {
        {
            std::scoped_lock<std::mutex> lck(mResponsesMtx);
            for (auto& responseWithId : newResponses)
            {
                // global id to Result
                responseWithId.response = Response(responseWithId.gid, responseWithId.response.getResult());

                mResponses[responseWithId.gid].emplace_back(std::move(responseWithId.response));
            }
        }
        mResponsesCv.notify_all();
    }

    void generatedRoles()
    {
        int contextNum = 0;
        int genrationNum = 0;
        int controllerNum = 0;
        for (int rank = 0; rank < mRolesPerRank.size(); rank++)
        {
            uint32_t role = mRolesPerRank[rank];
            if ((role & 0b001) != 0u)
            {
                if ((role & 0b010) != 0u)
                {
                    contextNum++;
                    mContextLeaderRanks.push_back(rank);
                }
                else
                {
                    genrationNum++;
                    mGenerationLeaderRanks.push_back(rank);
                }
            }
            if ((role & 0b100) != 0u)
            {
                controllerNum++;
                mControllerRank = rank;
            }
        }
        TLLM_CHECK_WITH_INFO(controllerNum == 1, "only one rank is controller but get %d controllerNum", controllerNum);
    }

    IdType generatedControlId()
    {
        return (mLastId++ % UINT64_MAX);
    }

    int selectContextLeaderRank()
    {
        static int leaderRank = 0;
        leaderRank = (leaderRank + 1) % mContextLeaderRanks.size();
        return mContextLeaderRanks[leaderRank];
    }

    int selectGenerationLeaderRank()
    {

        // TODO: for same reqId , need select specific generationLeader
        static int leaderRank = 0;
        leaderRank = (leaderRank + 1) % mGenerationLeaderRanks.size();
        return mGenerationLeaderRanks[leaderRank];
    }

    void ControllerSendThread()
    {
        // send request to context reqid
        // and send context pahse to generation

        TLLM_CUDA_CHECK(
            cudaSetDevice(mDeviceIdsThisInstance.at(COMM_SESSION.getRank() % (mDeviceIdsThisInstance.size()))));
        tensorrt_llm::common::setThreadName("ControllerSendThread");

        while (!mShutdown)
        {
            auto message = mControllerSendQueue.pop();
            if (message.id == MessageID::TERMINATION)
            {

                TLLM_LOG_DEBUG("controller get terminiation message in sendQueue");
                break;
            }
            else if (message.id == MessageID::PENDING_CONTEXT_REQUEST)
            {

                auto& reqWithIds = std::get<RequestsData>(message.data);
                auto packed = RequestWithId::serializeReqWithIds(reqWithIds.requests);
                int contextRank = selectContextLeaderRank();

                mWorldComm.send(&message.id, 1, tensorrt_llm::mpi::MpiType::kUINT64, contextRank, kM_CONTROLLER_ID_TAG);

                mWorldComm.send(packed.data(), packed.size(), tensorrt_llm::mpi::MpiType::kCHAR, contextRank,
                    kM_CONTROLLER_DATA_TAG);
            }
            else if (message.id == MessageID::PENDING_GENERATION_REQUEST)
            {

                auto& reqWithIds = std::get<RequestsData>(message.data);
                auto packed = RequestWithId::serializeReqWithIds(reqWithIds.requests);
                int generationRank = selectGenerationLeaderRank();

                mWorldComm.send(
                    &message.id, 1, tensorrt_llm::mpi::MpiType::kUINT64, generationRank, kM_CONTROLLER_ID_TAG);

                mWorldComm.send(packed.data(), packed.size(), tensorrt_llm::mpi::MpiType::kCHAR, generationRank,
                    kM_CONTROLLER_DATA_TAG);
            }
            else
            {
                TLLM_THROW("rank:%d, size:%d controller send Invalid message id:%ld", mWorldComm.getRank(),
                    mWorldComm.getSize(), static_cast<uint64_t>(message.id));
            }
        }
    }

    void ControllerRecvThread()
    {
#if ENABLE_MULTI_DEVICE
        tensorrt_llm::common::setThreadName("ControllerRecvThread");

        // recv response from context and push to sendQueue
        // recv response from generation and push to responseQueue and notify awaitResponse
        TLLM_CUDA_CHECK(
            cudaSetDevice(mDeviceIdsThisInstance.at(COMM_SESSION.getRank() % (mDeviceIdsThisInstance.size()))));

        while (!mShutdown)
        {

            MPI_Message msg;
            MPI_Status status;

            mWorldComm.mprobe(MPI_ANY_SOURCE, kM_INSTANCE_ID_TAG, &msg, &status);

            auto sourceRank{status.MPI_SOURCE};
            int32_t count;
            MPICHECK(MPI_Get_count(&status, MPI_UINT64_T, &count));
            TLLM_CHECK(count == 1);

            MessageID messageId;
            MPICHECK(MPI_Mrecv(&messageId, count, MPI_UINT64_T, &msg, &status));

            if (messageId == MessageID::TERMINATION)
            {
                TLLM_LOG_DEBUG("controller received termination message***************\n");
                break;
            }
            else if (messageId == MessageID::CONTEXT_RESPONSE)
            {
                mWorldComm.mprobe(sourceRank, kM_INSTANCE_DATA_TAG, &msg, &status);
                MPICHECK(MPI_Get_count(&status, MPI_CHAR, &count));
                std::vector<char> buffer(count);
                MPICHECK(MPI_Mrecv(buffer.data(), count, MPI_CHAR, &msg, &status));
                auto responseWithIds = deserializeResponseWithIds(buffer);
                //  enqueueTo sendQueue like enqueuRequest. . modify requestType and set ContextPhaseParams
                //  and push to sendQueue.
                std::vector<RequestWithId> requestWithIds;
                for (auto&& responseWithId : responseWithIds)
                {
                    auto reqId = responseWithId.gid;
                    auto& request = mRequestMap.at(reqId);

                    request.setRequestType(RequestType::REQUEST_TYPE_GENERATION_ONLY);
                    request.setContextPhaseParams(responseWithId.response.getResult().contextPhaseParams.value());
                    requestWithIds.push_back(RequestWithId{request, reqId});
                }
                mControllerSendQueue.push({MessageID::PENDING_GENERATION_REQUEST, RequestsData{requestWithIds}});
            }

            else if (messageId == MessageID::GENERATION_RESPONSE)
            {

                mWorldComm.mprobe(sourceRank, kM_INSTANCE_DATA_TAG, &msg, &status);
                MPICHECK(MPI_Get_count(&status, MPI_CHAR, &count));
                std::vector<char> buffer(count);
                MPICHECK(MPI_Mrecv(buffer.data(), count, MPI_CHAR, &msg, &status));

                auto responseWithIds = deserializeResponseWithIds(buffer);
                appendNewResponses(std::move(responseWithIds));
            }
            else
            {
                TLLM_THROW("rank:%d, size:%d controller recv Invalid message id:%ld", mWorldComm.getRank(),
                    mWorldComm.getSize(), static_cast<uint64_t>(messageId));
            }
        }
#endif
    }

    void InstanceLeaderSendThread()
    {
        tensorrt_llm::common::setThreadName("InstanceLeaderSendThread");

        TLLM_CUDA_CHECK(
            cudaSetDevice(mDeviceIdsThisInstance.at(COMM_SESSION.getRank() % (mDeviceIdsThisInstance.size()))));

        // pop senQueue and send response to controller

        while (!mShutdown)
        {
            auto message = mInstanceSendQueue.pop();
            if (message.id == MessageID::CONTEXT_RESPONSE || message.id == MessageID::GENERATION_RESPONSE)
            {
                auto& responseWithIds = std::get<ResponsesData>(message.data);
                auto packed = serializeResponseWithIds(responseWithIds.response);

                mWorldComm.send(
                    &message.id, 1, tensorrt_llm::mpi::MpiType::kUINT64, mControllerRank, kM_INSTANCE_ID_TAG);
                mWorldComm.send(packed.data(), packed.size(), tensorrt_llm::mpi::MpiType::kCHAR, mControllerRank,
                    kM_INSTANCE_DATA_TAG);
            }
            else if (message.id == MessageID::TERMINATION)
            {
                // break; no send
                TLLM_LOG_DEBUG(
                    "ranK:%d ,size:%d ,isContext:%d... Context or Generation leader get termination message in "
                    "sendQueue***************\n",
                    mWorldComm.getRank(), mWorldComm.getSize(), int(mIsContext));
                break;
            }
            else
            {
                TLLM_THROW("rank:%d, size:%d InstanceLeaderSendThread send Invalid message id:%ld",
                    mWorldComm.getRank(), mWorldComm.getSize(), static_cast<uint64_t>(message.id));
            }
        }
    }

    void InstanceLeaderRecvThread()
    {

#if ENABLE_MULTI_DEVICE
        tensorrt_llm::common::setThreadName("InstanceLeaderRecvThread");

        TLLM_CUDA_CHECK(
            cudaSetDevice(mDeviceIdsThisInstance.at(COMM_SESSION.getRank() % (mDeviceIdsThisInstance.size()))));

        // recv request from controller and enqueRequest to executor
        while (!mShutdown)
        {
            MPI_Message msg;
            MPI_Status status;
            auto sourceRank{mControllerRank};
            mWorldComm.mprobe(sourceRank, kM_CONTROLLER_ID_TAG, &msg, &status);

            int32_t count;
            MPICHECK(MPI_Get_count(&status, MPI_UINT64_T, &count));
            TLLM_CHECK(count == 1);

            MessageID messageId;
            MPICHECK(MPI_Mrecv(&messageId, count, MPI_UINT64_T, &msg, &status));

            if (messageId == MessageID::TERMINATION)
            {
                TLLM_LOG_DEBUG(
                    "ranK:%d ,size:%d ,isContext:%d ... Context or Generation leader revb termination message in "
                    "InstanceLeaderRecvThread***************\n",
                    mWorldComm.getRank(), mWorldComm.getSize(), int(mIsContext));
                shutDown();
                break;
            }
            else if (messageId == MessageID::PENDING_CONTEXT_REQUEST
                || messageId == MessageID::PENDING_GENERATION_REQUEST)
            {
                mWorldComm.mprobe(sourceRank, kM_CONTROLLER_DATA_TAG, &msg, &status);
                MPICHECK(MPI_Get_count(&status, MPI_CHAR, &count));
                std::vector<char> buffer(count);
                MPICHECK(MPI_Mrecv(buffer.data(), count, MPI_CHAR, &msg, &status));
                auto requestWithIds = RequestWithId::deserializeReqWithIds(buffer);
                for (auto&& requestWithId : requestWithIds)
                {

                    auto globalReqId = requestWithId.id;
                    if (mIsContext)
                    {
                        TLLM_CHECK(messageId == MessageID::PENDING_CONTEXT_REQUEST);
                        TLLM_CHECK(requestWithId.req.getRequestType() == RequestType::REQUEST_TYPE_CONTEXT_ONLY);
                    }
                    if (!mIsContext)
                    {
                        TLLM_CHECK(messageId == MessageID::PENDING_GENERATION_REQUEST);
                        TLLM_CHECK(requestWithId.req.getRequestType() == RequestType::REQUEST_TYPE_GENERATION_ONLY);
                    }
                    auto reqId = mExecutor->enqueueRequest(requestWithId.req);
                    {
                        std::scoped_lock<std::mutex> lock{mIdToGlbalMutex};
                        mInstanceIdToGlobalId[reqId] = globalReqId;
                    }
                }
            }

            else
            {
                TLLM_THROW("rank:%d, size:%d InstanceLeaderRecvThread send Invalid message id:%ld",
                    mWorldComm.getRank(), mWorldComm.getSize(), static_cast<uint64_t>(messageId));
            }
        }
#endif
    }

    void InstanceLeaderLoopThread()
    {

        tensorrt_llm::common::setThreadName("InstanceLeaderLoopThread");

        TLLM_CUDA_CHECK(
            cudaSetDevice(mDeviceIdsThisInstance.at(COMM_SESSION.getRank() % (mDeviceIdsThisInstance.size()))));

        // loop awaitResponse and enqueue into sendQueue
        while (!mShutdown)
        {
            std::chrono::milliseconds waitTime(1);

            auto responses = mExecutor->awaitResponses(waitTime);
            if (responses.empty())
            {
                continue;
            }
            std::vector<ResponseWithId> responseWithIds;
            for (auto&& response : responses)
            {
                auto reqId = response.getRequestId();
                IdType globalId{0};
                {
                    std::scoped_lock<std::mutex> lock{mIdToGlbalMutex};
                    globalId = mInstanceIdToGlobalId[reqId];
                }
                TLLM_CHECK(globalId != 0);
                responseWithIds.push_back(ResponseWithId{response, globalId});
            }

            if (mIsContext)
            {
                mInstanceSendQueue.push({MessageID::CONTEXT_RESPONSE, ResponsesData{responseWithIds}});
            }
            if ((!mIsContext))
            {
                mInstanceSendQueue.push({MessageID::GENERATION_RESPONSE, ResponsesData{responseWithIds}});
            }
        }
    }
};

namespace texec = tensorrt_llm::executor;

class DisaggExecutorOrchestrator
{
public:
    DisaggExecutorOrchestrator(std::vector<std::filesystem::path> const& ctxEnginePaths,
        std::vector<std::filesystem::path> const& genEnginePaths,
        std::vector<texec::ExecutorConfig> const& ctxExecutorConfigs,
        std::vector<texec::ExecutorConfig> const& genExecutorConfigs, bool hasContextAwaitThreads,
        bool hasGenAwaitThreads)
        : mhasContextAwaitThreads(hasContextAwaitThreads)
        , mhasGenAwaitThreads(hasGenAwaitThreads)
        , mContextAciveRequestNum(ctxEnginePaths.size())
        , mGenActiveRequestNum(genEnginePaths.size())
    {
        TLLM_CHECK(ctxEnginePaths.size() == ctxExecutorConfigs.size());
        TLLM_CHECK(genEnginePaths.size() == genExecutorConfigs.size());
        TLLM_CHECK(!(ctxEnginePaths.empty() || genEnginePaths.empty()));
        int worldRank = tensorrt_llm::mpi::MpiComm::world().getRank();
        int worldSize = tensorrt_llm::mpi::MpiComm::world().getSize();
        mIsOrchestrator = (worldRank == 0);
        auto contextNum = ctxEnginePaths.size();
        mContextReqIdToGlobalId = std::vector<std::unordered_map<IdType, IdType>>(contextNum);
        mContextMapMutexs = std::vector<std::mutex>(contextNum);
        auto genNum = genEnginePaths.size();
        mGenerationReqIdToGlobalId = std::vector<std::unordered_map<IdType, IdType>>(genNum);
        mGenerationMapMutexs = std::vector<std::mutex>(genNum);

        for (int cN = 0; cN < contextNum; cN++)
        {
            mContextExecutors.push_back(std::make_unique<texec::Executor>(
                ctxEnginePaths[cN], texec::ModelType::kDECODER_ONLY, ctxExecutorConfigs[cN]));
        }

        for (int gN = 0; gN < genNum; gN++)
        {
            mGenerationExecutors.push_back(std::make_unique<texec::Executor>(
                genEnginePaths[gN], texec::ModelType::kDECODER_ONLY, genExecutorConfigs[gN]));
        }

        if (mIsOrchestrator)
        {
            if (mhasContextAwaitThreads)
            {
                for (int contextIdx = 0; contextIdx < contextNum; contextIdx++)
                {
                    mContextThreads.emplace_back(
                        [this, contextIdx]() { this->waitResponseAndAppendThreadFun(true, contextIdx); });
                }
            }
            if (mhasGenAwaitThreads)
            {

                for (int genIdx = 0; genIdx < genNum; genIdx++)
                {
                    mGenerationThreads.emplace_back(
                        [this, genIdx]() { this->waitResponseAndAppendThreadFun(false, genIdx); });
                }
            }
        }
        tensorrt_llm::mpi::MpiComm::world().barrier();
    }

    std::vector<IdType> enqueueContext(std::vector<texec::Request> const& requests,
        std::optional<int> selectContextId = std::nullopt, bool batch = false)
    {

        std::vector<IdType> globalReqIds;
        for (auto const& request : requests)
        {
            globalReqIds.push_back(generatedControlId());
            TLLM_CHECK(request.getRequestType() == tensorrt_llm::executor::RequestType::REQUEST_TYPE_CONTEXT_ONLY);
        }

        if (batch)
        {
            size_t contextId = selectContextId.has_value() ? selectContextId.value() : selectContextExecutor();
            auto contextReqIds = mContextExecutors[contextId]->enqueueRequests(requests);
            mContextAciveRequestNum.at(contextId) += static_cast<int64_t>(contextReqIds.size());
            {
                std::scoped_lock<std::mutex> lock{mContextMapMutexs[contextId]};
                for (size_t i = 0; i < requests.size(); ++i)
                {
                    mContextReqIdToGlobalId[contextId][contextReqIds[i]] = globalReqIds[i];
                }
            }
        }
        else
        {
            for (size_t i = 0; i < requests.size(); ++i)
            {
                size_t contextId = selectContextId.has_value() ? selectContextId.value() : selectContextExecutor();

                auto contextReqId = mContextExecutors[contextId]->enqueueRequest(requests[i]);
                ++mContextAciveRequestNum.at(contextId);
                {
                    std::scoped_lock<std::mutex> lock{mContextMapMutexs[contextId]};
                    mContextReqIdToGlobalId[contextId][contextReqId] = globalReqIds[i];
                }
            }
        }
        return globalReqIds;
    }

    void enqueueGeneration(std::vector<texec::Request> const& requests, std::vector<IdType> const& globalRequestIds,
        std::optional<int> selectGenIdx = std::nullopt, bool batch = false)
    {

        TLLM_CHECK(globalRequestIds.size() == requests.size());

        for (auto const& request : requests)
        {

            TLLM_CHECK(request.getRequestType() == tensorrt_llm::executor::RequestType::REQUEST_TYPE_GENERATION_ONLY);
        }
        if (batch)
        {
            size_t genIdx = selectGenIdx.has_value() ? selectGenIdx.value() : selectGenerationExecutor();
            auto genReqIds = mGenerationExecutors[genIdx]->enqueueRequests(requests);
            mGenActiveRequestNum.at(genIdx) += static_cast<int64_t>(genReqIds.size());
            {
                std::scoped_lock<std::mutex> lock{mGenerationMapMutexs[genIdx]};
                for (size_t i = 0; i < requests.size(); ++i)
                {
                    mGenerationReqIdToGlobalId[genIdx][genReqIds[i]] = globalRequestIds[i];
                }
            }
        }
        else
        {
            for (size_t i = 0; i < requests.size(); ++i)
            {
                size_t genIdx = selectGenIdx.has_value() ? selectGenIdx.value() : selectGenerationExecutor();

                auto genReqId = mGenerationExecutors[genIdx]->enqueueRequest(requests[i]);
                ++mGenActiveRequestNum.at(genIdx);
                {
                    std::scoped_lock<std::mutex> lock{mGenerationMapMutexs[genIdx]};
                    mGenerationReqIdToGlobalId[genIdx][genReqId] = globalRequestIds[i];
                }
            }
        }
    }

    std::vector<ResponseWithId> awaitContextResponses(
        std::optional<int> contextIdx, std::optional<std::chrono::milliseconds> const& timeout)
    {

        std::vector<ResponseWithId> responses;

        if (mhasContextAwaitThreads)
        {

            std::unique_lock<std::mutex> lock(mResponsesContextMtx);
            auto pred = [&mShutdown = mShutdown, &resp = this->mContextResponses]() -> bool
            { return !resp.empty() || mShutdown; };
            auto storeResponses = [this, &resp = this->mContextResponses, &responses]()
            {
                responses = std::move(resp);
                resp.clear();
            };
            if (timeout)
            {
                if (mContextResponsesCV.wait_for(lock, timeout.value(), pred))
                {
                    storeResponses();
                }
            }
            else
            {
                mContextResponsesCV.wait(lock, pred);
                storeResponses();
            }
            TLLM_CHECK_WITH_INFO(
                !contextIdx.has_value(), "contextIdx should not be provided when mhasContextAwaitThreads is true");

            return responses;
        }

        if (contextIdx.has_value())
        {
            TLLM_CHECK(!mhasContextAwaitThreads);
            auto responseFromExecutor = mContextExecutors[contextIdx.value()]->awaitResponses(timeout);
            for (auto&& resp : responseFromExecutor)
            {

                auto reqId = resp.getRequestId();
                IdType globalId{0};

                {
                    std::scoped_lock<std::mutex> lock{mContextMapMutexs.at(contextIdx.value())};
                    globalId = mContextReqIdToGlobalId.at(contextIdx.value()).at(reqId);
                }
                TLLM_CHECK(globalId != 0);
                responses.emplace_back(std::move(resp), globalId);
            }
            mContextAciveRequestNum.at(contextIdx.value()) -= static_cast<int64_t>(responseFromExecutor.size());
            return responses;
        }
        TLLM_CHECK(timeout.has_value());
        auto timeouP = timeout.value() / mContextExecutors.size();
        for (size_t ci = 0; ci < mContextExecutors.size(); ci++)
        {
            auto responseFromExecutor = mContextExecutors.at(ci)->awaitResponses(timeouP);
            for (auto&& resp : responseFromExecutor)
            {
                auto reqId = resp.getRequestId();
                IdType globalId{0};

                {
                    std::scoped_lock<std::mutex> lock{mContextMapMutexs.at(ci)};
                    globalId = mContextReqIdToGlobalId.at(ci).at(reqId);
                }
                TLLM_CHECK(globalId != 0);
                responses.emplace_back(std::move(resp), globalId);
            }
            mContextAciveRequestNum.at(ci) -= static_cast<int64_t>(responseFromExecutor.size());
        }

        return responses;
    };

    std::vector<ResponseWithId> awaitGenerationResponses(
        std::optional<int> genIdx, std::optional<std::chrono::milliseconds> const& timeout)
    {

        std::vector<ResponseWithId> responses;

        if (mhasGenAwaitThreads)
        {

            std::unique_lock<std::mutex> lock(mResponseGenerationMtx);
            auto pred = [&mShutdown = mShutdown, &resp = this->mGenerationResponses]() -> bool
            { return !resp.empty() || mShutdown; };
            auto storeResponses = [this, &resp = this->mGenerationResponses, &responses]()
            {
                responses = std::move(resp);
                resp.clear();
            };
            if (timeout)
            {
                if (mGenerationResponsesCv.wait_for(lock, timeout.value(), pred))
                {
                    storeResponses();
                }
            }
            else
            {
                mGenerationResponsesCv.wait(lock, pred);
                storeResponses();
            }
            TLLM_CHECK_WITH_INFO(!genIdx.has_value(), "genIdx should not be provided when mhasGenAwaitThreads is true");
            return responses;
        }

        if (genIdx.has_value())
        {
            TLLM_CHECK(!mhasGenAwaitThreads);
            auto responseFromExecutor = mGenerationExecutors[genIdx.value()]->awaitResponses(timeout);
            for (auto&& resp : responseFromExecutor)
            {

                auto reqId = resp.getRequestId();
                IdType globalId{0};

                {
                    std::scoped_lock<std::mutex> lock{mGenerationMapMutexs.at(genIdx.value())};
                    globalId = mGenerationReqIdToGlobalId.at(genIdx.value()).at(reqId);
                }
                TLLM_CHECK(globalId != 0);
                if (resp.getResult().isFinal)
                {
                    --mGenActiveRequestNum.at(genIdx.value());
                }
                responses.emplace_back(std::move(resp), globalId);
            }
            return responses;
        }
        TLLM_CHECK(timeout.has_value());
        auto timeouP = timeout.value() / mGenerationExecutors.size();

        for (size_t gi = 0; gi < mGenerationExecutors.size(); gi++)
        {
            auto responseFromExecutor = mGenerationExecutors.at(gi)->awaitResponses(timeouP);
            for (auto&& resp : responseFromExecutor)
            {

                auto reqId = resp.getRequestId();
                IdType globalId{0};

                {
                    std::scoped_lock<std::mutex> lock{mGenerationMapMutexs.at(gi)};
                    globalId = mGenerationReqIdToGlobalId.at(gi).at(reqId);
                }
                TLLM_CHECK(globalId != 0);
                if (resp.getResult().isFinal)
                {
                    --mGenActiveRequestNum.at(gi);
                }
                responses.emplace_back(std::move(resp), globalId);
            }
        }

        return responses;
    };

    [[nodiscard]] bool canEnqueue() const
    {
        return mIsOrchestrator;
    }

    [[nodiscard]] std::vector<std::unique_ptr<texec::Executor>> const& getContextExecutors() const
    {
        return mContextExecutors;
    }

    [[nodiscard]] std::vector<std::unique_ptr<texec::Executor>> const& getGenExecutors() const
    {
        return mGenerationExecutors;
    }

    ~DisaggExecutorOrchestrator()
    {

        mShutdown = true;

        mContextResponsesCV.notify_all();
        mGenerationResponsesCv.notify_all();
        for (auto&& executor : mContextExecutors)
        {
            executor->shutdown();
        }
        for (auto&& executor : mGenerationExecutors)
        {
            executor->shutdown();
        }

        if (mIsOrchestrator)
        {
            if (mhasContextAwaitThreads)
            {
                for (auto&& contextThread : mContextThreads)
                {
                    if (contextThread.joinable())
                    {
                        contextThread.join();
                    }
                }
            }
            if (mhasGenAwaitThreads)
            {
                for (auto&& genThread : mGenerationThreads)
                {
                    if (genThread.joinable())
                    {
                        genThread.join();
                    }
                }
            }
        }
    }

private:
    IdType generatedControlId()
    {
        return (++mLastId % UINT64_MAX);
    };

    size_t selectContextExecutor()
    {
        auto contextIdx = std::distance(mContextAciveRequestNum.begin(),
            std::min_element(mContextAciveRequestNum.begin(), mContextAciveRequestNum.end()));
        return contextIdx;
    }

    size_t selectGenerationExecutor()
    {
        auto generationIdx = std::distance(
            mGenActiveRequestNum.begin(), std::min_element(mGenActiveRequestNum.begin(), mGenActiveRequestNum.end()));

        return generationIdx;
    }

    void appendNewContextResponse(std::vector<ResponseWithId>&& newResponses)
    {
        {
            std::scoped_lock<std::mutex> lock(mResponsesContextMtx);
            for (auto&& response : newResponses)
            {
                mContextResponses.emplace_back(std::move(response));
            }
        }
        mContextResponsesCV.notify_all();
    }

    void appendNewGenerationResponse(std::vector<ResponseWithId>&& newResponses)
    {
        {
            std::scoped_lock<std::mutex> lock(mResponseGenerationMtx);
            for (auto&& response : newResponses)
            {
                mGenerationResponses.emplace_back(std::move(response));
            }
        }
        mGenerationResponsesCv.notify_all();
    }

    void waitResponseAndAppendThreadFun(bool isContext, int executorIdx)
    {
        auto& executor = isContext ? mContextExecutors[executorIdx] : mGenerationExecutors[executorIdx];

        while (!mShutdown)
        {
            auto responses = executor->awaitResponses();

            if (responses.empty())
            {
                continue;
            }
            std::vector<ResponseWithId> responseWithIds;
            if (isContext)
            {
                for (auto&& response : responses)
                {
                    auto reqId = response.getRequestId();
                    IdType globalId{0};

                    {
                        std::scoped_lock<std::mutex> lock{mContextMapMutexs.at(executorIdx)};
                        globalId = mContextReqIdToGlobalId.at(executorIdx).at(reqId);
                    }
                    TLLM_CHECK(globalId != 0);
                    responseWithIds.emplace_back(std::move(response), globalId);
                }
                if (responseWithIds.size() > 0)
                {
                    mContextAciveRequestNum.at(executorIdx) -= static_cast<int64_t>(responseWithIds.size());
                    appendNewContextResponse(std::move(responseWithIds));
                }
            }
            else
            {

                for (auto&& response : responses)
                {
                    auto reqId = response.getRequestId();
                    IdType globalId{0};

                    {
                        std::scoped_lock<std::mutex> lock{mGenerationMapMutexs.at(executorIdx)};
                        globalId = mGenerationReqIdToGlobalId.at(executorIdx).at(reqId);
                    }
                    TLLM_CHECK(globalId != 0);
                    if (response.getResult().isFinal)
                    {
                        --mGenActiveRequestNum.at(executorIdx);
                    }
                    responseWithIds.emplace_back(std::move(response), globalId);
                }
                if (responseWithIds.size() > 0)
                {
                    appendNewGenerationResponse(std::move(responseWithIds));
                }
            }
        }
    };

    std::vector<std::unique_ptr<texec::Executor>> mContextExecutors;
    std::vector<std::unique_ptr<texec::Executor>> mGenerationExecutors;
    std::vector<std::thread> mContextThreads;
    std::vector<std::thread> mGenerationThreads;

    std::atomic<IdType> mLastId{0};
    std::vector<std::unordered_map<IdType, IdType>> mContextReqIdToGlobalId;
    std::vector<std::unordered_map<IdType, IdType>> mGenerationReqIdToGlobalId;
    std::vector<std::mutex> mContextMapMutexs;
    std::vector<std::mutex> mGenerationMapMutexs;
    std::vector<ResponseWithId> mContextResponses;
    std::condition_variable mContextResponsesCV;
    std::mutex mResponsesContextMtx;

    std::vector<ResponseWithId> mGenerationResponses;
    std::condition_variable mGenerationResponsesCv;
    std::mutex mResponseGenerationMtx;
    std::atomic<bool> mShutdown{false};
    std::atomic<bool> mhasContextAwaitThreads{false};
    std::atomic<bool> mhasGenAwaitThreads{false};
    std::vector<std::atomic<int64_t>> mContextAciveRequestNum;
    std::vector<std::atomic<int64_t>> mGenActiveRequestNum;
    bool mIsOrchestrator{false};
};
} // namespace tensorrt_llm::testing::executor::disaggexecutor
