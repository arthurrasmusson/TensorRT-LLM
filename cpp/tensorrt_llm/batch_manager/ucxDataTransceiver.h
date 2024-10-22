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

#include "dataTransceiver.h"

#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/peftCacheManagerConfig.h"
#include "tensorrt_llm/common/mpiUtils.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/runtime/common.h"
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <ucxx/api.h>
#include <ucxx/utils/sockaddr.h>
#include <ucxx/utils/ucx.h>

namespace tensorrt_llm::batch_manager
{
class UcxComm
{
public:
    using TPtrContainer = std::vector<UcxComm const*>;

    explicit UcxComm() {}

    explicit UcxComm(std::shared_ptr<ucxx::Endpoint> const& endpoint)
        : mEndpoint(endpoint)
    {
        if (mEndpoint)
        {
            initializeEndpointTag();
        }
    }

    virtual ~UcxComm() = default;

    virtual void sendBuffer(runtime::IBuffer const& buf) const;
    virtual void recvBuffer(runtime::IBuffer& buf) const;

    virtual RequestInfo recvRequestInfo() const;
    virtual void sendRequestInfo(RequestInfo const& info) const;

private:
    void initializeEndpointTag();
    void setRequestTag(LlmRequest::RequestIdType const requestId);

    static constexpr ucxx::Tag kID_TAG{1};
    static constexpr ucxx::Tag kDATA_TAG{2};

    // a send tag is defined as:
    // | local port (16 bits) | remote port (16 bits) | truncated request id (16 bits) | UCXComm flags (16 bits) |
    // a recv tag is defined as:
    // | remote port (16 bits) | local port (16 bits) | truncated request id (16 bits) | UCXComm flags (16 bits) |
    ucxx::Tag mSendTag{0};
    ucxx::Tag mRecvTag{0};

    // Set of bit map used to represent UCXComm flags
    ucxx::Tag mInfoTag{1};

    static constexpr ucxx::TagMask mEndpointMask{(((uint64_t) 1 << (32 + 1)) - 1) << 32};
    static constexpr ucxx::TagMask mRequestIdMask{(((uint64_t) 1 << (16 + 1)) - 1) << 16};
    static constexpr ucxx::TagMask mFlagMask{(((uint64_t) 1 << (16 + 1)) - 1)};

    mutable std::mutex mMtx;
    mutable std::condition_variable mCv;
    std::shared_ptr<ucxx::Endpoint> mEndpoint;
};

// Factory class for creating UcxComm object, UCX transceivers will construct UcxComm objects
// per connection and this factory class will be passed to the transceivers to achieve further
// polymorphism.
class UcxCommFactory
{
public:
    virtual ~UcxCommFactory() = default;

    virtual std::unique_ptr<UcxComm> create(std::shared_ptr<ucxx::Endpoint> const& endpoint)
    {
        return std::make_unique<UcxComm>(endpoint);
    }
};

template <typename TDataConfig>
class UcxDataSender final : public DataSender
{
public:
    using TFormatter = std::unique_ptr<IOFormatter<UcxComm, TDataConfig>>;

    UcxDataSender(std::unique_ptr<UcxCommFactory>&& factory, executor::kv_cache::CacheState selfCacheState,
        SizeType32 selfIndex, TFormatter formatter, uint16_t listenerPort = 0)
        : mFactory{std::move(factory)}
        , mFormatter{std::move(formatter)}
    {
        mSelfState.setCommState(
            executor::kv_cache::CommState{std::vector<executor::kv_cache::SocketState>{}, selfIndex});
        mSelfState.setCacheState(std::move(selfCacheState));
        mContext = ucxx::createContext({}, ucxx::Context::defaultFeatureFlags);
        mWorker = mContext->createWorker();
        // Ensure the progress thread has CUDA context initialized
        int device;
        TLLM_CUDA_CHECK(cudaGetDevice(&device));

        mWorker->setProgressThreadStartCallback(
            [device](void* arg) { TLLM_CUDA_CHECK(cudaSetDevice(device)); }, nullptr);
        mWorker->startProgressThread();
        startListener(listenerPort);
    }

    [[nodiscard]] RequestInfo recvRequestInfo() override
    {
        // Below can be initiated asynchronously when an endpoint is created..
        std::unique_ptr<UcxComm> comm;
        {
            std::unique_lock<std::mutex> lk(mMtx);
            mRequestCv.wait(lk, [this]() { return !mIncomingRequests.empty(); });
            comm = std::move(mIncomingRequests.front());
            mIncomingRequests.pop_front();
        }
        auto info = comm->recvRequestInfo();

        TLLM_CHECK_WITH_INFO(mFormatter->inquireSupport(
                                 mSelfState.getCacheState().value(), info.getTransState().getCacheState().value()),
            "Disagg server does not currently support these cacheState.");
        std::lock_guard<std::mutex> lk(mMtx);
        mRequestToComm.emplace(info.getRequestId(), std::move(comm));

        return info;
    }

    void sendSync(LlmRequest const& llmRequest) override
    {
        UcxComm* comm;
        {
            std::lock_guard<std::mutex> lk(mMtx);
            auto it = mRequestToComm.find(llmRequest.mRequestId);
            TLLM_CHECK_WITH_INFO(
                (it != mRequestToComm.end()), "sendSync() must be called with request returned by recvRequestInfo().");
            comm = it->second.get();
        }

        // TODO: fake destCacheState

        (*mFormatter)(llmRequest, {comm}, mSelfState.getCacheState().value(),
            mSelfState.getCommState().value().getSelfIdx(), mSelfState.getCacheState().value());

        {
            // For now, the connection will be dropped once the transfer is completed
            std::lock_guard<std::mutex> lk(mMtx);
            {
                // [WAR] Releasing endpoint soon after tagSend results in hanging,
                // postponing release at the moment to avoid that.
                auto it = mRequestToComm.find(llmRequest.mRequestId);
                reapFinishedComm(std::move(it->second));
            }
            mRequestToComm.erase(llmRequest.mRequestId);
        }
    }

    [[nodiscard]] executor::kv_cache::CommState const& getCommState() const override
    {
        return mSelfState.getCommState().value();
    }

    void setCommState(executor::kv_cache::CommState const& commState) override
    {
        mSelfState.setCommState(commState);
    }

    [[nodiscard]] bool availableRelease(LlmRequest const& llmRequest)
    {
        return true;
    }

private:
    static void listenerCallback(ucp_conn_request_h conn_request, void* arg)
    {
        auto sender = reinterpret_cast<UcxDataSender*>(arg);

        auto endpoint = sender->mListener->createEndpointFromConnRequest(conn_request);
        sender->addIncomingRequests(sender->mFactory->create(endpoint));
    }

    void startListener(uint16_t listenerPort)
    {
        mListener = mWorker->createListener(listenerPort, listenerCallback, this);
        mSelfState.setCommState(executor::kv_cache::CommState{mListener->getPort(), mListener->getIp()});
    }

    void addIncomingRequests(std::unique_ptr<UcxComm>&& incomingRequest)
    {
        {
            std::lock_guard<std::mutex> lk(mMtx);
            mIncomingRequests.emplace_back(std::move(incomingRequest));
        }
        mRequestCv.notify_all();
    }

    void reapFinishedComm(std::unique_ptr<UcxComm>&& comm)
    {
        // this WAR function assumes 'mMtx' is being held
        auto now = std::chrono::steady_clock::now();
        if (comm != nullptr)
        {
            mReapingComm.emplace_back(now, std::move(comm));
        }
        while (!mReapingComm.empty())
        {
            auto timePassed = std::chrono::duration<double, std::milli>(now - mReapingComm.front().first).count();
            if (timePassed < 200)
            {
                break;
            }
            mReapingComm.pop_front();
        }
    }

    std::unique_ptr<UcxCommFactory> mFactory;
    TFormatter mFormatter;

    std::shared_ptr<ucxx::Context> mContext;
    std::shared_ptr<ucxx::Worker> mWorker;
    std::shared_ptr<ucxx::Listener> mListener;

    std::mutex mMtx;
    std::condition_variable mRequestCv;
    std::deque<std::unique_ptr<UcxComm>> mIncomingRequests;
    std::map<LlmRequest::RequestIdType, std::unique_ptr<UcxComm>> mRequestToComm;
    executor::DataTransceiverState mSelfState;

    std::deque<std::pair<decltype(std::chrono::steady_clock::now()), std::unique_ptr<UcxComm>>> mReapingComm;
};

template <typename TDataConfig>
class UcxDataReceiver final : public DataReceiver
{
public:
    using TFormatter = std::unique_ptr<IOFormatter<UcxComm, TDataConfig>>;

    UcxDataReceiver(std::unique_ptr<UcxCommFactory>&& factory, executor::kv_cache::CacheState selfCacheState,
        SizeType32 selfIndex, TFormatter formatter)
        : mFactory{std::move(factory)}
        , mFormatter{std::move(formatter)}
    {
        mSelfState.setCommState(
            executor::kv_cache::CommState{std::vector<executor::kv_cache::SocketState>{}, selfIndex});
        mSelfState.setCacheState(std::move(selfCacheState));
        mContext = ucxx::createContext({}, ucxx::Context::defaultFeatureFlags);
        mWorker = mContext->createWorker();
        // Ensure the progress thread has CUDA context initialized
        int device;
        TLLM_CUDA_CHECK(cudaGetDevice(&device));

        mWorker->setProgressThreadStartCallback(
            [device](void* arg) { TLLM_CUDA_CHECK(cudaSetDevice(device)); }, nullptr);
        mWorker->startProgressThread();
    }

    void sendRequestInfo(LlmRequest const& llmRequest) override
    {
        // TODO: support hetergenous mapping
        auto requestId = llmRequest.getContextPhaseParams().value().getReqId();
        auto const& contextState = llmRequest.getDataTransceiverState();
        auto const& socketState = contextState.getCommState().value().getSocketState();
        std::vector<SizeType32> targetRanks{};

        auto const& destCacheState = contextState.getCacheState().value();

        targetRanks = mFormatter->getCounterparts(
            mSelfState.getCacheState().value(), mSelfState.getCommState().value().getSelfIdx(), destCacheState);
        TLLM_CHECK_WITH_INFO(mFormatter->inquireSupport(mSelfState.getCacheState().value(), destCacheState),
            "Disagg server does not currently support these cacheState.");
        auto& socketComm = socketState.at(targetRanks[0]);
        auto comm = mFactory->create(mWorker->createEndpointFromHostname(socketComm.mIp, socketComm.mPort));
        RequestInfo info{requestId, mSelfState};
        comm->sendRequestInfo(std::move(info));
        std::lock_guard<std::mutex> lk(mMtx);
        mRequestToComm.emplace(llmRequest.mRequestId, std::move(comm));
    }

    void receiveSync(LlmRequest const& llmRequest) override
    {
        UcxComm* comm;
        {
            std::lock_guard<std::mutex> lk(mMtx);
            auto it = mRequestToComm.find(llmRequest.mRequestId);
            TLLM_CHECK_WITH_INFO(
                (it != mRequestToComm.end()), "sendSync() must be called with request returned by recvRequestInfo().");
            comm = it->second.get();
        }
        auto const& contextState = llmRequest.getDataTransceiverState();

        auto const& destCacheState = contextState.getCacheState().value();

        (*mFormatter)(llmRequest, {comm}, mSelfState.getCacheState().value(),
            mSelfState.getCommState().value().getSelfIdx(), destCacheState);
        {
            // For now, the connection will be dropped once the transfer is completed
            std::lock_guard<std::mutex> lk(mMtx);
            mRequestToComm.erase(llmRequest.mRequestId);
        }
    }

private:
    std::unique_ptr<UcxCommFactory> mFactory;
    executor::DataTransceiverState mSelfState;
    TFormatter mFormatter;

    std::shared_ptr<ucxx::Context> mContext;
    std::shared_ptr<ucxx::Worker> mWorker;

    std::mutex mMtx;
    std::map<LlmRequest::RequestIdType, std::unique_ptr<UcxComm>> mRequestToComm;
};

// specify C linkage to allow isolating UCX features into separate shared library
// and dynamically loading on demand. This is to allow running fully-featured
// TRTLLM in UCX-less environment.
// This WAR results in additional UCX wrapper shared library to be shipped with TRTLLM,
// a cleaner resolution is to dynamically load the underlying UCX library, but it requires
// modification to implement the data responder against UCX directly, instead of UCXX
// (a C++ wrapper of UCX picked to simplify implementation). Implementing against UCXX
// adds one layer of indirection and makes dynamically loading the below functions a simpler WAR.
#if __cplusplus
extern "C"
{
#endif

    std::unique_ptr<DataResponder> makeUcxCacheResponder(executor::kv_cache::CacheState selfCacheState,
        SizeType32 selfIndex, kv_cache_manager::KVCacheManager* cacheManager);

    std::unique_ptr<DataRequester> makeUcxCacheRequester(executor::kv_cache::CacheState selfCacheState,
        SizeType32 selfIndex, kv_cache_manager::KVCacheManager* cacheManager);

#if __cplusplus
}
#endif

} // namespace tensorrt_llm::batch_manager
