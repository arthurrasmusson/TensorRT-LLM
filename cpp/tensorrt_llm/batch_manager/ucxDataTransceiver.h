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
#include "tensorrt_llm/executor/contextPhaseState.h"
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
    explicit UcxComm() {}

    explicit UcxComm(std::shared_ptr<ucxx::Endpoint> const& endpoint)
        : mEndpoint(endpoint)
    {
    }

    virtual ~UcxComm() = default;

    virtual void sendBuffer(runtime::IBuffer const& buf) const;
    virtual void recvBuffer(runtime::IBuffer& buf) const;

    virtual LlmRequest::RequestIdType recvRequestId() const;
    virtual void sendRequestId(LlmRequest::RequestIdType id) const;

private:
    static constexpr ucxx::Tag kID_TAG{1};
    static constexpr ucxx::Tag kDATA_TAG{2};

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

    UcxDataSender(std::unique_ptr<UcxCommFactory>&& factory, TFormatter formatter, uint16_t listenerPort = 0)
        : mFactory{std::move(factory)}
        , mFormatter{std::move(formatter)}
    {
        mContext = ucxx::createContext({}, ucxx::Context::defaultFeatureFlags);
        mWorker = mContext->createWorker();
        // Ensure the progress thread has CUDA context initialized
        mWorker->setProgressThreadStartCallback(cudaFree, nullptr);
        mWorker->startProgressThread();

        startListener(listenerPort);
    }

    [[nodiscard]] LlmRequest::RequestIdType recvRequestId() override
    {
        // Below can be initiated asynchronously when an endpoint is created..
        std::unique_ptr<UcxComm> comm;
        {
            std::unique_lock<std::mutex> lk(mMtx);
            mRequestCv.wait(lk, [this]() { return !mIncomingRequests.empty(); });
            comm = std::move(mIncomingRequests.front());
            mIncomingRequests.pop_front();
        }
        auto requestId = comm->recvRequestId();
        std::lock_guard<std::mutex> lk(mMtx);
        mRequestToComm.emplace(requestId, std::move(comm));
        return requestId;
    }

    void sendSync(LlmRequest const& llmRequest) override
    {
        UcxComm* comm;
        {
            std::lock_guard<std::mutex> lk(mMtx);
            auto it = mRequestToComm.find(llmRequest.mRequestId);
            TLLM_CHECK_WITH_INFO(
                (it != mRequestToComm.end()), "sendSync() must be called with request returned by recvRequestId().");
            comm = it->second.get();
        }
        (*mFormatter)(llmRequest, {comm});
        {
            // For now, the connection will be dropped once the transfer is completed
            std::lock_guard<std::mutex> lk(mMtx);
            mRequestToComm.erase(llmRequest.mRequestId);
        }
    }

    [[nodiscard]] executor::kv_cache::CommState const& getCommState() const override
    {
        return mCommState;
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
        mCommState = executor::kv_cache::CommState{mListener->getPort(), mListener->getIp()};
    }

    void addIncomingRequests(std::unique_ptr<UcxComm>&& incomingRequest)
    {
        {
            std::lock_guard<std::mutex> lk(mMtx);
            mIncomingRequests.emplace_back(std::move(incomingRequest));
        }
        mRequestCv.notify_all();
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
    executor::kv_cache::CommState mCommState;
};

template <typename TDataConfig>
class UcxDataReceiver final : public DataReceiver
{
public:
    using TFormatter = std::unique_ptr<IOFormatter<UcxComm, TDataConfig>>;

    UcxDataReceiver(std::unique_ptr<UcxCommFactory>&& factory, TFormatter formatter)
        : mFactory{std::move(factory)}
        , mFormatter{std::move(formatter)}
    {
        mContext = ucxx::createContext({}, ucxx::Context::defaultFeatureFlags);
        mWorker = mContext->createWorker();
        // Ensure the progress thread has CUDA context initialized
        mWorker->setProgressThreadStartCallback(cudaFree, nullptr);
        mWorker->startProgressThread();
    }

    void sendRequestId(LlmRequest const& llmRequest) override
    {
        auto requestId = llmRequest.getContextPhaseState().getReqId();
        auto& socketComm = llmRequest.getContextPhaseState().getCommState().value().getSocketState().at(0);
        auto comm = mFactory->create(mWorker->createEndpointFromHostname(socketComm.mIp, socketComm.mPort));
        comm->sendRequestId(requestId);
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
                (it != mRequestToComm.end()), "sendSync() must be called with request returned by recvRequestId().");
            comm = it->second.get();
        }
        (*mFormatter)(llmRequest, {comm});
        {
            // For now, the connection will be dropped once the transfer is completed
            std::lock_guard<std::mutex> lk(mMtx);
            mRequestToComm.erase(llmRequest.mRequestId);
        }
    }

private:
    std::unique_ptr<UcxCommFactory> mFactory;
    TFormatter mFormatter;

    std::shared_ptr<ucxx::Context> mContext;
    std::shared_ptr<ucxx::Worker> mWorker;

    std::mutex mMtx;
    std::map<LlmRequest::RequestIdType, std::unique_ptr<UcxComm>> mRequestToComm;
};

std::unique_ptr<DataResponder> makeUcxCacheResponder(kv_cache_manager::KVCacheManager* cacheManager);

std::unique_ptr<DataRequester> makeUcxCacheRequester(kv_cache_manager::KVCacheManager* cacheManager);

} // namespace tensorrt_llm::batch_manager
