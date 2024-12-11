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
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/mpiUtils.h"
#include "tensorrt_llm/executor/cache_transmission/cacheConcatenate.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/runtime/common.h"
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <ucxx/api.h>
#include <ucxx/utils/sockaddr.h>
#include <ucxx/utils/ucx.h>
#if __linux__
#include <arpa/inet.h>
#include <ifaddrs.h>
#endif
namespace tensorrt_llm::batch_manager
{

class UcxEndpoint
{
public:
    using TPtrContainer = std::vector<UcxEndpoint const*>;

    UcxEndpoint() = default;

    explicit UcxEndpoint(std::shared_ptr<ucxx::Endpoint> const& endpoint)
        : mEndpoint(endpoint)
    {
        if (mEndpoint)
        {
            initializeEndpointTag();
        }
    }

    virtual ~UcxEndpoint() = default;

    virtual void sendBuffer(runtime::IBuffer const& buf) const;
    virtual void recvBuffer(runtime::IBuffer& buf) const;

    virtual RequestInfo recvRequestInfo() const;
    virtual void sendRequestInfo(RequestInfo const& info) const;

private:
    void initializeEndpointTag(int maxTryTimes = 10);
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

// TODO: Integrate this class with `UcxEndpoint` and limit the dependency on UCX to this class.
class UcxComm : public executor::kv_cache::Communicator
{
public:
    [[nodiscard]] bool isThreadSafe() const noexcept override
    {
        return true;
    }

    void sendBuffer(runtime::IBuffer const& buf, executor::kv_cache::DataContext const& context,
        executor::kv_cache::ProcessInfo const& processInfo) const override
    {
        processInfo.getEndpoint()->sendBuffer(buf);
    }

    void recvBuffer(runtime::IBuffer& buf, executor::kv_cache::DataContext const& context,
        executor::kv_cache::ProcessInfo const& processInfo) const override
    {
        processInfo.getEndpoint()->recvBuffer(buf);
    }

    [[nodiscard]] std::unique_ptr<batch_manager::RequestInfo> recvRequestInfo(
        std::optional<executor::kv_cache::ProcessInfo> const& processInfo = std::nullopt) const override
    {
        return std::make_unique<batch_manager::RequestInfo>(processInfo.value().getEndpoint()->recvRequestInfo());
    }

    void sendRequestInfo(batch_manager::RequestInfo const& requestInfo,
        executor::kv_cache::ProcessInfo const& processInfo) const override
    {
        processInfo.getEndpoint()->sendRequestInfo(requestInfo);
    }
};

// Factory class for creating UcxEndpoint object, UCX transceivers will construct UcxEndpoint objects
// per connection and this factory class will be passed to the transceivers to achieve further
// polymorphism.
class UcxCommFactory
{
public:
    virtual ~UcxCommFactory() = default;

    virtual std::unique_ptr<UcxEndpoint> create(std::shared_ptr<ucxx::Endpoint> const& endpoint)
    {
        return std::make_unique<UcxEndpoint>(endpoint);
    }
};

class UcxDataSender final : public DataSender
{
public:
    using TFormatter = std::unique_ptr<IOFormatter>;

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
        std::unique_ptr<UcxEndpoint> comm;
        {
            std::unique_lock<std::mutex> lk(mMtx);
            if (mIncomingRequests.empty())
            {
                mRequestCv.wait(lk, [this]() { return !mIncomingRequests.empty(); });
            }
            comm = std::move(mIncomingRequests.front());
            mIncomingRequests.pop_front();
        }
        auto info = comm->recvRequestInfo();

        TLLM_CHECK_WITH_INFO(mFormatter->inquireSupport(
                                 mSelfState.getCacheState().value(), info.getTransState().getCacheState().value()),
            "Disagg server does not currently support these cacheState.");

        std::unique_lock<std::mutex> lk(mMtxForMap);
        auto peerTargetRanks
            = tensorrt_llm::executor::kv_cache::targetIRanks(info.getTransState().getCacheState().value(),
                mSelfState.getCacheState().value(), getCommState().getSelfIdx());
        auto const requestId = info.getRequestId();
        if (mRequestToComms.find(requestId) == mRequestToComms.end())
        {
            auto recvExpectCount = peerTargetRanks.size();
            mRequestToComms.emplace(requestId, std::vector<std::unique_ptr<UcxEndpoint>>());
            mRequestToComms[requestId].resize(recvExpectCount);
        }
        int peerIdx = std::distance(peerTargetRanks.begin(),
            std::find(
                peerTargetRanks.begin(), peerTargetRanks.end(), info.getTransState().getCommState()->getSelfIdx()));
        TLLM_CHECK_WITH_INFO((peerIdx >= 0) && (peerIdx < static_cast<int>(peerTargetRanks.size())),
            "Peer idx should be found in peerTargetRanks");
        mRequestToComms[requestId].at(peerIdx) = std::move(comm);
        return info;
    }

    void sendSync(LlmRequest const& llmRequest) override
    {
        std::vector<executor::kv_cache::ProcessInfo> comms;
        {
            std::unique_lock<std::mutex> lk(mMtxForMap);
            auto it = mRequestToComms.find(llmRequest.mRequestId);
            TLLM_CHECK_WITH_INFO(
                (it != mRequestToComms.end()), "sendSync() must be called with request returned by recvRequestInfo().");
            for (auto&& comm : it->second)
            {
                comms.push_back(executor::kv_cache::ProcessInfo{comm.get()});
            }
        }

        // TODO: fake destCacheState
        mFormatter->formatOutput(mComm, llmRequest, std::move(comms), mSelfState.getCacheState().value(),
            mSelfState.getCommState().value().getSelfIdx(), mSelfState.getCacheState().value());

        {
            // For now, the connection will be dropped once the transfer is completed
            std::lock_guard<std::mutex> lk(mMtxForMap);
            {
                // [WAR] Releasing endpoint soon after tagSend results in hanging,
                // postponing release at the moment to avoid that.
                auto it = mRequestToComms.find(llmRequest.mRequestId);
                reapFinishedComm(std::move(it->second));
            }
            mRequestToComms.erase(llmRequest.mRequestId);
        }
    }

    [[nodiscard]] executor::kv_cache::CommState const& getCommState() const override
    {
        return mSelfState.getCommState().value();
    }

    void setCommState(executor::kv_cache::CommState commState) override
    {
        mSelfState.setCommState(std::move(commState));
    }

    [[nodiscard]] size_t getCounterpartsCount(LlmRequest::RequestIdType requestId) const override
    {
        TLLM_CHECK(mRequestToComms.find(requestId) != mRequestToComms.end());
        return mRequestToComms.at(requestId).size();
    }

    void release(LlmRequest::RequestIdType requestId) override
    {
        std::unique_lock<std::mutex> lk(mMtxForMap);
        mRequestToComms.erase(requestId);
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
#if __linux__
        // query network interface

        struct ifaddrs *ifa, *ifaddr;
        void* tmpAddrPtr;

        TLLM_CHECK_WITH_INFO(getifaddrs(&ifaddr) == 0, " UCX startListener getifaddrs call failed\n");
        TLLM_CHECK_WITH_INFO((ifaddr != NULL), "UCX startListener getifaddrs call failed\n");
        int idx = 0;
        for (ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next)
        {
            // exclude docker intrface and loopback
            if (strcmp(ifa->ifa_name, "docker0") == 0 || strcmp(ifa->ifa_name, "lo") == 0)
            {
                continue;
            }
            if (ifa->ifa_addr && ifa->ifa_addr->sa_family == AF_INET)
            {
                tmpAddrPtr = &((struct sockaddr_in*) ifa->ifa_addr)->sin_addr;
                char buffer[INET_ADDRSTRLEN];
                inet_ntop(AF_INET, tmpAddrPtr, buffer, INET_ADDRSTRLEN);
                mNetInfoMap[std::string(ifa->ifa_name)].interface = std::string(ifa->ifa_name);
                mNetInfoMap[std::string(ifa->ifa_name)].ipv4 = std::string(buffer);
                if (mNetInfoMap[std::string(ifa->ifa_name)].idx == -1)
                {
                    mNetInfoMap[std::string(ifa->ifa_name)].idx = idx;
                    idx++;
                }
            }
            else if (ifa->ifa_addr && ifa->ifa_addr->sa_family == AF_INET6)
            {
                tmpAddrPtr = &((struct sockaddr_in6*) ifa->ifa_addr)->sin6_addr;
                char buffer[INET6_ADDRSTRLEN];
                inet_ntop(AF_INET6, tmpAddrPtr, buffer, INET6_ADDRSTRLEN);
                mNetInfoMap[std::string(ifa->ifa_name)].interface = ifa->ifa_name;
                mNetInfoMap[std::string(ifa->ifa_name)].ipv6 = std::string(buffer);
                if (mNetInfoMap[std::string(ifa->ifa_name)].idx == -1)
                {
                    mNetInfoMap[std::string(ifa->ifa_name)].idx = idx;
                    idx++;
                }
            }
        }
        std::string selectedIp;
        std::string userUCXInterface = common::getEnvUCXInterface();
        if (!userUCXInterface.empty())
        {
            if (mNetInfoMap.find(userUCXInterface) != mNetInfoMap.end())
            {
                selectedIp = mNetInfoMap[userUCXInterface].ipv4;
                if (selectedIp.empty())
                {
                    selectedIp = mNetInfoMap[userUCXInterface].ipv6;
                }
                TLLM_LOG_INFO("UCX listener started on interface:%s address: %s:%u",
                    mNetInfoMap[userUCXInterface].interface.c_str(), selectedIp.c_str(), mListener->getPort());
                mSelfState.setCommState(executor::kv_cache::CommState{mListener->getPort(), selectedIp});
                freeifaddrs(ifaddr);
                return;
            }

            TLLM_LOG_WARNING(
                "Invalid UCX interface specified: %s will use default interface", userUCXInterface.c_str());
        }
        std::map<int, NetINfoT> netInfoSortedMap;
        for (auto&& [key, netInfo] : mNetInfoMap)
        {
            netInfoSortedMap[netInfo.idx] = netInfo;
        }

        selectedIp = netInfoSortedMap[0].ipv4;
        if (selectedIp.empty())
        {
            selectedIp = netInfoSortedMap[0].ipv6;
        }
        TLLM_LOG_INFO("UCX listener started on interface:%s address: %s:%u", netInfoSortedMap[0].interface.c_str(),
            selectedIp.c_str(), mListener->getPort());

        mSelfState.setCommState(executor::kv_cache::CommState{mListener->getPort(), selectedIp});
        freeifaddrs(ifaddr);
#else
        mSelfState.setCommState(executor::kv_cache::CommState{mListener->getPort(), mListener->getIp()});
#endif
    }

    void addIncomingRequests(std::unique_ptr<UcxEndpoint>&& incomingRequest)
    {
        {
            std::lock_guard<std::mutex> lk(mMtx);
            mIncomingRequests.emplace_back(std::move(incomingRequest));
        }
        mRequestCv.notify_all();
    }

    void reapFinishedComm(std::vector<std::unique_ptr<UcxEndpoint>>&& comms)
    {
        // this WAR function assumes 'mMtx' is being held
        auto now = std::chrono::steady_clock::now();
        for (auto&& comm : comms)
        {
            if (comm != nullptr)
            {
                mReapingComm.emplace_back(now, std::move(comm));
            }
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

    struct NetINfoT
    {
        std::string interface;
        std::string ipv4;
        std::string ipv6;
        int idx = -1;
    };

    UcxComm mComm;
    std::unordered_map<std::string, NetINfoT> mNetInfoMap;
    std::unique_ptr<UcxCommFactory> mFactory;
    TFormatter mFormatter;

    std::shared_ptr<ucxx::Context> mContext;
    std::shared_ptr<ucxx::Worker> mWorker;
    std::shared_ptr<ucxx::Listener> mListener;

    std::mutex mMtx;
    std::condition_variable mRequestCv;
    std::deque<std::unique_ptr<UcxEndpoint>> mIncomingRequests;
    std::map<LlmRequest::RequestIdType, std::vector<std::unique_ptr<UcxEndpoint>>> mRequestToComms;
    std::mutex mMtxForMap;

    executor::DataTransceiverState mSelfState;
    std::deque<std::pair<decltype(std::chrono::steady_clock::now()), std::unique_ptr<UcxEndpoint>>> mReapingComm;
};

class UcxDataReceiver final : public DataReceiver
{
public:
    using TFormatter = std::unique_ptr<IOFormatter>;

    UcxDataReceiver(std::unique_ptr<UcxCommFactory>&& factory, executor::kv_cache::CacheState selfCacheState,
        SizeType32 selfIndex, TFormatter formatter)
        : mFactory{std::move(factory)}
        , mFormatter{std::move(formatter)}
        , mBufferManager{std::make_shared<runtime::CudaStream>()}
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
        initSelfIps();
    }

    void sendRequestInfo(LlmRequest const& llmRequest) override
    {
        // TODO: support hetergenous mapping
        auto requestId = llmRequest.getContextPhaseParams().value().getReqId();
        auto const& contextState = llmRequest.getDataTransceiverState();
        auto const& socketState = contextState.getCommState().value().getSocketState();

        auto const& destCacheState = contextState.getCacheState().value();

        auto targetRanks = mFormatter->getCounterparts(
            mSelfState.getCacheState().value(), mSelfState.getCommState().value().getSelfIdx(), destCacheState);
        TLLM_CHECK_WITH_INFO(mFormatter->inquireSupport(mSelfState.getCacheState().value(), destCacheState),
            "Disagg server does not currently support these cacheState.");
        std::vector<std::unique_ptr<UcxEndpoint>> comms;

        for (auto index : targetRanks)
        {
            auto& socketComm = socketState.at(index);
            if (mSelfIps.find(socketComm.mIp) != mSelfIps.end())
            {
                auto comm
                    = mFactory->create(mWorker->createEndpointFromHostname(std::string("127.0.0.1"), socketComm.mPort));
                comms.push_back(std::move(comm));
            }
            else
            {
                auto comm = mFactory->create(mWorker->createEndpointFromHostname(socketComm.mIp, socketComm.mPort));
                comms.push_back(std::move(comm));
            }
        }
        for (auto&& comm : comms)
        {
            comm->sendRequestInfo({requestId, mSelfState});
        }
        std::unique_lock<std::mutex> lk(mMtx);
        mRequestToComms.emplace(llmRequest.mRequestId, std::move(comms));
    }

    void receiveSync(LlmRequest const& llmRequest) override
    {
        std::vector<executor::kv_cache::ProcessInfo> comms;
        {
            std::unique_lock<std::mutex> lk(mMtx);
            auto it = mRequestToComms.find(llmRequest.mRequestId);
            TLLM_CHECK_WITH_INFO(
                (it != mRequestToComms.end()), "sendSync() must be called with request returned by recvRequestInfo().");
            for (auto&& comm : it->second)
            {
                comms.push_back(executor::kv_cache::ProcessInfo{comm.get()});
            }
        }
        auto const& contextState = llmRequest.getDataTransceiverState();
        TLLM_CHECK(contextState.getCommState());
        TLLM_CHECK(contextState.getCacheState());
        auto const& destCacheState = contextState.getCacheState().value();
        mFormatter->formatInput(mComm, llmRequest, std::move(comms), mSelfState.getCacheState().value(),
            mSelfState.getCommState().value().getSelfIdx(), destCacheState, mBufferManager);
        {
            // For now, the connection will be dropped once the transfer is completed
            std::unique_lock<std::mutex> lk(mMtx);
            mRequestToComms.erase(llmRequest.mRequestId);
        }
    }

private:
    void initSelfIps()
    {

#if __linux__
        struct ifaddrs *ifa, *ifaddr;
        void* tmpAddrPtr;

        TLLM_CHECK_WITH_INFO(getifaddrs(&ifaddr) == 0, " UCX initSelfIps getifaddrs call failed\n");
        TLLM_CHECK_WITH_INFO((ifaddr != NULL), "UCX initSelfIps getifaddrs call failed\n");

        for (ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next)
        {

            if (ifa->ifa_addr && ifa->ifa_addr->sa_family == AF_INET)
            {
                tmpAddrPtr = &((struct sockaddr_in*) ifa->ifa_addr)->sin_addr;
                char buffer[INET_ADDRSTRLEN];
                inet_ntop(AF_INET, tmpAddrPtr, buffer, INET_ADDRSTRLEN);

                mSelfIps.insert(std::string(buffer));
            }
            else if (ifa->ifa_addr && ifa->ifa_addr->sa_family == AF_INET6)
            {
                tmpAddrPtr = &((struct sockaddr_in6*) ifa->ifa_addr)->sin6_addr;
                char buffer[INET6_ADDRSTRLEN];
                inet_ntop(AF_INET6, tmpAddrPtr, buffer, INET6_ADDRSTRLEN);
                mSelfIps.insert(std::string(buffer));
            }
        }

        freeifaddrs(ifaddr);

#endif
    }

    UcxComm mComm;
    std::unique_ptr<UcxCommFactory> mFactory;
    executor::DataTransceiverState mSelfState;
    TFormatter mFormatter;

    std::shared_ptr<ucxx::Context> mContext;
    std::shared_ptr<ucxx::Worker> mWorker;

    std::mutex mMtx;
    std::map<LlmRequest::RequestIdType, std::vector<std::unique_ptr<UcxEndpoint>>> mRequestToComms;
    runtime::BufferManager mBufferManager;
    std::unordered_set<std::string> mSelfIps;
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
