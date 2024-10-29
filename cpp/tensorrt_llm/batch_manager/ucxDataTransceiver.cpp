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

#if ENABLE_UCX

#include "ucxDataTransceiver.h"

#include "tensorrt_llm/batch_manager/cacheFormatter.h"
#include <chrono>

namespace tensorrt_llm::batch_manager
{

using CacheState = tensorrt_llm::executor::kv_cache::CacheState;

template class UcxDataSender<CacheState>;
template class UcxDataReceiver<CacheState>;

#if __cplusplus
extern "C"
{
#endif

    std::unique_ptr<DataResponder> makeUcxCacheResponder(executor::kv_cache::CacheState selfCacheState,
        SizeType32 selfIndex, kv_cache_manager::KVCacheManager* cacheManager)
    {
        using namespace tensorrt_llm::batch_manager::kv_cache_manager;
        auto sender = std::make_unique<UcxDataSender<CacheState>>(std::make_unique<UcxCommFactory>(),
            std::move(selfCacheState), selfIndex, std::make_unique<CacheOutputFormatter<UcxComm>>(cacheManager));
        return std::make_unique<DataResponder>(std::move(sender));
    }

    std::unique_ptr<DataRequester> makeUcxCacheRequester(executor::kv_cache::CacheState selfCacheState,
        SizeType32 selfIndex, kv_cache_manager::KVCacheManager* cacheManager)
    {
        using namespace tensorrt_llm::batch_manager::kv_cache_manager;
        return std::make_unique<DataRequester>(
            std::make_unique<UcxDataReceiver<CacheState>>(std::make_unique<UcxCommFactory>(), std::move(selfCacheState),
                selfIndex, std::make_unique<CacheInputFormatter<UcxComm>>(cacheManager)));
    }

#if __cplusplus
}
#endif

void UcxComm::sendBuffer(runtime::IBuffer const& buf) const
{
    ucxx::Tag dataTag{mSendTag};

    // Guard to ensure CUDA context is initialized for UCX ops
    TLLM_CUDA_CHECK(cudaFree(0));
    TLLM_CHECK_WITH_INFO((mEndpoint), "sendBuffer called without established communicator channel.");
    auto completionCallback = [this](ucs_status_t, ucxx::RequestCallbackUserData) -> void { mCv.notify_all(); };
    auto req
        = mEndpoint->tagSend(const_cast<void*>(buf.data()), buf.getSizeInBytes(), dataTag, false, completionCallback);
    std::unique_lock<std::mutex> lk(mMtx);
    mCv.wait(lk, [&req]() { return req->isCompleted(); });
    // throw if there is error
    req->checkError();
}

void UcxComm::recvBuffer(runtime::IBuffer& buf) const
{
    ucxx::Tag dataTag{mRecvTag};

    // Guard to ensure CUDA context is initialized for UCX ops
    TLLM_CUDA_CHECK(cudaFree(0));
    TLLM_CHECK_WITH_INFO((mEndpoint), "recvBuffer called without established communicator channel.");
    auto completionCallback = [this](ucs_status_t, ucxx::RequestCallbackUserData) -> void { mCv.notify_all(); };
    auto req
        = mEndpoint->tagRecv(buf.data(), buf.getSizeInBytes(), dataTag, ucxx::TagMaskFull, false, completionCallback);
    std::unique_lock<std::mutex> lk(mMtx);
    mCv.wait(lk, [&req]() { return req->isCompleted(); });
    // throw if there is error
    req->checkError();
}

RequestInfo UcxComm::recvRequestInfo() const
{
    ucxx::Tag infoTag{mRecvTag | mInfoTag};
    ucxx::TagMask infoMask{mEndpointMask | mFlagMask};

    std::string serializedInfo;
    std::size_t infoSize{0};
    // Guard to ensure CUDA context is initialized for UCX ops
    TLLM_CUDA_CHECK(cudaFree(0));
    TLLM_CHECK_WITH_INFO((mEndpoint), "recvRequestInfo called without established communicator channel.");
    auto completionCallback = [this](ucs_status_t, ucxx::RequestCallbackUserData) -> void { mCv.notify_all(); };
    {
        auto req = mEndpoint->tagRecv(&infoSize, sizeof(infoSize), infoTag, infoMask, false, completionCallback);
        std::unique_lock<std::mutex> lk(mMtx);
        mCv.wait(lk, [&req]() { return req->isCompleted(); });
        // throw if there is error
        req->checkError();
        serializedInfo.resize(infoSize);
    }
    {
        auto req = mEndpoint->tagRecv(serializedInfo.data(), infoSize, infoTag, infoMask, false, completionCallback);
        std::unique_lock<std::mutex> lk(mMtx);
        mCv.wait(lk, [&req]() { return req->isCompleted(); });
        // throw if there is error
        req->checkError();
    }
    std::istringstream iss(serializedInfo);
    RequestInfo info{RequestInfo::deserialize(iss)};
    // [FIXME] setRequestTag(info.getRequestId());
    return info;
}

void UcxComm::sendRequestInfo(RequestInfo const& info) const
{
    // [FIXME] setRequestTag(info.getRequestId());
    ucxx::Tag infoTag{mSendTag | mInfoTag};

    // Guard to ensure CUDA context is initialized for UCX ops
    TLLM_CUDA_CHECK(cudaFree(0));

    std::ostringstream oss;
    RequestInfo::serialize(info, oss);
    auto const& serializedInfo = oss.str();
    std::size_t infoSize = serializedInfo.size();

    TLLM_CHECK_WITH_INFO((mEndpoint), "sendRequestInfo called without established communicator channel.");
    auto completionCallback = [this](ucs_status_t, ucxx::RequestCallbackUserData) -> void { mCv.notify_all(); };
    {
        auto req = mEndpoint->tagSend(&infoSize, sizeof(infoSize), infoTag, false, completionCallback);
        std::unique_lock<std::mutex> lk(mMtx);
        mCv.wait(lk, [&req]() { return req->isCompleted(); });
        // throw if there is error
        req->checkError();
    }
    {
        auto req = mEndpoint->tagSend(
            const_cast<char*>(serializedInfo.data()), infoSize, infoTag, false, completionCallback);
        std::unique_lock<std::mutex> lk(mMtx);
        mCv.wait(lk, [&req]() { return req->isCompleted(); });
        // throw if there is error
        req->checkError();
    }
}

void UcxComm::initializeEndpointTag(int maxTryTimes)
{
    // [FIXME] IP exchange seems to be more robust to ensure tag establishment,
    // i.e. different peers can use the same port number to connect with self worker
    // which results in identical tag if only self / peer port is used.

    // knowing that ucxx::Tag is uint64_t
    ucxx::Tag localPort{0};
    ucxx::Tag remotePort{0};
    char ipStr[INET6_ADDRSTRLEN];
    char portStr[INET6_ADDRSTRLEN];

    ucp_ep_attr_t ep_attr;
    ep_attr.field_mask = UCP_EP_ATTR_FIELD_LOCAL_SOCKADDR | UCP_EP_ATTR_FIELD_REMOTE_SOCKADDR;

    ucs_status_t status = ucp_ep_query(mEndpoint->getHandle(), &ep_attr);
    if (status == UCS_OK)
    {
        ucxx::utils::sockaddr_get_ip_port_str(&ep_attr.local_sockaddr, ipStr, portStr, INET6_ADDRSTRLEN);
        localPort = static_cast<ucxx::Tag>(std::stoull(portStr));

        ucxx::utils::sockaddr_get_ip_port_str(&ep_attr.remote_sockaddr, ipStr, portStr, INET6_ADDRSTRLEN);
        remotePort = static_cast<ucxx::Tag>(std::stoull(portStr));

        // Network port value is defined to fit in 16 bit.
        mSendTag = static_cast<ucxx::Tag>((localPort << (16 + 32)) | (remotePort << 32));
        mRecvTag = static_cast<ucxx::Tag>((remotePort << (16 + 32)) | (localPort << 32));
    }
    else
    {
        // [FIXME] better message
        if (status == UCS_ERR_NOT_CONNECTED && maxTryTimes > 0)
        {
            TLLM_LOG_WARNING("UCX connection has not been established yet. wait 100 ms before retrying. maxTryTimes:%d",
                maxTryTimes);
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            initializeEndpointTag(maxTryTimes - 1);
        }
        else
        {
            TLLM_LOG_WARNING("UCX data transceiver is not created by connecting to a socket address.");
        }
    }
}

void UcxComm::setRequestTag(LlmRequest::RequestIdType const requestId)
{
    uint16_t truncatedRequestId = requestId;
    mSendTag = static_cast<ucxx::Tag>((mSendTag & ~mRequestIdMask) | (truncatedRequestId << 16));
    mRecvTag = static_cast<ucxx::Tag>((mRecvTag & ~mRequestIdMask) | (truncatedRequestId << 16));
}

} // namespace tensorrt_llm::batch_manager

#endif
