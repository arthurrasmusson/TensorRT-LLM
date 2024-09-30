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

#include "ucxDataTransceiver.h"

#include "tensorrt_llm/batch_manager/cacheFormatter.h"

namespace tensorrt_llm::batch_manager
{

using CacheState = tensorrt_llm::executor::kv_cache::CacheState;

template class UcxDataSender<CacheState>;
template class UcxDataReceiver<CacheState>;

std::unique_ptr<DataResponder> makeUcxCacheResponder(
    executor::kv_cache::CacheState selfCacheState, SizeType32 selfIndex, kv_cache_manager::KVCacheManager* cacheManager)
{
    using namespace tensorrt_llm::batch_manager::kv_cache_manager;
    auto sender = std::make_unique<UcxDataSender<CacheState>>(std::make_unique<UcxCommFactory>(),
        std::move(selfCacheState), selfIndex, std::make_unique<CacheOutputFormatter<UcxComm>>(cacheManager));
    return std::make_unique<DataResponder>(std::move(sender));
}

std::unique_ptr<DataRequester> makeUcxCacheRequester(
    executor::kv_cache::CacheState selfCacheState, SizeType32 selfIndex, kv_cache_manager::KVCacheManager* cacheManager)
{
    using namespace tensorrt_llm::batch_manager::kv_cache_manager;
    return std::make_unique<DataRequester>(
        std::make_unique<UcxDataReceiver<CacheState>>(std::make_unique<UcxCommFactory>(), std::move(selfCacheState),
            selfIndex, std::make_unique<CacheInputFormatter<UcxComm>>(cacheManager)));
}

void UcxComm::sendBuffer(runtime::IBuffer const& buf) const
{
    // Guard to ensure CUDA context is initialized for UCX ops
    TLLM_CUDA_CHECK(cudaFree(0));
    TLLM_CHECK_WITH_INFO((mEndpoint), "sendBuffer called without established communicator channel.");
    auto completionCallback = [this](ucs_status_t, ucxx::RequestCallbackUserData) -> void { mCv.notify_all(); };
    auto req
        = mEndpoint->tagSend(const_cast<void*>(buf.data()), buf.getSizeInBytes(), kDATA_TAG, false, completionCallback);
    std::unique_lock<std::mutex> lk(mMtx);
    mCv.wait(lk, [&req]() { return req->isCompleted(); });
    // throw if there is error
    req->checkError();
}

void UcxComm::recvBuffer(runtime::IBuffer& buf) const
{
    // Guard to ensure CUDA context is initialized for UCX ops
    TLLM_CUDA_CHECK(cudaFree(0));
    TLLM_CHECK_WITH_INFO((mEndpoint), "recvBuffer called without established communicator channel.");
    auto completionCallback = [this](ucs_status_t, ucxx::RequestCallbackUserData) -> void { mCv.notify_all(); };
    auto req
        = mEndpoint->tagRecv(buf.data(), buf.getSizeInBytes(), kDATA_TAG, ucxx::TagMaskFull, false, completionCallback);
    std::unique_lock<std::mutex> lk(mMtx);
    mCv.wait(lk, [&req]() { return req->isCompleted(); });
    // throw if there is error
    req->checkError();
}

RequestInfo UcxComm::recvRequestInfo() const
{
    std::string serializedInfo;
    std::size_t infoSize{0};
    // Guard to ensure CUDA context is initialized for UCX ops
    TLLM_CUDA_CHECK(cudaFree(0));
    TLLM_CHECK_WITH_INFO((mEndpoint), "recvRequestInfo called without established communicator channel.");
    auto completionCallback = [this](ucs_status_t, ucxx::RequestCallbackUserData) -> void { mCv.notify_all(); };
    {
        auto req
            = mEndpoint->tagRecv(&infoSize, sizeof(infoSize), kID_TAG, ucxx::TagMaskFull, false, completionCallback);
        std::unique_lock<std::mutex> lk(mMtx);
        mCv.wait(lk, [&req]() { return req->isCompleted(); });
        // throw if there is error
        req->checkError();
        serializedInfo.resize(infoSize);
    }
    {
        auto req = mEndpoint->tagRecv(
            serializedInfo.data(), infoSize, kID_TAG, ucxx::TagMaskFull, false, completionCallback);
        std::unique_lock<std::mutex> lk(mMtx);
        mCv.wait(lk, [&req]() { return req->isCompleted(); });
        // throw if there is error
        req->checkError();
    }
    std::istringstream iss(serializedInfo);
    RequestInfo info{RequestInfo::deserialize(iss)};
    return info;
}

void UcxComm::sendRequestInfo(RequestInfo const& info) const
{
    // Guard to ensure CUDA context is initialized for UCX ops
    TLLM_CUDA_CHECK(cudaFree(0));

    std::ostringstream oss;
    RequestInfo::serialize(info, oss);
    auto const& serializedInfo = oss.str();
    std::size_t infoSize = serializedInfo.size();

    TLLM_CHECK_WITH_INFO((mEndpoint), "sendRequestInfo called without established communicator channel.");
    auto completionCallback = [this](ucs_status_t, ucxx::RequestCallbackUserData) -> void { mCv.notify_all(); };
    {
        auto req = mEndpoint->tagSend(&infoSize, sizeof(infoSize), kID_TAG, false, completionCallback);
        std::unique_lock<std::mutex> lk(mMtx);
        mCv.wait(lk, [&req]() { return req->isCompleted(); });
        // throw if there is error
        req->checkError();
    }
    {
        auto req = mEndpoint->tagSend(
            const_cast<char*>(serializedInfo.data()), infoSize, kID_TAG, false, completionCallback);
        std::unique_lock<std::mutex> lk(mMtx);
        mCv.wait(lk, [&req]() { return req->isCompleted(); });
        // throw if there is error
        req->checkError();
    }
}

} // namespace tensorrt_llm::batch_manager
