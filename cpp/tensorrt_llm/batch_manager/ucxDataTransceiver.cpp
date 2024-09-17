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

std::unique_ptr<DataResponder> makeUcxCacheResponder(kv_cache_manager::KVCacheManager* cacheManager)
{
    using namespace tensorrt_llm::batch_manager::kv_cache_manager;
    auto sender = std::make_unique<UcxDataSender<CacheState>>(
        std::make_unique<UcxCommFactory>(), std::make_unique<CacheOutputFormatter<UcxComm>>(cacheManager));
    return std::make_unique<DataResponder>(std::move(sender));
}

std::unique_ptr<DataRequester> makeUcxCacheRequester(kv_cache_manager::KVCacheManager* cacheManager)
{
    using namespace tensorrt_llm::batch_manager::kv_cache_manager;
    return std::make_unique<DataRequester>(std::make_unique<UcxDataReceiver<CacheState>>(
        std::make_unique<UcxCommFactory>(), std::make_unique<CacheInputFormatter<UcxComm>>(cacheManager)));
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

LlmRequest::RequestIdType UcxComm::recvRequestId() const
{
    LlmRequest::RequestIdType id;
    // Guard to ensure CUDA context is initialized for UCX ops
    TLLM_CUDA_CHECK(cudaFree(0));
    TLLM_CHECK_WITH_INFO((mEndpoint), "recvRequestId called without established communicator channel.");
    auto completionCallback = [this](ucs_status_t, ucxx::RequestCallbackUserData) -> void { mCv.notify_all(); };
    auto req = mEndpoint->tagRecv(
        &id, sizeof(LlmRequest::RequestIdType), kID_TAG, ucxx::TagMaskFull, false, completionCallback);
    std::unique_lock<std::mutex> lk(mMtx);
    mCv.wait(lk, [&req]() { return req->isCompleted(); });
    // throw if there is error
    req->checkError();
    return id;
}

void UcxComm::sendRequestId(LlmRequest::RequestIdType id) const
{
    // Guard to ensure CUDA context is initialized for UCX ops
    TLLM_CUDA_CHECK(cudaFree(0));
    TLLM_CHECK_WITH_INFO((mEndpoint), "sendRequestId called without established communicator channel.");
    auto completionCallback = [this](ucs_status_t, ucxx::RequestCallbackUserData) -> void { mCv.notify_all(); };
    auto req = mEndpoint->tagSend(&id, sizeof(LlmRequest::RequestIdType), kID_TAG, false, completionCallback);
    std::unique_lock<std::mutex> lk(mMtx);
    mCv.wait(lk, [&req]() { return req->isCompleted(); });
    // throw if there is error
    req->checkError();
}

} // namespace tensorrt_llm::batch_manager
