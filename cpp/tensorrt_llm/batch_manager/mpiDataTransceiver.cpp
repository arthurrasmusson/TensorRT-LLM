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

#include "mpiDataTransceiver.h"

namespace tensorrt_llm::batch_manager
{

MpiResponder::MpiResponder(std::vector<std::unique_ptr<DataSender>> senders, MpiComm const& comm,
    std::vector<std::unique_ptr<DataContext>> requesterContexts)
    : mComm{std::addressof(comm)}
    , mSenders{std::move(senders)}
{
#if !ENABLE_MULTI_DEVICE
    TLLM_THROW("MPI responder requires multi-device support.");
#endif
    TLLM_CHECK_WITH_INFO(
        (requesterContexts.size() == 1 && mSenders.size() == 1), "Currently only one engine transfer is supported.");
    for (auto& context : requesterContexts)
    {
        TLLM_CHECK(context);
        mRequestHandlers.emplace_back(std::make_unique<RequestHandler>(std::move(context), mSenders));
    }
    // TODO: Supports multiple engines and allocates a thread to each engine.
    mResponseFuture = std::async(std::launch::async, &MpiResponder::response, this);
}

void MpiResponder::response()
{
    mComm->setCudaDevice();
    while (!mTerminate || !mAnyReady)
    {
        if (!mAnyReady)
        {
            std::unique_lock lk(mResponderMutex);
            mResponderCv.wait(lk, [&]() { return (mAnyReady || mTerminate); });
        }
        if (mTerminate)
        {
            break;
        }
        if (!isSending())
        {
            mCurrentRequestId = recvRequestId();
        }
        auto it = getCurrentResponse();
        if (it != mReadyResponses.end())
        {
            send(it);
        }
        else
        {
            std::unique_lock lk(mResponderMutex);
            mResponderCv.wait(lk, [&]() { return (mAnyReady || mTerminate); });
        }
    }
}

// After the responder terminates, the requester cannot send new requests, otherwise it will cause MPI
// exceptions.
void MpiResponder::terminate()
{
    {
        std::unique_lock lk(mResponderMutex);
        mTerminate = true;
    }
    // We don't have to wait for the future. If another thread is sending data, it won't pay attention
    // to the terminate flag.
    mResponderCv.notify_all();
}

std::map<MpiResponder::RequestIdType, MpiResponder::Response>::iterator MpiResponder::getCurrentResponse()
{
    std::unique_lock lk(mResponderMutex);
    return mReadyResponses.find(getCurrentRequestId());
}

void MpiResponder::removeResponse(std::map<RequestIdType, Response>::iterator it)
{
    std::unique_lock lk(mResponderMutex);
    mReadyResponses.erase(it);
    if (mReadyResponses.empty())
    {
        mAnyReady = false;
    }
}

std::future<void> MpiResponder::respondAndSendAsync(LlmRequest const& llmRequest)
{
    std::promise<void> promise;
    auto future = promise.get_future();
    {
        std::unique_lock lk(mResponderMutex);
        mReadyResponses.emplace(llmRequest.mRequestId, Response{std::addressof(llmRequest), std::move(promise)});
        mAnyReady = true;
    }
    mResponderCv.notify_all();
    return future;
}

MpiResponder::RequestIdType MpiResponder::recvRequestId()
{
    TLLM_CHECK_WITH_INFO(
        mRequestHandlers.size() == 1, "Currently, only single-group receiver-sender cache transfers are supported.");
    auto const& ranks = mRequestHandlers.front()->getContext().getRanks();
    std::optional<MpiResponder::RequestIdType> reqId{};
    for (auto rank : ranks)
    {
        MpiResponder::RequestIdType id{mComm->recvRequestId(rank)};
        if (reqId)
        {
            TLLM_CHECK(reqId.value() == id);
        }
        else
        {
            reqId = id;
        }
    }
    return reqId.value();
}

void MpiResponder::send(std::map<RequestIdType, Response>::iterator it)
{
    TLLM_CHECK_WITH_INFO(
        mRequestHandlers.size() == 1, "Currently, only single-group receiver-sender cache transfers are supported.");
    auto const& handler = *mRequestHandlers.front();
    handler(*it->second.mRequest);
    it->second.mPromise.set_value();
    removeResponse(it);
    mCurrentRequestId = std::nullopt;
}

MpiRequester::MpiRequester(std::vector<std::unique_ptr<DataReceiver>> receivers, MpiComm const& comm,
    std::vector<std::unique_ptr<DataContext>> responderContexts)
    : mComm{std::addressof(comm)}
    , mReceivers{std::move(receivers)}
{
#if !ENABLE_MULTI_DEVICE
    TLLM_THROW("MPI requester requires multi-device support.");
#endif
    TLLM_CHECK_WITH_INFO(
        (mReceivers.size() == 1 && responderContexts.size() == 1), "Currently only one engine transfer is supported.");
    for (auto& context : responderContexts)
    {
        TLLM_CHECK(context);
        mResponseHandlers.emplace_back(std::make_unique<ResponseHandler>(std::move(context), mReceivers));
    }
}

void MpiRequester::requestSync(LlmRequest const& llmRequest, DataContext const& context)
{
    mComm->setCudaDevice();
    auto const& responders = context.getRanks();
    for (auto const responderRank : responders)
    {
        mComm->sendRequestId(llmRequest.mRequestId, responderRank);
    }
    auto const& handler = *mResponseHandlers.front();
    handler(llmRequest);
}

} // namespace tensorrt_llm::batch_manager
