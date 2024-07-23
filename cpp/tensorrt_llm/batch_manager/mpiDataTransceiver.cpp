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

void MpiComm::sendRequestId(const LlmRequest::RequestIdType requestId, const SizeType32 responderRank) const
{
    MpiId id{MpiId::REQUEST_SEND};
    mComm->send(std::addressof(id), 1, mpi::MpiType::kUINT64, responderRank, kMPI_ID_TAG);
    mComm->send(std::addressof(requestId), 1, mpi::MpiType::kUINT64, responderRank, kMPI_ID_TAG);
}

std::pair<int, LlmRequest::RequestIdType> MpiComm::recvRequestId() const
{
#if ENABLE_MULTI_DEVICE
    MpiId id;
    MPI_Status status;
    MPI_Recv(std::addressof(id), 1, MPI_INT64_T, MPI_ANY_SOURCE, kMPI_ID_TAG, static_cast<MPI_Comm>(*mComm),
        std::addressof(status));
    TLLM_CHECK(id == MpiId::REQUEST_SEND);
    auto requesterRank{status.MPI_SOURCE};
    LlmRequest::RequestIdType requestId;
    mComm->recv(std::addressof(requestId), 1, mpi::MpiType::kUINT64, requesterRank, kMPI_ID_TAG);
    return {requesterRank, requestId};
#else
    TLLM_THROW("Multi device support is disabled.");
#endif
}

LlmRequest::RequestIdType MpiComm::recvRequestId(const SizeType32 requesterRank) const
{
    MpiId id;
    LlmRequest::RequestIdType requestId;
    mComm->recv(std::addressof(id), 1, mpi::MpiType::kUINT64, requesterRank, kMPI_ID_TAG);
    TLLM_CHECK(id == MpiId::REQUEST_SEND);
    mComm->recv(std::addressof(requestId), 1, mpi::MpiType::kUINT64, requesterRank, kMPI_ID_TAG);
    return requestId;
}

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
        auto handler = std::make_unique<RequestHandler>(std::move(context), mSenders);
        for (auto const rank : handler->getContext().getRanks())
        {
            bool success{false};
            std::tie(std::ignore, success) = mRankToHandler.insert({rank, handler.get()});
            TLLM_CHECK(success);
        }
        mRequestHandlers.emplace_back(std::move(handler));
    }
    // TODO: Supports multiple engines and allocates a thread to each engine.
    mResponseFuture = std::async(std::launch::async, &MpiResponder::response, this);
}

void MpiResponder::response()
{
    try
    {
        mComm->setCudaDevice();
        while (!mTerminate || !mAnyReady)
        {
            if (!mAnyReady)
            {
                std::unique_lock lk(mCondMutex);
                mResponderCv.wait(lk, [&]() { return (mAnyReady || mTerminate); });
            }
            if (mTerminate)
            {
                break;
            }
            if (!isSending())
            {
                mCurrentRequest = recvRequestId();
            }
            auto it = getCurrentResponse();
            if (it != mReadyResponses.end())
            {
                send(it);
            }
            else
            {
                std::unique_lock lk(mCondMutex);
                mResponderCv.wait(lk, [&]() { return (mAnyReady || mTerminate); });
            }
        }
    }
    catch (std::exception const&)
    {
        for (auto& it : mReadyResponses)
        {
            it.second.mPromise.set_exception(std::current_exception());
        }
    }
}

// After the responder terminates, the requester cannot send new requests, otherwise it will cause MPI
// exceptions.
void MpiResponder::terminate()
{
    {
        std::unique_lock lk(mCondMutex);
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
    std::unique_lock lkResp(mResponderMutex);
    mReadyResponses.erase(it);
    if (mReadyResponses.empty())
    {
        std::unique_lock lkCond(mCondMutex);
        mAnyReady = false;
    }
}

std::future<void> MpiResponder::respondAndSendAsync(LlmRequest const& llmRequest)
{
    std::promise<void> promise;
    auto future = promise.get_future();
    {
        std::unique_lock lkResp(mResponderMutex);
        mReadyResponses.emplace(llmRequest.mRequestId, Response{std::addressof(llmRequest), std::move(promise)});
        std::unique_lock lkCond(mCondMutex);
        mAnyReady = true;
    }
    mResponderCv.notify_all();
    return future;
}

std::pair<MpiResponder::RequestIdType, MpiResponder::RequestHandler const*> MpiResponder::recvRequestId()
{
    auto const [requesterRank, requestId] = mComm->recvRequestId();
    for (auto const rank : mRankToHandler.at(requesterRank)->getContext().getRanks())
    {
        if (rank == requesterRank)
        {
            continue;
        }
        MpiResponder::RequestIdType id{mComm->recvRequestId(rank)};
        TLLM_CHECK(requestId == id);
    }
    return {requestId, mRankToHandler.at(requesterRank)};
}

void MpiResponder::send(std::map<RequestIdType, Response>::iterator it)
{
    auto const& handler = *getCurrentHandler();
    handler(*it->second.mRequest);
    it->second.mPromise.set_value();
    removeResponse(it);
    mCurrentRequest = std::nullopt;
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
        (responderContexts.size() == 1 && mReceivers.size() == 1), "Currently only one engine transfer is supported.");
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
    for (auto const& handler : mResponseHandlers)
    {
        if (handler->getContext().getRanks() == context.getRanks())
        {
            (*handler)(llmRequest);
            break;
        }
    }
}

} // namespace tensorrt_llm::batch_manager
