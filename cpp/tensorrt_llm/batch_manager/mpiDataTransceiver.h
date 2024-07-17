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

#include <map>

#include "dataTransceiver.h"
#include "tensorrt_llm/common/mpiUtils.h"

namespace tensorrt_llm::batch_manager
{

class MpiComm
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;

    explicit MpiComm(mpi::MpiComm const& comm)
        : mComm{std::addressof(comm)}
    {
        cudaGetDevice(&mDeviceId);
    }

    virtual ~MpiComm() = default;

    virtual void sendRequestId(const LlmRequest::RequestIdType requestId, const SizeType32 responderRank) const
    {
        MpiId id{MpiId::REQUEST_SEND};
        mComm->send(std::addressof(id), 1, mpi::MpiType::kUINT64, responderRank, kMPI_ID_TAG);
        mComm->send(std::addressof(requestId), 1, mpi::MpiType::kUINT64, responderRank, kMPI_ID_TAG);
    }

    [[nodiscard]] virtual LlmRequest::RequestIdType recvRequestId(const SizeType32 requesterRank) const
    {
        MpiId id;
        LlmRequest::RequestIdType requestId;
        mComm->recv(std::addressof(id), 1, mpi::MpiType::kUINT64, requesterRank, kMPI_ID_TAG);
        TLLM_CHECK(id == MpiId::REQUEST_SEND);
        mComm->recv(std::addressof(requestId), 1, mpi::MpiType::kUINT64, requesterRank, kMPI_ID_TAG);
        return requestId;
    }

    virtual void sendBuffer(runtime::IBuffer const& buf, int dest) const
    {
        mComm->send(buf, dest, kMPI_DATA_TAG);
    }

    virtual void recvBuffer(runtime::IBuffer& buf, int dest) const
    {
        mComm->recv(buf, dest, kMPI_DATA_TAG);
    }

    virtual void setCudaDevice() const
    {
        cudaSetDevice(mDeviceId);
    }

    [[nodiscard]] virtual int getRank() const
    {
        return mComm->getRank();
    }

    [[nodiscard]] virtual int getSize() const
    {
        return mComm->getSize();
    }

private:
    enum class MpiId : uint64_t
    {
        REQUEST_SEND = 1,
        TERMINATION = 2
    };

    static constexpr int32_t kMPI_ID_TAG{127};
    static constexpr int32_t kMPI_DATA_TAG{1023};

    mpi::MpiComm const* mComm{};
    int mDeviceId{-1};
};

// Use MPI to transfer KV cache between processes. Its design is based on a few assumptions:
// 1. MPI is thread-safe, but by default, it only supports sequential transmission.
// 2. The efficiency of MPI transmission is not strongly correlated with the size of the data
//    being transmitted.
class MpiResponder final : public DataResponder
{
public:
    using RankIdType = int;
    using RequestIdType = LlmRequest::RequestIdType;

    MpiResponder(std::vector<std::unique_ptr<DataSender>> senders, MpiComm const& comm,
        std::vector<std::unique_ptr<DataContext>> requesterContexts);

    [[nodiscard]] std::future<void> respondAndSendAsync(LlmRequest const& llmRequest) override;

    ~MpiResponder()
    {
        terminate();
    }

private:
    struct Response
    {
        LlmRequest const* mRequest;
        std::promise<void> mPromise;
    };

    class RequestHandler
    {
    public:
        RequestHandler(
            std::unique_ptr<DataContext> responseContext, std::vector<std::unique_ptr<DataSender>> const& senders)
            : mContext{std::move(responseContext)}
        {
            TLLM_CHECK(mContext);
            for (auto const& sender : senders)
            {
                TLLM_CHECK(sender);
                if (sender->inquireSupport(mContext.get()))
                {
                    mSender = sender.get();
                    break;
                }
            }
            TLLM_CHECK_WITH_INFO(mSender, "There is no suitable sender available for selection.");
        }

        void operator()(LlmRequest const& request) const
        {
            return mSender->send(request, *mContext);
        }

        [[nodiscard]] DataContext const& getContext() const
        {
            return *mContext;
        }

    private:
        std::unique_ptr<DataContext> mContext;
        DataSender* mSender{};
    };

    void response();

    void terminate();

    [[nodiscard]] RequestIdType recvRequestId();

    void send(std::map<RequestIdType, Response>::iterator it);

    void removeResponse(std::map<RequestIdType, Response>::iterator it);

    [[nodiscard]] bool isSending() const
    {
        return mCurrentRequestId.has_value();
    }

    [[nodiscard]] RequestIdType getCurrentRequestId() const
    {
        return mCurrentRequestId.value();
    }

    [[nodiscard]] std::map<RequestIdType, Response>::iterator getCurrentResponse();

    MpiComm const* mComm{};
    std::vector<std::unique_ptr<DataSender>> mSenders;
    std::vector<std::unique_ptr<RequestHandler>> mRequestHandlers;
    std::optional<RequestIdType> mCurrentRequestId;
    std::map<RequestIdType, Response> mReadyResponses;

    std::mutex mResponderMutex;
    std::atomic<bool> mAnyReady{false};
    std::condition_variable mResponderCv;
    std::future<void> mResponseFuture;
    std::atomic<bool> mTerminate{false};
};

class MpiRequester final : public DataRequester
{
public:
    using RankIdType = int;

    MpiRequester(std::vector<std::unique_ptr<DataReceiver>> receivers, MpiComm const& comm,
        std::vector<std::unique_ptr<DataContext>> responderContexts);

    [[nodiscard]] std::future<void> requestAndReceiveAsync(
        LlmRequest const& llmRequest, DataContext const& context) override
    {
        // TODO: Modify the implementation here to avoid frequent thread creation.
        return std::async(
            std::launch::async, &MpiRequester::requestSync, this, std::cref(llmRequest), std::cref(context));
    }

private:
    class ResponseHandler
    {
    public:
        ResponseHandler(
            std::unique_ptr<DataContext> requestContext, std::vector<std::unique_ptr<DataReceiver>> const& receivers)
            : mContext{std::move(requestContext)}
        {
            TLLM_CHECK(mContext);
            for (auto const& receiver : receivers)
            {
                TLLM_CHECK(receiver);
                if (receiver->inquireSupport(mContext.get()))
                {
                    mReceiver = receiver.get();
                    break;
                }
            }
            TLLM_CHECK_WITH_INFO(mReceiver, "There is no suitable receiver available for selection.");
        }

        void operator()(LlmRequest const& request) const
        {
            return mReceiver->receive(request, *mContext);
        }

        [[nodiscard]] DataContext const& getContext() const
        {
            return *mContext;
        }

    private:
        std::unique_ptr<DataContext> mContext;
        DataReceiver* mReceiver{};
    };

    void requestSync(LlmRequest const& llmRequest, DataContext const& context);

    MpiComm const* mComm{};
    std::vector<std::unique_ptr<DataReceiver>> mReceivers;
    std::vector<std::unique_ptr<ResponseHandler>> mResponseHandlers;
};

} // namespace tensorrt_llm::batch_manager
