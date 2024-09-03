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

#include "tensorrt_llm/batch_manager/dataTransceiver.h"
#include <map>

namespace tensorrt_llm::batch_manager
{

class DataResponder::Impl
{
public:
    using RequestIdType = LlmRequest::RequestIdType;

    Impl(std::unique_ptr<DataSender> sender)
        : mSender{std::move(sender)}
    {
        TLLM_CHECK(mSender);
        TLLM_CUDA_CHECK(cudaGetDevice(&mDeviceId));
        mResponseFuture = std::async(std::launch::async, &Impl::response, this);
    }

    [[nodiscard]] std::future<void> respondAndSendAsync(LlmRequest const& llmRequest)
    {
        std::promise<void> promise;
        auto future = promise.get_future();
        {
            {
                std::unique_lock lkResp(mResponderMutex);
                mReadyResponses.emplace(
                    llmRequest.mRequestId, Response{std::addressof(llmRequest), std::move(promise)});
            }
            std::unique_lock lkCond(mCondMutex);
            mAnyReady = true;
        }
        mResponderCv.notify_all();
        return future;
    }

    ~Impl()
    {
        terminate();
    }

private:
    struct Response
    {
        LlmRequest const* mRequest;
        std::promise<void> mPromise;
    };

    void response()
    {
        try
        {
            TLLM_CUDA_CHECK(cudaSetDevice(mDeviceId));
            while (!mTerminate || !mAnyReady)
            {
                if (!mAnyReady)
                {
                    std::unique_lock lk(mCondMutex);
                    mResponderCv.wait(lk, [this]() { return (mAnyReady || mTerminate); });
                }
                if (mTerminate)
                {
                    break;
                }
                if (!isSending() && !mReadyResponses.empty())
                {
                    mCurrentRequest = mSender->recvRequestId();
                }
                auto it = getCurrentResponse();
                if (it != mReadyResponses.end())
                {
                    mSender->sendSync(*it->second.mRequest);
                    it->second.mPromise.set_value();
                    removeResponse(it);
                    mCurrentRequest = std::nullopt;
                }
                else
                {
                    std::unique_lock lk(mCondMutex);
                    mResponderCv.wait(lk, [this]() { return (mAnyReady || mTerminate); });
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

    void terminate()
    {
        {
            std::unique_lock lk(mCondMutex);
            mTerminate = true;
        }
        // We don't have to wait for the future. If another thread is sending data, it won't pay attention
        // to the terminate flag.
        mResponderCv.notify_all();
    }

    void removeResponse(std::map<RequestIdType, Response>::iterator it)
    {
        {
            std::unique_lock lkResp(mResponderMutex);
            mReadyResponses.erase(it);
        }
        if (mReadyResponses.empty())
        {
            std::unique_lock lkCond(mCondMutex);
            mAnyReady = false;
        }
    }

    [[nodiscard]] bool isSending() const
    {
        return mCurrentRequest.has_value();
    }

    [[nodiscard]] RequestIdType getCurrentRequestId() const
    {
        return mCurrentRequest.value();
    }

    [[nodiscard]] std::map<RequestIdType, Response>::iterator getCurrentResponse()
    {
        std::unique_lock lk(mResponderMutex);
        return mReadyResponses.find(getCurrentRequestId());
    }

private:
    std::optional<RequestIdType> mCurrentRequest;
    std::map<RequestIdType, Response> mReadyResponses;
    std::mutex mResponderMutex, mCondMutex;
    std::atomic<bool> mAnyReady{false}, mTerminate{false};
    std::condition_variable mResponderCv;
    std::future<void> mResponseFuture;
    std::unique_ptr<DataSender> mSender;
    int mDeviceId{-1};
};

class DataRequester::Impl
{
public:
    Impl(std::unique_ptr<DataReceiver> receiver)
        : mReceiver{std::move(receiver)}
    {
        TLLM_CHECK(mReceiver);
        TLLM_CUDA_CHECK(cudaGetDevice(&mDeviceId));
    }

    [[nodiscard]] std::future<void> requestAndReceiveAsync(LlmRequest const& llmRequest)
    {
        // TODO: Modify the implementation here to avoid frequent thread creation.
        return std::async(std::launch::async, &DataRequester::Impl::requestSync, this, std::cref(llmRequest));
    }

private:
    void requestSync(LlmRequest const& llmRequest)
    {
        TLLM_CUDA_CHECK(cudaSetDevice(mDeviceId));
        mReceiver->sendRequestId(llmRequest);
        mReceiver->receiveSync(llmRequest);
    }

    std::unique_ptr<DataReceiver> mReceiver;
    int mDeviceId{-1};
};

DataResponder::DataResponder(std::unique_ptr<DataSender> sender)
    : mImpl{std::make_unique<Impl>(std::move(sender))}
{
}

std::future<void> DataResponder::respondAndSendAsync(LlmRequest const& llmRequest) const
{
    return mImpl->respondAndSendAsync(llmRequest);
}

DataResponder::~DataResponder() = default;

DataRequester::DataRequester(std::unique_ptr<DataReceiver> receiver)
    : mImpl{std::make_unique<Impl>(std::move(receiver))}
{
}

std::future<void> DataRequester::requestAndReceiveAsync(LlmRequest const& llmRequest) const
{
    return mImpl->requestAndReceiveAsync(llmRequest);
}

DataRequester::~DataRequester() = default;

} // namespace tensorrt_llm::batch_manager
