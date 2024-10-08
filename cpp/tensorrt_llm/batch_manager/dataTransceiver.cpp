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
#include "tensorrt_llm/batch_manager/utils/staticThreadPool.h"
#include "tensorrt_llm/common/utils.h"
#include <map>

namespace tensorrt_llm::batch_manager
{

RequestInfo::RequestInfo(LlmRequest::RequestIdType requestId, executor::DataTransceiverState transState)
    : mRequestId{requestId}
    , mTransState{std::move(transState)}
{
}

bool RequestInfo::operator==(RequestInfo const& rhs) const
{
    return mRequestId == rhs.mRequestId && mTransState == rhs.mTransState;
}

LlmRequest::RequestIdType RequestInfo::getRequestId() const noexcept
{
    return mRequestId;
}

executor::DataTransceiverState const& RequestInfo::getTransState() const noexcept
{
    return mTransState;
}

void RequestInfo::serialize(RequestInfo const& requestInfo, std::ostream& os)
{
    namespace su = executor::serialize_utils;
    su::serialize(requestInfo.mRequestId, os);
    su::serialize(requestInfo.mTransState, os);
}

RequestInfo RequestInfo::deserialize(std::istream& is)
{
    namespace su = executor::serialize_utils;
    auto requestId = su::deserialize<decltype(mRequestId)>(is);
    auto transState = su::deserialize<decltype(mTransState)>(is);
    return RequestInfo{requestId, std::move(transState)};
}

std::size_t RequestInfo::serializedSize(RequestInfo const& requestInfo)
{
    namespace su = executor::serialize_utils;
    std::size_t totalSize = 0;
    totalSize += su::serializedSize(requestInfo.mRequestId);
    totalSize += su::serializedSize(requestInfo.mTransState);
    return totalSize;
}

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

    [[nodiscard]] executor::kv_cache::CommState const& getCommState() const
    {
        return mSender->getCommState();
    }

    void setCommState(executor::kv_cache::CommState const& commState)
    {
        mSender->setCommState(commState);
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
            tensorrt_llm::common::setThreadName("dataTransResp");
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
                    mCurrentRequest = mSender->recvRequestInfo().getRequestId();
                }
                auto it = getCurrentResponse();
                if (it != mReadyResponses.end())
                {
                    mSender->sendSync(*it->second.mRequest);
                    if (mSender->availableRelease(*it->second.mRequest))
                    {
                        it->second.mPromise.set_value();
                        removeResponse(it);
                    }
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
        return mThreadPool.execute(&DataRequester::Impl::requestSync, this, std::cref(llmRequest));
    }

private:
    void requestSync(LlmRequest const& llmRequest)
    {
        TLLM_CUDA_CHECK(cudaSetDevice(mDeviceId));
        mReceiver->sendRequestInfo(llmRequest);
        mReceiver->receiveSync(llmRequest);
    }

    std::unique_ptr<DataReceiver> mReceiver;
    utils::StaticThreadPool mThreadPool{10};
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

executor::kv_cache::CommState const& DataResponder::getCommState() const
{
    return mImpl->getCommState();
}

void DataResponder::setCommState(executor::kv_cache::CommState const& commState)
{
    mImpl->setCommState(commState);
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
