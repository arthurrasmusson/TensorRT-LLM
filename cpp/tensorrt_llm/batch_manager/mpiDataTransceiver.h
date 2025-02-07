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

#include "cacheFormatter.h"
#include "dataTransceiver.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/mpiUtils.h"
#include "tensorrt_llm/executor/cache_transmission/cacheConcatenate.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/cudaEvent.h"
#include "tensorrt_llm/runtime/cudaStream.h"

namespace tensorrt_llm::batch_manager
{

class MpiComm : public executor::kv_cache::Communicator
{
public:
    using TPtrContainer = std::vector<std::unique_ptr<MpiComm>>;
    enum class Id : uint64_t
    {
        REQUEST_SEND = 1,
        TERMINATION = 2
    };

    static constexpr int32_t kID_TAG{19};
    static constexpr int32_t KINFO_SIZE_TAG{22};
    static constexpr int32_t KINFO_TAG{32};

    static constexpr int32_t kDATA_TAG{43};

    MpiComm(mpi::MpiComm const& comm)
        : mComm{std::addressof(comm)}
    {
        TLLM_CHECK(mComm);
    }

    [[nodiscard]] bool isThreadSafe() const noexcept override
    {
        return true;
    }

    void recvBuffer(runtime::IBuffer& buf, executor::kv_cache::DataContext const& context,
        executor::kv_cache::ProcessInfo const& processInfo) const override
    {
        int dataTag = ((context.getRequestId() & 0xFFF) << 12) | (kDATA_TAG & 0xFF);
        mComm->recv(buf, processInfo.getRank(), dataTag);
    }

    void sendBuffer(runtime::IBuffer const& buf, executor::kv_cache::DataContext const& context,
        executor::kv_cache::ProcessInfo const& processInfo) const override
    {
        int dataTag = ((context.getRequestId() & 0xFFF) << 12) | (kDATA_TAG & 0xFF);
        mComm->send(buf, processInfo.getRank(), dataTag);
    }

    [[nodiscard]] std::unique_ptr<RequestInfo> recvRequestInfo(
        std::optional<executor::kv_cache::ProcessInfo> const& processInfo = std::nullopt) const override;

    void sendRequestInfo(RequestInfo const& info, executor::kv_cache::ProcessInfo const& processInfo) const override;

private:
    mpi::MpiComm const* mComm{};
};

class MpiDataSender : public DataSender
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using TFormatter = std::unique_ptr<IOFormatter>;
    using RequestMapInfo = std::vector<std::pair<int, executor::DataTransceiverState>>;

    template <typename... TArgs>
    MpiDataSender(mpi::MpiComm const& comm, executor::kv_cache::CacheState selfCacheState, SizeType32 selfIndex,
        TArgs... formatters)
        : mComm{comm}
        , mSelfState{std::move(selfCacheState),
              executor::kv_cache::CommState{
                  tensorrt_llm::mpi::getWorldRanks(tensorrt_llm::mpi::MpiComm::session()), selfIndex}}
        , mBufferManager{std::make_shared<runtime::CudaStream>()}
    {
        mFormatters.emplace_back(std::move(formatters)...);
        TLLM_CHECK(mFormatters.size() == 1);
    }

    [[nodiscard]] RequestInfo recvRequestInfo() override;

    void sendSync(LlmRequest const& llmRequest) override;

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
        auto it = mRequestToComms.find(requestId);
        TLLM_CHECK(it != mRequestToComms.end());
        return mRequestToComms.at(requestId).size();
    }

    void release(LlmRequest::RequestIdType requestId) override
    {
        auto it = mRequestToComms.find(requestId);
        TLLM_CHECK(it != mRequestToComms.end());
        std::unique_lock<std::mutex> lk(mMtxForMap);
        mRequestToComms.erase(it);
    }

private:
    MpiComm mComm;
    std::map<LlmRequest::RequestIdType, RequestMapInfo> mRequestToComms;
    std::vector<TFormatter> mFormatters;
    executor::DataTransceiverState mSelfState;
    std::mutex mMtxForMap;
    runtime::BufferManager mBufferManager;
};

class MpiDataReceiver : public DataReceiver
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using TFormatter = std::unique_ptr<IOFormatter>;

    template <typename... TArgs>
    MpiDataReceiver(mpi::MpiComm const& comm, executor::kv_cache::CacheState selfCacheState, SizeType32 selfIndex,
        TArgs... formatters)
        : mComm{comm}
        , mSelfState{std::move(selfCacheState),
              executor::kv_cache::CommState{
                  tensorrt_llm::mpi::getWorldRanks(tensorrt_llm::mpi::MpiComm::session()), selfIndex}}
    {
        mFormatters.emplace_back(std::move(formatters)...);
        TLLM_CHECK(mFormatters.size() == 1);
    }

    void sendRequestInfo(LlmRequest const& llmRequest) override;

    void receiveSync(LlmRequest const& llmRequest) override;

private:
    MpiComm mComm;
    std::vector<TFormatter> mFormatters;
    executor::DataTransceiverState mSelfState;

    struct ReceiveCacheResource
    {

        runtime::BufferManager mBufferManager;
        runtime::CudaEvent mCudaEvent;

        ReceiveCacheResource(runtime::BufferManager&& bufferManager, runtime::CudaEvent&& cudaEvent)
            : mBufferManager(bufferManager)
            , mCudaEvent(std::move(cudaEvent))
        {
        }
    };

    std::unique_ptr<ReceiveCacheResource> const& getReceiveCacheResource(LlmRequest const& llmRequest);

    std::unordered_map<std::string, std::unique_ptr<ReceiveCacheResource>> mProcessToResources;
    std::mutex mProcessIoResouceMutex;
};

} // namespace tensorrt_llm::batch_manager
