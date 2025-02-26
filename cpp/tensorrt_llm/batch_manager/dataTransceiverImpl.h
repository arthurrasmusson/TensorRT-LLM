/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include "tensorrt_llm/batch_manager/dataTransceiver.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/executor/cache_transmission/cacheConcatenate.h"

namespace tensorrt_llm::batch_manager
{

static std::string getConnectionName(const std::string exec, size_t rank);

class DataSenderImpl : public DataSender
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using RequestMapInfo
        = std::vector<std::pair<executor::kv_cache::Connection const*, executor::DataTransceiverState>>;

    DataSenderImpl(executor::kv_cache::ConnectionManager* manager, executor::kv_cache::CacheState selfCacheState,
        SizeType32 selfIndex, std::unique_ptr<IOFormatter> formatter);

    [[nodiscard]] RequestInfo recvRequestInfo() override;

    void sendSync(LlmRequest const& llmRequest) override;

    [[nodiscard]] executor::kv_cache::CommState const& getCommState() const override;

    void setCommState(executor::kv_cache::CommState commState) override;

    [[nodiscard]] size_t getCounterpartsCount(LlmRequest::RequestIdType requestId) const override;

    void release(LlmRequest::RequestIdType requestId) override;

private:
    enum class Id : uint64_t
    {
        REQUEST_SEND = 1,
        TERMINATION = 2
    };

    static constexpr int32_t kID_TAG{19};
    static constexpr int32_t kINFO_SIZE_TAG{22};
    static constexpr int32_t kINFO_TAG{32};

    executor::kv_cache::ConnectionManager* mManager;
    std::map<LlmRequest::RequestIdType, RequestMapInfo> mRequestToComms;
    executor::DataTransceiverState mSelfState;
    std::unique_ptr<IOFormatter> mFormatter;
    std::mutex mMtxForMap;
    runtime::BufferManager mBufferManager;
};

class DataReceiverImpl : public DataReceiver
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;

    DataReceiverImpl(executor::kv_cache::ConnectionManager* manager, executor::kv_cache::CacheState selfCacheState,
        SizeType32 selfIndex, std::unique_ptr<IOFormatter> formatter);

    void sendRequestInfo(LlmRequest const& llmRequest) override;

    void receiveSync(LlmRequest const& llmRequest) override;

private:
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

    static void sendRequestInfo(executor::kv_cache::Connection const* connection, RequestInfo const& info);

    std::unique_ptr<ReceiveCacheResource> const& getReceiveCacheResource(LlmRequest const& llmRequest);

    executor::kv_cache::ConnectionManager* mManager;
    executor::DataTransceiverState mSelfState;
    std::unique_ptr<IOFormatter> mFormatter;

    enum class Id : uint64_t
    {
        REQUEST_SEND = 1,
        TERMINATION = 2
    };

    static constexpr int32_t kID_TAG{19};
    static constexpr int32_t kINFO_SIZE_TAG{22};
    static constexpr int32_t kINFO_TAG{32};

    std::unordered_map<std::string, std::unique_ptr<ReceiveCacheResource>> mProcessToResources;
    std::mutex mProcessIoResouceMutex;
};

} // namespace tensorrt_llm::batch_manager
