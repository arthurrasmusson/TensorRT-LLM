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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/mpiUtils.h"
#include "tensorrt_llm/executor/cache_transmission/cacheConcatenate.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"

namespace tensorrt_llm::batch_manager
{

class MpiComm
{
public:
    using TPtrContainer = std::vector<std::unique_ptr<MpiComm>>;
    enum class Id : uint64_t
    {
        REQUEST_SEND = 1,
        TERMINATION = 2
    };

    static constexpr int32_t kID_TAG{127};
    static constexpr int32_t kDATA_TAG{1023};

    MpiComm(mpi::MpiComm const& comm, int rank)
        : mComm{std::addressof(comm)}
        , mRank{rank}
    {
        TLLM_CHECK(mComm);
    }

    void recvBuffer(runtime::IBuffer& buf) const
    {
        mComm->recv(buf, mRank, kDATA_TAG);
    }

    void sendBuffer(runtime::IBuffer const& buf) const
    {
        mComm->send(buf, mRank, kDATA_TAG);
    }

private:
    mpi::MpiComm const* mComm{};
    int mRank;
};

template <typename TDataState>
class MpiDataSender : public DataSender
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using TFormatter = std::unique_ptr<IOFormatter<MpiComm, TDataState>>;
    using RequestMapInfo = std::pair<int, executor::DataTransceiverState>;

    template <typename... TArgs>
    MpiDataSender(mpi::MpiComm const& comm, executor::kv_cache::CacheState selfCacheState, SizeType32 selfIndex,
        TArgs... formatters)
        : mComm{std::addressof(comm)}
        , mSelfState{std::move(selfCacheState),
              executor::kv_cache::CommState{
                  tensorrt_llm::mpi::getWorldRanks(tensorrt_llm::mpi::MpiComm::session()), selfIndex}}
    {
        mFormatters.emplace_back(std::move(formatters)...);
        TLLM_CHECK(mFormatters.size() == 1);
    }

    [[nodiscard]] RequestInfo recvRequestInfo() override
    {
#if ENABLE_MULTI_DEVICE
        MpiComm::Id id;
        MPI_Status status;
        MPI_Recv(std::addressof(id), 1, MPI_INT64_T, MPI_ANY_SOURCE, MpiComm::kID_TAG, static_cast<MPI_Comm>(*mComm),
            std::addressof(status));
        TLLM_CHECK(id == MpiComm::Id::REQUEST_SEND);
        auto requesterRank{status.MPI_SOURCE};
        std::size_t infoSize{0};
        mComm->recv(std::addressof(infoSize), 1, mpi::MpiType::kUINT64, requesterRank, MpiComm::kID_TAG);

        std::string serializedInfo;
        serializedInfo.resize(infoSize);
        mComm->recv(serializedInfo.data(), infoSize, mpi::MpiType::kCHAR, requesterRank, MpiComm::kID_TAG);
        std::istringstream iss(serializedInfo);
        RequestInfo info{RequestInfo::deserialize(iss)};
        LlmRequest::RequestIdType requestId = info.getRequestId();

        mRequestToRank[requestId] = {status.MPI_SOURCE, info.getTransState()};
        TLLM_CHECK_WITH_INFO(mFormatters[0]->inquireSupport(
                                 mSelfState.getCacheState().value(), info.getTransState().getCacheState().value()),
            "Disagg server does not currently support these cacheState.");
        if (mRequestRemainSendCount.find(requestId) == mRequestRemainSendCount.end()
            || (mRequestRemainSendCount[requestId] == 0))
        {
            mRequestRemainSendCount[requestId]
                = executor::kv_cache::targetIRanks(info.getTransState().getCacheState().value(),
                    mSelfState.getCacheState().value(), mSelfState.getCommState().value().getSelfIdx())
                      .size();
        }
        return info;
#else
        TLLM_THROW("Multi device support is disabled.");
#endif
    }

    void sendSync(LlmRequest const& llmRequest) override
    {
        auto rank = mRequestToRank[llmRequest.mRequestId].first;
        auto&& dataTransceiverState = mRequestToRank[llmRequest.mRequestId].second;
        auto const& formatter = mFormatters.front();
        std::vector<std::unique_ptr<MpiComm>> comms;
        // TODO: fix me
        comms.emplace_back(std::make_unique<MpiComm>(*mComm, rank));
        (*formatter)(llmRequest, std::move(comms), mSelfState.getCacheState().value(),
            mSelfState.getCommState().value().getSelfIdx(), dataTransceiverState.getCacheState().value());
        if (mRequestRemainSendCount.find(llmRequest.mRequestId) != mRequestRemainSendCount.end())
        {
            TLLM_CHECK_WITH_INFO(mRequestRemainSendCount[llmRequest.mRequestId] > 0,
                "sendSync kvcache  with request id %ld count should >0 but get %d ", llmRequest.mRequestId,
                mRequestRemainSendCount[llmRequest.mRequestId]);
            mRequestRemainSendCount[llmRequest.mRequestId]--;
        }
    }

    [[nodiscard]] executor::kv_cache::CommState const& getCommState() const override
    {
        return mSelfState.getCommState().value();
    }

    void setCommState(executor::kv_cache::CommState const& commState) override
    {
        mSelfState.setCommState(commState);
    }

    bool availableRelease(LlmRequest const& llmRequest) override
    {
        if (mRequestRemainSendCount.find(llmRequest.mRequestId) == mRequestRemainSendCount.end())
        {
            return true;
        }
        if (mRequestRemainSendCount[llmRequest.mRequestId] == 0)
        {
            mRequestRemainSendCount.erase(llmRequest.mRequestId);
            return true;
        }
        return false;
    }

private:
    mpi::MpiComm const* mComm{};
    std::map<LlmRequest::RequestIdType, RequestMapInfo> mRequestToRank;
    std::vector<TFormatter> mFormatters;
    executor::DataTransceiverState mSelfState;
    std::unordered_map<LlmRequest::RequestIdType, int> mRequestRemainSendCount;
};

template <typename TDataState>
class MpiDataReceiver : public DataReceiver
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using TFormatter = std::unique_ptr<IOFormatter<MpiComm, TDataState>>;

    template <typename... TArgs>
    MpiDataReceiver(mpi::MpiComm const& comm, executor::kv_cache::CacheState selfCacheState, SizeType32 selfIndex,
        TArgs... formatters)
        : mComm{std::addressof(comm)}
        , mSelfState{std::move(selfCacheState),
              executor::kv_cache::CommState{
                  tensorrt_llm::mpi::getWorldRanks(tensorrt_llm::mpi::MpiComm::session()), selfIndex}}
    {
        mFormatters.emplace_back(std::move(formatters)...);
        TLLM_CHECK(mFormatters.size() == 1);
    }

    void sendRequestInfo(LlmRequest const& llmRequest) override
    {
        uint64_t requestId = llmRequest.getContextPhaseParams().value().getReqId();
        auto const& contextState = llmRequest.getDataTransceiverState();
        auto const& mpiState = contextState.getCommState().value().getMpiState();
        auto const& destCacheState = contextState.getCacheState().value();
        MpiComm::Id id{MpiComm::Id::REQUEST_SEND};
        auto const& formatter = mFormatters.front();

        RequestInfo info{requestId, mSelfState};
        std::ostringstream oss;
        RequestInfo::serialize(info, oss);
        auto const& serializedInfo = oss.str();
        std::size_t infoSize = serializedInfo.size();

        TLLM_CHECK_WITH_INFO(
            mpiState.mRanks.size() > 0, "At present, cache transfer is only supported for ranks of size >0 .");

        TLLM_CHECK_WITH_INFO(formatter->inquireSupport(mSelfState.getCacheState().value(), destCacheState),
            "Disagg server does not currently support these cacheState.");
        for (auto index : formatter->getCounterparts(
                 mSelfState.getCacheState().value(), mSelfState.getCommState().value().getSelfIdx(), destCacheState))
        {
            TLLM_CHECK(mpiState.mRanks.size() > static_cast<std::size_t>(index));
            tensorrt_llm::executor::SizeType32 responderRank = mpiState.mRanks.at(index);
            mComm->send(std::addressof(id), 1, mpi::MpiType::kUINT64, responderRank, MpiComm::kID_TAG);
            mComm->send(std::addressof(infoSize), 1, mpi::MpiType::kUINT64, responderRank, MpiComm::kID_TAG);
            mComm->send(serializedInfo.data(), infoSize, mpi::MpiType::kCHAR, responderRank, MpiComm::kID_TAG);
        }
    }

    void receiveSync(LlmRequest const& llmRequest) override
    {
        auto const& contextState = llmRequest.getDataTransceiverState();
        auto const& mpiState = contextState.getCommState().value().getMpiState();
        auto const& destCacheState = contextState.getCacheState().value();
        auto const& formatter = mFormatters.front();
        std::vector<std::unique_ptr<MpiComm>> comms;
        for (auto index : formatter->getCounterparts(
                 mSelfState.getCacheState().value(), mSelfState.getCommState().value().getSelfIdx(), destCacheState))
        {
            TLLM_CHECK(mpiState.mRanks.size() > static_cast<std::size_t>(index));
            comms.emplace_back(std::make_unique<MpiComm>(*mComm, mpiState.mRanks.at(index)));
        }
        (*formatter)(llmRequest, std::move(comms), mSelfState.getCacheState().value(),
            mSelfState.getCommState().value().getSelfIdx(), destCacheState);
    }

private:
    mpi::MpiComm const* mComm{};
    std::vector<TFormatter> mFormatters;
    executor::DataTransceiverState mSelfState;
};

} // namespace tensorrt_llm::batch_manager
