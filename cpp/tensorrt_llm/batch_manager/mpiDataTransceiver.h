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
#include "tensorrt_llm/common/mpiUtils.h"
#include "tensorrt_llm/executor/contextPhaseState.h"

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

    template <typename... TArgs>
    MpiDataSender(mpi::MpiComm const& comm, executor::kv_cache::CacheState selfCacheState, SizeType32 selfIndex,
        TArgs... formatters)
        : mComm{std::addressof(comm)}
        , mSelfCacheState{std::move(selfCacheState)}
        , mSelfIdx{selfIndex}
        , mCommState{tensorrt_llm::mpi::getWorldRanks(tensorrt_llm::mpi::MpiComm::session())}
    {
        mFormatters.emplace_back(std::move(formatters)...);
        TLLM_CHECK(mFormatters.size() == 1);
    }

    [[nodiscard]] LlmRequest::RequestIdType recvRequestId() override
    {
#if ENABLE_MULTI_DEVICE
        MpiComm::Id id;
        MPI_Status status;
        MPI_Recv(std::addressof(id), 1, MPI_INT64_T, MPI_ANY_SOURCE, MpiComm::kID_TAG, static_cast<MPI_Comm>(*mComm),
            std::addressof(status));
        TLLM_CHECK(id == MpiComm::Id::REQUEST_SEND);
        auto requesterRank{status.MPI_SOURCE};
        LlmRequest::RequestIdType requestId;
        mComm->recv(std::addressof(requestId), 1, mpi::MpiType::kUINT64, requesterRank, MpiComm::kID_TAG);
        mRequestToRank[requestId] = status.MPI_SOURCE;
        return requestId;
#else
        TLLM_THROW("Multi device support is disabled.");
#endif
    }

    void sendSync(LlmRequest const& llmRequest) override
    {
        auto rank = mRequestToRank[llmRequest.mRequestId];
        auto const& formatter = mFormatters.front();
        std::vector<std::unique_ptr<MpiComm>> comms;
        // TODO: fix me
        comms.emplace_back(std::make_unique<MpiComm>(*mComm, rank));
        (*formatter)(llmRequest, std::move(comms));
    }

    [[nodiscard]] executor::kv_cache::CommState const& getCommState() const override
    {
        return mCommState;
    }

private:
    mpi::MpiComm const* mComm{};
    std::map<LlmRequest::RequestIdType, int> mRequestToRank;
    executor::kv_cache::CacheState mSelfCacheState;
    SizeType32 mSelfIdx{};
    std::vector<TFormatter> mFormatters;
    executor::kv_cache::CommState mCommState;
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
        , mSelfCacheState{std::move(selfCacheState)}
        , mSelfIdx{selfIndex}
    {
        mFormatters.emplace_back(std::move(formatters)...);
        TLLM_CHECK(mFormatters.size() == 1);
    }

    void sendRequestId(LlmRequest const& llmRequest) override
    {
        auto const& contextState = llmRequest.getContextPhaseState();
        uint64_t requestId = contextState.getReqId();
        auto const& mpiState = contextState.getCommState().value().getMpiState();
        auto const& destCacheState = contextState.getCacheState().value();
        MpiComm::Id id{MpiComm::Id::REQUEST_SEND};
        auto const& formatter = mFormatters.front();
        TLLM_CHECK_WITH_INFO(
            mpiState.mRanks.size() == 1, "At present, cache transfer is only supported for ranks of size 2.");

        for (auto index : formatter->getCounterparts(mSelfCacheState, mSelfIdx, destCacheState))
        {
            TLLM_CHECK(mpiState.mRanks.size() > static_cast<std::size_t>(index));
            tensorrt_llm::executor::SizeType32 responderRank = mpiState.mRanks.at(index);
            mComm->send(std::addressof(id), 1, mpi::MpiType::kUINT64, responderRank, MpiComm::kID_TAG);
            mComm->send(std::addressof(requestId), 1, mpi::MpiType::kUINT64, responderRank, MpiComm::kID_TAG);
        }
    }

    void receiveSync(LlmRequest const& llmRequest) override
    {
        auto const& contextState = llmRequest.getContextPhaseState();
        auto const& mpiState = contextState.getCommState().value().getMpiState();
        auto const& destCacheState = contextState.getCacheState().value();
        auto const& formatter = mFormatters.front();
        std::vector<std::unique_ptr<MpiComm>> comms;
        for (auto index : formatter->getCounterparts(mSelfCacheState, mSelfIdx, destCacheState))
        {
            TLLM_CHECK(mpiState.mRanks.size() > static_cast<std::size_t>(index));
            comms.emplace_back(std::make_unique<MpiComm>(*mComm, mpiState.mRanks.at(index)));
        }
        (*formatter)(llmRequest, std::move(comms));
    }

private:
    mpi::MpiComm const* mComm{};
    executor::kv_cache::CacheState mSelfCacheState;
    SizeType32 mSelfIdx{};
    std::vector<TFormatter> mFormatters;
};

} // namespace tensorrt_llm::batch_manager
