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

#include "cacheTransceiver.h"
#include "dataTransceiver.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/common/mpiUtils.h"
#include "tensorrt_llm/executor/contextPhaseState.h"

namespace tensorrt_llm::batch_manager
{

class MpiComm
{
public:
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

template <typename TDataConfig>
class MpiDataSender : public DataSender
{
public:
    using TFormatter = std::unique_ptr<IOFormatter<MpiComm, TDataConfig>>;

    MpiDataSender(mpi::MpiComm const& comm, TFormatter formatter)
        : mComm{std::addressof(comm)}
    {
        mFormatter = std::move(formatter);
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
        MpiComm comm{*mComm, rank};
        (*mFormatter)(llmRequest, {std::addressof(comm)});
    }

private:
    mpi::MpiComm const* mComm{};
    std::map<LlmRequest::RequestIdType, int> mRequestToRank;
    TFormatter mFormatter;
};

template <typename TDataConfig>
class MpiDataReceiver : public DataReceiver
{
public:
    using TFormatter = std::unique_ptr<IOFormatter<MpiComm, TDataConfig>>;

    MpiDataReceiver(mpi::MpiComm const& comm, TFormatter formatter)
        : mComm{std::addressof(comm)}
    {
        mFormatter = std::move(formatter);
    }

    void sendRequestId(LlmRequest const& llmRequest) override
    {
        uint64_t requestId = llmRequest.getContextPhaseState().getReqId();
        tensorrt_llm::executor::SizeType32 responderRank
            = llmRequest.getContextPhaseState().getMpiComm().mRanks.front();
        MpiComm::Id id{MpiComm::Id::REQUEST_SEND};
        mComm->send(std::addressof(id), 1, mpi::MpiType::kUINT64, responderRank, MpiComm::kID_TAG);
        mComm->send(std::addressof(requestId), 1, mpi::MpiType::kUINT64, responderRank, MpiComm::kID_TAG);
    }

    void receiveSync(LlmRequest const& llmRequest) override
    {
        auto rank = llmRequest.getContextPhaseState().getMpiComm().mRanks.front();
        MpiComm comm{*mComm, rank};
        (*mFormatter)(llmRequest, {std::addressof(comm)});
    }

private:
    mpi::MpiComm const* mComm{};
    TFormatter mFormatter;
};

std::unique_ptr<DataResponder> makeMpiCacheResponder(
    mpi::MpiComm const& comm, kv_cache_manager::KVCacheManager* cacheManager);

std::unique_ptr<DataRequester> makeMpiCacheRequester(
    mpi::MpiComm const& comm, kv_cache_manager::KVCacheManager* cacheManager);

} // namespace tensorrt_llm::batch_manager
