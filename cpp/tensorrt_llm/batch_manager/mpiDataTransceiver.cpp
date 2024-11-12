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

#include "tensorrt_llm/batch_manager/mpiDataTransceiver.h"

namespace tensorrt_llm::batch_manager
{

[[nodiscard]] std::unique_ptr<RequestInfo> MpiComm::recvRequestInfo(
    std::optional<executor::kv_cache::ProcessInfo> const& processInfo) const
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
    auto info = std::make_unique<RequestInfo>(RequestInfo::deserialize(iss));
    auto const& commState = info->getTransState().getCommState();
    TLLM_CHECK(requesterRank == commState->getMpiState().mRanks[commState->getSelfIdx()]);
    return info;
#else
    TLLM_THROW("Multi device support is disabled.");
#endif
}

void MpiComm::sendRequestInfo(RequestInfo const& info, executor::kv_cache::ProcessInfo const& processInfo) const
{
    std::ostringstream oss;
    RequestInfo::serialize(info, oss);
    auto const& serializedInfo = oss.str();
    std::size_t infoSize = serializedInfo.size();

    MpiComm::Id id{MpiComm::Id::REQUEST_SEND};
    auto responderRank = processInfo.getRank();
    mComm->send(std::addressof(id), 1, mpi::MpiType::kUINT64, responderRank, MpiComm::kID_TAG);
    mComm->send(std::addressof(infoSize), 1, mpi::MpiType::kUINT64, responderRank, MpiComm::kID_TAG);
    mComm->send(serializedInfo.data(), infoSize, mpi::MpiType::kCHAR, responderRank, MpiComm::kID_TAG);
}

[[nodiscard]] RequestInfo MpiDataSender::recvRequestInfo()
{
#if ENABLE_MULTI_DEVICE
    auto info = *mComm.recvRequestInfo();
    LlmRequest::RequestIdType requestId = info.getRequestId();
    TLLM_CHECK_WITH_INFO(mFormatters[0]->inquireSupport(
                             mSelfState.getCacheState().value(), info.getTransState().getCacheState().value()),
        "Disagg server does not currently support these cacheState.");
    auto peerRelativeRanks = executor::kv_cache::targetIRanks(info.getTransState().getCacheState().value(),
        mSelfState.getCacheState().value(), mSelfState.getCommState().value().getSelfIdx());
    if (mRequestRemainSendCount.find(requestId) == mRequestRemainSendCount.end()
        || (mRequestRemainSendCount[requestId] == 0))
    {
        int recvExpectCount = peerRelativeRanks.size();
        mRequestRemainSendCount[requestId] = recvExpectCount;
        mRequestToComms.emplace(requestId, RequestMapInfo{});
        mRequestToComms[requestId].resize(recvExpectCount);
    }
    int peerIdx = std::distance(peerRelativeRanks.begin(),
        std::find(
            peerRelativeRanks.begin(), peerRelativeRanks.end(), info.getTransState().getCommState()->getSelfIdx()));
    auto requesterRank = info.getTransState().getCommState()->getMpiState().mRanks[peerRelativeRanks[peerIdx]];
    mRequestToComms[requestId].at(peerIdx) = {requesterRank, info.getTransState()};
    return info;
#else
    TLLM_THROW("Multi device support is disabled.");
#endif
}

void MpiDataSender::sendSync(LlmRequest const& llmRequest)
{
    // TODO: A reqeustId of context instance can only be requested by one instance of gen.
    if (mRequestRemainSendCount.find(llmRequest.mRequestId) != mRequestRemainSendCount.end())
    {
        TLLM_CHECK_WITH_INFO(mRequestRemainSendCount[llmRequest.mRequestId] > 0,
            "sendSync kvcache  with request id %ld count should >0 but get %d ", llmRequest.mRequestId,
            mRequestRemainSendCount[llmRequest.mRequestId]);
        mRequestRemainSendCount[llmRequest.mRequestId]--;
    }
    else
    {
        TLLM_THROW("Sender does not receive the requstInfo message before sending the data");
    }
    if (mRequestRemainSendCount[llmRequest.mRequestId] > 0)
    {
        return; // Wait until  the requests from all rank of the gen instance have been received.
    }
    auto const& formatter = mFormatters.front();
    std::vector<executor::kv_cache::ProcessInfo> processInfos;
    for (auto&& [rank, dataTransceiverState] : mRequestToComms[llmRequest.mRequestId])
    {
        processInfos.emplace_back(executor::kv_cache::ProcessInfo{rank});
    }
    auto&& dataTransceiverState = mRequestToComms[llmRequest.mRequestId].at(0).second;
    formatter->formatOutput(mComm, llmRequest, std::move(processInfos), mSelfState.getCacheState().value(),
        mSelfState.getCommState().value().getSelfIdx(), dataTransceiverState.getCacheState().value());
}

bool MpiDataSender::availableRelease(LlmRequest const& llmRequest)
{
    if (mRequestRemainSendCount.find(llmRequest.mRequestId) == mRequestRemainSendCount.end())
    {
        return true;
    }
    if (mRequestRemainSendCount[llmRequest.mRequestId] == 0)
    {
        mRequestRemainSendCount.erase(llmRequest.mRequestId);
        mRequestToComms.erase(llmRequest.mRequestId);
        return true;
    }
    return false;
}

void MpiDataReceiver::sendRequestInfo(LlmRequest const& llmRequest)
{
    uint64_t requestId = llmRequest.getContextPhaseParams().value().getReqId();
    auto const& contextState = llmRequest.getDataTransceiverState();
    auto const& mpiState = contextState.getCommState().value().getMpiState();
    TLLM_CHECK_WITH_INFO(
        mpiState.mRanks.size() > 0, "At present, cache transfer is only supported for ranks of size >0 .");

    auto const& formatter = mFormatters.front();
    auto const& destCacheState = contextState.getCacheState().value();
    TLLM_CHECK_WITH_INFO(formatter->inquireSupport(mSelfState.getCacheState().value(), destCacheState),
        "Disagg server does not currently support these cacheState.");
    for (auto index : formatter->getCounterparts(
             mSelfState.getCacheState().value(), mSelfState.getCommState().value().getSelfIdx(), destCacheState))
    {
        TLLM_CHECK(mpiState.mRanks.size() > static_cast<std::size_t>(index));
        mComm.sendRequestInfo(
            RequestInfo{requestId, mSelfState}, executor::kv_cache::ProcessInfo{mpiState.mRanks.at(index)});
    }
}

void MpiDataReceiver::receiveSync(LlmRequest const& llmRequest)
{
    auto const& contextState = llmRequest.getDataTransceiverState();
    auto const& mpiState = contextState.getCommState().value().getMpiState();
    auto const& destCacheState = contextState.getCacheState().value();
    auto const& formatter = mFormatters.front();
    std::vector<executor::kv_cache::ProcessInfo> processInfos;
    for (auto index : formatter->getCounterparts(
             mSelfState.getCacheState().value(), mSelfState.getCommState().value().getSelfIdx(), destCacheState))
    {
        TLLM_CHECK(mpiState.mRanks.size() > static_cast<std::size_t>(index));
        processInfos.emplace_back(executor::kv_cache::ProcessInfo{mpiState.mRanks.at(index)});
    }
    formatter->formatInput(mComm, llmRequest, std::move(processInfos), mSelfState.getCacheState().value(),
        mSelfState.getCommState().value().getSelfIdx(), destCacheState, mBufferManager);
}

} // namespace tensorrt_llm::batch_manager
