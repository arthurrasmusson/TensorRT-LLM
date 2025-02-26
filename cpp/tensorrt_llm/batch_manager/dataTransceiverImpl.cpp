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

#include "tensorrt_llm/batch_manager/dataTransceiverImpl.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"

namespace tensorrt_llm::batch_manager
{

DataSenderImpl::DataSenderImpl(executor::kv_cache::ConnectionManager* manager,
    executor::kv_cache::CacheState selfCacheState, SizeType32 selfIndex, std::unique_ptr<IOFormatter> formatter)
    : mManager{manager}
    , mSelfState{std::move(selfCacheState),
          executor::kv_cache::CommState{
              tensorrt_llm::mpi::getWorldRanks(tensorrt_llm::mpi::MpiComm::session()), selfIndex}}
    , mFormatter(std::move(formatter))
    , mBufferManager{std::make_shared<runtime::CudaStream>()}
{
}

[[nodiscard]] RequestInfo DataSenderImpl::recvRequestInfo()
{
    using DataContext = tensorrt_llm::executor::kv_cache::DataContext;
    Id id;
    auto const* connection = mManager->recvConnect(DataContext{kID_TAG}, &id, sizeof(id));
    TLLM_CHECK(id == Id::REQUEST_SEND);
    std::uint64_t infoSize{0};
    connection->recv(executor::kv_cache::DataContext{kINFO_SIZE_TAG}, &infoSize, sizeof(infoSize));
    std::string serializedInfo;
    serializedInfo.resize(infoSize);
    connection->recv(executor::kv_cache::DataContext{kINFO_TAG}, serializedInfo.data(), infoSize);
    std::istringstream iss(serializedInfo);
    auto info = RequestInfo::deserialize(iss);

    auto requestId = info.getRequestId();
    TLLM_CHECK_WITH_INFO(
        mFormatter->inquireSupport(mSelfState.getCacheState().value(), info.getTransState().getCacheState().value()),
        "Disagg server does not currently support these cacheState.");
    auto peerRelativeRanks = executor::kv_cache::targetIRanks(info.getTransState().getCacheState().value(),
        mSelfState.getCacheState().value(), mSelfState.getCommState().value().getSelfIdx())
                                 .mIRanks;
    int peerIdx = std::distance(peerRelativeRanks.begin(),
        std::find(
            peerRelativeRanks.begin(), peerRelativeRanks.end(), info.getTransState().getCommState()->getSelfIdx()));
    {
        std::unique_lock<std::mutex> lk(mMtxForMap);
        auto it = mRequestToComms.find(requestId);
        if (it == mRequestToComms.end())
        {
            int recvExpectCount = peerRelativeRanks.size();
            {
                it = mRequestToComms.emplace(requestId, RequestMapInfo{}).first;
                it->second.resize(recvExpectCount);
            }
        }
        it->second[peerIdx] = {connection, info.getTransState()};
    }
    return info;
}

void DataSenderImpl::sendSync(LlmRequest const& llmRequest)
{
    std::vector<executor::kv_cache::Connection const*> connections;
    auto const& reqToComm = mRequestToComms.at(llmRequest.mRequestId);
    for (auto&& [connection, dataTransceiverState] : reqToComm)
    {
        connections.emplace_back(connection);
    }
    auto&& dataTransceiverState = reqToComm.at(0).second;
    mFormatter->formatOutput(llmRequest, std::move(connections), mSelfState.getCacheState().value(),
        mSelfState.getCommState().value().getSelfIdx(), dataTransceiverState.getCacheState().value(), mBufferManager);
}

[[nodiscard]] executor::kv_cache::CommState const& DataSenderImpl::getCommState() const
{
    return mSelfState.getCommState().value();
}

void DataSenderImpl::setCommState(executor::kv_cache::CommState commState)
{
    mSelfState.setCommState(std::move(commState));
}

[[nodiscard]] size_t DataSenderImpl::getCounterpartsCount(LlmRequest::RequestIdType requestId) const
{
    auto it = mRequestToComms.find(requestId);
    TLLM_CHECK(it != mRequestToComms.end());
    return mRequestToComms.at(requestId).size();
}

void DataSenderImpl::release(LlmRequest::RequestIdType requestId)
{
    auto it = mRequestToComms.find(requestId);
    TLLM_CHECK(it != mRequestToComms.end());
    std::unique_lock<std::mutex> lk(mMtxForMap);
    mRequestToComms.erase(it);
}

DataReceiverImpl::DataReceiverImpl(executor::kv_cache::ConnectionManager* manager,
    executor::kv_cache::CacheState selfCacheState, SizeType32 selfIndex, std::unique_ptr<IOFormatter> formatter)
    : mManager{manager}
    , mSelfState{std::move(selfCacheState),
          executor::kv_cache::CommState{
              tensorrt_llm::mpi::getWorldRanks(tensorrt_llm::mpi::MpiComm::session()), selfIndex}}
    , mFormatter(std::move(formatter))
{
    TLLM_CHECK(mManager);
    TLLM_CHECK(mFormatter);
}

void DataReceiverImpl::sendRequestInfo(LlmRequest const& llmRequest)
{
    uint64_t requestId = llmRequest.getContextPhaseParams().value().getReqId();
    auto const& contextState = llmRequest.getDataTransceiverState();
    auto const& commState = contextState.getCommState().value();
    auto const& destCacheState = contextState.getCacheState().value();
    TLLM_CHECK_WITH_INFO(mFormatter->inquireSupport(mSelfState.getCacheState().value(), destCacheState),
        "Disagg server does not currently support these cacheState.");
    for (auto index : mFormatter->getCounterparts(
             mSelfState.getCacheState().value(), mSelfState.getCommState().value().getSelfIdx(), destCacheState))
    {
        auto const* connection = mManager->getConnections(commState).at(index);
        sendRequestInfo(connection, RequestInfo{requestId, mSelfState});
    }
}

void DataReceiverImpl::receiveSync(LlmRequest const& llmRequest)
{
    auto const& contextState = llmRequest.getDataTransceiverState();
    auto const& commState = contextState.getCommState().value();
    auto const& destCacheState = contextState.getCacheState().value();
    std::vector<tensorrt_llm::executor::kv_cache::Connection const*> connections;
    for (auto index : mFormatter->getCounterparts(
             mSelfState.getCacheState().value(), mSelfState.getCommState().value().getSelfIdx(), destCacheState))
    {
        auto const* connection = mManager->getConnections(commState).at(index);
        connections.emplace_back(connection);
    }
    auto const& resource = getReceiveCacheResource(llmRequest);
    mFormatter->formatInput(llmRequest, std::move(connections), mSelfState.getCacheState().value(),
        mSelfState.getCommState().value().getSelfIdx(), destCacheState, resource->mBufferManager);
}

void DataReceiverImpl::sendRequestInfo(executor::kv_cache::Connection const* connection, RequestInfo const& info)
{
    std::ostringstream oss;
    RequestInfo::serialize(info, oss);
    auto const& serializedInfo = oss.str();
    const std::size_t infoSize = serializedInfo.size();
    Id id{Id::REQUEST_SEND};
    connection->send(executor::kv_cache::DataContext{kID_TAG}, &id, sizeof(id));
    connection->send(executor::kv_cache::DataContext{kINFO_SIZE_TAG}, &infoSize, sizeof(infoSize));
    connection->send(executor::kv_cache::DataContext{kINFO_TAG}, serializedInfo.data(), infoSize);
}

std::unique_ptr<DataReceiverImpl::ReceiveCacheResource> const& DataReceiverImpl::getReceiveCacheResource(
    LlmRequest const& llmRequest)
{
    std::scoped_lock<std::mutex> lock(mProcessIoResouceMutex);
    TLLM_CHECK(llmRequest.getDataTransceiverState().getCommState().has_value());
    std::string processString = llmRequest.getDataTransceiverState().getCommState()->toString();

    if (common::getEnvRequestKVCacheSerial())
    {
        processString = "default";
    }
    if (mProcessToResources.find(processString) == mProcessToResources.end())
    {
        mProcessToResources.emplace(processString,
            std::make_unique<ReceiveCacheResource>(
                runtime::BufferManager{std::make_shared<runtime::CudaStream>()}, runtime::CudaEvent{}));
    }

    return mProcessToResources.at(processString);
}

} // namespace tensorrt_llm::batch_manager
