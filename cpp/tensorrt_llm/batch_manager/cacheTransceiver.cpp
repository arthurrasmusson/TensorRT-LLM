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

#define UCX_WRAPPER_LIB_NAME "tensorrt_llm_ucx_wrapper"

#if defined(_WIN32)
#include <windows.h>
#define dllOpen(name) LoadLibrary(name ".dll")
#define dllClose(handle) FreeLibrary(static_cast<HMODULE>(handle))
#define dllGetSym(handle, name) static_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle), name))
#else // For non-Windows platforms
#include <dlfcn.h>
#define dllOpen(name) dlopen("lib" name ".so", RTLD_LAZY)
#define dllClose(handle) dlclose(handle)
#define dllGetSym(handle, name) dlsym(handle, name)
#endif // defined(_WIN32)

#include "cacheTransceiver.h"
#include "tensorrt_llm/batch_manager/contextProgress.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/executor/serializeUtils.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include <algorithm>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <unordered_set>

namespace tensorrt_llm::batch_manager
{

std::mutex CacheTransceiver::mDllMutex;

CacheTransceiver::CacheTransceiver(kv_cache_manager::BaseKVCacheManager* cacheManager, CommType commType,
    runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig)
    : mCommType{commType}
{
    using namespace tensorrt_llm::batch_manager::kv_cache_manager;
    mMpiGroupComm = std::addressof(tensorrt_llm::mpi::MpiComm::session());

    if (worldConfig.isPipelineParallel())
    {
        mMpiGroupPipeParaComm = std::make_shared<tensorrt_llm::mpi::MpiComm>(
            mMpiGroupComm->split(worldConfig.getTensorParallelRank(), worldConfig.getPipelineParallelRank()));
    }
    if (worldConfig.isTensorParallel())
    {
        mMpiGroupTensorParaComm = std::make_shared<tensorrt_llm::mpi::MpiComm>(
            mMpiGroupComm->split(worldConfig.getPipelineParallelRank(), worldConfig.getTensorParallelRank()));
    }
    mCacheState = std::make_unique<executor::kv_cache::CacheState>(modelConfig, worldConfig);
    if (mCommType == CommType::MPI)
    {
        mMpiWorldComm = std::addressof(tensorrt_llm::mpi::MpiComm::world());
        mDataResponder = std::make_unique<DataResponder>(std::make_unique<MpiDataSender>(
            *mMpiWorldComm, *mCacheState, worldConfig.getRank(), std::make_unique<CacheFormatter>(cacheManager)));
        mDataRequester = std::make_unique<DataRequester>(std::make_unique<MpiDataReceiver>(
            *mMpiWorldComm, *mCacheState, worldConfig.getRank(), std::make_unique<CacheFormatter>(cacheManager)));
    }
    else if (mCommType == CommType::UCX)
    {
        {
            std::lock_guard<std::mutex> lock(mDllMutex);
            mWrapperLibHandle = dllOpen(UCX_WRAPPER_LIB_NAME);
            TLLM_CHECK_WITH_INFO(mWrapperLibHandle != nullptr, "UCX wrapper library is not open correctly.");
            auto load_sym = [](void* handle, char const* name)
            {
                void* ret = dllGetSym(handle, name);
                TLLM_CHECK_WITH_INFO(ret != nullptr,
                    "Unable to load UCX wrapper library symbol, possible cause is that TensorRT-LLM library is not "
                    "built with UCX support, please rebuild in UCX-enabled environment.");
                return ret;
            };
            std::unique_ptr<DataResponder> (*makeUcxCacheResponder)(
                executor::kv_cache::CacheState, SizeType32, kv_cache_manager::BaseKVCacheManager*);
            std::unique_ptr<DataRequester> (*makeUcxCacheRequester)(
                executor::kv_cache::CacheState, SizeType32, kv_cache_manager::BaseKVCacheManager*);
            *(void**) (&makeUcxCacheResponder) = load_sym(mWrapperLibHandle, "makeUcxCacheResponder");
            *(void**) (&makeUcxCacheRequester) = load_sym(mWrapperLibHandle, "makeUcxCacheRequester");
            mDataResponder = makeUcxCacheResponder(*mCacheState, worldConfig.getRank(), cacheManager);
            mDataRequester = makeUcxCacheRequester(*mCacheState, worldConfig.getRank(), cacheManager);
        }
        namespace su = tensorrt_llm::executor::serialize_utils;

        if (mMpiGroupComm->getSize() > 1)
        {
            mMpiGroupComm->barrier();
            executor::kv_cache::CommState commState = mDataResponder->getCommState();
            std::ostringstream oStream;
            su::serialize(commState, oStream);
            auto str = oStream.str();
            std::vector<char> buffer(str.begin(), str.end());
            std::vector<SizeType32> sizeofBuffer(mMpiGroupComm->getSize());
            SizeType32 bufferSize = buffer.size();
            mMpiGroupComm->allgather(&bufferSize, sizeofBuffer.data(), 1, mpi::MpiType::kINT32);
            SizeType32 recvBufferSize = std::accumulate(sizeofBuffer.begin(), sizeofBuffer.end(), 0);
            std::vector<char> recvBuffer(recvBufferSize);
            std::vector<int> displs(mMpiGroupComm->getSize());
            for (int r = 0; r < mMpiGroupComm->getSize(); r++)
            {
                displs[r] = (r == 0) ? 0 : (displs[r - 1] + sizeofBuffer[r - 1]);
            }
            mMpiGroupComm->allgatherv(buffer.data(), bufferSize, mpi::MpiType::kCHAR, recvBuffer.data(), sizeofBuffer,
                displs, mpi::MpiType::kCHAR);

            // deserialize
            std::vector<executor::kv_cache::CommState> commSessionCommState(mMpiGroupComm->getSize());
            std::vector<executor::kv_cache::SocketState> socketStates;
            for (int i = 0; i < mMpiGroupComm->getSize(); i++)
            {
                std::vector<char> serBuffer(
                    recvBuffer.begin() + displs[i], recvBuffer.begin() + (displs[i] + sizeofBuffer[i]));
                su::VectorWrapBuf<char> strbuf(serBuffer);
                std::istream is(&strbuf);
                commSessionCommState[i] = su::deserialize<executor::kv_cache::CommState>(is);
                TLLM_CHECK_WITH_INFO(
                    commSessionCommState[i].getSocketState().size() == 1, "getSocketState size should be 1");
                socketStates.push_back(commSessionCommState[i].getSocketState()[0]);
            }
            executor::kv_cache::CommState allCommState{socketStates, worldConfig.getRank()};
            mDataResponder->setCommState(std::move(allCommState));
        }
    }
    else
    {
        TLLM_THROW("Unsupported communication type.");
    }
    initializeCommState();
}

CacheTransceiver::~CacheTransceiver()
{
    if (mWrapperLibHandle)
    {
        std::lock_guard<std::mutex> lock(mDllMutex);
        dllClose(mWrapperLibHandle);
    }
}

void CacheTransceiver::initializeCommState()
{
    mCommState = std::addressof(mDataResponder->getCommState());
}

void CacheTransceiver::setContextState(LlmRequest* llmRequest)
{
    TLLM_CHECK(llmRequest && llmRequest->isContextOnlyRequest());
    auto contextState = std::make_unique<executor::DataTransceiverState>();
    contextState->setCommState(*mCommState);
    contextState->setCacheState(*mCacheState);
    llmRequest->setContextPhaseParams(executor::ContextPhaseParams{{}, llmRequest->mRequestId, contextState.release()});
}

void CacheTransceiver::respondAndSendAsync(LlmRequest* llmRequest)
{
    TLLM_CHECK(llmRequest && llmRequest->isContextOnlyRequest());
    llmRequest->setState(LlmRequestState::kDISAGG_CONTEXT_TRANS_IN_PROGRESS);
    if (mResponderFutures.find(llmRequest) != mResponderFutures.end())
    {
        if (llmRequest->getContextProgress() == nullptr)
        {
            TLLM_LOG_WARNING("Request %ld is already responding", llmRequest->mRequestId);
        }
        return;
    }
    setContextState(llmRequest);
    auto future = mDataResponder->respondAndSendAsync(*llmRequest);
    mResponderFutures.insert({llmRequest, std::move(future)});
}

void CacheTransceiver::respondAndSendLayerWise(
    RequestVector const& requests, std::shared_ptr<ContextProgress> const& progress)
{
    for (auto& llmRequest : requests)
    {
        TLLM_CHECK(llmRequest && llmRequest->isContextOnlyRequest());
        TLLM_CHECK(mResponderFutures.find(llmRequest.get()) == mResponderFutures.end());
        llmRequest->setContextProgress(progress);
        TLLM_LOG_DEBUG("Request %ld is sending layer-wise", llmRequest->mRequestId);

        llmRequest->setState(LlmRequestState::kDISAGG_CONTEXT_INIT_AND_TRANS);
        setContextState(llmRequest.get());
        auto future = mDataResponder->respondAndSendAsync(*llmRequest);
        mResponderFutures.emplace(llmRequest.get(), std::move(future));
    }
}

void CacheTransceiver::requestAndReceiveSync(LlmRequest* llmRequest)
{
    TLLM_CHECK(llmRequest && llmRequest->isGenerationOnlyRequest());
    {
        auto future = mDataRequester->requestAndReceiveAsync(*llmRequest);
        future.get();
    }
    llmRequest->setState(LlmRequestState::KDISAGG_GENERATION_TRANS_COMPLETE);
}

void CacheTransceiver::requestAndReceiveAsync(LlmRequest* llmRequest)
{
    TLLM_CHECK(llmRequest && llmRequest->isGenerationOnlyRequest());

    if (std::find_if(mRequesterFutures.begin(), mRequesterFutures.end(),
            [llmRequest](auto const& pair) { return pair.first->mRequestId == llmRequest->mRequestId; })
        != mRequesterFutures.end())
    {
        TLLM_LOG_WARNING("request id %zu already in mRequestFutres", llmRequest->mRequestId);
        return;
    }

    auto future = mDataRequester->requestAndReceiveAsync(*llmRequest);
    mRequesterFutures.emplace_back(llmRequest, std::move(future));
    llmRequest->setState(LlmRequestState::kDISAGG_GENERATION_TRANS_IN_PROGRESS);
}

std::vector<LlmRequest::RequestIdType> gatherRequestIds(
    mpi::MpiComm const& mpiComm, std::vector<LlmRequest::RequestIdType> const& requestIds)
{
    int localSize = static_cast<int>(requestIds.size());
    std::vector<int> sizes(mpiComm.getSize());
    mpiComm.allgather(&localSize, sizes.data(), 1, mpi::MpiType::kINT32);
    // std::vector<LlmRequest::RequestIdType> all_data(total_size);
    std::vector<int> displs(mpiComm.getSize());
    int totalSize = 0;
    for (int i = 0; i < mpiComm.getSize(); i++)
    {
        displs[i] = totalSize;
        totalSize += sizes[i];
    }
    std::vector<LlmRequest::RequestIdType> retData(totalSize);
    mpiComm.allgatherv(requestIds.data(), static_cast<int>(requestIds.size()), mpi::MpiType::kUINT64, retData.data(),
        sizes, displs, mpi::MpiType::kUINT64);
    return retData;
}

void CacheTransceiver::checkContextTransferStatus(bool blocking)
{
    std::vector<LlmRequest::RequestIdType> contextCompleteRequestIds;
    for (auto&& [request, future] : mResponderFutures)
    {
        if (future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
        {
            contextCompleteRequestIds.push_back(request->mRequestId);
        }
    }

    std::unordered_map<LlmRequest::RequestIdType, int> frequencyMap;
    if ((mMpiGroupTensorParaComm) && mMpiGroupTensorParaComm->getSize() > 1)
    {

        auto gatherRequestIdVec = gatherRequestIds(*mMpiGroupTensorParaComm, contextCompleteRequestIds);
        for (auto&& requestId : gatherRequestIdVec)
        {
            frequencyMap[requestId]++;
        }
    }
    else
    {
        for (auto&& requestId : contextCompleteRequestIds)
        {
            frequencyMap[requestId]++;
        }
    }
    std::vector<std::pair<LlmRequest::RequestIdType, int>> freqVec(frequencyMap.begin(), frequencyMap.end());

    std::sort(freqVec.begin(), freqVec.end(),
        [](std::pair<LlmRequest::RequestIdType, int> const& left,
            std::pair<LlmRequest::RequestIdType, int> const& right) { return left.second > right.second; });
    std::unordered_set<LlmRequest::RequestIdType> toCompleteIdSet;
    for (auto&& [requestId, freq] : freqVec)
    {
        if (freq == ((mMpiGroupTensorParaComm) ? mMpiGroupTensorParaComm->getSize() : 1))
        {
            toCompleteIdSet.insert(requestId);
        }
    }
    for (auto it = mResponderFutures.begin(); it != mResponderFutures.end();)
    {
        if (blocking || (toCompleteIdSet.find(it->first->mRequestId) != toCompleteIdSet.end()))
        {
            it->second.get();
            it->first->setState(LlmRequestState::kDISAGG_CONTEXT_COMPLETE);
            it = mResponderFutures.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

void CacheTransceiver::checkGenTransferStatus(int atLeastRequestNum)
{

    bool blockAll = atLeastRequestNum < 0;
    std::vector<LlmRequest::RequestIdType> genTransferReadyRequestIds;
    for (auto&& [request, future] : mRequesterFutures)
    {
        if (future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
        {
            genTransferReadyRequestIds.push_back(request->mRequestId);
        }
    }
    std::unordered_map<LlmRequest::RequestIdType, int> frequencyMap;

    std::vector<LlmRequest::RequestIdType> toBlockRequestIds;
    if ((mMpiGroupComm) && mMpiGroupComm->getSize() > 1)
    {
        auto gatherRequestIdVec = gatherRequestIds(*mMpiGroupComm, genTransferReadyRequestIds);
        for (auto&& requestId : gatherRequestIdVec)
        {
            frequencyMap[requestId]++;
        }
    }
    else
    {
        for (auto&& requestId : genTransferReadyRequestIds)
        {
            frequencyMap[requestId]++;
        }
    }

    std::vector<std::pair<LlmRequest::RequestIdType, int>> freqVec(frequencyMap.begin(), frequencyMap.end());

    std::sort(freqVec.begin(), freqVec.end(),
        [](std::pair<LlmRequest::RequestIdType, int> const& left,
            std::pair<LlmRequest::RequestIdType, int> const& right) { return left.second > right.second; });
    std::unordered_set<LlmRequest::RequestIdType> toCompleteIdSet;
    size_t idx = 0;
    while (atLeastRequestNum > static_cast<int>(toCompleteIdSet.size()))
    {
        if (idx >= freqVec.size())
        {
            break;
        }
        toCompleteIdSet.insert(freqVec.at(idx).first);
        TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), " checkGenTransferStatus atLest form freqVec requestId: %zu ",
            freqVec.at(idx).first);
        idx++;
    }
    idx = 0;

    // insert order
    while (atLeastRequestNum > static_cast<int>(toCompleteIdSet.size()))
    {
        if (idx >= mRequesterFutures.size())
        {
            break;
        }
        if (toCompleteIdSet.find(mRequesterFutures.at(idx).first->mRequestId) == toCompleteIdSet.end())
        {
            toCompleteIdSet.insert(mRequesterFutures.at(idx).first->mRequestId);
            TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
                " checkGenTransferStatus atLest form RequesterFeature requestId: %zu atLeastRequestNum:%d",
                mRequesterFutures.at(idx).first->mRequestId, atLeastRequestNum);
        }
        idx++;
    }
    for (auto&& [requestId, freq] : freqVec)
    {
        if (freq == ((mMpiGroupComm != nullptr) ? mMpiGroupComm->getSize() : 1))
        {
            toCompleteIdSet.insert(requestId);
        }
        TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), " checkGenTransferStatus freqVec requestId: %zu,freq:%d  ",
            requestId, freq);
    }
    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
        " checkGenTransferStatus toCompleteIdSet size: %zu,atLeastRequestNum:%d  ", toCompleteIdSet.size(),
        atLeastRequestNum);
    for (auto it = mRequesterFutures.begin(); it != mRequesterFutures.end();)
    {
        if (blockAll || toCompleteIdSet.find(it->first->mRequestId) != toCompleteIdSet.end())
        {
            it->second.get();

            TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
                "****it->first->mRequestId:%ld , context request id:%ld***********get feature***",
                it->first->mRequestId, it->first->getContextPhaseParams().value().getReqId());
            it->first->setState(LlmRequestState::KDISAGG_GENERATION_TRANS_COMPLETE);
            it = mRequesterFutures.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

bool CacheTransceiver::checkGenTransferComplete() const
{
    return mRequesterFutures.empty();
}

} // namespace tensorrt_llm::batch_manager
