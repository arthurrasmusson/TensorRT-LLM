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
#include "tensorrt_llm/common/mpiUtils.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/executor/serializeUtils.h"
#include <cstddef>
#include <numeric>
#include <unordered_set>

namespace tensorrt_llm::batch_manager
{

std::mutex CacheTransceiver::mDllMutex;

CacheTransceiver::CacheTransceiver(kv_cache_manager::KVCacheManager* cacheManager, CommType commType,
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
                executor::kv_cache::CacheState, SizeType32, kv_cache_manager::KVCacheManager*);
            std::unique_ptr<DataRequester> (*makeUcxCacheRequester)(
                executor::kv_cache::CacheState, SizeType32, kv_cache_manager::KVCacheManager*);
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
    if (mResponderFutures.find(llmRequest) != mResponderFutures.end())
    {
        return;
    }
    setContextState(llmRequest);
    llmRequest->mState = LlmRequestState::kDISAGG_CONTEXT_TRANS_IN_PROGRESS;
    auto future = mDataResponder->respondAndSendAsync(*llmRequest);
    mResponderFutures.insert({llmRequest, std::move(future)});
}

// RequestVector are passed by value (move construct) to avoid dangling reference
void CacheTransceiver::respondAndSendLayerWise(RequestVector requests, std::unique_ptr<ContextProgress> progress)
{
    // TODO: send layer-wise request early in gen instances, and check future here

    int const numLayers = progress->getNumLayers();
    TLLM_LOG_DEBUG("CacheTransceiver::respondAndSendLayerWise - expect %d layers", numLayers);
    for (int i = 0; i < numLayers; i++)
    {
        TLLM_LOG_TRACE("CacheTransceiver::respondAndSendLayerWise - waiting kv cache for layer %d", i);
        progress->wait(i);
        // TODO: actually send the cache per request
        for (auto& llmRequest : requests)
        {
            TLLM_CHECK(llmRequest);
            TLLM_LOG_INFO("CacheTransceiver::respondAndSendLayerWise - sending kv cache for layer %d of request %zu", i,
                llmRequest->mRequestId);
            // TLLM_CHECK(llmRequest && llmRequest->isContextOnlyRequest());

            // TODO: actually send the cache per request
        }
    }
}

void CacheTransceiver::requestAndReceiveSync(LlmRequest* llmRequest)
{
    TLLM_CHECK(llmRequest && llmRequest->isGenerationOnlyRequest());
    {
        llmRequest->setKvCacheTransferStart(std::chrono::steady_clock::now());
        auto future = mDataRequester->requestAndReceiveAsync(*llmRequest);
        future.get();
        llmRequest->setKvCacheTransferEnd(std::chrono::steady_clock::now());
    }
    llmRequest->mState = tensorrt_llm::batch_manager::LlmRequestState::kGENERATION_IN_PROGRESS;
}

void CacheTransceiver::checkTranferStatus(bool blocking)
{
    // mMpiCommTensorPara
    std::vector<LlmRequest::RequestIdType> contextCompleteRequestIds;

    if ((!mMpiGroupTensorParaComm) || (mMpiGroupTensorParaComm->getRank() == 0))
    {
        for (auto it = mResponderFutures.begin(); it != mResponderFutures.end();)
        {
            if (it->second.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready || blocking)
            {

                it->second.get();

                it->first->mState = tensorrt_llm::batch_manager::LlmRequestState::kDISAGG_CONTEXT_COMPLETE;
                contextCompleteRequestIds.push_back(it->first->mRequestId);
                it = mResponderFutures.erase(it);
            }
            else
            {
                ++it;
            }
        }
    }

    if (mMpiGroupTensorParaComm && (mMpiGroupTensorParaComm->getSize() > 1))
    {
        mMpiGroupTensorParaComm->bcast(contextCompleteRequestIds, 0);
        if (!(contextCompleteRequestIds.empty()) && (mMpiGroupTensorParaComm->getRank() > 0))
        {
            std::unordered_set<LlmRequest::RequestIdType> toCompleteIdSet{
                contextCompleteRequestIds.begin(), contextCompleteRequestIds.end()};
            for (auto it = mResponderFutures.begin(); it != mResponderFutures.end();)
            {
                if (toCompleteIdSet.find(it->first->mRequestId) != toCompleteIdSet.end())
                {
                    it->second.get();
                    it->first->mState = tensorrt_llm::batch_manager::LlmRequestState::kDISAGG_CONTEXT_COMPLETE;
                    it = mResponderFutures.erase(it);
                }
                else
                {
                    ++it;
                }
            }
        }

        mMpiGroupTensorParaComm->barrier();
    }
}

} // namespace tensorrt_llm::batch_manager
