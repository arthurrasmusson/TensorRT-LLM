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

#include "cacheTransceiver.h"
#if ENABLE_UCX
#include "ucxDataTransceiver.h"
#endif

namespace tensorrt_llm::batch_manager
{

CacheTransceiver::CacheTransceiver(kv_cache_manager::KVCacheManager* cacheManager, CommType commType,
    runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig)
    : mCommType{commType}
{
    using namespace tensorrt_llm::batch_manager::kv_cache_manager;
    mMpiGroupComm = std::addressof(tensorrt_llm::mpi::MpiComm::session());
    mCacheState = std::make_unique<executor::kv_cache::CacheState>(modelConfig, worldConfig);
    if (mCommType == CommType::MPI)
    {
        mMpiWorldComm = std::addressof(tensorrt_llm::mpi::MpiComm::world());
        mDataResponder = std::make_unique<DataResponder>(
            std::make_unique<MpiDataSender<executor::kv_cache::CacheState>>(*mMpiWorldComm, *mCacheState,
                worldConfig.getRank(), std::make_unique<CacheOutputFormatter<MpiComm>>(cacheManager)));
        mDataRequester = std::make_unique<DataRequester>(
            std::make_unique<MpiDataReceiver<executor::kv_cache::CacheState>>(*mMpiWorldComm, *mCacheState,
                worldConfig.getRank(), std::make_unique<CacheInputFormatter<MpiComm>>(cacheManager)));
    }
    else if (mCommType == CommType::UCX)
    {
#if ENABLE_UCX
        mDataResponder = makeUcxCacheResponder(cacheManager);
        mDataRequester = makeUcxCacheRequester(cacheManager);
#else
        TLLM_THROW("To use UCX, the ENABLE_UCX option must be enabled during code building.");
#endif
    }
    else
    {
        TLLM_THROW("Unsupported communication type.");
    }
    initializeCommState();
}

void CacheTransceiver::initializeCommState()
{
    mCommState = std::addressof(mDataResponder->getCommState());
}

void CacheTransceiver::setContextState(LlmRequest* llmRequest)
{
    TLLM_CHECK(llmRequest && llmRequest->isContextOnlyRequest());
    auto contextState = std::make_unique<executor::ContextPhaseState>(llmRequest->mRequestId);
    contextState->setCommState(*mCommState);
    contextState->setCacheState(*mCacheState);
    llmRequest->setContextPhaseParams(executor::ContextPhaseParams{{}, contextState.release()});
}

void CacheTransceiver::respondAndSendAsync(LlmRequest* llmRequest)
{
    TLLM_CHECK(llmRequest && llmRequest->isContextOnlyRequest());
    if (mResponderFutures.find(llmRequest) != mResponderFutures.end())
    {
        return;
    }
    setContextState(llmRequest);
    llmRequest->mState = REQUEST_STATE_DISAGG_CONTEXT_TRANS_IN_PROGRESS;
    auto future = mDataResponder->respondAndSendAsync(*llmRequest);
    mResponderFutures.insert({llmRequest, std::move(future)});
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
    llmRequest->mState = tensorrt_llm::batch_manager::REQUEST_STATE_GENERATION_IN_PROGRESS;
}

void CacheTransceiver::checkTranferStatus(bool blocking)
{
    for (auto it = mResponderFutures.begin(); it != mResponderFutures.end();)
    {
        if (it->second.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready || blocking)
        {
            it->second.get();
            it->first->mState = REQUEST_STATE_DISAGG_CONTEXT_COMPLETE;
            it = mResponderFutures.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

} // namespace tensorrt_llm::batch_manager
