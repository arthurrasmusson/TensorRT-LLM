/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "inflightBatchingUtils.h"
#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"

namespace tensorrt_llm::batch_manager::utils
{
using ITensor = runtime::ITensor;

TensorPtr collectRequestIds(RequestVector const& contextRequests, RequestVector const& generationRequests)
{
    auto const numRequests = static_cast<ITensor::DimType64>(contextRequests.size() + generationRequests.size());
    auto requestIds
        = runtime::BufferManager::cpu(ITensor::makeShape({numRequests}), runtime::TRTDataType<RequestIdType>::value);
    auto requestIdsRange = runtime::BufferRange<RequestIdType>(*requestIds);
    auto batchIdx{0};
    for (auto const& requests : {contextRequests, generationRequests})
    {
        for (auto const& request : requests)
        {
            requestIdsRange[batchIdx++] = request->mRequestId;
        }
    }
    return requestIds;
}

void setupMedusaLogits(std::vector<TensorPtr>& medusaLogitsHeads, TensorPtr& medusaLogitsDevice, SizeType32 medusaHeads,
    SizeType32 logitsIndex, SizeType32 numLogits)
{
    for (SizeType32 hi = 0; hi < medusaHeads; ++hi)
    {
        TensorPtr logitsHead = ITensor::slice(medusaLogitsDevice, hi, 1);
        logitsHead->squeeze(0);
        medusaLogitsHeads[hi] = ITensor::slice(logitsHead, logitsIndex, numLogits);
    }
}

void copyLastContextLogits(
    TensorPtr const& contextLogits, LlmRequest& llmReq, runtime::BufferManager const& bufferManager)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const numLogits = contextLogits->getShape().d[0];
    for (int beam = 0; beam < llmReq.mSamplingConfig.beamWidth; beam++)
    {
        // [beamWidth, mMaxNewTokens, vocabSizePadded] -> [numLogits, vocabSizePadded]
        auto beamHostTensorPtr = ITensor::slice(llmReq.getGenerationLogitsHost(), {beam, 0}, numLogits);
        bufferManager.copy(*contextLogits, *beamHostTensorPtr);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void copyGenerationLogits(RuntimeBuffers const& genRuntimeBuffers, runtime::BufferManager const& bufferManager,
    LlmRequest& llmReq, std::size_t batchIdx, bool beforeDecoder, std::vector<SizeType32> const& numDroppedTokens)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    TLLM_CHECK_WITH_INFO(
        !beforeDecoder || numDroppedTokens.empty(), "numDroppedTokens are only possible after decoder.");

    auto const reqBeamWidth = llmReq.mSamplingConfig.beamWidth;
    TLLM_CHECK_WITH_INFO(numDroppedTokens.empty() || numDroppedTokens.size() == static_cast<size_t>(reqBeamWidth),
        "Dropped tokens have to be defined for all beams.");

    auto const fragmentSize = llmReq.getGenerationLogitsFragmentsSize();

    // Merge logits fragments on device
    TensorPtr transposeBufferPtr = ITensor::slice(genRuntimeBuffers.cacheTransposedGenerationLogits, batchIdx, 1);
    transposeBufferPtr->squeeze(0); // [beamWidth, GENERATION_LOGITS_BUFFER_LENGTH, vocabSize]
    auto cachePointerDevice = ITensor::slice(genRuntimeBuffers.cacheGenerationFragmentPointerDevice, batchIdx, 1);
    auto cachePointerHost = ITensor::slice(genRuntimeBuffers.cacheGenerationFragmentPointerHost, batchIdx, 1);
    tensorrt_llm::runtime::kernels::mergeLogitsFragments(bufferManager, *transposeBufferPtr,
        llmReq.getGenerationLogitsFragments(), *cachePointerDevice, *cachePointerHost, 0, 1, reqBeamWidth,
        bufferManager.getStream(), 0);
    llmReq.clearGenerationLogitsFragments();

    // Copy logits to host
    for (SizeType32 beam = 0; beam < reqBeamWidth; beam++)
    {
        auto const droppedSize = !numDroppedTokens.empty() ? numDroppedTokens.at(beam) : 0;
        // Ignore logits of dropped tokens
        auto const beamFragmentSize = fragmentSize - droppedSize;
        // If this function is called before the decoder, the request does not contain the generated token of the
        // current iteration, so we add 1 to the number of tokens.
        auto const numGenerationToken
            = static_cast<SizeType32>(beforeDecoder) + llmReq.getNumTokens(beam) - llmReq.mPromptLen;
        auto const hostOffset = numGenerationToken - beamFragmentSize;

        // [beamWidth, GENERATION_LOGITS_BUFFER_LENGTH, vocabSizePadded] -> [beamFragmentSize, vocabSizePadded]
        auto beamDeviceTensorPtr = ITensor::slice(transposeBufferPtr, {beam, 0}, beamFragmentSize);
        // [beamWidth, mMaxNewTokens, vocabSizePadded] -> [beamFragmentSize, vocabSizePadded]
        auto beamHostTensorPtr = ITensor::slice(llmReq.getGenerationLogitsHost(), {beam, hostOffset}, beamFragmentSize);
        bufferManager.copy(*beamDeviceTensorPtr, *beamHostTensorPtr);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void copyStreamingGenerationLogits(runtime::BufferManager const& bufferManager, LlmRequest& llmReq)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    // If llmRequest is streaming, directly copy to host.
    // Only one token's logits needs to be copied each time.
    TLLM_CHECK(llmReq.getGenerationLogitsFragmentsSize() == 1);

    SizeType32 numGenerationToken = llmReq.getMaxBeamNumTokens() - llmReq.mPromptLen;
    TensorPtr const& generationLogitsHost
        = llmReq.getGenerationLogitsHost(); // [mMaxNewTokens (or 1), beamWidth, vocabSizePadded]

    TensorPtr hostTensorPtr
        = ITensor::slice(generationLogitsHost, numGenerationToken, 1); // [1, beamWidth, vocabSizePadded]
    TensorPtr deviceTensorPtr = *(llmReq.getGenerationLogitsFragments().begin());

    bufferManager.copy(*deviceTensorPtr, *hostTensorPtr);
    llmReq.clearGenerationLogitsFragments();
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void terminateRequest(SequenceSlotManager& seqSlotManager, LlmRequest& llmReq, SizeType32 maxInputLen,
    OptionalRef<kv_cache_manager::KVCacheManager> kvCacheManager,
    OptionalRef<kv_cache_manager::KVCacheManager> crossKvCacheManager,
    OptionalRef<BasePeftCacheManager> peftCacheManager, bool pause)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    // If a sequence slot is associated with this request id, free it
    seqSlotManager.freeSequenceSlot(llmReq.mRequestId);
    // Remove the sequence from kvCacheManager
    auto const requestId = llmReq.mRequestId;
    if (kvCacheManager)
    {
        kvCacheManager->removeSequence(requestId, llmReq);
    }
    if (crossKvCacheManager)
    {
        crossKvCacheManager->removeSequence(requestId, llmReq);
    }
    if (pause && !llmReq.isGenerationCompleteState())
    {
        llmReq.pause(maxInputLen);
    }
    else
    {
        TLLM_LOG_DEBUG("terminated: request %lu, paused: %d", requestId, pause);
    }

    if (peftCacheManager)
    {
        peftCacheManager->markRequestDone(llmReq, pause);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void CudaGraphExecutor::create(cudaGraph_t const& graph)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    assert(mInstance == nullptr);
    TLLM_CUDA_CHECK(cudaGraphInstantiate(&mInstance, graph, nullptr, nullptr, 0));
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void CudaGraphExecutor::uploadToStream(runtime::CudaStream const& stream)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    assert(hasInstance());
    TLLM_CUDA_CHECK(cudaGraphUpload(mInstance, stream.get()));
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void CudaGraphExecutor::launch(runtime::CudaStream const& stream)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TLLM_CUDA_CHECK(cudaGraphLaunch(mInstance, stream.get()));
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

bool CudaGraphExecutor::update(cudaGraph_t const& graph)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    return cudaGraphExecUpdate(mInstance, graph, nullptr) != cudaSuccess;
}

void CudaGraphExecutor::clear()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    if (mInstance != nullptr)
    {
        TLLM_CUDA_CHECK(cudaGraphExecDestroy(mInstance));
        mInstance = nullptr;
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void CudaGraphExecutor::prepareNextGraph(std::shared_ptr<runtime::TllmRuntime>& runtime, SizeType32 nextContextId)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto& stream = runtime->getStream();

    cudaGraph_t nextGraph;
    TLLM_CUDA_CHECK(cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeThreadLocal));
    runtime->executeContext(nextContextId);
    TLLM_CUDA_CHECK(cudaStreamEndCapture(stream.get(), &nextGraph));

    if (hasInstance())
    {
        if (update(nextGraph))
        {
            clear();
            create(nextGraph);
        }
    }
    else
    {
        create(nextGraph);
    }

    TLLM_CUDA_CHECK(cudaGraphDestroy(nextGraph));
    uploadToStream(stream);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

std::optional<std::shared_ptr<CudaGraphExecutor>> CudaGraphExecutorCache::get(BatchState const& state)
{
    auto it = mMap.find(state);
    if (it == mMap.end())
    {
        return std::nullopt;
    }
    mCache.splice(mCache.begin(), mCache, it->second);
    return it->second->second;
}

void CudaGraphExecutorCache::put(BatchState const& state, std::shared_ptr<CudaGraphExecutor> const& value)
{
    auto it = mMap.find(state);
    if (it != mMap.end())
    {
        mCache.erase(it->second);
    }
    mCache.emplace_front(BatchStateGraphExecutorPair{state, value});
    mMap[state] = mCache.begin();

    if (static_cast<runtime::SizeType32>(mMap.size()) > mCapacity)
    {
        auto lastState = mCache.back().first;
        mCache.pop_back();
        mMap.erase(lastState);
    }
}

} // namespace tensorrt_llm::batch_manager::utils
