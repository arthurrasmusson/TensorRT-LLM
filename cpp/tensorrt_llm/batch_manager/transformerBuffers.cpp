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

#include "transformerBuffers.h"

#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"

using namespace tensorrt_llm::runtime;
namespace tk = tensorrt_llm::kernels;

namespace tensorrt_llm::batch_manager
{

TransformerBuffers::TransformerBuffers(SizeType32 maxBatchSize, SizeType32 maxBeamWidth, SizeType32 maxAttentionWindow,
    SizeType32 sinkTokenLen, SizeType32 multiBlockModeVal, runtime::TllmRuntime const& runtime,
    runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig)
{
    auto const& manager = runtime.getBufferManager();
    auto const& engine = runtime.getEngine();

    positionIds = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);

    auto const localNbLayers = modelConfig.getNbAttentionLayers(worldConfig.getPipelineParallelism());

    cacheIndirection
        = manager.gpu(ITensor::makeShape({maxBatchSize, maxBeamWidth, maxAttentionWindow}), nvinfer1::DataType::kINT32);

    auto const kvCacheBlockOffsetsType = engine.getTensorDataType("kv_cache_block_offsets");
    kvCacheBlockOffsetsHost = manager.emptyTensor(MemoryType::kPINNED, kvCacheBlockOffsetsType);
    kvCacheBlockOffsetsDevice = manager.emptyTensor(MemoryType::kGPU, kvCacheBlockOffsetsType);

    if (modelConfig.useCrossAttention())
    {
        crossKvCacheBlockOffsetsHost = manager.emptyTensor(MemoryType::kPINNED, kvCacheBlockOffsetsType);
        crossKvCacheBlockOffsetsDevice = manager.emptyTensor(MemoryType::kGPU, kvCacheBlockOffsetsType);
    }

    fillValuesAlt = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kINT32);
    seqSlotsAlt = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kINT32);

    cacheIndirBatchedCopySrcOffsets
        = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kINT32);
    cacheIndirBatchedCopyDstOffsets
        = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kINT32);
    cacheIndirBatchedCopySizes = BufferManager::pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kINT32);

    pastKeyValueLengths = manager.emptyTensor(MemoryType::kCPU, nvinfer1::DataType::kINT32);

    maxAttentionWindows = BufferManager::cpu(ITensor::makeShape({localNbLayers}), nvinfer1::DataType::kINT32);
    auto maxAttentionWindowsRange = BufferRange<SizeType32>(*maxAttentionWindows);
    std::fill(maxAttentionWindowsRange.begin(), maxAttentionWindowsRange.end(), maxAttentionWindow);

    sinkTokenLengths = BufferManager::cpu(ITensor::makeShape({1}), nvinfer1::DataType::kINT32);
    bufferCast<SizeType32>(*sinkTokenLengths)[0] = sinkTokenLen;

    SizeType32 perfKnobSize = 16;
    runtimePerfKnobsHost = BufferManager::cpu(ITensor::makeShape({perfKnobSize}), nvinfer1::DataType::kINT64);
    auto runtimePerfKnobsHostPtr = bufferCast<int64_t>(*runtimePerfKnobsHost);
    std::fill_n(runtimePerfKnobsHostPtr, perfKnobSize, -1);
    runtimePerfKnobsHostPtr[0] = multiBlockModeVal;
}

void TransformerBuffers::reshape(SizeType32 numSequences)
{
    pastKeyValueLengths->reshape(ITensor::makeShape({numSequences}));

    auto cacheBlockOffsetsShape = kvCacheBlockOffsetsHost->getShape();
    if (cacheBlockOffsetsShape.nbDims > 0)
    {
        cacheBlockOffsetsShape.d[0] = numSequences;
        kvCacheBlockOffsetsHost->reshape(cacheBlockOffsetsShape);
        kvCacheBlockOffsetsDevice->reshape(cacheBlockOffsetsShape);
    }
    else
    {
        TLLM_LOG_DEBUG("kvCacheBlockOffsets not allocated yet");
    }

    if (crossKvCacheBlockOffsetsHost)
    {
        TLLM_CHECK_WITH_INFO(
            crossKvCacheBlockOffsetsDevice, "crossKvCacheBlockOffsetsDevice is empty for model with cross attention!");
        auto crossCacheBlockOffsetsShape = crossKvCacheBlockOffsetsHost->getShape();
        if (crossCacheBlockOffsetsShape.nbDims > 0)
        {
            crossCacheBlockOffsetsShape.d[0] = numSequences;
            crossKvCacheBlockOffsetsHost->reshape(crossCacheBlockOffsetsShape);
            crossKvCacheBlockOffsetsDevice->reshape(crossCacheBlockOffsetsShape);
        }
        else
        {
            TLLM_LOG_DEBUG("crossKvCacheBlockOffsets not allocated yet");
        }
    }
}

void TransformerBuffers::reshapeKvTensors(SizeType32 maxBatchSize, SizeType32 maxBeamWidth, SizeType32 maxBlocksPerSeq,
    runtime::TllmRuntime const& runtime, KvCacheType kvCacheType)
{
    auto const& manager = runtime.getBufferManager();

    // allocate with max shape during init
    if (kvCacheType == KvCacheType::kSELF)
    {
        auto const cacheBlockOffsetsShape = ITensor::makeShape({maxBatchSize * maxBeamWidth, 2, maxBlocksPerSeq});

        kvCacheBlockOffsetsHost->reshape(cacheBlockOffsetsShape);
        manager.setZero(*kvCacheBlockOffsetsHost);

        kvCacheBlockOffsetsDevice->reshape(cacheBlockOffsetsShape);
        manager.setZero(*kvCacheBlockOffsetsDevice);
    }
    else if (kvCacheType == KvCacheType::kCROSS)
    {
        auto const crossCacheBlockOffsetsShape = ITensor::makeShape({maxBatchSize * maxBeamWidth, 2, maxBlocksPerSeq});

        crossKvCacheBlockOffsetsHost->reshape(crossCacheBlockOffsetsShape);
        manager.setZero(*crossKvCacheBlockOffsetsHost);

        crossKvCacheBlockOffsetsDevice->reshape(crossCacheBlockOffsetsShape);
        manager.setZero(*crossKvCacheBlockOffsetsDevice);
    }
}

void TransformerBuffers::setKvPoolPointers(kv_cache_manager::KVCacheManager& kvCacheManager)
{
    if (kvCacheManager.isCrossKv())
    {
        crossKvCacheBlockPoolPointers = kvCacheManager.getBlockPoolPointers();
    }
    else
    {
        kvCacheBlockPoolPointers = kvCacheManager.getBlockPoolPointers();
    }
}

void TransformerBuffers::getBuffers(TensorMap& inputBuffers) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(transformerBuffersGetBuffers);

    inputBuffers.insert_or_assign("position_ids", positionIds);
    inputBuffers.insert_or_assign("host_past_key_value_lengths", pastKeyValueLengths);
    inputBuffers.insert_or_assign("cache_indirection", cacheIndirection);
    inputBuffers.insert_or_assign("host_sink_token_length", sinkTokenLengths);

    inputBuffers.insert_or_assign("host_max_attention_window_sizes", maxAttentionWindows);
    inputBuffers.insert_or_assign("kv_cache_block_offsets", kvCacheBlockOffsetsDevice);
    inputBuffers.insert_or_assign("host_kv_cache_block_offsets", kvCacheBlockOffsetsHost);
    inputBuffers.insert_or_assign("host_kv_cache_pool_pointers", kvCacheBlockPoolPointers);
    inputBuffers.insert_or_assign("host_runtime_perf_knobs", runtimePerfKnobsHost);

    if (crossKvCacheBlockPoolPointers)
    {
        inputBuffers.insert_or_assign("cross_kv_cache_block_offsets", crossKvCacheBlockOffsetsDevice);
        inputBuffers.insert_or_assign("host_cross_kv_cache_block_offsets", crossKvCacheBlockOffsetsHost);
        inputBuffers.insert_or_assign("host_cross_kv_cache_pool_pointers", crossKvCacheBlockPoolPointers);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TransformerBuffers::copyPositionIds(
    runtime::TllmRuntime const& runtime, std::vector<SizeType32> const& positionIdsHost, bool isChatGlm)
{
    auto const& manager = runtime.getBufferManager();
    if (isChatGlm)
    {
        positionIds->reshape(ITensor::makeShape({2, static_cast<int>(positionIdsHost.size()) / 2}));
    }
    else
    {
        positionIds->reshape(ITensor::makeShape({static_cast<int>(positionIdsHost.size())}));
    }
    manager.copy(positionIdsHost.data(), *positionIds);
}

void TransformerBuffers::resetCacheIndirection(RequestVector const& contextRequests, SizeType32 maxBeamWidth,
    SizeType32 maxAttentionWindow, TensorPtr const& decoderCacheIndirectionInput,
    TensorPtr const& decoderCacheIndirectionOutput, TllmRuntime const& runtime)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(resetCacheIndirection);
    auto const& stream = runtime.getStream();

    auto const numContextRequests = contextRequests.size();

    std::fill_n(bufferCast<SizeType32>(*fillValuesAlt), numContextRequests, 0);
    std::transform(contextRequests.begin(), contextRequests.end(), bufferCast<SizeType32>(*seqSlotsAlt),
        [](auto const& llmReq) { return llmReq->mSeqSlot.value(); });

    auto const seqSlotsView = ITensor::slice(seqSlotsAlt, 0, numContextRequests);
    runtime::kernels::invokeFillBatch<std::int32_t>(*decoderCacheIndirectionInput, *seqSlotsView,
        static_cast<long>(maxBeamWidth) * maxAttentionWindow, *fillValuesAlt, stream);
    runtime::kernels::invokeFillBatch<std::int32_t>(*decoderCacheIndirectionOutput, *seqSlotsView,
        static_cast<long>(maxBeamWidth) * maxAttentionWindow, *fillValuesAlt, stream);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TransformerBuffers::copyKvBlockOffsets(RequestVector const& contextRequests, RequestVector const& genRequests,
    kv_cache_manager::KVCacheManager const* kvCacheManager, kv_cache_manager::KVCacheManager const* crossKvCacheManager,
    TllmRuntime const& runtime)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(copyKvBlockPointers);

    auto const& manager = runtime.getBufferManager();
    auto const& cudaStream = manager.getStream();

    SizeType32 constexpr contextBeamWidth{1};
    SizeType32 numSequences{0};
    SizeType32 maxBlockCount{0};
    SizeType32 maxCrossBlockCount{0};
    for (auto const& requests : {contextRequests, genRequests})
    {
        for (auto const& llmReq : requests)
        {
            // Get position of the current sequence in the KV cache
            auto const seqSlot = llmReq->mSeqSlot.value();
            auto const isContextRequest = llmReq->isContextInitState();
            auto const beamWidth = isContextRequest ? contextBeamWidth : llmReq->mSamplingConfig.beamWidth;
            auto const maxBeamBlockCount
                = kvCacheManager->copyBlockOffsets(*kvCacheBlockOffsetsHost, numSequences, seqSlot, beamWidth);
            maxBlockCount = std::max(maxBlockCount, maxBeamBlockCount);
            if (crossKvCacheBlockOffsetsHost)
            {
                auto const maxCrossBeamBlockCount = crossKvCacheManager->copyBlockOffsets(
                    *crossKvCacheBlockOffsetsHost, numSequences, seqSlot, beamWidth);
                maxCrossBlockCount = std::max(maxCrossBlockCount, maxCrossBeamBlockCount);
            }
            numSequences += beamWidth;
        }
    }

    // requests' block offsets collected as [totalNumSequences, 2, maxBlocksPerSeq], copy to device
    auto copyOffsetsToDevice = [&cudaStream](TensorPtr& offsetsHost, TensorPtr& offsetsDevice, SizeType32 maxBlockCount)
    {
        // shape should be [totalNumSequences, 2, maxBlocksPerSeq]
        auto const& offsetsShape = offsetsHost->getShape();
        auto const maxBlocksPerSeq = offsetsShape.d[2];
        auto const offsetsTypeSize = tensorrt_llm::common::getDTypeSize(offsetsHost->getDataType());
        auto const copyPitch = maxBlocksPerSeq * offsetsTypeSize;
        auto const copyHeight = offsetsShape.d[0] * offsetsShape.d[1];
        auto const copyWidth = maxBlockCount * offsetsTypeSize;
        auto* srcPtr = bufferCast<tk::KVCacheIndex>(*offsetsHost);
        auto* dstPtr = bufferCast<tk::KVCacheIndex>(*offsetsDevice);

        TLLM_CUDA_CHECK(cudaMemcpy2DAsync(
            dstPtr, copyPitch, srcPtr, copyPitch, copyWidth, copyHeight, cudaMemcpyHostToDevice, cudaStream.get()));
    };

    copyOffsetsToDevice(kvCacheBlockOffsetsHost, kvCacheBlockOffsetsDevice, maxBlockCount);
    if (crossKvCacheBlockOffsetsHost)
    {
        copyOffsetsToDevice(crossKvCacheBlockOffsetsHost, crossKvCacheBlockOffsetsDevice, maxCrossBlockCount);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TransformerBuffers::copyCacheIndirection(
    RequestVector const& genRequests, TensorPtr const& decoderCacheIndirectionOutput, TllmRuntime const& runtime)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(copyCacheIndirection);
    auto const& stream = runtime.getStream();

    auto const numGenerationRequests = genRequests.size();

    auto batchedCopySrcOffsets = BufferRange<SizeType32>(*cacheIndirBatchedCopySrcOffsets);
    auto batchedCopyDstOffsets = BufferRange<SizeType32>(*cacheIndirBatchedCopyDstOffsets);
    auto batchedCopySizes = BufferRange<SizeType32>(*cacheIndirBatchedCopySizes);

    auto cacheIndirShape = decoderCacheIndirectionOutput->getShape();
    cacheIndirShape.d[0] = 1;
    auto const copySize = static_cast<SizeType32>(ITensor::volume(cacheIndirShape));

    std::transform(genRequests.begin(), genRequests.end(), batchedCopySrcOffsets.begin(),
        [copySize](auto const& llmReq) { return llmReq->mSeqSlot.value() * copySize; });
    std::generate_n(
        batchedCopyDstOffsets.begin(), numGenerationRequests, [copySize, i = 0]() mutable { return (i++) * copySize; });
    std::fill_n(batchedCopySizes.begin(), numGenerationRequests, copySize);

    auto const batchedCopySrcOffsetsSlice = ITensor::slice(cacheIndirBatchedCopySrcOffsets, 0, numGenerationRequests);
    auto const batchedCopyDstOffsetsSlice = ITensor::slice(cacheIndirBatchedCopyDstOffsets, 0, numGenerationRequests);
    auto const batchedCopySizesSlice = ITensor::slice(cacheIndirBatchedCopySizes, 0, numGenerationRequests);
    runtime::kernels::invokeCopyBatch(*decoderCacheIndirectionOutput, *cacheIndirection, *batchedCopySrcOffsetsSlice,
        *batchedCopyDstOffsetsSlice, *batchedCopySizesSlice, copySize, stream);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

} // namespace tensorrt_llm::batch_manager
