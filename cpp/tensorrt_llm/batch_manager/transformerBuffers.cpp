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
#include "tensorrt_llm/kernels/attentionMask.h"
#include "tensorrt_llm/kernels/contextFusedMultiHeadAttention/fmhaPackedMask.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"

using namespace tensorrt_llm::runtime;
namespace tk = tensorrt_llm::kernels;

namespace tensorrt_llm::batch_manager
{

TransformerBuffers::TransformerBuffers(SizeType32 maxBatchSize, SizeType32 maxBeamWidth,
    std::vector<SizeType32> maxAttentionWindowVec, SizeType32 maxAttentionWindow, SizeType32 sinkTokenLen,
    executor::ExtendedRuntimePerfKnobConfig const& extendedRuntimePerfKnobConfig, runtime::TllmRuntime const& runtime,
    runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig)
{
    auto const& manager = runtime.getBufferManager();
    auto const& engine = runtime.getEngine();

    positionIds = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);

    auto const localNbAttnLayers
        = modelConfig.getNbAttentionLayers(worldConfig.getPipelineParallelism(), worldConfig.getPipelineParallelRank());
    // find the index of the first attention layer in the current rank
    auto const firstLayerId = modelConfig.countLowerRankLayers(runtime::ModelConfig::LayerType::kATTENTION,
        worldConfig.getPipelineParallelism(), worldConfig.getPipelineParallelRank());

    cacheIndirection
        = manager.gpu(ITensor::makeShape({maxBatchSize, maxBeamWidth, maxAttentionWindow}), nvinfer1::DataType::kINT32);

    maxInputLen = modelConfig.getMaxInputLen();
    maxEncoderOutputLen = modelConfig.getMaxEncoderLen();
    maxNumTokens = modelConfig.getMaxNumTokens().value();

    if (modelConfig.isKVCacheEnabled())
    {
        auto const kvCacheBlockOffsetsType = engine.getTensorDataType("kv_cache_block_offsets");
        kvCacheBlockOffsetsHost = manager.emptyTensor(MemoryType::kPINNEDPOOL, kvCacheBlockOffsetsType);
        kvCacheBlockOffsetsDevice = manager.emptyTensor(MemoryType::kGPU, kvCacheBlockOffsetsType);

        if (modelConfig.useCrossAttention())
        {
            crossKvCacheBlockOffsetsHost = manager.emptyTensor(MemoryType::kPINNEDPOOL, kvCacheBlockOffsetsType);
            crossKvCacheBlockOffsetsDevice = manager.emptyTensor(MemoryType::kGPU, kvCacheBlockOffsetsType);
            crossAttentionMaskDevice = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kBOOL);
            crossAttentionPackedMaskDevice = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
            crossAttentionCuQSeqLensDevice = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
            crossAttentionPackedMaskCuMaskRowsDevice
                = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);

            // Pinned memory for batch copy of attention masks.
            // There will be paddings in the dim1, so copy it by tokens.
            crossAttentionMaskCopySrcOffsets
                = manager.pinnedPool(ITensor::makeShape({maxNumTokens}), nvinfer1::DataType::kINT64);
            crossAttentionMaskCopyDstOffsets
                = manager.pinnedPool(ITensor::makeShape({maxNumTokens}), nvinfer1::DataType::kINT64);
            crossAttentionMaskCopySizes
                = manager.pinnedPool(ITensor::makeShape({maxNumTokens}), nvinfer1::DataType::kINT64);
        }
    }

    fillValuesAlt = manager.pinnedPool(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kINT32);
    fillValuesAltDevice = manager.gpu(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kINT32);
    seqSlotsAlt = manager.pinnedPool(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kINT32);
    seqSlotsAltDevice = manager.gpu(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kINT32);

    cacheIndirBatchedCopySrcOffsets
        = manager.pinnedPool(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kINT64);
    cacheIndirBatchedCopyDstOffsets
        = manager.pinnedPool(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kINT64);
    cacheIndirBatchedCopySizes = manager.pinnedPool(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kINT64);

    pastKeyValueLengths = manager.emptyTensor(MemoryType::kCPU, nvinfer1::DataType::kINT32);

    maxAttentionWindows = BufferManager::cpu(ITensor::makeShape({localNbAttnLayers}), nvinfer1::DataType::kINT32);
    auto maxAttentionWindowsPtr = bufferCast<SizeType32>(*maxAttentionWindows);
    auto const attentionWindowLength = maxAttentionWindowVec.size();
    for (SizeType32 i = 0; i < localNbAttnLayers; ++i)
    {
        maxAttentionWindowsPtr[i] = maxAttentionWindowVec[(firstLayerId + i) % attentionWindowLength];
    }

    sinkTokenLengths = BufferManager::cpu(ITensor::makeShape({1}), nvinfer1::DataType::kINT32);
    bufferCast<SizeType32>(*sinkTokenLengths)[0] = sinkTokenLen;

    SizeType32 perfKnobSize = 16;
    runtimePerfKnobsHost = BufferManager::cpu(ITensor::makeShape({perfKnobSize}), nvinfer1::DataType::kINT64);
    auto runtimePerfKnobsHostPtr = bufferCast<int64_t>(*runtimePerfKnobsHost);
    std::fill_n(runtimePerfKnobsHostPtr, perfKnobSize, -1);
    SizeType32 multiBlockModeVal = extendedRuntimePerfKnobConfig.getMultiBlockMode() ? 1 : 0;
    SizeType32 enableContextFMHAFP32AccVal = extendedRuntimePerfKnobConfig.getEnableContextFMHAFP32Acc() ? 1 : 0;
    runtimePerfKnobsHostPtr[0] = multiBlockModeVal;
    runtimePerfKnobsHostPtr[1] = enableContextFMHAFP32AccVal;
}

void TransformerBuffers::reshape(SizeType32 numSequences, SizeType32 numInputTokens)
{
    pastKeyValueLengths->reshape(ITensor::makeShape({numSequences}));

    if (kvCacheBlockOffsetsHost)
    {
        auto cacheBlockOffsetsShape = kvCacheBlockOffsetsHost->getShape();
        if (cacheBlockOffsetsShape.nbDims > 0)
        {
            cacheBlockOffsetsShape.d[1] = numSequences;
            kvCacheBlockOffsetsHost->reshape(cacheBlockOffsetsShape);
            kvCacheBlockOffsetsDevice->reshape(cacheBlockOffsetsShape);
        }
        else
        {
            TLLM_LOG_DEBUG("kvCacheBlockOffsets not allocated yet");
        }
    }

    if (crossKvCacheBlockOffsetsHost)
    {
        TLLM_CHECK_WITH_INFO(
            crossKvCacheBlockOffsetsDevice, "crossKvCacheBlockOffsetsDevice is empty for model with cross attention!");
        auto crossCacheBlockOffsetsShape = crossKvCacheBlockOffsetsHost->getShape();
        if (crossCacheBlockOffsetsShape.nbDims > 0)
        {
            crossCacheBlockOffsetsShape.d[1] = numSequences;
            crossKvCacheBlockOffsetsHost->reshape(crossCacheBlockOffsetsShape);
            crossKvCacheBlockOffsetsDevice->reshape(crossCacheBlockOffsetsShape);
        }
        else
        {
            TLLM_LOG_DEBUG("crossKvCacheBlockOffsets not allocated yet");
        }
    }

    if (crossAttentionMaskDevice)
    {
        auto crossAttentionMaskShape = crossAttentionMaskDevice->getShape();
        if (crossAttentionMaskShape.nbDims > 0)
        {
            crossAttentionMaskShape.d[0] = numInputTokens;
            crossAttentionMaskDevice->reshape(crossAttentionMaskShape);
            crossAttentionMaskCopySrcOffsets->reshape(ITensor::makeShape({numInputTokens}));
            crossAttentionMaskCopyDstOffsets->reshape(ITensor::makeShape({numInputTokens}));
            crossAttentionMaskCopySizes->reshape(ITensor::makeShape({numInputTokens}));
        }
        else
        {
            TLLM_LOG_DEBUG("crossAttentionMaskDevice not allocated yet");
        }
    }

    if (crossAttentionPackedMaskDevice)
    {
        auto crossAttentionMaskPackedShape = crossAttentionPackedMaskDevice->getShape();
        if (crossAttentionMaskPackedShape.nbDims > 0)
        {
            crossAttentionMaskPackedShape.d[0] = numInputTokens;
            crossAttentionPackedMaskDevice->reshape(crossAttentionMaskPackedShape);
        }
        else
        {
            TLLM_LOG_DEBUG("crossAttentionPackedMaskDevice not allocated yet");
        }
    }
}

void TransformerBuffers::reshapeKvTensors(SizeType32 maxBatchSize, SizeType32 maxBeamWidth, SizeType32 maxBlocksPerSeq,
    runtime::TllmRuntime const& runtime, kv_cache_manager::KVCacheManager const& kvCacheManager)
{
    auto const& manager = runtime.getBufferManager();
    auto const& blockManager = kvCacheManager.getBlockManager();
    auto const kvCacheType = blockManager.getCacheType();
    auto const numPools = blockManager.getNumPools();

    // allocate with max shape during init
    if (kvCacheType == KvCacheType::kSELF)
    {
        auto const cacheBlockOffsetsShape
            = ITensor::makeShape({numPools, maxBatchSize * maxBeamWidth, 2, maxBlocksPerSeq});

        kvCacheBlockOffsetsHost->reshape(cacheBlockOffsetsShape);
        manager.setZero(*kvCacheBlockOffsetsHost);

        kvCacheBlockOffsetsDevice->reshape(cacheBlockOffsetsShape);
        manager.setZero(*kvCacheBlockOffsetsDevice);
    }
    else if (kvCacheType == KvCacheType::kCROSS)
    {
        auto const crossCacheBlockOffsetsShape
            = ITensor::makeShape({numPools, maxBatchSize * maxBeamWidth, 2, maxBlocksPerSeq});

        crossKvCacheBlockOffsetsHost->reshape(crossCacheBlockOffsetsShape);
        manager.setZero(*crossKvCacheBlockOffsetsHost);

        crossKvCacheBlockOffsetsDevice->reshape(crossCacheBlockOffsetsShape);
        manager.setZero(*crossKvCacheBlockOffsetsDevice);

        // TODO: make cross-attention works with chunked context.
        crossAttentionMaskDevice->reshape(ITensor::makeShape({maxNumTokens, maxEncoderOutputLen}));
        manager.setZero(*crossAttentionMaskDevice);

        // Only context attention needs this, so allocate it by shape [maxBatchSize, maxInputLen, maxEncoderOutputLen].
        auto [packedMaskM, packedMaskN] = tk::roundUpPackedMaskMNDims(maxInputLen, maxEncoderOutputLen);
        crossAttentionPackedMaskDevice->reshape(ITensor::makeShape({maxBatchSize * packedMaskM, packedMaskN}));
        manager.setZero(*crossAttentionPackedMaskDevice);

        crossAttentionCuQSeqLensDevice->reshape(ITensor::makeShape({maxBatchSize + 1}));
        manager.setZero(*crossAttentionCuQSeqLensDevice);

        crossAttentionPackedMaskCuMaskRowsDevice->reshape(ITensor::makeShape({maxBatchSize + 1}));
        manager.setZero(*crossAttentionPackedMaskCuMaskRowsDevice);
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
    inputBuffers.insert_or_assign("host_runtime_perf_knobs", runtimePerfKnobsHost);

    if (crossKvCacheBlockOffsetsHost)
    {
        inputBuffers.insert_or_assign("cross_kv_cache_block_offsets", crossKvCacheBlockOffsetsDevice);
        inputBuffers.insert_or_assign("host_cross_kv_cache_block_offsets", crossKvCacheBlockOffsetsHost);
        inputBuffers.insert_or_assign("host_cross_kv_cache_pool_pointers", crossKvCacheBlockPoolPointers);
        inputBuffers.insert_or_assign("host_cross_kv_cache_pool_mapping", crossKvCacheBlockPoolMapping);
        inputBuffers.insert_or_assign("cross_attention_mask", crossAttentionMaskDevice);
        inputBuffers.insert_or_assign("cross_attention_packed_mask", crossAttentionPackedMaskDevice);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TransformerBuffers::reshapePositionIds(std::vector<SizeType32> const& positionIdsHost, bool isChatGlm)
{
    if (isChatGlm)
    {
        positionIds->reshape(ITensor::makeShape({2, static_cast<int>(positionIdsHost.size()) / 2}));
    }
    else
    {
        positionIds->reshape(ITensor::makeShape({static_cast<int>(positionIdsHost.size())}));
    }
}

void TransformerBuffers::copyPositionIds(
    runtime::TllmRuntime const& runtime, std::vector<SizeType32> const& positionIdsHost, bool isChatGlm)
{
    auto const& manager = runtime.getBufferManager();
    manager.copy(positionIdsHost.data(), *positionIds);
}

void TransformerBuffers::copyPositionIds(runtime::TllmRuntime const& runtime,
    std::vector<SizeType32> const& positionIdsHost, bool isChatGlm, TensorPtr decoderPositionIds)
{
    auto const& manager = runtime.getBufferManager();
    if (isChatGlm)
    {
        positionIds->reshape(ITensor::makeShape({2, static_cast<int>(positionIdsHost.size()) / 2}));
        manager.copy(positionIdsHost.data(), *positionIds);
    }
    else if (decoderPositionIds == nullptr)
    {
        positionIds->reshape(ITensor::makeShape({static_cast<int>(positionIdsHost.size())}));
        manager.copy(positionIdsHost.data(), *positionIds);
    }
    else
    {
        // concat context phase and generation phase positionIds.
        ITensor::DimType64 contextPositionIdsLen = static_cast<ITensor::DimType64>(positionIdsHost.size());
        ITensor::DimType64 generationPositionIdsLen = ITensor::volume(decoderPositionIds->getShape());
        positionIds->reshape(ITensor::makeShape({contextPositionIdsLen + generationPositionIdsLen}));
        manager.copy(positionIdsHost.data(), *ITensor::slice(positionIds, 0, contextPositionIdsLen));
        manager.copy(*decoderPositionIds, *ITensor::slice(positionIds, contextPositionIdsLen));
    }
}

void TransformerBuffers::resetCacheIndirection(RequestVector const& contextRequests, SizeType32 maxBeamWidth,
    SizeType32 maxAttentionWindow, TensorPtr const& decoderCacheIndirectionInput,
    TensorPtr const& decoderCacheIndirectionOutput, TllmRuntime const& runtime)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(resetCacheIndirection);
    auto const& manager = runtime.getBufferManager();
    auto const& stream = manager.getStream();

    auto const numContextRequests = contextRequests.size();

    std::fill_n(bufferCast<SizeType32>(*fillValuesAlt), numContextRequests, 0);
    std::transform(contextRequests.begin(), contextRequests.end(), bufferCast<SizeType32>(*seqSlotsAlt),
        [](auto const& llmReq) { return llmReq->mSeqSlot.value(); });

    auto const seqSlotsHostView = ITensor::slice(seqSlotsAlt, 0, numContextRequests);
    auto seqSlotsDeviceView = ITensor::slice(seqSlotsAltDevice, 0, numContextRequests);
    manager.copy(*seqSlotsHostView, *seqSlotsDeviceView);
    manager.copy(*fillValuesAlt, *fillValuesAltDevice);
    runtime::kernels::invokeFillBatch<std::int32_t>(*decoderCacheIndirectionInput, *seqSlotsDeviceView,
        static_cast<long>(maxBeamWidth) * maxAttentionWindow, *fillValuesAltDevice, stream);
    runtime::kernels::invokeFillBatch<std::int32_t>(*decoderCacheIndirectionOutput, *seqSlotsDeviceView,
        static_cast<long>(maxBeamWidth) * maxAttentionWindow, *fillValuesAltDevice, stream);
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
            auto const requestId = llmReq->mRequestId;
            auto const isContextRequest = llmReq->isContextInitState();
            auto const beamWidth = isContextRequest ? contextBeamWidth : llmReq->mSamplingConfig.beamWidth;
            auto const maxBeamBlockCount
                = kvCacheManager->copyBlockOffsets(*kvCacheBlockOffsetsHost, numSequences, requestId);
            maxBlockCount = std::max(maxBlockCount, maxBeamBlockCount);
            if (crossKvCacheBlockOffsetsHost)
            {
                auto const maxCrossBeamBlockCount
                    = crossKvCacheManager->copyBlockOffsets(*crossKvCacheBlockOffsetsHost, numSequences, requestId);
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
        auto const maxBlocksPerSeq = offsetsShape.d[3];
        auto const offsetsTypeSize = tensorrt_llm::common::getDTypeSize(offsetsHost->getDataType());
        auto const copyPitch = maxBlocksPerSeq * offsetsTypeSize;
        auto const copyHeight = offsetsShape.d[0] * offsetsShape.d[1] * offsetsShape.d[2];
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

    auto batchedCopySrcOffsets = BufferRange<SizeType64>(*cacheIndirBatchedCopySrcOffsets);
    auto batchedCopyDstOffsets = BufferRange<SizeType64>(*cacheIndirBatchedCopyDstOffsets);
    auto batchedCopySizes = BufferRange<SizeType64>(*cacheIndirBatchedCopySizes);

    auto cacheIndirShape = decoderCacheIndirectionOutput->getShape();
    cacheIndirShape.d[0] = 1;
    auto const copySize = static_cast<SizeType64>(ITensor::volume(cacheIndirShape));

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

void TransformerBuffers::copyCrossAttentionMasks(RequestVector const& contextRequests, RequestVector const& genRequests,
    TensorPtr const& decoderContextLengthsDevice, TensorPtr const& encoderInputLengths,
    SizeType32 maxDecoderContextLength, SizeType32 maxEncoderInputLengthInBatch, TllmRuntime const& runtime)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const& manager = runtime.getBufferManager();

    // Reshape the tensor to make sure the dim1 matches maxEncoderInputLengthInBatch.
    auto crossAttentionMaskShape = crossAttentionMaskDevice->getShape();
    crossAttentionMaskShape.d[1] = maxEncoderInputLengthInBatch;
    crossAttentionMaskDevice->reshape(crossAttentionMaskShape);
    // Set crossAttentionMask to true by default if it is not provided.
    manager.setMem(*crossAttentionMaskDevice, 1);

    // Check if all context requests have cross attention mask.
    bool allCrossAttentionMaskProvided = true;
    for (auto const& llmReq : contextRequests)
    {
        auto const& crossAttentionMaskRequest = llmReq->getCrossAttentionMask();
        if (bufferCastOrNull<bool>(crossAttentionMaskRequest) == nullptr)
        {
            allCrossAttentionMaskProvided = false;
            break;
        }
    }

    // If not all requests have cross attention mask, let us create the default ones. And it will be overwritten if some
    // requests have cross attention mask.
    auto const& stream = runtime.getStream();
    if (!allCrossAttentionMaskProvided)
    {
        tk::AttentionMaskParams<bool> attentionMaskParams;
        memset((void*) &attentionMaskParams, 0, sizeof(attentionMaskParams));
        // Set parameters.
        attentionMaskParams.mask = bufferCastOrNull<bool>(crossAttentionMaskDevice);
        attentionMaskParams.cuQSeqLens = bufferCastOrNull<SizeType32>(crossAttentionCuQSeqLensDevice);
        attentionMaskParams.actualQSeqLens = bufferCastOrNull<SizeType32>(decoderContextLengthsDevice);
        attentionMaskParams.actualKvSeqLens = bufferCastOrNull<SizeType32>(encoderInputLengths);
        attentionMaskParams.attentionMaskType = tk::AttentionMaskType::PADDING;
        attentionMaskParams.batchSize = contextRequests.size();
        attentionMaskParams.maxQSeqLen = maxDecoderContextLength;
        attentionMaskParams.maxKvSeqLen = maxEncoderInputLengthInBatch;
        // Launch the kernel.
        tk::invokeBuildAttentionMask(attentionMaskParams, stream.get());
        sync_check_cuda_error();
    }

    // Use the first request's cross attention mask tensor's pointer address as the primary source pointer.
    auto const& attentionMaskSrc = !contextRequests.empty() ? contextRequests[0]->getCrossAttentionMask()
                                                            : genRequests[0]->getCrossAttentionMask();
    bool const* primarySrcPtr = bufferCastOrNull<bool>(attentionMaskSrc);

    // Pinned-memory buffer preparation for batch copy.
    auto batchedCopySrcOffsets = BufferRange<SizeType64>(*crossAttentionMaskCopySrcOffsets);
    auto batchedCopyDstOffsets = BufferRange<SizeType64>(*crossAttentionMaskCopyDstOffsets);
    auto batchedCopySizes = BufferRange<SizeType64>(*crossAttentionMaskCopySizes);
    // Requests with cross-attention-mask don't need to copy.
    manager.setZero(*crossAttentionMaskCopySizes);

    SizeType32 numTokens = 0;
    for (auto const& llmReq : contextRequests)
    {
        auto const& crossAttentionMaskRequest = llmReq->getCrossAttentionMask();
        if (bufferCastOrNull<bool>(crossAttentionMaskRequest) != nullptr)
        {
            auto const position = llmReq->getContextCurrentPosition();
            auto const size = llmReq->getContextChunkSize();
            SizeType64 crossAttentionMaskRequestDim0
                = static_cast<SizeType64>(crossAttentionMaskRequest->getShape().d[0]);
            SizeType64 crossAttentionMaskRequestDim1
                = static_cast<SizeType64>(crossAttentionMaskRequest->getShape().d[1]);
            TLLM_LOG_DEBUG("copyCrossAttentionMasks (shape [%d, %d]) from contextRequests position %d chunkSize %d",
                crossAttentionMaskRequestDim0, crossAttentionMaskRequestDim1, position, size);
            if ((position + size - 1) >= crossAttentionMaskRequestDim0)
            {
                TLLM_LOG_WARNING(
                    "The provided crossAttentionMask input is not complete for context phases, the last row will be "
                    "used by default.");
            }
            for (SizeType32 tokenId = position; tokenId < position + size; tokenId++)
            {
                batchedCopySrcOffsets.begin()[numTokens]
                    = static_cast<SizeType64>(bufferCastOrNull<bool>(crossAttentionMaskRequest) - primarySrcPtr)
                    + std::min(crossAttentionMaskRequestDim0 - 1, static_cast<SizeType64>(tokenId))
                        * crossAttentionMaskRequestDim1;
                batchedCopyDstOffsets.begin()[numTokens]
                    = numTokens * static_cast<SizeType64>(maxEncoderInputLengthInBatch);
                batchedCopySizes.begin()[numTokens] = crossAttentionMaskRequestDim1;
                numTokens++;
            }
        }
        else
        {
            TLLM_LOG_WARNING(
                "CrossAttentionMask is not provided for the request. Default padding attention mask will be created.");
        }
    }

    for (auto const& llmReq : genRequests)
    {
        auto const promptLen = llmReq->mPromptLen;
        auto const decodingIter = llmReq->getDecodingIter();
        auto const& crossAttentionMaskRequest = llmReq->getCrossAttentionMask();
        if (bufferCastOrNull<bool>(crossAttentionMaskRequest) != nullptr)
        {
            SizeType64 crossAttentionMaskRequestDim0
                = static_cast<SizeType64>(crossAttentionMaskRequest->getShape().d[0]);
            SizeType64 crossAttentionMaskRequestDim1
                = static_cast<SizeType64>(crossAttentionMaskRequest->getShape().d[1]);
            TLLM_LOG_DEBUG("copyCrossAttentionMasks (shape [%d, %d]) from genRequests decodingIter %d",
                crossAttentionMaskRequestDim0, crossAttentionMaskRequestDim1, decodingIter);
            batchedCopySrcOffsets.begin()[numTokens]
                = static_cast<SizeType64>(bufferCastOrNull<bool>(crossAttentionMaskRequest) - primarySrcPtr)
                + std::min(crossAttentionMaskRequestDim0 - 1, static_cast<SizeType64>(promptLen + decodingIter - 1))
                    * crossAttentionMaskRequestDim1;
            if (promptLen + decodingIter - 1 >= crossAttentionMaskRequestDim0)
            {
                TLLM_LOG_WARNING(
                    "The provided crossAttentionMask input is not complete for generation phases, the last row will be "
                    "used by default.");
            }
            batchedCopyDstOffsets.begin()[numTokens]
                = numTokens * static_cast<SizeType64>(maxEncoderInputLengthInBatch);
            batchedCopySizes.begin()[numTokens] = crossAttentionMaskRequestDim1;
            numTokens++;
        }
        else
        {
            TLLM_LOG_WARNING(
                "CrossAttentionMask is not provided for the generation request. Full valid attentionMask will be used "
                "by default.");
        }
    }

    // Copy all requests' attention mask in one kernel.
    if (attentionMaskSrc != nullptr)
    {
        runtime::kernels::invokeCopyBatch(*attentionMaskSrc, *crossAttentionMaskDevice,
            *crossAttentionMaskCopySrcOffsets, *crossAttentionMaskCopyDstOffsets, *crossAttentionMaskCopySizes,
            maxEncoderInputLengthInBatch, stream);
    }
    sync_check_cuda_error();

    // The packed mask is only needed by context requests now.
    if (!contextRequests.empty())
    {
        // Set the parameters for creating packed mask for context FMHA.
        tk::PackedMaskParams<bool> maskParams;
        memset((void*) &maskParams, 0, sizeof(maskParams));
        maskParams.maskInput = bufferCastOrNull<bool>(crossAttentionMaskDevice);
        maskParams.cuQSeqLens = bufferCastOrNull<SizeType32>(crossAttentionCuQSeqLensDevice);
        maskParams.packedMask = bufferCastOrNull<uint32_t>(crossAttentionPackedMaskDevice);
        maskParams.cuMaskRows = bufferCastOrNull<SizeType32>(crossAttentionPackedMaskCuMaskRowsDevice);
        maskParams.actualQSeqLens = bufferCastOrNull<SizeType32>(decoderContextLengthsDevice);
        maskParams.actualKvSeqLens = bufferCastOrNull<SizeType32>(encoderInputLengths);
        maskParams.batchSize = contextRequests.size();
        maskParams.maxQSeqLen = maxDecoderContextLength;
        maskParams.maxKvSeqLen = maxEncoderInputLengthInBatch;
        maskParams.attentionMaskType = tk::ContextAttentionMaskType::CUSTOM_MASK;
        maskParams.validPosVal = true;

        // Launch the pack mask kernel.
        tk::invokeBuildPackedMask(maskParams, stream.get());
        sync_check_cuda_error();
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

} // namespace tensorrt_llm::batch_manager
