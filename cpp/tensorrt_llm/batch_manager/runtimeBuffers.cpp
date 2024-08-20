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

#include "runtimeBuffers.h"

#include "rnnStateManager.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/transformerBuffers.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/common/stlUtils.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"
#include "tensorrt_llm/runtime/utils/sessionUtils.h"

#include <algorithm>
#include <iterator>
#include <numeric>
#include <utility>
#include <vector>

using namespace tensorrt_llm::runtime;

namespace tensorrt_llm::batch_manager
{

RuntimeBuffers::RuntimeBuffers(SizeType32 maxBatchSize, SizeType32 maxBeamWidth,
    std::vector<SizeType32> maxAttentionWindowVec, SizeType32 maxAttentionWindow, SizeType32 sinkTokenLen,
    executor::ExtendedRuntimePerfKnobConfig const& extendedRuntimePerfKnobConfig, TensorPtr allReduceWorkspace,
    TllmRuntime const& runtime, ModelConfig const& modelConfig, WorldConfig const& worldConfig,
    executor::DecodingConfig const& decodingConfig, std::optional<SizeType32> maxNumTokens)
    : mAllReduceWorkspace{std::move(allReduceWorkspace)}
{
    if (worldConfig.isTensorParallel())
    {
        TLLM_CHECK(mAllReduceWorkspace);
    }

    create(maxBatchSize, maxBeamWidth, maxAttentionWindowVec, maxAttentionWindow, sinkTokenLen,
        extendedRuntimePerfKnobConfig, runtime, modelConfig, worldConfig, decodingConfig);

    // pre-allocate
    setMaxBufferSizes(maxBatchSize, maxBeamWidth, modelConfig, maxNumTokens);
    auto const maxBatchTokens = getNumTokens();
    reshape(runtime, modelConfig, worldConfig);
    inputsIds->reshape(ITensor::makeShape({maxBatchTokens}));
    if (worldConfig.isPipelineParallel())
    {
        auto const hiddenSize = modelConfig.getHiddenSize() * worldConfig.getTensorParallelism();
        auto const hiddenStatesShape = ITensor::makeShape({maxBatchTokens, hiddenSize});
        hiddenStates->reshape(hiddenStatesShape);
    }
}

void RuntimeBuffers::reshape(TllmRuntime const& runtime, ModelConfig const& modelConfig, WorldConfig const& worldConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(runtimeBuffersReshape);

    auto const numRequests = getNumRequests();
    auto const numSequences = getNumSequences();

    if (worldConfig.isLastPipelineParallelRank())
    {
        auto const vocabSizePadded = modelConfig.getVocabSizePadded(worldConfig.getSize());

        if (modelConfig.computeContextLogits() && (numContextRequests > 0))
        {
            // Only when need to return context logits, and there are new requests will execute context phase,
            // logits buffer need to be re-allocated with size of [numContextTokens + numGenSequences, vocabSizePadded]
            auto const& engine = runtime.getEngine();
            auto const& manager = runtime.getBufferManager();
            auto const logitsType = engine.getTensorDataType("logits");
            logits = manager.gpu(ITensor::makeShape({numContextTokens + numGenSequences, vocabSizePadded}), logitsType);
        }
        else if (modelConfig.computeGenerationLogits() && modelConfig.getSpeculativeDecodingMode().isNone())
        {
            // If need to return generation logits, re-point the logit buffer to avoid overwrite,
            // so we could write back GENERATION_LOGITS_BUFFER_LENGTH steps' logits together
            // logits shape: [1, maxBatchSize * maxBeamWidth, vocabSizePadded]
            // which is large enough to cover both numContextRequests and numGenSequences
            logits = ITensor::slice(cacheGenerationLogits, cacheGenerationLogitsOffset, 1);
            cacheGenerationLogitsOffset = (cacheGenerationLogitsOffset + 1) % GENERATION_LOGITS_BUFFER_LENGTH;
            logits->squeeze(0);
        }
        else
        {
            logits->reshape(ITensor::makeShape({numLogits, vocabSizePadded}));
        }
    }

    requestTypes->reshape(ITensor::makeShape({numSequences}));
    contextLengthsHost->reshape(ITensor::makeShape({numSequences}));
    contextLengthsDevice->reshape(ITensor::makeShape({numSequences}));
    decoderInputLengthsHost->reshape(ITensor::makeShape({numSequences}));
    sequenceLengthsHost->reshape(ITensor::makeShape({numSequences}));
    sequenceLengthsDevice->reshape(ITensor::makeShape({numSequences}));

    lastTokenIdsHost->reshape(ITensor::makeShape({numLogits}));
    lastTokenIdsDevice->reshape(ITensor::makeShape({numLogits}));
    logitsIdsHost->reshape(ITensor::makeShape({numLogits}));
    logitsIdsDevice->reshape(ITensor::makeShape({numLogits}));

    if (transformerBuffers)
    {
        transformerBuffers->reshape(numSequences);
    }

    if (rnnStateBuffers)
    {
        rnnStateBuffers->reshape(numSequences);
    }

    if (modelConfig.useCrossAttention())
    {
        encoderBuffers->reshape();
    }

    if (modelConfig.useLoraPlugin())
    {
        loraBuffers.reshape(numSequences);
    }

    if (medusaBuffers)
    {
        medusaBuffers->reshape(
            numContextRequests, numGenRequests, modelConfig.getSpeculativeDecodingModulePtr()->getMaxDecodingTokens());
    }

    if (lookaheadBuffers)
    {
        lookaheadBuffers->reshape(
            numContextRequests, numGenRequests, modelConfig.getSpeculativeDecodingModulePtr()->getMaxDecodingTokens());
    }

    if (explicitDraftTokensBuffers)
    {
        explicitDraftTokensBuffers->reshape(numContextRequests, numGenRequests, modelConfig);
    }

    seqSlots->reshape(ITensor::makeShape({numRequests}));
    sortedSeqSlots->reshape(ITensor::makeShape({numRequests}));
    seqSlotRemappingHost->reshape(ITensor::makeShape({numRequests}));
    seqSlotRemappingDevice->reshape(ITensor::makeShape({numRequests}));

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void RuntimeBuffers::create(SizeType32 maxBatchSize, SizeType32 maxBeamWidth,
    std::vector<SizeType32> maxAttentionWindowVec, SizeType32 maxAttentionWindow, SizeType32 sinkTokenLen,
    executor::ExtendedRuntimePerfKnobConfig const& extendedRuntimePerfKnobConfig, TllmRuntime const& runtime,
    ModelConfig const& modelConfig, WorldConfig const& worldConfig, executor::DecodingConfig const& decodingConfig)
{
    auto const& manager = runtime.getBufferManager();
    auto const& engine = runtime.getEngine();

    if (modelConfig.isTransformerBased())
    {
        transformerBuffers.emplace(maxBatchSize, maxBeamWidth, maxAttentionWindowVec, maxAttentionWindow, sinkTokenLen,
            extendedRuntimePerfKnobConfig, runtime, modelConfig, worldConfig);
    }
    if (modelConfig.isRnnBased())
    {
        rnnStateBuffers.emplace(maxBatchSize, runtime);
    }

    auto constexpr nvTokenIdType = TRTDataType<TokenIdType>::value;
    inputsIds = manager.emptyTensor(MemoryType::kGPU, nvTokenIdType);
    decoderInputsIds = manager.emptyTensor(MemoryType::kGPU, nvTokenIdType);

    if (worldConfig.isLastPipelineParallelRank())
    {
        auto logitsType = engine.getTensorDataType("logits");
        logits = manager.emptyTensor(MemoryType::kGPU, logitsType);
    }

    seqSlotRemappingHost = manager.emptyTensor(MemoryType::kPINNEDPOOL, nvinfer1::DataType::kINT32);
    seqSlotRemappingDevice = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);

    // TODO(rkobus) check which tensors can be allocated as pinned for max size
    requestTypes = manager.emptyTensor(MemoryType::kCPU, nvinfer1::DataType::kINT32);

    contextLengthsHost = manager.emptyTensor(MemoryType::kCPU, nvinfer1::DataType::kINT32);
    contextLengthsDevice = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
    decoderInputLengthsHost = manager.emptyTensor(MemoryType::kCPU, nvinfer1::DataType::kINT32);
    sequenceLengthsHost = manager.emptyTensor(MemoryType::kCPU, nvinfer1::DataType::kINT32);
    sequenceLengthsDevice = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);

    lastTokenIdsHost = manager.emptyTensor(MemoryType::kCPU, nvinfer1::DataType::kINT32);
    lastTokenIdsDevice = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
    logitsIdsHost = manager.emptyTensor(MemoryType::kCPU, nvinfer1::DataType::kINT32);
    logitsIdsDevice = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);

    inputsIds = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);

    if (worldConfig.isPipelineParallel())
    {
        hiddenStates = manager.emptyTensor(MemoryType::kGPU, modelConfig.getDataType());
    }

    fillValues = manager.pinnedPool(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kINT32);
    fillValuesDevice = manager.gpu(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kINT32);
    seqSlots = manager.pinnedPool(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kINT32);
    seqSlotsDevice = manager.gpu(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kINT32);
    sortedSeqSlots = manager.pinnedPool(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kINT32);

    cacheIndirDecoderIOBatchedCopySrcOffsets
        = manager.pinnedPool(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kINT32);
    cacheIndirDecoderIOBatchedCopyDstOffsets
        = manager.pinnedPool(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kINT32);
    cacheIndirDecoderIOBatchedCopySizes
        = manager.pinnedPool(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kINT32);
    mCacheIndirDecoderIOBatchedCopySrcOffsetsSliceDevice
        = manager.gpu(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kINT32);
    mCacheIndirDecoderIOBatchedCopyDstOffsetsSliceDevice
        = manager.gpu(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kINT32);
    mCacheIndirDecoderIOBatchedCopyCopySizesDevice
        = manager.gpu(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kINT32);

    // Pre-allocate buffer for saving generation logits for model w/o draft tokens
    if (modelConfig.computeGenerationLogits()
        && (modelConfig.getSpeculativeDecodingMode().isDraftTokensExternal()
            || modelConfig.getSpeculativeDecodingMode().isNone())
        && worldConfig.isLastPipelineParallelRank())
    {
        auto const vocabSizePadded = modelConfig.getVocabSizePadded(worldConfig.getSize());
        auto const logitsType = engine.getTensorDataType("logits");

        cacheTransposedGenerationLogits = manager.gpu(
            ITensor::makeShape({maxBatchSize, maxBeamWidth, GENERATION_LOGITS_BUFFER_LENGTH, vocabSizePadded}),
            logitsType);
        cacheGenerationLogits = manager.gpu(
            ITensor::makeShape({GENERATION_LOGITS_BUFFER_LENGTH, maxBatchSize * maxBeamWidth, vocabSizePadded}),
            logitsType);

        cacheGenerationFragmentPointerDevice = manager.gpu(
            ITensor::makeShape({maxBatchSize, GENERATION_LOGITS_BUFFER_LENGTH}), nvinfer1::DataType::kINT64);
        cacheGenerationFragmentPointerHost = manager.pinnedPool(
            ITensor::makeShape({maxBatchSize, GENERATION_LOGITS_BUFFER_LENGTH}), nvinfer1::DataType::kINT64);
    }

    if (modelConfig.useCrossAttention())
    {
        encoderBuffers.emplace();
        encoderBuffers->create(maxBatchSize, modelConfig, runtime);
    }

    if (modelConfig.usePromptTuning())
    {
        promptTuningBuffers.create(maxBatchSize, manager, modelConfig, worldConfig);
    }

    if (modelConfig.useLoraPlugin())
    {
        loraBuffers.create(maxBatchSize, maxBeamWidth, runtime, modelConfig, worldConfig);
    }

    if (modelConfig.getSpeculativeDecodingMode().isMedusa())
    {
        medusaBuffers.emplace(maxBatchSize, maxBeamWidth, manager, modelConfig, worldConfig, decodingConfig, runtime);
    }

    if (modelConfig.getSpeculativeDecodingMode().isLookaheadDecoding())
    {
        lookaheadBuffers.emplace(
            maxBatchSize, maxBeamWidth, manager, modelConfig, worldConfig, decodingConfig, runtime);
    }

    if (modelConfig.getSpeculativeDecodingMode().isExplicitDraftTokens())
    {
        explicitDraftTokensBuffers.emplace(
            maxBatchSize, maxBeamWidth, manager, modelConfig, worldConfig, decodingConfig, runtime);
    }
}

void RuntimeBuffers::setMaxBufferSizes(SizeType32 maxBatchSize, SizeType32 maxBeamWidth,
    runtime::ModelConfig const& modelConfig, std::optional<SizeType32> maxNumRuntimeTokens)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    // maxNumSequences is reached when all requests are in generation
    numContextRequests = 0;
    numGenSequences = maxBatchSize * maxBeamWidth;

    auto const maxDraftTokens = modelConfig.getMaxDecodingDraftTokens();
    auto const maxNumContextTokens = maxBatchSize * modelConfig.getMaxInputLen();
    auto const maxNumGenTokens = maxBatchSize * std::max(1 + maxDraftTokens, maxBeamWidth);
    auto maxNumModelTokens = modelConfig.getMaxNumTokens();
    // this is only used for computeContextLogits, do not set here
    numContextTokens = 0;
    // set maxNumTokens for pre-allocation
    numGenTokens
        = maxNumRuntimeTokens.value_or(maxNumModelTokens.value_or(std::max(maxNumContextTokens, maxNumGenTokens)));

    // Draft tokens cannot be combined with beam search
    numLogits = maxBatchSize * std::max(1 + maxDraftTokens, maxBeamWidth);

    if (modelConfig.useCrossAttention())
    {
        encoderBuffers->setMaxBufferSizes(maxBatchSize, modelConfig);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void RuntimeBuffers::setBufferSizes(RequestVector const& contextRequests, RequestVector const& genRequests)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(runtimeBuffersSetBufferSizes);

    // set context sizes
    numContextRequests = static_cast<SizeType32>(contextRequests.size());
    auto numContextLogits = numContextRequests;
    numContextTokens = 0;
    maxContextLength = 0;
    for (auto const& llmReq : contextRequests)
    {
        auto const draftLength = llmReq->isLastContextChunk() ? llmReq->getNumDraftTokens() : 0;
        numContextLogits += draftLength;

        auto const contextChunkSize
            = llmReq->isFullContextRequest() ? llmReq->mPromptLen : llmReq->getContextChunkSize();
        numContextTokens += contextChunkSize + draftLength;
        if (maxContextLength < llmReq->mPromptLen)
            maxContextLength = llmReq->mPromptLen;
    }

    // set generation sizes
    numGenRequests = static_cast<SizeType32>(genRequests.size());
    numGenSequences = 0;
    numGenTokens = 0;
    for (auto const& llmReq : genRequests)
    {
        auto const reqBeamWidth = llmReq->mSamplingConfig.beamWidth;
        numGenSequences += reqBeamWidth;
        auto const draftLen = llmReq->getNumDraftTokens();
        numGenTokens += draftLen + reqBeamWidth;
    }

    numLogits = numContextLogits + numGenTokens;

    if (encoderBuffers)
    {
        encoderBuffers->setBufferSizes(contextRequests, genRequests);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void RuntimeBuffers::setFromInputs(RequestVector const& contextRequests, RequestVector const& genRequests,
    SizeType32 maxBeamWidth, SizeType32 maxAttentionWindow, DecoderBuffers& decoderBuffers,
    kv_cache_manager::KVCacheManager* kvCacheManagerPtr, kv_cache_manager::KVCacheManager* crossKvCacheManagerPtr,
    rnn_state_manager::RnnStateManager* rnnStateManagerPtr, PeftTable const& peftTable,
    runtime::TllmRuntime const& runtime, runtime::ModelConfig const& modelConfig,
    runtime::WorldConfig const& worldConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(runtimeBuffersSetFromInputs);

    auto const& manager = runtime.getBufferManager();
    auto const& stream = runtime.getStream();

    SizeType32 totalInputSize = 0;
    std::vector<TokenIdType> inputHost;
    std::vector<TokenIdType> decoderInputHost;
    std::vector<SizeType32> positionIdsHost;
    std::vector<SizeType32> positionIdsHostRow2;
    auto* hostRequestTypes = bufferCast<SizeType32>(*requestTypes);
    auto* contextLengthsHostPtr = bufferCast<SizeType32>(*contextLengthsHost);
    auto* decoderInputLengthsHostPtr = bufferCast<SizeType32>(*decoderInputLengthsHost);
    auto* sequenceLengthsHostPtr = bufferCast<SizeType32>(*sequenceLengthsHost);
    auto* pastKeyValueLengthsPtr
        = transformerBuffers ? bufferCast<SizeType32>(*transformerBuffers->pastKeyValueLengths) : nullptr;
    SizeType32 totalNumLogits{0};
    auto* logitsIdsHostPtr = bufferCast<SizeType32>(*logitsIdsHost);
    bool const isChatGlm = modelConfig.getModelVariant() == ModelConfig::ModelVariant::kChatGlm;
    bool const isGlm = modelConfig.getModelVariant() == ModelConfig::ModelVariant::kGlm;

    // sequence length fill common loop
    {
        NVTX3_SCOPED_RANGE(sequenceLenLoop);
        auto* seqSlotIndices = bufferCast<SizeType32>(*seqSlots);
        auto* fillValuesPtr = bufferCast<SizeType32>(*fillValues);

        SizeType32 batchIdx{0};
        for (auto const& requests : {contextRequests, genRequests})
        {
            for (auto const& llmReq : requests)
            {
                auto const currentSequenceLen = llmReq->mPromptLen + llmReq->getMaxNumGeneratedTokens();
                // Get position of the current sequence in the KV cache
                auto const seqSlot = llmReq->mSeqSlot.value();
                seqSlotIndices[batchIdx] = seqSlot;
                fillValuesPtr[batchIdx] = currentSequenceLen;
                ++batchIdx;
            }
        }

        TLLM_CHECK(seqSlots->getSize() == static_cast<std::size_t>(batchIdx));
        TensorPtr seqSlotsDeviceSlice = ITensor::slice(seqSlotsDevice, 0, batchIdx);
        manager.copy(*ITensor::slice(seqSlots, 0, batchIdx), *seqSlotsDeviceSlice);
        manager.copy(*fillValues, *fillValuesDevice);
        runtime::kernels::invokeFillBatch<SizeType32>(
            *decoderBuffers.sequenceLengths, *seqSlotsDeviceSlice, maxBeamWidth, *fillValuesDevice, stream);
    }

    // context preparation loop
    if (!contextRequests.empty())
    {
        NVTX3_SCOPED_RANGE(contextPrepareLoop);
        numContextLogits.resize(contextRequests.size());

        SizeType32 batchIdx{0};
        for (auto const& llmReq : contextRequests)
        {
            TLLM_CHECK_WITH_INFO(llmReq->isContextInitState(), "The request should be in context phase.");
            TLLM_CHECK_WITH_INFO(
                llmReq->getMaxNumGeneratedTokens() == 0, "Context request should not have generated tokens.");

            SizeType32 constexpr requestType{0};
            hostRequestTypes[batchIdx] = requestType;

            auto const promptLen = llmReq->mPromptLen;
            auto const& reqTokens = llmReq->getTokens(0);
            auto const& draftTokens = llmReq->getDraftTokens();
            auto const draftLength = llmReq->getNumDraftTokens();

            decoderInputHost.insert(decoderInputHost.end(), reqTokens.begin(), reqTokens.end());
            decoderInputLengthsHostPtr[batchIdx] = promptLen;

            auto const contextChunkSize
                = llmReq->isFullContextRequest() ? llmReq->mPromptLen : llmReq->getContextChunkSize();
            auto const beginCompute = llmReq->getContextCurrentPosition();
            auto const endCompute = beginCompute + contextChunkSize;
            inputHost.insert(inputHost.end(), reqTokens.begin() + beginCompute, reqTokens.begin() + endCompute);

            logitsIdsHostPtr[totalNumLogits++] = contextChunkSize;
            numContextLogits.at(batchIdx) = modelConfig.computeContextLogits() ? contextChunkSize : 1;

            if (llmReq->isLastContextChunk())
            {
                inputHost.insert(inputHost.end(), draftTokens->begin(), draftTokens->end());
                std::fill_n(logitsIdsHostPtr + totalNumLogits, draftLength, 1);
                totalNumLogits += draftLength;
                decoderInputHost.insert(decoderInputHost.end(), draftTokens->begin(), draftTokens->end());
            }
            auto const inputLength = contextChunkSize + (llmReq->isLastContextChunk() ? draftLength : 0);
            contextLengthsHostPtr[batchIdx] = inputLength;
            auto const sequenceLen = inputLength + llmReq->getContextCurrentPosition();
            sequenceLengthsHostPtr[batchIdx] = sequenceLen;

            if (pastKeyValueLengthsPtr)
            {
                pastKeyValueLengthsPtr[batchIdx] = beginCompute + inputLength;
            }

            if (isChatGlm) // ChatGLM-6B
            {
                // Using 2D Position Encoding, shape of positionIds is doubled than gpt.
                positionIdsHost.resize(totalInputSize + inputLength);
                std::iota(std::begin(positionIdsHost) + totalInputSize, std::end(positionIdsHost), 0);
                positionIdsHost.back() = positionIdsHost.back() - 1;

                positionIdsHostRow2.resize(totalInputSize + inputLength);
                positionIdsHostRow2.back() = 1;
            }
            else if (isGlm)
            {
                // iterate over inputIds to find mask id position
                auto start = inputHost.begin() + totalInputSize;
                auto end = start + inputLength;
                auto it
                    = std::find_if(start, end, [](SizeType32 id) { return id == 50260 || id == 50263 || id == 50264; });
                if (it != end)
                {
                    llmReq->mMaskPosition = std::distance(start, it);
                }
                else
                {
                    llmReq->mMaskPosition = maxContextLength;
                }

                // Using 2D Position Encoding, shape of positionIds is doubled than gpt.
                positionIdsHost.resize(totalInputSize + inputLength);
                std::iota(std::begin(positionIdsHost) + totalInputSize, std::end(positionIdsHost), 0);
                positionIdsHost.back() = llmReq->mMaskPosition;

                positionIdsHostRow2.resize(totalInputSize + inputLength);
                positionIdsHostRow2.back() = 1;
            }
            else // GPT / ChatGLM2-6B / ChatGLM3-6B
            {
                positionIdsHost.resize(totalInputSize + inputLength);
                std::iota(std::begin(positionIdsHost) + totalInputSize,
                    std::begin(positionIdsHost) + totalInputSize + inputLength, beginCompute);
            }
            totalInputSize += inputLength;
            ++batchIdx;
        }

        if (rnnStateBuffers)
        {
            rnnStateBuffers->fillSlotMappings(contextRequests, rnnStateManagerPtr);
        }

        if (transformerBuffers && maxBeamWidth > 1)
        {
            transformerBuffers->resetCacheIndirection(contextRequests, maxBeamWidth, maxAttentionWindow,
                decoderBuffers.cacheIndirectionInput, decoderBuffers.cacheIndirectionOutput, runtime);
        }
    }

    // generation preparation loop
    if (!genRequests.empty())
    {
        NVTX3_SCOPED_RANGE(genPrepareLoop);

        auto const numContextRequests = static_cast<SizeType32>(contextRequests.size());
        auto numSequences = numContextRequests;
        for (auto const& llmReq : genRequests)
        {
            auto const reqBeamWidth = llmReq->mSamplingConfig.beamWidth;

            SizeType32 constexpr requestType{1};
            std::fill_n(hostRequestTypes + numSequences, reqBeamWidth, requestType);

            // Get position of the current sequence in the KV cache
            auto const seqSlot = llmReq->mSeqSlot.value();

            auto const draftLength = llmReq->getNumDraftTokens();
            auto const& draftTokens = llmReq->getDraftTokens();
            auto const numLogits = draftLength + reqBeamWidth;
            TLLM_CHECK(draftLength == 0 || reqBeamWidth == 1);

            auto const promptLen = llmReq->mPromptLen;
            auto const sequenceLen = promptLen + llmReq->getMaxNumGeneratedTokens();

            for (int beam = 0; beam < reqBeamWidth; ++beam)
            {
                auto const lastToken = llmReq->getLastTokens(beam);
                auto const numTokens = llmReq->getNumTokens(beam);
                inputHost.push_back(lastToken);
                decoderInputHost.push_back(lastToken);

                // If model updates generation position ids do not append them here.
                if (!modelConfig.getSpeculativeDecodingMode().updatesPositionIds())
                {
                    if (isChatGlm) // ChatGLM-6B
                    {
                        positionIdsHost.push_back(static_cast<SizeType32>(promptLen - 2));
                        positionIdsHostRow2.push_back(static_cast<SizeType32>(sequenceLen - promptLen + 1));
                    }
                    else if (isGlm)
                    {
                        positionIdsHost.push_back(llmReq->mMaskPosition);
                        positionIdsHostRow2.push_back(static_cast<SizeType32>(sequenceLen - promptLen + 1));
                    }
                    else // GPT / ChatGLM2-6B / ChatGLM3-6B / BART
                    {
                        // positionIds is just the size of tokens -1
                        positionIdsHost.push_back(numTokens - 1);
                    }
                }

                if (draftLength > 0)
                {
                    inputHost.insert(inputHost.end(), draftTokens->begin(), draftTokens->end());
                    decoderInputHost.insert(decoderInputHost.end(), draftTokens->begin(), draftTokens->end());
                }
            }

            SizeType32 pastKeyValueLength = sequenceLen - 1;

            std::fill_n(decoderInputLengthsHostPtr + numSequences, reqBeamWidth, draftLength + 1);
            if (pastKeyValueLengthsPtr)
            {
                std::fill_n(pastKeyValueLengthsPtr + numSequences, reqBeamWidth, pastKeyValueLength);
            }
            totalInputSize += numLogits;

            std::fill_n(logitsIdsHostPtr + totalNumLogits, numLogits, 1);

            totalNumLogits += numLogits;

            if (rnnStateBuffers)
            {
                auto& rnnStateManager = *rnnStateManagerPtr;
                rnnStateManager.fillSlotMapping(*rnnStateBuffers->slotMappingHost, numSequences, seqSlot, reqBeamWidth);
            }
            numSequences += reqBeamWidth;
        }

        if (transformerBuffers && maxBeamWidth > 1)
        {
            transformerBuffers->copyCacheIndirection(genRequests, decoderBuffers.cacheIndirectionOutput, runtime);
        }

        numSequences = numContextRequests;
        for (auto const& llmReq : genRequests)
        {
            auto const reqBeamWidth = llmReq->mSamplingConfig.beamWidth;

            auto const draftLength = llmReq->getNumDraftTokens();

            auto const contextQLength = llmReq->mPromptLen + draftLength;
            auto const sequenceLen = contextQLength + llmReq->getMaxNumGeneratedTokens();

            std::fill_n(contextLengthsHostPtr + numSequences, reqBeamWidth, contextQLength);
            std::fill_n(sequenceLengthsHostPtr + numSequences, reqBeamWidth, sequenceLen);
            numSequences += reqBeamWidth;
        }
        if (modelConfig.getSpeculativeDecodingMode().needsKVCacheRewind())
        {
            auto remappingSeqSlotIndices = BufferRange<SizeType32>(*seqSlotRemappingHost);
            auto const* seqSlotIndices = bufferCast<SizeType32>(*seqSlots);

            std::iota(remappingSeqSlotIndices.begin(), remappingSeqSlotIndices.end(), 0);
            std::sort(remappingSeqSlotIndices.begin(), remappingSeqSlotIndices.end(),
                [&seqSlotIndices](SizeType32 a, SizeType32 b) { return seqSlotIndices[a] < seqSlotIndices[b]; });
            manager.copy(*seqSlotRemappingHost, *seqSlotRemappingDevice);

            manager.copy(*seqSlots, *sortedSeqSlots);
            auto sortedSeqSlotIndices = BufferRange<SizeType32>(*sortedSeqSlots);
            std::sort(sortedSeqSlotIndices.begin(), sortedSeqSlotIndices.end());
        }
        if (modelConfig.getSpeculativeDecodingMode().isLookaheadDecoding())
        {
            // copy from lookahead decoding buffer
            lookaheadBuffers->setFromInputs(numContextRequests, numGenRequests, *requestTypes, *seqSlots,
                decoderBuffers.lookaheadBuffers.value(), runtime, modelConfig, worldConfig);
        }
    }

    if (isChatGlm || isGlm)
    {
        positionIdsHost.reserve(totalInputSize * 2);
        positionIdsHost.insert(positionIdsHost.end(), positionIdsHostRow2.begin(), positionIdsHostRow2.end());
    }

    if (transformerBuffers && kvCacheManagerPtr)
    {
        transformerBuffers->copyKvBlockOffsets(
            contextRequests, genRequests, kvCacheManagerPtr, crossKvCacheManagerPtr, runtime);
    }

    if (modelConfig.useCrossAttention())
    {
        encoderBuffers->fill(contextRequests, genRequests, manager);
    }
    if (modelConfig.usePromptTuning())
    {
        promptTuningBuffers.fill(contextRequests, genRequests, manager, modelConfig.usePackedInput());
    }
    if (modelConfig.useLoraPlugin())
    {
        loraBuffers.fill(contextRequests, genRequests, peftTable, manager, modelConfig, worldConfig);
    }

    {
        NVTX3_SCOPED_RANGE(bufferCopies);
        inputsIds->reshape(ITensor::makeShape({totalInputSize}));
        manager.copy(inputHost.data(), *inputsIds);
        decoderInputsIds->reshape(ITensor::makeShape({static_cast<int>(decoderInputHost.size())}));
        manager.copy(decoderInputHost.data(), *decoderInputsIds);
        // In generation phase, device ptr of context lengths need to be tiled.
        manager.copy(*contextLengthsHost, *contextLengthsDevice);
        manager.copy(*sequenceLengthsHost, *sequenceLengthsDevice);
        manager.copy(*logitsIdsHost, *logitsIdsDevice);
        auto const logitsIdsHostRange = BufferRange<SizeType32>(*logitsIdsHost);
        auto lastTokenIdsHostRange = BufferRange<SizeType32>(*lastTokenIdsHost);
        common::stl_utils::inclusiveScan(
            logitsIdsHostRange.begin(), logitsIdsHostRange.end(), lastTokenIdsHostRange.begin());
        manager.copy(*lastTokenIdsHost, *lastTokenIdsDevice);
        if (transformerBuffers)
        {
            TensorPtr decoderPositionIds = modelConfig.getSpeculativeDecodingMode().isLookaheadDecoding()
                ? ITensor::slice(lookaheadBuffers->positionIdsDevice, 0, numGenRequests)
                : nullptr;
            transformerBuffers->copyPositionIds(runtime, positionIdsHost, isChatGlm || isGlm, decoderPositionIds);
        }
        if (rnnStateBuffers)
        {
            rnnStateBuffers->copySlotMappingH2D(runtime);
        }
    }

    if (worldConfig.isPipelineParallel())
    {
        auto const hiddenSize = modelConfig.getHiddenSize() * worldConfig.getTensorParallelism();
        auto const hiddenStatesShape = ITensor::makeShape({totalInputSize, hiddenSize});
        hiddenStates->reshape(hiddenStatesShape);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void RuntimeBuffers::prepareExplicitDraftTokenBuffers(DecoderBuffers& decoderBuffers, TllmRuntime const& runtime,
    ModelConfig const& modelConfig, WorldConfig const& worldConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    TLLM_CHECK(explicitDraftTokensBuffers);

    explicitDraftTokensBuffers->setFromInputs(numContextRequests, numGenRequests, *requestTypes, *seqSlots,
        decoderBuffers.explicitDraftTokensBuffers, *transformerBuffers->positionIds, runtime, modelConfig, worldConfig);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

std::tuple<SizeType32, RuntimeBuffers::TensorMap const&, RuntimeBuffers::TensorMap&> RuntimeBuffers::prepareStep(
    RequestVector const& contextRequests, RequestVector const& genRequests, SizeType32 maxBeamWidth,
    SizeType32 maxAttentionWindow, DecoderBuffers& decoderBuffers, kv_cache_manager::KVCacheManager* kvCacheManager,
    kv_cache_manager::KVCacheManager* crossKvCacheManager, rnn_state_manager::RnnStateManager* rnnStateManager,
    PeftTable const& peftTable, TllmRuntime const& runtime, ModelConfig const& modelConfig,
    WorldConfig const& worldConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(runtimeBuffersPrepareStep);

    setBufferSizes(contextRequests, genRequests);
    reshape(runtime, modelConfig, worldConfig);

    setFromInputs(contextRequests, genRequests, maxBeamWidth, maxAttentionWindow, decoderBuffers, kvCacheManager,
        crossKvCacheManager, rnnStateManager, peftTable, runtime, modelConfig, worldConfig);

    fillIOMaps(rnnStateManager, modelConfig, worldConfig);

    auto const numTokens = getNumTokens();
    auto const optProfileId = runtime.getOptProfileId(numTokens, ModelConfig::getOptProfilesSplitPoints());
    setContextIndex(optProfileId);
    TLLM_LOG_DEBUG("numTokens: %d, optProfileId: %d", numTokens, optProfileId);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return {optProfileId, inputMap, outputMap};
}

void RuntimeBuffers::fillIOMaps(
    rnn_state_manager::RnnStateManager* rnnStateManager, ModelConfig const& modelConfig, WorldConfig const& worldConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(runtimeBuffersFillIOMaps);

    inputMap.clear();
    outputMap.clear();

    if (transformerBuffers)
    {
        transformerBuffers->getBuffers(inputMap);
    }
    if (rnnStateBuffers)
    {
        rnnStateBuffers->getBuffers(rnnStateManager, inputMap, modelConfig, worldConfig);
    }

    if (worldConfig.isLastPipelineParallelRank())
    {
        // feed a view to TensorRT runtime so reshaping does not change logits buffer
        outputMap.insert_or_assign("logits", ITensor::view(logits));
    }
    else
    {
        outputMap.insert_or_assign("hidden_states_output", hiddenStates);
    }

    if (worldConfig.isFirstPipelineParallelRank())
    {
        inputMap.insert_or_assign("input_ids", inputsIds);
    }
    else
    {
        inputMap.insert_or_assign("hidden_states_input", hiddenStates);
    }

    inputMap.insert_or_assign("last_token_ids", lastTokenIdsDevice);

    inputMap.insert_or_assign("host_request_types", requestTypes);
    // In the generation phase, we still pass context lengths.
    inputMap.insert_or_assign("context_lengths", contextLengthsDevice);
    inputMap.insert_or_assign("host_context_lengths", contextLengthsHost);
    inputMap.insert_or_assign("sequence_length", sequenceLengthsDevice);

    if (worldConfig.isTensorParallel())
    {
        inputMap.insert_or_assign("all_reduce_workspace", mAllReduceWorkspace);
    }

    if (modelConfig.useCrossAttention())
    {
        encoderBuffers->insertInputTensors(inputMap);
    }

    if (modelConfig.usePromptTuning())
    {
        auto const& promptTuningParams = promptTuningBuffers.mPromptTuningParams;
        inputMap.insert_or_assign("prompt_embedding_table", promptTuningParams.embeddingTable);
        inputMap.insert_or_assign("tasks", promptTuningParams.tasks);
        inputMap.insert_or_assign("prompt_vocab_size", promptTuningParams.vocabSize);
    }

    if (modelConfig.useLoraPlugin())
    {
        loraBuffers.insertInputTensors(inputMap, loraBuffers.mLoraWeightsPointersHost,
            loraBuffers.mLoraAdapterSizesHost, modelConfig, worldConfig);
    }

    if (medusaBuffers)
    {
        medusaBuffers->insertInputTensors(inputMap, outputMap, worldConfig);
    }

    if (lookaheadBuffers)
    {
        lookaheadBuffers->insertInputTensors(inputMap, outputMap, worldConfig);
    }

    if (explicitDraftTokensBuffers)
    {
        explicitDraftTokensBuffers->insertInputTensors(inputMap, outputMap, worldConfig);
    }

    // runtime::utils::printTensorMap(std::cerr, inputBuffers);
    // runtime::utils::printTensorMap(std::cerr, outputBuffers);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

} // namespace tensorrt_llm::batch_manager
