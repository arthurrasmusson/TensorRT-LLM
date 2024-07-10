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

#include "encoderBuffers.h"

#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"

using namespace tensorrt_llm::runtime;

namespace tensorrt_llm::batch_manager
{

EncoderBuffers::EncoderBuffers(
    SizeType32 maxBatchSize, ModelConfig const& modelConfig, WorldConfig const& worldConfig, TllmRuntime const& runtime)
{
    // init empty buffers on cpu/gpu/pinned
    init(maxBatchSize, modelConfig, worldConfig, runtime);

    // pre-allocate based on max buffer sizes
    // Note: pre-allocation can be done directly instead of empty-->reshape, but it is ok extract the common reshape()
    // utility because the buffer shapes can be dynamically set during runtime as well
    initBufferSizes(maxBatchSize, modelConfig, worldConfig, runtime);
}

void EncoderBuffers::init(
    SizeType32 maxBatchSize, ModelConfig const& modelConfig, WorldConfig const& worldConfig, TllmRuntime const& runtime)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const& manager = runtime.getBufferManager();

    inputIds = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
    // in PP, only rank 0 needs the following input fields
    if (modelConfig.usePositionEmbedding() && worldConfig.isFirstPipelineParallelRank())
    {
        positionIds = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
        positionIdsReserved.resize(maxBatchSize * modelConfig.getMaxInputLen());
        std::iota(positionIdsReserved.begin(), positionIdsReserved.end(), 0);
    }
    if (modelConfig.useTokenTypeEmbedding() && worldConfig.isFirstPipelineParallelRank())
    {
        tokenTypeIds = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
        tokenTypeIdsReserved.resize(maxBatchSize * modelConfig.getMaxInputLen());
        std::fill(tokenTypeIdsReserved.begin(), tokenTypeIdsReserved.end(), 0);
    }

    inputLengths = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
    maxInputLength = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);

    auto hiddenStatesType = modelConfig.getDataType();
    if (worldConfig.isPipelineParallel())
    {
        hiddenStates = manager.emptyTensor(MemoryType::kGPU, hiddenStatesType);
    }
    if (worldConfig.isLastPipelineParallelRank())
    {
        encoderOutput = manager.emptyTensor(MemoryType::kGPU, hiddenStatesType);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void EncoderBuffers::initBufferSizes(
    SizeType32 maxBatchSize, ModelConfig const& modelConfig, WorldConfig const& worldConfig, TllmRuntime const& runtime)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    // get buffer shape based on max values
    numRequests = maxBatchSize;
    numTokens = maxBatchSize * modelConfig.getMaxInputLen();
    maxInputLengthInBatch = modelConfig.getMaxInputLen();

    // update buffer shapes
    reshape(runtime, modelConfig, worldConfig);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void EncoderBuffers::updateBufferSizes(RequestVector const& requests, ModelConfig const& modelConfig,
    WorldConfig const& worldConfig, TllmRuntime const& runtime)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    numRequests = requests.size();
    numTokens = 0;
    maxInputLengthInBatch = 0;

    // get buffer shape based on actual batched requests
    for (auto const& req : requests)
    {
        auto reqLength = req->getEncoderLen();
        numTokens += reqLength;
        maxInputLengthInBatch = std::max(maxInputLengthInBatch, reqLength);

        // update request-owned external buffer for each request
        if (worldConfig.isPipelineParallel())
        {
            req->getEncoderHiddenStates()->reshape(
                ITensor::makeShape({reqLength, modelConfig.getHiddenSize() * worldConfig.getTensorParallelism()}));
        }
        if (worldConfig.isLastPipelineParallelRank())
        {
            req->getEncoderOutput()->reshape(
                ITensor::makeShape({reqLength, modelConfig.getHiddenSize() * worldConfig.getTensorParallelism()}));
        }
    }

    // update buffer shapes
    reshape(runtime, modelConfig, worldConfig);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void EncoderBuffers::reshape(TllmRuntime const& runtime, ModelConfig const& modelConfig, WorldConfig const& worldConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    inputIds->reshape(ITensor::makeShape({numTokens}));
    if (positionIds)
    {
        positionIds->reshape(ITensor::makeShape({numTokens}));
    }
    if (tokenTypeIds)
    {
        tokenTypeIds->reshape(ITensor::makeShape({numTokens}));
    }

    inputLengths->reshape(ITensor::makeShape({numRequests}));
    maxInputLength->reshape(ITensor::makeShape({maxInputLengthInBatch}));

    if (worldConfig.isPipelineParallel())
    {
        hiddenStates->reshape(
            ITensor::makeShape({numTokens, modelConfig.getHiddenSize() * worldConfig.getTensorParallelism()}));
    }
    if (worldConfig.isLastPipelineParallelRank())
    {
        encoderOutput->reshape(
            ITensor::makeShape({numTokens, modelConfig.getHiddenSize() * worldConfig.getTensorParallelism()}));
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void EncoderBuffers::setFromInputs(RequestVector const& requests, ModelConfig const& modelConfig,
    WorldConfig const& worldConfig, TllmRuntime const& runtime)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(encoderBuffersSetFromInputs);

    if (!worldConfig.isFirstPipelineParallelRank())
    {
        return;
    }

    auto const& manager = runtime.getBufferManager();

    std::vector<TokenIdType> inputIdsAll;
    std::vector<SizeType32> positionIdsAll;
    std::vector<SizeType32> tokenTypeIdsAll;
    std::vector<SizeType32> inputLengthsAll;
    // use shape to indicates max input length, content is not important
    // TODO: change to a scalar value for this from engine side
    std::vector<SizeType32> maxInputLengthAll(maxInputLengthInBatch, 0);

    // collect inputs in batched requests
    for (auto const& req : requests)
    {
        auto const& reqTokens = *req->getEncoderTokens().value();
        auto reqLength = reqTokens.size();
        inputIdsAll.insert(inputIdsAll.end(), reqTokens.begin(), reqTokens.end());
        if (positionIds)
        {
            positionIdsAll.insert(
                positionIdsAll.end(), positionIdsReserved.begin(), positionIdsReserved.begin() + reqLength);
        }
        if (tokenTypeIds)
        {
            tokenTypeIdsAll.insert(
                tokenTypeIdsAll.end(), tokenTypeIdsReserved.begin(), tokenTypeIdsReserved.begin() + reqLength);
        }
        inputLengthsAll.emplace_back(reqLength);
    }

    // copy inputs from host to device
    {
        NVTX3_SCOPED_RANGE(bufferCopies);
        manager.copy(inputIdsAll.data(), *inputIds);
        if (positionIds)
        {
            manager.copy(positionIdsAll.data(), *positionIds);
        }
        if (tokenTypeIds)
        {
            manager.copy(tokenTypeIdsAll.data(), *tokenTypeIds);
        }
        manager.copy(inputLengthsAll.data(), *inputLengths);
        manager.copy(maxInputLengthAll.data(), *maxInputLength);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void EncoderBuffers::fillIOMaps(ModelConfig const& modelConfig, WorldConfig const& worldConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(runtimeBuffersFillIOMaps);

    inputMap.clear();
    outputMap.clear();

    // inputs
    if (worldConfig.isFirstPipelineParallelRank())
    {
        inputMap.insert_or_assign("input_ids", inputIds);
        if (positionIds)
        {
            inputMap.insert_or_assign("position_ids", positionIds);
        }
        if (tokenTypeIds)
        {
            inputMap.insert_or_assign("token_type_ids", tokenTypeIds);
        }
    }
    else
    {
        inputMap.insert_or_assign("hidden_states_input", hiddenStates);
    }
    inputMap.insert_or_assign("input_lengths", inputLengths);
    inputMap.insert_or_assign("max_input_length", maxInputLength);

    // outputs
    if (worldConfig.isLastPipelineParallelRank())
    {
        outputMap.insert_or_assign("encoder_output", encoderOutput);
    }
    else
    {
        outputMap.insert_or_assign("hidden_states_output", hiddenStates);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

std::pair<EncoderBuffers::TensorMap const&, EncoderBuffers::TensorMap&> EncoderBuffers::prepareIO(
    RequestVector const& requests, ModelConfig const& modelConfig, WorldConfig const& worldConfig,
    TllmRuntime const& runtime)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    updateBufferSizes(requests, modelConfig, worldConfig, runtime);

    setFromInputs(requests, modelConfig, worldConfig, runtime);

    fillIOMaps(modelConfig, worldConfig);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);

    return {inputMap, outputMap};
}

void EncoderBuffers::rearrangeOutputs(RequestVector const& requests, ModelConfig const& modelConfig,
    WorldConfig const& worldConfig, TllmRuntime const& runtime)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(encoderBuffersRearrangeOutput);

    auto const& manager = runtime.getBufferManager();

    SizeType32 offset = 0, size = 0;
    for (auto const& req : requests)
    {
        // copy from internal buffer to request-owned external buffers
        size = req->getEncoderLen();
        if (worldConfig.isPipelineParallel())
        {
            manager.copy(*ITensor::slice(hiddenStates, offset, size), *req->getEncoderHiddenStates());
        }
        if (worldConfig.isLastPipelineParallelRank())
        {
            manager.copy(*ITensor::slice(encoderOutput, offset, size), *req->getEncoderOutput());
        }
        offset += size;
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void EncoderBuffers::create(SizeType32 maxBatchSize, ModelConfig const& modelConfig, TllmRuntime const& runtime)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const& manager = runtime.getBufferManager();

    inputLengths = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
    maxInputLength = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);

    hiddenSize = modelConfig.getEncoderHiddenSize(); // full hidden size
    // assume encoder & decoder use the same data type
    encoderOutput = manager.emptyTensor(MemoryType::kGPU, modelConfig.getDataType());
    encoderOutputReserved = manager.gpu(ITensor::makeShape({1, hiddenSize}), modelConfig.getDataType());

    crossKvCacheGen = manager.gpu(ITensor::makeShape({1}), nvinfer1::DataType::kBOOL);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void EncoderBuffers::setMaxBufferSizes(SizeType32 maxBatchSize, runtime::ModelConfig const& modelConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    numRequests = maxBatchSize;
    numTokens = maxBatchSize * modelConfig.getMaxEncoderLen();
    maxInputLengthInBatch = modelConfig.getMaxEncoderLen();

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void EncoderBuffers::setBufferSizes(RequestVector const& contextRequests, RequestVector const& genRequests)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    numRequests = 0;           /// total number of requests that need encoder information (context requests +
                               /// generation requests * beam width)
    numTokens = 0;             /// total number of encoder tokens across context requests
    maxInputLengthInBatch = 1; /// maximum encoder length in a batch

    for (auto const& llmReq : contextRequests)
    {
        numRequests += 1;
        numTokens += llmReq->getEncoderLen();
        maxInputLengthInBatch = std::max(maxInputLengthInBatch, llmReq->getEncoderLen());
    }

    for (auto const& llmReq : genRequests)
    {
        auto const reqBeamWidth = llmReq->mSamplingConfig.beamWidth;
        numRequests += reqBeamWidth; // tile by beam width
        maxInputLengthInBatch = std::max(maxInputLengthInBatch, llmReq->getEncoderLen());
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void EncoderBuffers::reshape()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    inputLengths->reshape(ITensor::makeShape({numRequests}));
    maxInputLength->reshape(ITensor::makeShape({maxInputLengthInBatch}));
    encoderOutput->reshape(ITensor::makeShape({numTokens, hiddenSize}));

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void EncoderBuffers::fill(
    RequestVector const& ctxRequests, RequestVector const& genRequests, runtime::BufferManager const& manager)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(encoderBufferCopies);

    std::vector<SizeType32> inputLengthsAll;
    std::vector<SizeType32> maxInputLengthAll(maxInputLength->getShape().d[0], 0);

    SizeType32 offset = 0, size = 0;
    for (auto const& requests : {ctxRequests, genRequests})
    {
        for (auto const& llmReq : requests)
        {
            // 1. only ctx requests should gather the encoder output
            // 2. only gen requests should tile encoder input lengths info by beam width
            bool isCtx = llmReq->isContextInitState();
            if (isCtx)
            {
                size = llmReq->getEncoderLen();
                auto const encoderOutputSlice = runtime::ITensor::slice(encoderOutput, offset, size);
                manager.copy(*llmReq->getEncoderOutput(), *encoderOutputSlice);
                offset += size;

                inputLengthsAll.emplace_back(llmReq->getEncoderLen());
            }
            else
            {
                auto const reqBeamWidth = llmReq->mSamplingConfig.beamWidth;
                std::fill_n(std::back_inserter(inputLengthsAll), reqBeamWidth,
                    llmReq->getEncoderLen()); // although encoder output is not needed, gen phase still needs the
                                              // encoder length info for cross kv cache. Also tile by beam width
            }
        }
    }
    manager.copy(inputLengthsAll.data(), *inputLengths);
    manager.copy(maxInputLengthAll.data(), *maxInputLength);
    // crossKvCacheGen unused in engine for now, use default tensor

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void EncoderBuffers::insertInputTensors(TensorMap& inputMap)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    inputMap.insert_or_assign("encoder_output", encoderOutput);
    inputMap.insert_or_assign("encoder_input_lengths", inputLengths);
    inputMap.insert_or_assign("encoder_max_input_length", maxInputLength);
    inputMap.insert_or_assign("cross_kv_cache_gen", crossKvCacheGen);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}
} // namespace tensorrt_llm::batch_manager
