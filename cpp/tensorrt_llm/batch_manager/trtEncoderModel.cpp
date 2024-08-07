/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "trtEncoderModel.h"
#include "encoderBuffers.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/tllmLogger.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"
#include "tensorrt_llm/runtime/utils/sessionUtils.h"

#include <algorithm>
#include <cstddef>
#include <vector>

using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm::mpi;

namespace tensorrt_llm::batch_manager
{

TrtEncoderModel::TrtEncoderModel(runtime::ModelConfig const& modelConfig, WorldConfig const& worldConfig,
    runtime::RawEngine const& rawEngine, std::shared_ptr<nvinfer1::ILogger> logger,
    TrtGptModelOptionalParams const& optionalParams)
    : TrtGptModel(modelConfig, worldConfig, optionalParams)
    , mModelConfig{modelConfig}
    , mWorldConfig{worldConfig}
    , mDevice{runtime::utils::initDevice(worldConfig)}
    , mLogger{logger ? std::move(logger) : std::make_shared<TllmLogger>()}
    , mRuntime{std::make_shared<TllmRuntime>(rawEngine, mLogger.get(), optionalParams.gpuWeightsPercent)}
    , mMicroBatchId(0)
    , mCopyBufferManager{std::make_shared<CudaStream>()}
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    if (mWorldConfig.isPipelineParallel())
    {
        TLLM_THROW("Pipeline parallelism is currently not supported for encoder models.");
        mNumMicroBatches = mWorldConfig.getPipelineParallelism();
    }
    else
    {
        mNumMicroBatches = isTtrOverlap() ? 2 : 1;
    }

    mNumBuffers = mNumMicroBatches;

    createRuntimeContexts();

    createBuffers();

    if (mWorldConfig.isPipelineParallel())
    {
        auto const& commSession = COMM_SESSION;
        mMpiCommPipelinePara = std::make_shared<tensorrt_llm::mpi::MpiComm>(
            commSession.split(mWorldConfig.getTensorParallelRank(), mWorldConfig.getPipelineParallelRank()));
    }

    mMicroBatchScheduledRequests.resize(mNumMicroBatches);
    // mEncoderWaitEvents.resize(mNumMicroBatches);

    // set noScheduleUntilState to REQUEST_STATE_ENCODER_INIT for encoder model
    auto PeftCacheManager = std::make_shared<NoOpPeftCacheManager>();
    // when null kv cache manager is given, request scheduler will use MaxRequests as capacity scheduler, i.e. no
    // handling of maximizing utlization or pause/evict
    // TODO: finer control on encoder requests scheduling
    mRequestScheduler = std::make_shared<batch_scheduler::RequestScheduler>(getMaxBatchSize(), mNumMicroBatches,
        nullptr, nullptr, PeftCacheManager, optionalParams.schedulerConfig, mModelConfig.getMaxNumTokens(),
        std::nullopt, mModelConfig.getMaxInputLen(), REQUEST_STATE_ENCODER_INIT, REQUEST_STATE_CONTEXT_INIT);

    mHiddenSize = modelConfig.getHiddenSize();

    mMaxInputLen = mModelConfig.getMaxInputLen();
    TLLM_LOG_INFO("TRTEncoderModel mMaxInputLen: reset to %d from build config.", mMaxInputLen);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

BufferManager const& TrtEncoderModel::getBufferManager() const
{
    return mRuntime->getBufferManager();
}

BufferManager::CudaStreamPtr TrtEncoderModel::getRuntimeStreamPtr() const
{
    return mRuntime->getStreamPtr();
}

void TrtEncoderModel::setLayerProfiler()
{
    TLLM_CHECK(mRuntime);
    mRuntime->setLayerProfiler();
}

std::string TrtEncoderModel::getLayerProfileInfo() const
{
    TLLM_CHECK(mRuntime);
    return mRuntime->getLayerProfileInfo();
}

void TrtEncoderModel::createRuntimeContexts()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    mRuntime->clearContexts();
    auto const numProfiles = mRuntime->getNbProfiles();
    TLLM_CHECK_WITH_INFO(numProfiles == 1, "Encoder only expects one optimization profile");
    for (auto i = 0; i < numProfiles; ++i)
    {
        mRuntime->addContext(i);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtEncoderModel::executeContext(SizeType32 runtimeContextId)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(executeContext);
    auto enqueueSuccessful = mRuntime->executeContext(runtimeContextId);
    if (!enqueueSuccessful)
    {
        throw std::runtime_error("Executing TRT engine failed!");
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtEncoderModel::createBuffers()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    for (SizeType32 i = 0; i < mNumBuffers; ++i)
    {
        mBuffers.emplace_back(
            std::make_shared<EncoderBuffers>(getMaxBatchSize(), mModelConfig, mWorldConfig, *mRuntime));
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtEncoderModel::executeBatch(ScheduledRequests const& scheduledRequests)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(executeBatch);

    // encoder model only have one optimization profile for now, so no optimization profile switch
    SizeType32 optProfileIndex = 0;
    auto const bufferId = getBufferId();
    if (!scheduledRequests.contextRequests.empty())
    {
        // engine I/O
        auto [inputMap, outputMap]
            = mBuffers[bufferId]->prepareIO(scheduledRequests.contextRequests, mModelConfig, mWorldConfig, *mRuntime);
        mRuntime->setInputTensors(optProfileIndex, inputMap);
        mRuntime->setOutputTensors(optProfileIndex, outputMap);

        // engine run
        executeContext(optProfileIndex);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtEncoderModel::rearrangeOutputs(ScheduledRequests const& scheduledRequests)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(rearrangeOutputs);

    auto const bufferId = getBufferId();
    if (!scheduledRequests.contextRequests.empty())
    {
        mBuffers[bufferId]->rearrangeOutputs(scheduledRequests.contextRequests, mModelConfig, mWorldConfig, *mRuntime);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtEncoderModel::forwardSync()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(forwardSync);

    auto const device = mWorldConfig.getDevice();
    TLLM_CUDA_CHECK(cudaSetDevice(device));

    auto& currRequests = mMicroBatchScheduledRequests.at(mMicroBatchId);
    // auto& encoderWaitEvent = mEncoderWaitEvents.at(mMicroBatchId);

    if (!currRequests.empty())
    {
        if (!mWorldConfig.isPipelineParallel() || !mWorldConfig.isLastPipelineParallelRank())
        {
            // TLLM_CHECK_WITH_INFO(mEncStepAsyncSndHdl.get() == nullptr, "encoderSync handle must be nullptr.");
            // // Wait for encoding for requests in flight for the current micro batch
            // mEncStepAsyncSndHdl = encoderSync(currRequests, encoderWaitEvent);
        }
        else
        {
        }

        NVTX3_SCOPED_RANGE(pauseFlaggedCurrRequests);
        for (auto const& requests : {currRequests.contextRequests})
        {
            for (auto const& llmReq : requests)
            {
                auto const reqId = llmReq->mRequestId;
                mInflightReqIds.erase(reqId);
                TLLM_LOG_DEBUG("request ID %u removed from ENCODER inflight set", reqId);

                // If a request in encoder phase had been flagged to be paused, pause it right away
                if (mReqIdsToPause.find(reqId) != mReqIdsToPause.end())
                {
                    terminateRequest(llmReq, true);
                    mReqIdsToPause.erase(reqId);
                }
            }
        }
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtEncoderModel::forwardAsync(RequestList const& activeRequests)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const device = mWorldConfig.getDevice();
    TLLM_CUDA_CHECK(cudaSetDevice(device));

    try
    {
        auto& currRequests = mMicroBatchScheduledRequests.at(mMicroBatchId);
        // auto& encoderWaitEvent = mEncoderWaitEvents.at(mMicroBatchId);

        // Get a new set of requests for encoder
        // The scheduler will not include any requests that are already in flight for encoder models
        // TODO: add pause handling logic
        TLLM_LOG_DEBUG("Running ENCODER request scheduler");
        RequestVector requestsToPause;
        std::tie(currRequests.contextRequests, std::ignore, requestsToPause)
            = mRequestScheduler->scheduleRequests(activeRequests, mInflightReqIds);

        {
            NVTX3_SCOPED_RANGE(pauseRequestsFlaggedByScheduler);
            // Loop over requests flagged to be paused, and if not in flight pause it right away
            for (auto const& llmReq : requestsToPause)
            {
                auto const reqId = llmReq->mRequestId;
                if (mInflightReqIds.find(reqId) == mInflightReqIds.end())
                {
                    // Not in flight, can terminate right away
                    terminateRequest(llmReq, true);
                }
                else
                {
                    // In flight, add to set for pausing later
                    mReqIdsToPause.insert(reqId);
                }
            }
        }

        TLLM_CHECK(currRequests.size() <= static_cast<size_t>(getMaxBatchSize()));

        if (!currRequests.empty())
        {
            TLLM_LOG_DEBUG("Running ENCODER model with batch size: %u", currRequests.size());
            {
                NVTX3_SCOPED_RANGE(updateInflightReqIds);
                // Add to set of requests in flight
                for (auto const& requests : {currRequests.contextRequests})
                {
                    for (auto const& llmReq : requests)
                    {
                        TLLM_LOG_DEBUG("request ID %u added to ENCODER inflight set", llmReq->mRequestId);
                        mInflightReqIds.insert(llmReq->mRequestId);
                    }
                }
            }

            executeBatch(currRequests);

            sync_check_cuda_error();

            rearrangeOutputs(currRequests);

            sync_check_cuda_error();

            // encoderWaitEvent = encoderStepAsync(currRequests);

            for (auto const& requests : {currRequests.contextRequests})
            {
                for (auto const& llmReq : requests)
                {
                    if (llmReq->isEncoderInitState())
                    {
                        llmReq->mState = REQUEST_STATE_CONTEXT_INIT;
                        TLLM_LOG_DEBUG("request ID: %u finishes encoder phase", llmReq->mRequestId);
                    }
                }
            }
        }

        // TODO: PP handling
        if (!currRequests.empty())
        {
            if (mWorldConfig.isPipelineParallel() && mWorldConfig.isLastPipelineParallelRank())
            {
                // TLLM_CHECK_WITH_INFO(mEncStepAsyncSndHdl.get() == nullptr, "decoderSync handle must be nullptr.");
                // Wait for encoding for requests in flight for the current micro batch
                // mEncStepAsyncSndHdl = encoderSync(currRequests, encoderWaitEvent);
            }
        }

        // Update the micro batch ID
        mMicroBatchId = (mMicroBatchId + 1) % mNumMicroBatches;
    }
    // In case of error, we need to free the batch slot associated with those requests
    catch (std::exception const& e)
    {
        for (auto const& llmReq : activeRequests)
        {
            terminateRequest(llmReq);
        }
        throw;
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtEncoderModel::terminateRequest(std::shared_ptr<LlmRequest> const& llmReq, bool pause)
{
    // For encoder-only models, just change req state here. might need to do more when using an asynced forward
    // For enc-dec models, only remove cross kv cache after decoder
    // genenration has finished
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    if (llmReq->mState == REQUEST_STATE_ENCODER_INIT)
    {
        llmReq->mState = REQUEST_STATE_CONTEXT_INIT;
    }
    else
    {
        TLLM_LOG_DEBUG("Non-encoder request terminated in encoder model: id %lu", llmReq->mRequestId);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtEncoderModel::fillEncoderOutputSync(RequestVector const& requestList, TensorMap outputTensors)
{
    auto const totalTokensNb = outputTensors["encoder_output"]->getShape().d[0];
    auto const encoderOutputDtype = mRuntime->getEngine().getTensorDataType("encoder_output");
    SizeType32 const bytesPerValue = (encoderOutputDtype == nvinfer1::DataType::kFLOAT) ? 4 : 2;
    std::vector<std::byte> encoderOutputHost(
        totalTokensNb * mHiddenSize * bytesPerValue * mWorldConfig.getTensorParallelism());
    TLLM_CHECK_WITH_INFO(encoderOutputHost.size() > 0, "Encoder output size is 0!");
    getBufferManager().copy(*(outputTensors["encoder_output"]), reinterpret_cast<void*>(encoderOutputHost.data()));
    getBufferManager().getStream().synchronize(); // TODO: change engine call to async to improve perf. Also
                                                  // need to store output buffers, cuda events, etc.

    auto encoderOutputHostPtr = encoderOutputHost.data();
    for (auto const& llmReq : requestList)
    {
        SizeType32 const seqLen = llmReq->getEncoderLen();
        TensorPtr currentEncoderOutput
            = mCopyBufferManager.copyFrom(reinterpret_cast<half const*>(encoderOutputHostPtr),
                ITensor::makeShape({seqLen, mHiddenSize * mWorldConfig.getTensorParallelism()}), MemoryType::kCPU);
        llmReq->setEncoderOutputHost(currentEncoderOutput);
        encoderOutputHostPtr += seqLen * mHiddenSize * bytesPerValue * mWorldConfig.getTensorParallelism();

        if (llmReq->mState == REQUEST_STATE_ENCODER_INIT)
        {
            llmReq->mState = REQUEST_STATE_CONTEXT_INIT;
        }
        else
        {
            TLLM_LOG_DEBUG("Non-encoder request terminated in encoder model: id %lu", llmReq->mRequestId);
        }
    }
}

void TrtEncoderModel::executeBatch(RequestVector const& requestList)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(executeBatch);

    std::vector<TokenIdType> inputIdsHost;
    std::vector<SizeType32> positionIdsHost;
    std::vector<SizeType32> inputLengthsHost;
    inputLengthsHost.reserve(requestList.size());
    SizeType32 maxInputLengthHost = 0; // scalar tensor input to engine

    for (auto const& llmReq : requestList)
    {
        auto const& reqTokens = *(llmReq->getEncoderTokens().value());
        inputIdsHost.insert(inputIdsHost.end(), reqTokens.begin(), reqTokens.end());
        positionIdsHost.reserve(positionIdsHost.size() + reqTokens.size());
        auto const newReqPosBegin = positionIdsHost.end();
        positionIdsHost.resize(positionIdsHost.size() + reqTokens.size());
        std::iota(newReqPosBegin, positionIdsHost.end(), 0);
        inputLengthsHost.push_back(reqTokens.size());
        maxInputLengthHost = std::max(maxInputLengthHost, static_cast<SizeType32>(reqTokens.size()));
    }

    // Engine inputs
    TensorPtr inputIds;
    TensorPtr positionIds;
    TensorPtr hiddenStatesInput;
    TensorPtr inputLengths = getBufferManager().copyFrom(
        inputLengthsHost, ITensor::makeShape({static_cast<SizeType32>(inputLengthsHost.size())}), MemoryType::kGPU);
    // use shape of maxInputLength to indicates max length, content is not important
    TensorPtr maxInputLength = getBufferManager().gpu(
        ITensor::makeShape({maxInputLengthHost}), nvinfer1::DataType::kINT32); // TODO: use view instead?

    SizeType32 totalNbTokens = inputIdsHost.size();
    // engine outputs
    TensorPtr rankOutput
        = getBufferManager().gpu(ITensor::makeShape({totalNbTokens, mHiddenSize * mWorldConfig.getTensorParallelism()}),
            mModelConfig.getDataType()); // TODO: use view instead?

    TensorMap inputTensors{
        std::make_pair("max_input_length", maxInputLength), std::make_pair("input_lengths", inputLengths)};

    if (mWorldConfig.isFirstPipelineParallelRank())
    {
        inputIds = getBufferManager().copyFrom(inputIdsHost, ITensor::makeShape({totalNbTokens}), MemoryType::kGPU);
        positionIds
            = getBufferManager().copyFrom(positionIdsHost, ITensor::makeShape({totalNbTokens}), MemoryType::kGPU);
        inputTensors.emplace("input_ids", inputIds);
        inputTensors.emplace("position_ids", positionIds);
    }
    else
    {
        hiddenStatesInput = getBufferManager().gpu(
            ITensor::makeShape({totalNbTokens, mHiddenSize * mWorldConfig.getTensorParallelism()}),
            mModelConfig.getDataType()); // TODO: use view instead?

        inputTensors.emplace("hidden_states_input", hiddenStatesInput);
    }

    auto const outputName = mWorldConfig.isLastPipelineParallelRank() ? "encoder_output" : "hidden_states_output";
    TensorMap outputTensors{std::make_pair(outputName, rankOutput)};

    // Set input / output tensors to context, encoder model only have one context
    mRuntime->setInputTensors(0, inputTensors);
    mRuntime->setOutputTensors(0, outputTensors);

    executeContext(0);

    // copy encoder output to llmRequest, if last PP rank
    // dispatch result to each llmReq, only needed by the last PP rank
    // TODO: more dtypes support
    if (mWorldConfig.isLastPipelineParallelRank())
    {
        fillEncoderOutputSync(requestList, outputTensors);
    }
    else
    {
        getBufferManager().getStream().synchronize();
    }

    // Update the micro batch ID for next microbatches
    mMicroBatchId = (mMicroBatchId + 1) % mWorldConfig.getPipelineParallelism();
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtEncoderModel::forward(RequestVector& activeRequests)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const device = mWorldConfig.getDevice();
    TLLM_CUDA_CHECK(cudaSetDevice(device));

    try
    {
        if (activeRequests.empty())
        {
            return;
        }

        executeBatch(activeRequests);
    }
    catch (std::exception const& e)
    {
        for (auto& req : activeRequests)
        {
            terminateRequest(req);
        }
        throw;
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtEncoderModel::setLogitsPostProcessorBatched(
    std::optional<LogitsPostProcessorBatched> logitsPostProcessorBatched)
{
    TLLM_CHECK_WITH_INFO(!logitsPostProcessorBatched.has_value(), "TrtEncoderModel does not use logits processor.");
}

} //  namespace tensorrt_llm::batch_manager
