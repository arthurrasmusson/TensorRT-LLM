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

#include "tensorrt_llm/common/logger.h"
#ifndef TOP_LEVEL_DIR
#error "Define TOP_LEVEL_DIR"
#endif

#include "tensorrt_llm/batch_manager/BatchManager.h"
#include "tensorrt_llm/batch_manager/GptManager.h"
#include "tensorrt_llm/batch_manager/inferenceRequest.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/namedTensor.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaProfilerUtils.h"
#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/common/stlUtils.h"
#include "tensorrt_llm/common/stringUtils.h"
#include "tensorrt_llm/common/timestampUtils.h"
#include "tensorrt_llm/executor/requestUtils.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/loraUtils.h"
#include "tensorrt_llm/runtime/memoryCounters.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/worldConfig.h"
#include "trtGptModel.h"
#include "trtGptModelFactory.h"

#include <nlohmann/json.hpp>

#include <cstring>
#include <cuda_profiler_api.h>
#include <filesystem>
#include <functional>
#include <optional>
#include <stdexcept>
#include <utility>

using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm::batch_manager::inference_request;

namespace tc = tensorrt_llm::common;

namespace
{

using namespace tensorrt_llm::batch_manager;

template <typename T>
void isCallbackEmptyThrowWithErrorMessage(T const& cb, std::string const& msg)
{
    if (!cb)
    {
        throw std::invalid_argument(msg);
    }
}

template <typename>
struct is_std_vector : std::false_type
{
};

template <typename T, typename A>
struct is_std_vector<std::vector<T, A>> : std::true_type
{
};

template <typename T>
constexpr bool is_std_vector_v = is_std_vector<T>::value;

template <typename T>
std::optional<T> getScalarValueFromTensor(NamedTensor const& nt, std::vector<int64_t> const& expectedShape)
{
    auto const& t = nt.tensor;
    if (t == nullptr || t->getSize() == 0)
    {
        return std::nullopt;
    }

    TLLM_CHECK_WITH_INFO(t->shapeEquals(expectedShape.data(), expectedShape.size()),
        "Invalid shape for %s. expected: %s, supplied: %s", nt.name.c_str(), tc::vec2str(expectedShape).c_str(),
        ITensor::toString(t->getShape()).c_str());
    TLLM_CHECK(t->getSize() > 0);
    return bufferCast<T>(*t)[0];
}

template <typename OptionalType, typename TensorType = OptionalType>
void setOptionalValueFromScalarTensor(
    std::optional<OptionalType>& optional, NamedTensor const& namedTensor, std::vector<int64_t> const& expectedShape)
{
    auto value = getScalarValueFromTensor<TensorType>(namedTensor, expectedShape);
    if (value.has_value())
    {
        if constexpr (std::is_scalar_v<OptionalType>)
        {
            optional = value;
        }
        else if constexpr (is_std_vector_v<OptionalType>)
        {
            optional = OptionalType{static_cast<typename OptionalType::value_type>(value.value())};
        }
        else
        {
            static_assert(
                !sizeof(OptionalType), "OptionalType is not supported by setOptionalValueFromScalarTensor().");
        }
    }
}

template <typename ValueType, typename TensorType = ValueType>
void setValueFromScalarTensor(
    ValueType& value, NamedTensor const& namedTensor, std::vector<int64_t> const& expectedShape, bool isOptional)
{
    auto returned_value = getScalarValueFromTensor<TensorType>(namedTensor, expectedShape);
    TLLM_CHECK_WITH_INFO(
        isOptional || returned_value.has_value(), "Failed to get value from tensor: %s", namedTensor.name.c_str());
    if (returned_value.has_value())
    {
        value = returned_value.value();
    }
}

std::optional<GptManager::TensorPtr> getOptionalTensor(GptManager::TensorPtr const& tensorPtr)
{
    if (!tensorPtr || tensorPtr->getSize() == 0)
        return std::nullopt;

    return tensorPtr;
}

} // namespace

namespace tensorrt_llm::batch_manager
{

std::ostream& operator<<(std::ostream& os, TrtGptModelType modelType)
{
    switch (modelType)
    {
    case TrtGptModelType::InflightBatching: os << "Inflight Batching"; break;
    case TrtGptModelType::InflightFusedBatching: os << "Inflight Fused Batching"; break;
    case TrtGptModelType::V1: os << "V1"; break;
    default: TLLM_THROW("Unsupported model type");
    }
    return os;
}

/* Public member functions */
GptManager::GptManager(std::filesystem::path const& trtEnginePath, TrtGptModelType modelType,
    GetInferenceRequestsCallback getInferenceRequestsCb, SendResponseCallback sendResponseCb,
    PollStopSignalCallback pollStopSignalCb, ReturnBatchManagerStatsCallback returnBatchManagerStatsCb,
    TrtGptModelOptionalParams const& optionalParams, std::optional<uint64_t> terminateReqId, bool excludeInputInOutput)
    : mGetInferenceRequestsCb(std::move(getInferenceRequestsCb))
    , mSendResponseCb(std::move(sendResponseCb))
    , mPollStopSignalCb(std::move(pollStopSignalCb))
    , mReturnBatchManagerStatsCb(std::move(returnBatchManagerStatsCb))
    , mTrtGptModel{TrtGptModelFactory::create(trtEnginePath, modelType, optionalParams)}
    , mTerminateReqId(terminateReqId)
    , mIterationCounter(0)
    , mExcludeInputInOutput(excludeInputInOutput)
    , shutdown_requested_{false}
{
    TLLM_LOG_WARNING(
        "GptManager is deprecated and will be removed in a future release."
        " Please use the executor API instead (cpp/include/tensorrt_llm/executor).");

    isCallbackEmptyThrowWithErrorMessage(mGetInferenceRequestsCb, "GptManager ctor: getInferenceRequestsCb is empty.");
    isCallbackEmptyThrowWithErrorMessage(mSendResponseCb, "GptManager ctor: sendResponseCb is empty.");

    worker_thread_ = std::make_shared<std::thread>(&GptManager::decoupled_execution_loop, this);

    auto const displayRuntimeOptions = std::getenv("TLLM_GPTM_DISPLAY_RUNTIME_OPTIONS") != nullptr;
    if (displayRuntimeOptions)
    {
        using tensorrt_llm::common::stl_utils::toString;

        TLLM_LOG_WARNING("GptManager runtime options:");
        TLLM_LOG_WARNING("TrtGptModelType: %s", toString(modelType).c_str());
        TLLM_LOG_WARNING("Terminate Request ID: %s",
            terminateReqId.has_value() ? std::to_string(terminateReqId.value()).c_str() : "nullopt");
    }
}

BatchManagerErrorCode_t GptManager::shutdown()
{
    shutdown_requested_ = true;
    return waitUntilTerminate();
}

SizeType32 GptManager::getNumActiveRequests()
{
    return mActiveRequests.size();
}

void GptManager::setLayerProfiler()
{
    mTrtGptModel->setLayerProfiler();
}

std::string GptManager::getLayerProfileInfo() const
{
    return mTrtGptModel->getLayerProfileInfo();
}

GptManager::~GptManager()
{
    shutdown();
}

SizeType32 GptManager::getMaxInputLen() const
{
    return mTrtGptModel->getMaxInputLen();
}

SizeType32 GptManager::getMaxSequenceLen() const
{
    return mTrtGptModel->getMaxSequenceLen();
}

SizeType32 GptManager::getMaxNumSequences() const
{
    return mTrtGptModel->getMaxNumSequences();
}

SizeType32 GptManager::getMaxDraftLen() const
{
    return mTrtGptModel->getMaxDraftLen();
}

BatchManagerErrorCode_t GptManager::waitUntilTerminate()
{
    if (worker_thread_->joinable())
    {
        worker_thread_->join();
    }
    return BatchManagerErrorCode_t::STATUS_SUCCESS;
}

std::shared_ptr<std::vector<TokenIdType>> GptManager::getReqInputTokens(std::shared_ptr<InferenceRequest> newReq)
{
    auto tokens = std::make_shared<std::vector<TokenIdType>>();

    auto const& inputIds = newReq->getInputIdsNamed();
    auto const& t = inputIds.tensor;
    TLLM_CHECK_WITH_INFO(t != nullptr, "Input ids are not defined. Set tensor %s", inputIds.name.c_str());
    if (t->getShape().nbDims == 2)
    {
        TLLM_CHECK_WITH_INFO(
            t->getShape().d[0] == 1, "Expected batch dimension to be 1 for each request for %s", inputIds.name.c_str());
    }
    else
    {
        TLLM_CHECK_WITH_INFO(t->getShape().nbDims == 1, "Invalid shape for %s", inputIds.name.c_str());
    }

    auto tokenRange = BufferRange<TokenIdType>(*t);
    return std::make_shared<std::vector<TokenIdType>>(tokenRange.begin(), tokenRange.end());
}

void GptManager::validateLlmRequest(
    LlmRequest& req, runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig) const
{
    auto const draftLen = req.getNumDraftTokens();
    if (req.mPromptLen + draftLen > getMaxInputLen())
    {
        if (draftLen == 0)
        {
            TLLM_THROW("Prompt length (%d) exceeds maximum input length (%d).", req.mPromptLen, getMaxInputLen());
        }
        else
        {
            TLLM_THROW("Prompt length + number of draft tokens (%d + %d) exceeds maximum input length (%d).",
                req.mPromptLen, draftLen, getMaxInputLen());
        }
    }

    if (req.mPromptLen + req.mMaxNewTokens > getMaxSequenceLen())
    {
        auto const maxNewTokens = getMaxSequenceLen() - req.mPromptLen;
        TLLM_LOG_WARNING(
            "Prompt length + number of requested output tokens (%d + %d) exceeds maximum sequence length (%d). "
            "Number of requested output tokens is changed to (%d).",
            req.mPromptLen, req.mMaxNewTokens, getMaxSequenceLen(), maxNewTokens);
        req.mMaxNewTokens = maxNewTokens;
    }

    if (req.mSamplingConfig.beamWidth == 0)
    {
        std::string err
            = "Requested value: 0 for beamWidth is invalid. To de-activate beam searching "
              "set beamWidth to 1 instead.";
        throw std::runtime_error(err);
    }

    if (req.getLoraWeights().has_value() || req.getLoraConfig().has_value() || req.getLoraTaskId().has_value())
    {
        lora::loraValidateRequestTensors(
            req.getLoraTaskId(), req.getLoraWeights(), req.getLoraConfig(), modelConfig, worldConfig);
    }
}

std::shared_ptr<LlmRequest> GptManager::fillLlmRequest(std::shared_ptr<InferenceRequest> newReq)
{
    auto tokens = getReqInputTokens(newReq);

    std::optional<std::shared_ptr<std::vector<TokenIdType>>> draftTokens;
    std::optional<TensorPtr> draftLogits;
    {
        auto optDraftTokenTensor = getOptionalTensor(newReq->getDraftInputIdsUnchecked());
        if (optDraftTokenTensor)
        {
            auto t = optDraftTokenTensor.value();
            TLLM_CHECK_WITH_INFO(t->getShape().nbDims == 2, "Invalid shape for draft tokens tensor");
            TLLM_CHECK_WITH_INFO(t->getShape().d[0] == 1, "Expected batch dimension to be 1 for each draft tokens");
            TLLM_CHECK_WITH_INFO(t->getMemoryType() != MemoryType::kGPU, "Expected draft tokens to be in CPU memory.");
            auto tokenRange = BufferRange<TokenIdType>(*t);
            draftTokens = std::make_shared<std::vector<TokenIdType>>(tokenRange.begin(), tokenRange.end());
        }
        draftLogits = getOptionalTensor(newReq->getDraftLogitsUnchecked());
    }

    std::optional<SizeType32> endId(std::nullopt);
    std::optional<SizeType32> padId(std::nullopt);
    std::optional<SizeType32> beamWidth(std::nullopt);
    std::optional<SizeType32> promptVocabSize(std::nullopt);

    SamplingConfig samplingConfig;

    SizeType32 maxNewTokens;
    setValueFromScalarTensor<SizeType32>(maxNewTokens, newReq->getMaxNewTokensNamed(), {1, 1}, false);

    setOptionalValueFromScalarTensor<SizeType32>(beamWidth, newReq->getBeamWidthNamed(), {1});
    samplingConfig.beamWidth = beamWidth.value_or(1);
    setOptionalValueFromScalarTensor<SizeType32>(endId, newReq->getEndIdNamed(), {1});
    setOptionalValueFromScalarTensor<SizeType32>(padId, newReq->getPadIdNamed(), {1});

    setOptionalValueFromScalarTensor<std::vector<float>, float>(
        samplingConfig.temperature, newReq->getTemperatureNamed(), {1});

    setOptionalValueFromScalarTensor<std::vector<SizeType32>, int32_t>(
        samplingConfig.topK, newReq->getRuntimeTopKNamed(), {1});
    setOptionalValueFromScalarTensor<std::vector<float>, float>(
        samplingConfig.topP, newReq->getRuntimeTopPNamed(), {1});
    setOptionalValueFromScalarTensor<std::vector<float>, float>(
        samplingConfig.lengthPenalty, newReq->getLengthPenaltyNamed(), {1});
    setOptionalValueFromScalarTensor<std::vector<SizeType32>, int32_t>(
        samplingConfig.earlyStopping, newReq->getEarlyStoppingNamed(), {1});
    setOptionalValueFromScalarTensor<std::vector<float>, float>(
        samplingConfig.repetitionPenalty, newReq->getRepetitionPenaltyNamed(), {1});
    setOptionalValueFromScalarTensor<std::vector<SizeType32>, int32_t>(
        samplingConfig.minLength, newReq->getMinLengthNamed(), {1});
    setOptionalValueFromScalarTensor<std::vector<float>, float>(
        samplingConfig.presencePenalty, newReq->getPresencePenaltyNamed(), {1});
    setOptionalValueFromScalarTensor<std::vector<float>, float>(
        samplingConfig.frequencyPenalty, newReq->getFrequencyPenaltyNamed(), {1});
    setOptionalValueFromScalarTensor<std::vector<uint64_t>, uint64_t>(
        samplingConfig.randomSeed, newReq->getRandomSeedNamed(), {1});
    setOptionalValueFromScalarTensor<std::vector<SizeType32>, int32_t>(
        samplingConfig.noRepeatNgramSize, newReq->getNoRepeatNgramSizeNamed(), {1});

    std::optional<bool> returnLogProbs = false;
    setOptionalValueFromScalarTensor<bool, bool>(returnLogProbs, newReq->getReturnLogProbsNamed(), {1});

    std::optional<bool> returnContextLogits = false;
    setOptionalValueFromScalarTensor<bool, bool>(returnContextLogits, newReq->getReturnContextLogitsNamed(), {1});

    std::optional<bool> returnGenerationLogits = false;
    setOptionalValueFromScalarTensor<bool, bool>(returnGenerationLogits, newReq->getReturnGenerationLogitsNamed(), {1});

    auto promptEmbeddingTable = getOptionalTensor(newReq->getPromptEmbeddingTableUnchecked());
    setOptionalValueFromScalarTensor<SizeType32>(promptVocabSize, newReq->getPromptVocabSizeNamed(), {1});

    // LoRA tensors
    std::optional<std::uint64_t> loraTaskId(std::nullopt);
    setOptionalValueFromScalarTensor<std::uint64_t>(loraTaskId, newReq->getLoraTaskIdNamed(), {1});
    auto optLoraWeights = getOptionalTensor(newReq->getLoraWeightsUnchecked());
    auto optLoraConfig = getOptionalTensor(newReq->getLoraConfigUnchecked());

    auto embeddingBias = getOptionalTensor(newReq->getEmbeddingBiasUnchecked());
    auto badWordsList = getOptionalTensor(newReq->getBadWordsListUnchecked());
    auto stopWordsList = getOptionalTensor(newReq->getStopWordsListUnchecked());
    auto lookaheadConfig = newReq->getLookaheadConfig();

    TLLM_CHECK_WITH_INFO(samplingConfig.beamWidth == 1 || !newReq->isStreaming(),
        "Streaming mode is only supported with beam width of 1.");

    auto r = std::make_shared<LlmRequest>(newReq->getRequestId(), maxNewTokens, tokens, samplingConfig,
        newReq->isStreaming(), endId, padId, embeddingBias, badWordsList, stopWordsList, promptEmbeddingTable,
        promptVocabSize, loraTaskId, optLoraWeights, optLoraConfig, lookaheadConfig, returnLogProbs.value(),
        returnContextLogits.value(), returnGenerationLogits.value(), draftTokens, draftLogits,
        false /* FIXME: exclude input in output */, newReq->getLogitsPostProcessor());

    return r;
}

BatchManagerErrorCode_t GptManager::fetchNewRequests()
{
    NVTX3_SCOPED_RANGE(fetchNewRequests);
    // TODO: Update request states as discussed (Chris/Patrice) such that requests are not terminated early
    // due to reaching specified number of tokens per draft request (this will increment until the original
    // request is complete.) We don't want the request terminated and KV cached evicted after say 5 drafted tokens.
    if (!shutdown_requested_)
    {
        // Ask for getMaxNumSequences() - mActiveRequests.size()
        SizeType32 maxNumNewRequests
            = std::max(getMaxNumSequences() - static_cast<SizeType32>(mActiveRequests.size()), 0);
        auto newRequests = mGetInferenceRequestsCb(maxNumNewRequests);

        // Store the new requests to be added to the request list
        for (auto const& newReq : newRequests)
        {
            try
            {
                if (newReq->getRequestId() == mTerminateReqId)
                {
                    return BatchManagerErrorCode_t::STATUS_TERMINATE;
                }
                auto r = fillLlmRequest(newReq);

                r->validate(getMaxInputLen(), getMaxSequenceLen(), getMaxDraftLen());

                auto const vocabSizePadded
                    = mTrtGptModel->getModelConfig().getVocabSizePadded(mTrtGptModel->getWorldConfig().getSize());
                auto logitsDtype = mTrtGptModel->getModelConfig().getLogitsDtype();

                // Allocate host memory for context logits
                if (r->getReturnContextLogits())
                {
                    if (!mTrtGptModel->getModelConfig().computeContextLogits())
                    {
                        TLLM_THROW("Return context logit need to build engine with gather_context_logits");
                    }
                    r->allocContextLogitsHost(vocabSizePadded, logitsDtype);
                }

                // Allocate host memory for generation logits
                if (r->getReturnGenerationLogits())
                {
                    if (!mTrtGptModel->getModelConfig().computeGenerationLogits())
                    {
                        TLLM_THROW("Return generation logit need to build engine with gather_generation_logits");
                    }
                    if (mTrtGptModel->getModelConfig().getSpeculativeDecodingMode().isDraftTokensExternal())
                    {
                        r->allocTargetModelAcceptedTokenLogitsHost(vocabSizePadded, logitsDtype);
                    }
                    else
                    {
                        r->allocGenerationLogitsHost(vocabSizePadded, logitsDtype);
                    }
                }

                auto requestId = r->mRequestId;
                if (mActiveRequestsIds.find(requestId) != mActiveRequestsIds.end())
                {
                    std::string err = "  Request ID " + std::to_string(requestId)
                        + " already exist in request table. Will not be added.";
                    throw std::runtime_error(err);
                }

                mTrtGptModel->updatePeftCache(r);

                tensorrt_llm::executor::insertRequestInOrder(mActiveRequests, r);
                mActiveRequestsIds.insert(requestId);
            }
            catch (std::exception const& e)
            {
                std::string err = std::string("Cannot process new request: ") + e.what();
                TLLM_LOG_ERROR(err);
                // Call the response callback so that this request gets removed from workItems
                mSendResponseCb(static_cast<uint64_t>(newReq->getRequestId()), {}, true, err);
            }
        }
    }
    return BatchManagerErrorCode_t::STATUS_SUCCESS;
}

BatchManagerErrorCode_t GptManager::pollStopSignals()
{
    // Get id of all stopped requests
    auto stoppedReqIds = mPollStopSignalCb();

    // Check if one of the active request is stopped
    for (auto& req : mActiveRequests)
    {
        auto requestId = req->mRequestId;
        if (stoppedReqIds.find(requestId) != stoppedReqIds.end())
        {
            req->mState = REQUEST_STATE_GENERATION_COMPLETE;
            mTrtGptModel->terminateRequest(req);
        }
    }
    return BatchManagerErrorCode_t::STATUS_SUCCESS;
}

BatchManagerErrorCode_t GptManager::returnBatchManagerStats()
{
    NVTX3_SCOPED_RANGE(returnBatchManagerStats);
    nlohmann::json statsJson;
    // Timestamp
    statsJson["Timestamp"] = tensorrt_llm::common::getCurrentTimestamp();
    // Iteration counter
    statsJson["Iteration Counter"] = mIterationCounter;
    // Active request count
    auto numActiveRequests = mActiveRequests.size();
    statsJson["Active Request Count"] = numActiveRequests;
    // Max number of requests
    auto maxNumRequests = getMaxNumSequences();
    statsJson["Max Request Count"] = maxNumRequests;
    // Runtime memory allocation statistics
    auto& memoryCounters = MemoryCounters::getInstance();
    statsJson["Runtime GPU Memory Usage"] = memoryCounters.getGpu();
    statsJson["Runtime CPU Memory Usage"] = memoryCounters.getCpu();
    statsJson["Runtime Pinned Memory Usage"] = memoryCounters.getPinned();
    // KVCacheManager statistics
    auto const& kvCacheManager = mTrtGptModel->getKVCacheManager();
    if (kvCacheManager)
    {
        auto kvCacheStats = kvCacheManager->getKvCacheStats();
        statsJson["Max KV cache blocks"] = kvCacheStats.maxNumBlocks;
        statsJson["Free KV cache blocks"] = kvCacheStats.freeNumBlocks;
        statsJson["Used KV cache blocks"] = kvCacheStats.usedNumBlocks;
        statsJson["Tokens per KV cache block"] = kvCacheStats.toksPerBlock;
        statsJson["Allocated total KV cache blocks"] = kvCacheStats.allocTotalBlocks;
        statsJson["Allocated new KV cache blocks"] = kvCacheStats.allocNewBlocks;
        statsJson["Reused KV cache blocks"] = kvCacheStats.reusedBlocks;
    }
    auto const modelType = mTrtGptModel->getModelType();
    if (modelType == TrtGptModelType::InflightBatching || modelType == TrtGptModelType::InflightFusedBatching)
    {
        auto ifbGptModel = std::dynamic_pointer_cast<TrtGptModelInflightBatching>(mTrtGptModel);
        auto lastIterationStatsIFB = ifbGptModel->getLastIterationStats();
        statsJson["Scheduled Requests"] = lastIterationStatsIFB.scheduledRequests.size();
        statsJson["Context Requests"] = lastIterationStatsIFB.numCtxRequests;
        statsJson["Total Context Tokens"] = lastIterationStatsIFB.numCtxTokens;
        statsJson["Generation Requests"] = lastIterationStatsIFB.numGenRequests;
        statsJson["MicroBatch ID"] = lastIterationStatsIFB.microBatchId;
        statsJson["Paused Requests"] = lastIterationStatsIFB.pausedRequests.size();
        statsJson["Average Number of Decoded Tokens Per Iteration"] = lastIterationStatsIFB.avgNumDecodedTokensPerIter;
    }
    else if (modelType == TrtGptModelType::V1)
    {
        auto v1GptModel = std::dynamic_pointer_cast<TrtGptModelV1>(mTrtGptModel);
        auto lastIterationStatsV1 = v1GptModel->getLastIterationStats();
        statsJson["Scheduled Requests"] = lastIterationStatsV1.numScheduledRequests;
        statsJson["Context Requests"] = lastIterationStatsV1.numScheduledRequests;
        statsJson["Total Context Tokens"] = lastIterationStatsV1.numCtxTokensInBatch;
        statsJson["Total Generation Tokens"] = lastIterationStatsV1.numGenTokensInBatch;
        statsJson["Empty Generation Slots"] = lastIterationStatsV1.emptyGenSlots;
    }
    else
    {
        TLLM_LOG_ERROR("Invalid modelType");
        return BatchManagerErrorCode_t::STATUS_FAILED;
    }

    std::string statsJsonStr = statsJson.dump();
    mReturnBatchManagerStatsCb(statsJsonStr);

    return BatchManagerErrorCode_t::STATUS_SUCCESS;
}

BatchManagerErrorCode_t GptManager::forwardSync()
{
    try
    {
        mTrtGptModel->forwardSync();
    }
    catch (std::exception const& e)
    {
        return BatchManagerErrorCode_t::STATUS_FAILED;
    }
    return BatchManagerErrorCode_t::STATUS_SUCCESS;
}

BatchManagerErrorCode_t GptManager::forwardAsync(
    RequestList& activeRequests, std::unordered_set<uint64_t>& activeRequestsIds)
{
    try
    {
        mTrtGptModel->forwardAsync(activeRequests);
    }
    catch (std::exception const& e)
    {
        // If we encountered a failure in forwardAsync, we need to call the responseCallback with an
        // error so that the requests are removed from the workItems list
        std::string err = std::string("Encountered an error in forwardAsync function: ") + e.what();
        TLLM_LOG_ERROR("%s", err.c_str());
        for (auto it = activeRequests.cbegin(); it != activeRequests.cend();)
        {
            auto const& llmReq = (*it);
            // Call the response callback so that requests get removed from workItems
            mSendResponseCb(static_cast<uint64_t>(llmReq->mRequestId), {}, true, err);
            // Remove from the requestList
            activeRequestsIds.erase(llmReq->mRequestId);
            it = activeRequests.erase(it);
        }
        return BatchManagerErrorCode_t::STATUS_FAILED;
    }
    return BatchManagerErrorCode_t::STATUS_SUCCESS;
}

BatchManagerErrorCode_t GptManager::returnCompletedRequests()
{
    NVTX3_SCOPED_RANGE(returnCompletedRequests);
    // Loop over the requests in the table
    // Temporarily just prints the completed requests.
    for (auto it = mActiveRequests.cbegin(); it != mActiveRequests.cend();)
    {
        auto& llmReq = *(*it);
        auto const& requestId = llmReq.mRequestId;
        // Treat V1 as non-streaming until we integrate with onTokenGenerated callback
        bool const isStreaming = llmReq.isStreaming() && mTrtGptModel->getModelType() != TrtGptModelType::V1;
        if (llmReq.isGenerationCompleteState() || (isStreaming && llmReq.isGenerationInProgressState()))
        {
            // Call response callback
            bool const isFinalResponse = llmReq.isGenerationCompleteState();
            auto const nbBeams = llmReq.mSamplingConfig.beamWidth;
            auto const maxNbTokens = llmReq.getMaxBeamNumTokens();
            // FIXME(nkorobov): For streaming we do not allow beam search and
            // streaming index calculation here applies only for sampling
            // getNumTokensPerIteration takes accepted draft tokens into account
            int nbTokensOut = isStreaming ? std::max(llmReq.getNumTokensPerIteration(), 1) : maxNbTokens;
            if (mExcludeInputInOutput && !isStreaming)
            {
                nbTokensOut -= llmReq.getOrigPromptLen();
            }

            ITensor::SharedPtr outputIdsTensor{
                BufferManager::cpu(ITensor::makeShape({nbBeams, nbTokensOut}), nvinfer1::DataType::kINT32)};
            ITensor::SharedPtr sequenceLengthTensor{
                BufferManager::cpu(ITensor::makeShape({nbBeams}), nvinfer1::DataType::kINT32)};

            // Position of the first token to be sent
            auto tokenPos = maxNbTokens - nbTokensOut;
            auto const& bufferManager = mTrtGptModel->getBufferManager();
            for (SizeType32 beam = 0; beam < nbBeams; ++beam)
            {
                auto const tokens = llmReq.getTokens(beam);
                // Number of tokens to be sent
                auto nbTokens = isStreaming ? (tokenPos + 1 - llmReq.getMaxSentTokenLen())
                                            : static_cast<runtime::SizeType32>(tokens.size());
                // Take accepted draft tokens into account when streaming
                auto const numAcceptedTokens = std::max(0, llmReq.getNumTokensPerIteration() - 1);
                nbTokens += isStreaming ? numAcceptedTokens : 0;
                if (mExcludeInputInOutput && !isStreaming)
                {
                    nbTokens -= llmReq.getOrigPromptLen();
                }
                auto outputIdsBeam = ITensor::slice(outputIdsTensor, beam, 1);
                if (nbTokens > 0)
                {
                    outputIdsBeam->resize(nbTokens);
                    bufferManager.copy(tokens.data() + tokenPos, *outputIdsBeam, MemoryType::kCPU);
                }
                else
                {
                    auto const beamLength = outputIdsBeam->getSize();
                    auto* beamData = bufferCast<TokenIdType>(*outputIdsBeam);
                    std::fill(beamData, beamData + beamLength, llmReq.mEndId.value_or(-1));
                }
                // Correct next token position by accepted draft tokens
                tokenPos += numAcceptedTokens;
                auto* sequenceLengthPtr = bufferCast<SizeType32>(*sequenceLengthTensor);
                sequenceLengthPtr[beam] = nbTokens;
            }
            outputIdsTensor->unsqueeze(0);
            sequenceLengthTensor->unsqueeze(0);
            std::list<NamedTensor> output_tensors{
                {outputIdsTensor, kOutputIdsTensorName}, {sequenceLengthTensor, kSequenceLengthTensorName}};

            // We need to return those tensors even if not requested since Triton doesn't support optional output
            // tensors
            ITensor::SharedPtr cumLogProbsTensor{
                BufferManager::cpu(ITensor::makeShape({1, nbBeams}), nvinfer1::DataType::kFLOAT)};

            auto nbProbsTokensOut = maxNbTokens - llmReq.getOrigPromptLen();
            ITensor::SharedPtr logProbsTensor{
                BufferManager::cpu(ITensor::makeShape({1, nbBeams, nbProbsTokensOut}), nvinfer1::DataType::kFLOAT)};

            bufferManager.setZero(*cumLogProbsTensor);
            bufferManager.setZero(*logProbsTensor);

            if (llmReq.getReturnContextLogits())
            {
                TLLM_CHECK_WITH_INFO(
                    !llmReq.isStreaming(), "Return context logits is not supported with streaming mode");

                TensorPtr const& contextLogitsHost = llmReq.getContextLogitsHost();
                auto contextLogitsShape = contextLogitsHost->getShape();
                TLLM_CHECK(contextLogitsShape.nbDims == 2);
                auto promptLength = contextLogitsShape.d[0];
                auto vocabSizePadded = contextLogitsShape.d[1];
                TensorPtr contextLogitsHostView = ITensor::view(
                    llmReq.getContextLogitsHost(), ITensor::makeShape({1, promptLength, vocabSizePadded}));
                output_tensors.emplace_back(NamedTensor({contextLogitsHostView, kContextLogitsName}));
            }
            else
            {
                // Return a dummy tensor to cooperate with triton backend
                ITensor::SharedPtr dummyContextLogitsHost{
                    BufferManager::cpu(ITensor::makeShape({1, 1, 1}), nvinfer1::DataType::kFLOAT)};
                bufferManager.setZero(*dummyContextLogitsHost);
                output_tensors.emplace_back(NamedTensor({dummyContextLogitsHost, kContextLogitsName}));
            }

            if (llmReq.getReturnGenerationLogits())
            {
                TLLM_CHECK_WITH_INFO(!llmReq.isStreaming(),
                    "GptManager is deprecated, please use executor API to return generation logits under streaming "
                    "mode");

                TensorPtr const& generationLogitsHost = llmReq.getGenerationLogitsHost();
                auto const generationLogitsShape = generationLogitsHost->getShape();
                TLLM_CHECK(generationLogitsShape.nbDims == 3);
                // Reshape to be a dim=4 tensor to meet dimensional requirements
                auto const newShape = ITensor::unsqueeze(generationLogitsShape, 0);
                TensorPtr generationLogitsHostView = ITensor::view(llmReq.getGenerationLogitsHost(), newShape);
                output_tensors.emplace_back(NamedTensor({generationLogitsHostView, kGenerationLogitsName}));
            }
            else
            {
                // Return a dummy tensor to cooperate with triton backend
                ITensor::SharedPtr dummyGenerationLogitsHost{
                    BufferManager::cpu(ITensor::makeShape({1, 1, 1, 1}), nvinfer1::DataType::kFLOAT)};
                bufferManager.setZero(*dummyGenerationLogitsHost);
                output_tensors.emplace_back(NamedTensor({dummyGenerationLogitsHost, kGenerationLogitsName}));
            }

            if (llmReq.returnLogProbs())
            {
                // cumLogProbs
                auto const& cumLogProbs = llmReq.getCumLogProbs();
                TLLM_CHECK(cumLogProbs.size() == static_cast<std::size_t>(nbBeams));
                bufferManager.copy(cumLogProbs.data(), *cumLogProbsTensor);

                // logProbs
                for (SizeType32 beam = 0; beam < nbBeams; ++beam)
                {
                    auto logProbs = llmReq.getLogProbs(beam);
                    auto numTokens = llmReq.getNumTokens(beam);
                    auto nbProbTokensOut = numTokens - llmReq.getOrigPromptLen();

                    TensorPtr logProbsTensorView = ITensor::view(logProbsTensor);
                    logProbsTensorView->squeeze(0);
                    std::memcpy(ITensor::slice(logProbsTensorView, beam, 1)->data(), logProbs.data(),
                        nbProbTokensOut * sizeof(float));
                }
            }

            output_tensors.emplace_back(logProbsTensor, kLogProbsTensorName);
            output_tensors.emplace_back(cumLogProbsTensor, kCumLogProbsTensorName);

            if (llmReq.isGenerationCompleteState() || (isStreaming && tokenPos + 1 > llmReq.getMaxSentTokenLen()))
            {
                mSendResponseCb(static_cast<uint64_t>(requestId), output_tensors, isFinalResponse, "");
                llmReq.setMaxSentTokenLen(tokenPos + 1);
            }
            if (isFinalResponse)
            {
                mActiveRequestsIds.erase(requestId);
                it = mActiveRequests.erase(it);
            }
            else
            {
                ++it;
            }
        }
        else
        {
            ++it;
        }
    }
    return BatchManagerErrorCode_t::STATUS_SUCCESS;
}

#define WATCHDOG_MAX 2

void GptManager::decoupled_execution_loop()
{
    // TODO: Implement the watchdog cleanly
    // int32_t watchdog = 0;
    auto const [profileIterIdxs, stopIterIdxs] = tensorrt_llm::common::populateIterationIndexes(
        kPROFILE_START_STOP_ENV_VAR_NAME, kLEGACY_PROFILE_START_STOP_ENV_VAR_NAME);

    while (!shutdown_requested_ || !mActiveRequests.empty())
    {
        auto const profileIter = !profileIterIdxs.empty() && (profileIterIdxs.count(mIterationCounter) > 0);
        auto const stopIter = !stopIterIdxs.empty() && (stopIterIdxs.count(mIterationCounter) > 0);

        BatchManagerErrorCode_t status;

        if (profileIter)
        {
            cudaProfilerStart();
        }
        if (!mActiveRequests.empty())
        {
            status = forwardSync(); // also updates request state at the end of the step
            if (status != BatchManagerErrorCode_t::STATUS_SUCCESS)
            {
                TLLM_LOG_WARNING("Sync decoder function failed, continuing.");
                continue;
            }
        }

        if (mPollStopSignalCb)
        {
            status = pollStopSignals();
            if (status != BatchManagerErrorCode_t::STATUS_SUCCESS)
            {
                break;
            }
        }

        status = returnCompletedRequests();
        if (status != BatchManagerErrorCode_t::STATUS_SUCCESS)
        {
            TLLM_LOG_WARNING("Returning completed requests failed, continuing.");
            continue;
        }

        // Note: fetchNewRequests currently never returns failure, just a potentially empty list of requests
        status = fetchNewRequests();
        if (status == BatchManagerErrorCode_t::STATUS_TERMINATE)
        {
            TLLM_LOG_INFO("Terminate signal received, worker thread exiting.");
            break;
        }
        else if (status != BatchManagerErrorCode_t::STATUS_SUCCESS)
        {
            TLLM_LOG_WARNING("Fetching new requests failed, continuing.");
            continue;
        }

        if (!mActiveRequests.empty())
        {
            status = forwardAsync(
                mActiveRequests, mActiveRequestsIds); // also updates request state at the end of the step
            mIterationCounter++;
            if (status != BatchManagerErrorCode_t::STATUS_SUCCESS)
            {
                TLLM_LOG_WARNING("forwardAsync function failed, continuing.");
                continue;
            }
        }

        if (mReturnBatchManagerStatsCb)
        {
            // Don't return stats when no requests were processed in this iteration
            if (!mActiveRequests.empty())
            {
                status = returnBatchManagerStats();
                if (status != BatchManagerErrorCode_t::STATUS_SUCCESS)
                {
                    TLLM_LOG_WARNING("Returning batch manager stats failed, continuing.");
                    continue;
                }
            }
        }
        if (stopIter)
        {
            cudaProfilerStop();
        }
    }
}

namespace kv_cache_manager
{
std::ostream& operator<<(std::ostream& os, KvCacheConfig const& config)
{
    os << "  maxTokens: " << config.maxTokens.value_or(0) << "\n";
    os << "  maxAttentionWindow: ";
    if (config.maxAttentionWindowVec.has_value())
    {
        os << tc::vec2str(config.maxAttentionWindowVec.value()) << "\n";
    }
    else
    {
        os << "0\n";
    }
    os << "  sinkTokenLength: " << config.sinkTokenLength.value_or(0) << "\n";
    os << "  freeGpuMemoryFraction: " << config.freeGpuMemoryFraction.value_or(0) << "\n";
    os << "  enableBlockReuse: " << config.enableBlockReuse << "\n";
    os << "  useUvm: " << config.useUvm << "\n";
    os << "  hostCacheSize: " << config.hostCacheSize.value_or(0) << "\n";
    os << "  onboardBlocks: " << config.onboardBlocks << "\n";
    return os;
}
} // namespace kv_cache_manager

std::ostream& operator<<(std::ostream& os, TrtGptModelOptionalParams const& options)
{
    os << "TrtGptModelOptionalParams:\n";
    os << "  kvCacheConfig:\n";
    os << options.kvCacheConfig;
    os << "  enableTrtOverlap: " << options.enableTrtOverlap << "\n";
    os << "  deviceIds: ";
    if (options.deviceIds)
    {
        os << "[";
        for (auto const& deviceId : *options.deviceIds)
        {
            os << deviceId << ", ";
        }
        os << "]\n";
    }
    else
    {
        os << "nullopt\n";
    }
    os << "  normalizeLogProbs: " << options.normalizeLogProbs << "\n";
    os << "  enableChunkedContext: " << options.enableChunkedContext << "\n";

    // NOTE: peftCacheManagerConfig and decodingConfig are ignored temporarily
    return os;
}

} // namespace tensorrt_llm::batch_manager
