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
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/requestImpl.h"
#include "tensorrt_llm/executor/tensor.h"
#include "tensorrt_llm/executor/types.h"

namespace tensorrt_llm::executor
{

Request::Request(VecTokens inputTokenIds, SizeType32 maxTokens, bool streaming, SamplingConfig const& samplingConfig,
    OutputConfig const& outputConfig, std::optional<SizeType32> const& endId, std::optional<SizeType32> const& padId,
    std::optional<std::vector<SizeType32>> positionIds, std::optional<std::list<VecTokens>> badWords,
    std::optional<std::list<VecTokens>> stopWords, std::optional<Tensor> embeddingBias,
    std::optional<ExternalDraftTokensConfig> externalDraftTokensConfig, std::optional<PromptTuningConfig> pTuningConfig,
    std::optional<LoraConfig> loraConfig, std::optional<LookaheadDecodingConfig> lookaheadConfig,
    std::optional<KvCacheRetentionConfig> kvCacheRetentionConfig, std::optional<std::string> logitsPostProcessorName,
    std::optional<VecTokens> encoderInputTokenIds, std::optional<IdType> clientId, bool returnAllGeneratedTokens,
    float priority, RequestType type, std::optional<ContextPhaseParams> contextPhaseParams,
    std::optional<Tensor> encoderInputFeatures, std::optional<SizeType32> encoderOutputLength,
    std::optional<Tensor> crossAttentionMask, SizeType32 numReturnSequences, std::optional<EagleConfig> eagleConfig,
    std::optional<Tensor> skipCrossAttnBlocks)
    : mImpl(std::make_unique<Impl>(std::move(inputTokenIds), maxTokens, streaming, samplingConfig, outputConfig, endId,
        padId, std::move(positionIds), std::move(badWords), std::move(stopWords), std::move(embeddingBias),
        std::move(externalDraftTokensConfig), std::move(pTuningConfig), std::move(loraConfig),
        std::move(lookaheadConfig), std::move(kvCacheRetentionConfig), std::move(logitsPostProcessorName),
        std::move(encoderInputTokenIds), clientId, returnAllGeneratedTokens, priority, type,
        std::move(contextPhaseParams), std::move(encoderInputFeatures), encoderOutputLength, crossAttentionMask,
        numReturnSequences, eagleConfig, skipCrossAttnBlocks))
{
}

Request::~Request() = default;

Request::Request(Request const& other)
    : mImpl(std::make_unique<Impl>(*other.mImpl))
{
}

Request::Request(Request&& other) noexcept = default;

Request& Request::operator=(Request const& other)
{
    if (this != &other)
    {
        mImpl = std::make_unique<Impl>(*other.mImpl);
    }
    return *this;
}

Request& Request::operator=(Request&& other) noexcept = default;

VecTokens Request::getInputTokenIds() const
{
    return mImpl->getInputTokenIds();
}

SizeType32 Request::getMaxTokens() const
{
    return mImpl->getMaxNewTokens();
}

SizeType32 Request::getMaxNewTokens() const
{
    TLLM_LOG_WARNING("getMaxNewTokens is being deprecated; please use getMaxTokens instead.");
    return mImpl->getMaxNewTokens();
}

bool Request::getStreaming() const
{
    return mImpl->getStreaming();
}

SamplingConfig Request::getSamplingConfig() const
{
    return mImpl->getSamplingConfig();
}

OutputConfig Request::getOutputConfig() const
{
    return mImpl->getOutputConfig();
}

std::optional<SizeType32> Request::getEndId() const
{
    return mImpl->getEndId();
}

std::optional<SizeType32> Request::getPadId() const
{
    return mImpl->getPadId();
}

std::optional<std::vector<SizeType32>> Request::getPositionIds() const
{
    return mImpl->getPositionIds();
}

std::optional<std::list<VecTokens>> Request::getBadWords() const
{
    return mImpl->getBadWords();
}

std::optional<std::list<VecTokens>> Request::getStopWords() const
{
    return mImpl->getStopWords();
}

std::optional<Tensor> Request::getEmbeddingBias() const
{
    return mImpl->getEmbeddingBias();
}

std::optional<ExternalDraftTokensConfig> Request::getExternalDraftTokensConfig() const
{
    return mImpl->getExternalDraftTokensConfig();
}

std::optional<PromptTuningConfig> Request::getPromptTuningConfig() const
{
    return mImpl->getPromptTuningConfig();
}

std::optional<LoraConfig> Request::getLoraConfig() const
{
    return mImpl->getLoraConfig();
}

std::optional<LookaheadDecodingConfig> Request::getLookaheadConfig() const
{
    return mImpl->getLookaheadConfig();
}

std::optional<KvCacheRetentionConfig> Request::getKvCacheRetentionConfig() const
{
    return mImpl->getKvCacheRetentionConfig();
}

std::optional<std::string> Request::getLogitsPostProcessorName() const
{
    return mImpl->getLogitsPostProcessorName();
}

std::optional<VecTokens> Request::getEncoderInputTokenIds() const
{
    return mImpl->getEncoderInputTokenIds();
}

std::optional<IdType> Request::getClientId() const
{
    return mImpl->getClientId();
}

PriorityType Request::getPriority() const
{
    return mImpl->getPriority();
}

bool Request::getReturnAllGeneratedTokens() const
{
    return mImpl->getReturnAllGeneratedTokens();
}

RequestType Request::getRequestType() const
{
    return mImpl->getRequestType();
}

std::optional<ContextPhaseParams> const& Request::getContextPhaseParams() const
{
    return mImpl->getContextPhaseParams();
}

std::optional<Tensor> Request::getEncoderInputFeatures() const
{
    return mImpl->getEncoderInputFeatures();
}

std::optional<SizeType32> Request::getEncoderOutputLength() const
{
    return mImpl->getEncoderOutputLength();
}

std::optional<Tensor> Request::getCrossAttentionMask() const
{
    return mImpl->getCrossAttentionMask();
}

SizeType32 Request::getNumReturnSequences() const
{
    TLLM_LOG_WARNING(
        "'Request.getNumReturnSequences' will be deprecated. Please directly use "
        "'SamplingConfig.getNumReturnSequences' instead.");
    return mImpl->getNumReturnSequences().value_or(1);
}

std::optional<EagleConfig> Request::getEagleConfig() const
{
    return mImpl->getEagleConfig();
}

std::optional<Tensor> Request::getSkipCrossAttnBlocks() const
{
    return mImpl->getSkipCrossAttnBlocks();
}

void Request::setStreaming(bool streaming)
{
    return mImpl->setStreaming(streaming);
}

void Request::setSamplingConfig(SamplingConfig const& config)
{
    return mImpl->setSamplingConfig(config);
}

void Request::setOutputConfig(OutputConfig const& outputConfig)
{
    return mImpl->setOutputConfig(outputConfig);
}

void Request::setEndId(SizeType32 endId)
{
    return mImpl->setEndId(endId);
}

void Request::setPadId(SizeType32 padId)
{
    return mImpl->setPadId(padId);
}

void Request::setPositionIds(std::vector<SizeType32> const& positionIds)
{
    return mImpl->setPositionIds(positionIds);
}

void Request::setBadWords(std::list<VecTokens> const& badWords)
{
    return mImpl->setBadWords(badWords);
}

void Request::setStopWords(std::list<VecTokens> const& stopWords)
{
    return mImpl->setStopWords(stopWords);
}

void Request::setEmbeddingBias(Tensor const& embeddingBias)
{
    return mImpl->setEmbeddingBias(embeddingBias);
}

void Request::setExternalDraftTokensConfig(ExternalDraftTokensConfig const& specDecodingConfig)
{
    return mImpl->setExternalDraftTokensConfig(specDecodingConfig);
}

void Request::setPromptTuningConfig(PromptTuningConfig const& pTuningConfig)
{
    return mImpl->setPromptTuningConfig(pTuningConfig);
}

void Request::setLoraConfig(LoraConfig const& loraConfig)
{
    return mImpl->setLoraConfig(loraConfig);
}

void Request::setLookaheadConfig(LookaheadDecodingConfig const& lookaheadConfig)
{
    return mImpl->setLookaheadConfig(lookaheadConfig);
}

void Request::setKvCacheRetentionConfig(KvCacheRetentionConfig const& kvCacheRetentionConfig)
{
    return mImpl->setKvCacheRetentionConfig(kvCacheRetentionConfig);
}

void Request::setLogitsPostProcessorName(std::string const& logitsPostProcessorName)
{
    return mImpl->setLogitsPostProcessorName(logitsPostProcessorName);
}

void Request::setEncoderInputTokenIds(VecTokens const& encoderInputTokenIds)
{
    return mImpl->setEncoderInputTokenIds(encoderInputTokenIds);
}

void Request::setClientId(IdType clientId)
{
    return mImpl->setClientId(clientId);
}

void Request::setPriority(PriorityType priority)
{
    return mImpl->setPriority(priority);
}

void Request::setReturnAllGeneratedTokens(bool returnAllGeneratedTokens)
{
    return mImpl->setReturnAllGeneratedTokens(returnAllGeneratedTokens);
}

void Request::setRequestType(RequestType const& requestType)
{
    mImpl->setRequestType(requestType);
}

void Request::setContextPhaseParams(ContextPhaseParams contextPhaseParams)
{
    mImpl->setContextPhaseParams(std::move(contextPhaseParams));
}

void Request::setEncoderInputFeatures(Tensor encoderInputFeatures)
{
    return mImpl->setEncoderInputFeatures(encoderInputFeatures);
}

void Request::setEncoderOutputLength(SizeType32 encoderOutputLength)
{
    return mImpl->setEncoderOutputLength(encoderOutputLength);
}

void Request::setCrossAttentionMask(Tensor crossAttentionMask)
{
    return mImpl->setCrossAttentionMask(crossAttentionMask);
}

void Request::setNumReturnSequences(SizeType32 numReturnSequences)
{
    TLLM_LOG_WARNING(
        "'Request.setNumReturnSequences' will be deprecated. Please directly use "
        "'SamplingConfig.setNumReturnSequences' instead.");
    mImpl->setNumReturnSequences(numReturnSequences);
}

void Request::setEagleConfig(std::optional<EagleConfig> const& eagleConfig)
{
    mImpl->setEagleConfig(eagleConfig);
}

void Request::setSkipCrossAttnBlocks(Tensor skipCrossAttnBlocks)
{
    return mImpl->setSkipCrossAttnBlocks(skipCrossAttnBlocks);
}

} // namespace tensorrt_llm::executor
