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

#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/requestImpl.h"
#include "tensorrt_llm/executor/tensor.h"
#include "tensorrt_llm/executor/types.h"

namespace tensorrt_llm::executor
{

Request::Request(VecTokens inputTokenIds, SizeType32 maxNewTokens, bool streaming, SamplingConfig const& samplingConfig,
    OutputConfig const& outputConfig, std::optional<SizeType32> const& endId, std::optional<SizeType32> const& padId,
    std::optional<std::list<VecTokens>> badWords, std::optional<std::list<VecTokens>> stopWords,
    std::optional<Tensor> embeddingBias, std::optional<ExternalDraftTokensConfig> externalDraftTokensConfig,
    std::optional<PromptTuningConfig> pTuningConfig, std::optional<LoraConfig> loraConfig,
    std::optional<std::string> logitsPostProcessorName, std::optional<VecTokens> encoderInputTokenIds,
    std::optional<IdType> clientId, bool returnAllGeneratedTokens, float priority)
    : mImpl(std::make_unique<Impl>(std::move(inputTokenIds), maxNewTokens, streaming, samplingConfig, outputConfig,
        endId, padId, std::move(badWords), std::move(stopWords), std::move(embeddingBias),
        std::move(externalDraftTokensConfig), std::move(pTuningConfig), std::move(loraConfig),
        std::move(logitsPostProcessorName), std::move(encoderInputTokenIds), clientId, returnAllGeneratedTokens,
        priority))
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

SizeType32 Request::getMaxNewTokens() const
{
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

} // namespace tensorrt_llm::executor
