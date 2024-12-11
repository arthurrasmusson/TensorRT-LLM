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

#pragma once

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/serializeUtils.h"
#include "tensorrt_llm/executor/tensor.h"
#include "tensorrt_llm/executor/types.h"

namespace su = tensorrt_llm::executor::serialize_utils;

namespace tensorrt_llm::executor
{
class Request::Impl
{

public:
    // Constructor
    Impl(VecTokens inputTokenIds, SizeType32 maxNewTokens, bool streaming, SamplingConfig const& samplingConfig,
        OutputConfig const& outputConfig, std::optional<TokenIdType> const& endId,
        std::optional<TokenIdType> const& padId, std::optional<std::vector<SizeType32>> positionIds,
        std::optional<std::list<VecTokens>> badWords, std::optional<std::list<VecTokens>> stopWords,
        std::optional<Tensor> embeddingBias, std::optional<ExternalDraftTokensConfig> externalDraftTokensConfig,
        std::optional<PromptTuningConfig> pTuningConfig, std::optional<MropeConfig> mRopeConfig,
        std::optional<LoraConfig> loraConfig, std::optional<LookaheadDecodingConfig> lookaheadConfig,
        std::optional<KvCacheRetentionConfig> kvCacheRetentionConfig,
        std::optional<std::string> logitsPostProcessorName, std::optional<VecTokens> encoderInputTokenIds,
        std::optional<IdType> clientId, bool returnAllGeneratedTokens, PriorityType priority, RequestType type,
        std::optional<ContextPhaseParams> contextPhaseParams, std::optional<Tensor> encoderInputFeatures,
        std::optional<SizeType32> encoderOutputLength, std::optional<Tensor> crossAttentionMask,
        SizeType32 numReturnSequences, std::optional<EagleConfig> eagleConfig,
        std::optional<Tensor> skipCrossAttnBlocks, std::optional<GuidedDecodingParams> guidedDecodingParams,
        std::optional<MillisecondsType> allottedTimeMs)
        : mInputTokenIds(std::move(inputTokenIds))
        , mMaxNewTokens(maxNewTokens)
        , mStreaming(streaming)
        , mSamplingConfig(samplingConfig)
        , mOutputConfig(outputConfig)
        , mEndId(endId)
        , mPadId(padId)
        , mPositionIds(std::move(positionIds))
        , mBadWords(std::move(badWords))
        , mStopWords(std::move(stopWords))
        , mEmbeddingBias(checkEmbeddingBias(std::move(embeddingBias)))
        , mExternalDraftTokensConfig(std::move(externalDraftTokensConfig))
        , mPTuningConfig(std::move(pTuningConfig))
        , mMropeConfig(std::move(mRopeConfig))
        , mLoraConfig(std::move(loraConfig))
        , mLookaheadConfig(std::move(lookaheadConfig))
        , mKvCacheRetentionConfig(std::move(kvCacheRetentionConfig))
        , mLogitsPostProcessorName(std::move(logitsPostProcessorName))
        , mEncoderInputTokenIds(std::move(encoderInputTokenIds))
        , mClientId(clientId)
        , mReturnAllGeneratedTokens(returnAllGeneratedTokens)
        , mPriority(priority)
        , mType(type)
        , mContextPhaseParams(contextPhaseParams)
        , mEncoderInputFeatures(encoderInputFeatures)
        , mEncoderOutputLength(encoderOutputLength)
        , mCrossAttentionMask(crossAttentionMask)
        , mNumReturnSequences(numReturnSequences)
        , mEagleConfig(eagleConfig)
        , mSkipCrossAttnBlocks(skipCrossAttnBlocks)
        , mGuidedDecodingParams(std::move(guidedDecodingParams))
        , mAllottedTimeMs(allottedTimeMs)
    {
        validate();
    }

    ~Impl() = default;

    void serialize(std::ostream& ostream) const
    {
        visitMembers([&ostream](auto const& member) { su::serialize(member, ostream); });
    }

    [[nodiscard]] size_t serializedSize() const
    {
        size_t totalSize = 0;
        visitMembers([&totalSize](auto const& member) { totalSize += su::serializedSize(member); });
        return totalSize;
    }

    VecTokens getInputTokenIds() const
    {
        return mInputTokenIds;
    }

    SizeType32 getMaxNewTokens() const
    {
        return mMaxNewTokens;
    }

    bool getStreaming() const
    {
        return mStreaming;
    }

    SamplingConfig getSamplingConfig() const
    {
        return mSamplingConfig;
    }

    OutputConfig getOutputConfig() const
    {
        return mOutputConfig;
    }

    std::optional<SizeType32> getEndId() const
    {
        return mEndId;
    }

    std::optional<SizeType32> getPadId() const
    {
        return mPadId;
    }

    std::optional<std::vector<SizeType32>> getPositionIds() const
    {
        return mPositionIds;
    }

    std::optional<std::list<VecTokens>> getBadWords() const
    {
        return mBadWords;
    }

    std::optional<std::list<VecTokens>> getStopWords() const
    {
        return mStopWords;
    }

    std::optional<Tensor> getEmbeddingBias() const
    {
        return mEmbeddingBias;
    }

    std::optional<ExternalDraftTokensConfig> getExternalDraftTokensConfig() const
    {
        return mExternalDraftTokensConfig;
    }

    std::optional<PromptTuningConfig> getPromptTuningConfig() const
    {
        return mPTuningConfig;
    }

    std::optional<MropeConfig> getMropeConfig() const
    {
        return mMropeConfig;
    }

    std::optional<LoraConfig> getLoraConfig() const
    {
        return mLoraConfig;
    }

    std::optional<LookaheadDecodingConfig> getLookaheadConfig() const
    {
        return mLookaheadConfig;
    }

    std::optional<KvCacheRetentionConfig> getKvCacheRetentionConfig() const
    {
        return mKvCacheRetentionConfig;
    }

    std::optional<std::string> getLogitsPostProcessorName() const
    {
        return mLogitsPostProcessorName;
    }

    std::optional<VecTokens> getEncoderInputTokenIds() const
    {
        return mEncoderInputTokenIds;
    }

    std::optional<IdType> getClientId() const
    {
        return mClientId;
    }

    PriorityType getPriority() const
    {
        return mPriority;
    }

    [[nodiscard]] std::optional<MillisecondsType> getAllottedTimeMs() const
    {
        return mAllottedTimeMs;
    }

    [[nodiscard]] bool getReturnAllGeneratedTokens() const
    {
        return mReturnAllGeneratedTokens;
    }

    RequestType getRequestType() const
    {
        return mType;
    }

    std::optional<ContextPhaseParams> const& getContextPhaseParams() const
    {
        return mContextPhaseParams;
    }

    std::optional<Tensor> getEncoderInputFeatures() const
    {
        return mEncoderInputFeatures;
    }

    std::optional<Tensor> getCrossAttentionMask() const
    {
        return mCrossAttentionMask;
    }

    std::optional<SizeType32> getEncoderOutputLength() const
    {
        return mEncoderOutputLength;
    }

    std::optional<SizeType32> getNumReturnSequences() const
    {
        TLLM_LOG_WARNING(
            "The 'getNumReturnSequences' method in the Request class is deprecated and will be removed in a future "
            "release. Please use 'getNumReturnSequences' directly from the 'SamplingConfig' object.");
        return mSamplingConfig.getNumReturnSequences();
    }

    std::optional<EagleConfig> getEagleConfig() const
    {
        return mEagleConfig;
    }

    std::optional<Tensor> getSkipCrossAttnBlocks() const
    {
        return mSkipCrossAttnBlocks;
    }

    [[nodiscard]] std::optional<GuidedDecodingParams> getGuidedDecodingParams() const
    {
        return mGuidedDecodingParams;
    }

    void setStreaming(bool streaming)
    {
        mStreaming = streaming;
    }

    void setSamplingConfig(SamplingConfig const& config)
    {
        mSamplingConfig = config;
    }

    void setOutputConfig(OutputConfig const& outputConfig)
    {
        mOutputConfig = outputConfig;
    }

    void setEndId(SizeType32 endId)
    {
        mEndId = endId;
    }

    void setPadId(SizeType32 padId)
    {
        mPadId = padId;
    }

    void setPositionIds(std::vector<SizeType32> const& positionIds)
    {
        mPositionIds = positionIds;
    }

    void setBadWords(std::list<VecTokens> const& badWords)
    {
        mBadWords = badWords;
    }

    void setStopWords(std::list<VecTokens> const& stopWords)
    {
        mStopWords = stopWords;
    }

    void setEmbeddingBias(Tensor const& embeddingBias)
    {
        mEmbeddingBias = checkEmbeddingBias(embeddingBias);
    }

    void setExternalDraftTokensConfig(ExternalDraftTokensConfig const& externalDraftTokensConfig)
    {
        mExternalDraftTokensConfig = externalDraftTokensConfig;
    }

    void setPromptTuningConfig(PromptTuningConfig const& pTuningConfig)
    {
        mPTuningConfig = pTuningConfig;
    }

    void setMropeConfig(MropeConfig const& mRopeConfig)
    {
        mMropeConfig = mRopeConfig;
    }

    void setLoraConfig(LoraConfig const& loraConfig)
    {
        mLoraConfig = loraConfig;
    }

    void setLookaheadConfig(LookaheadDecodingConfig const& lookaheadConfig)
    {
        mLookaheadConfig = lookaheadConfig;
    }

    void setKvCacheRetentionConfig(KvCacheRetentionConfig const& kvCacheRetentionConfig)
    {
        mKvCacheRetentionConfig = kvCacheRetentionConfig;
    }

    void setLogitsPostProcessorName(std::string const& logitsPostProcessorName)
    {
        mLogitsPostProcessorName = logitsPostProcessorName;
    }

    void setEncoderInputTokenIds(VecTokens const& encoderInputTokenIds)
    {
        mEncoderInputTokenIds = encoderInputTokenIds;
    }

    void setClientId(IdType clientId)
    {
        mClientId = clientId;
    }

    void setPriority(PriorityType priority)
    {
        mPriority = priority;
    }

    void setReturnAllGeneratedTokens(bool returnAllGeneratedTokens)
    {
        mReturnAllGeneratedTokens = returnAllGeneratedTokens;
    }

    void setRequestType(RequestType requestType)
    {
        mType = requestType;
    }

    void setContextPhaseParams(ContextPhaseParams contextPhaseParams)
    {
        mContextPhaseParams = std::move(contextPhaseParams);
    }

    void setEncoderInputFeatures(Tensor encoderInputFeatures)
    {
        mEncoderInputFeatures = encoderInputFeatures;
    }

    void setCrossAttentionMask(Tensor crossAttentionMask)
    {
        mCrossAttentionMask = crossAttentionMask;
    }

    void setEncoderOutputLength(SizeType32 encoderOutputLength)
    {
        mEncoderOutputLength = encoderOutputLength;
    }

    void setNumReturnSequences(SizeType32 numReturnSequences)
    {
        TLLM_LOG_WARNING(
            "The 'setNumReturnSequences' method in the Request class is deprecated and will be removed in a future "
            "release. Please use 'setNumReturnSequences' directly on the 'SamplingConfig' object.");
        mNumReturnSequences = numReturnSequences;
        mSamplingConfig.setNumReturnSequences(numReturnSequences);
    }

    void setEagleConfig(std::optional<EagleConfig> eagleConfig)
    {
        mEagleConfig = eagleConfig;
    }

    void setSkipCrossAttnBlocks(Tensor skipCrossAttnBlocks)
    {
        mSkipCrossAttnBlocks = skipCrossAttnBlocks;
    }

    void setGuidedDecodingParams(GuidedDecodingParams const& guidedDecodingParams)
    {
        mGuidedDecodingParams = guidedDecodingParams;
    }

    void setAllottedTimeMs(MillisecondsType allottedTimeMs)
    {
        mAllottedTimeMs = allottedTimeMs;
    }

private:
    void validate()
    {
        TLLM_CHECK(!mInputTokenIds.empty());
        TLLM_CHECK(mMaxNewTokens > 0);

        // Show warning message unless mNumReturnSequences is the default value.
        if (mNumReturnSequences > 1)
        {
            TLLM_LOG_WARNING(
                "The 'numReturnSequences' in the Request class is deprecated and will be removed in a future release. "
                "Please set the number of return sequences directly in 'SamplingConfig'.");
            mSamplingConfig.setNumReturnSequences(mNumReturnSequences);
        }

        if (mGuidedDecodingParams.has_value() && mSamplingConfig.getBeamWidth() > 1)
        {
            TLLM_THROW("Guided decoding does not support with beam search.");
        }
    }

    static std::optional<Tensor> checkEmbeddingBias(std::optional<Tensor> bias)
    {
        if (bias)
        {
            TLLM_CHECK(bias.value().getShape().size() == 1);
            TLLM_CHECK(bias.value().getDataType() == DataType::kFP32);
        }
        return bias;
    }

    template <typename Lambda>
    void visitMembers(Lambda const& lambda) const
    {
        lambda(mInputTokenIds);
        lambda(mMaxNewTokens);
        lambda(mStreaming);
        lambda(mSamplingConfig);
        lambda(mOutputConfig);
        lambda(mEndId);
        lambda(mPadId);
        lambda(mPositionIds);
        lambda(mBadWords);
        lambda(mStopWords);
        lambda(mEmbeddingBias);
        lambda(mExternalDraftTokensConfig);
        lambda(mPTuningConfig);
        lambda(mMropeConfig);
        lambda(mLoraConfig);
        lambda(mLookaheadConfig);
        lambda(mKvCacheRetentionConfig);
        lambda(mLogitsPostProcessorName);
        lambda(mEncoderInputTokenIds);
        lambda(mClientId);
        lambda(mReturnAllGeneratedTokens);
        lambda(mPriority);
        lambda(mType);
        lambda(mContextPhaseParams);
        lambda(mEncoderInputFeatures);
        lambda(mEncoderOutputLength);
        lambda(mCrossAttentionMask);
        lambda(mNumReturnSequences);
        lambda(mEagleConfig);
        lambda(mSkipCrossAttnBlocks);
        lambda(mGuidedDecodingParams);
        lambda(mAllottedTimeMs ? std::make_optional(mAllottedTimeMs->count()) : std::nullopt);
    }

    VecTokens mInputTokenIds;
    SizeType32 mMaxNewTokens;
    bool mStreaming;
    SamplingConfig mSamplingConfig;
    OutputConfig mOutputConfig;
    std::optional<SizeType32> mEndId;
    std::optional<SizeType32> mPadId;
    std::optional<std::vector<SizeType32>> mPositionIds;
    std::optional<std::list<VecTokens>> mBadWords;
    std::optional<std::list<VecTokens>> mStopWords;
    std::optional<Tensor> mEmbeddingBias;
    std::optional<ExternalDraftTokensConfig> mExternalDraftTokensConfig;
    std::optional<PromptTuningConfig> mPTuningConfig;
    std::optional<MropeConfig> mMropeConfig;
    std::optional<LoraConfig> mLoraConfig;
    std::optional<LookaheadDecodingConfig> mLookaheadConfig;
    std::optional<KvCacheRetentionConfig> mKvCacheRetentionConfig;
    std::optional<std::string> mLogitsPostProcessorName;
    std::optional<VecTokens> mEncoderInputTokenIds;
    std::optional<IdType> mClientId;
    bool mReturnAllGeneratedTokens;
    PriorityType mPriority;
    RequestType mType;
    std::optional<ContextPhaseParams> mContextPhaseParams;
    std::optional<Tensor> mEncoderInputFeatures;
    std::optional<SizeType32> mEncoderOutputLength;
    std::optional<Tensor> mCrossAttentionMask;
    SizeType32 mNumReturnSequences;
    std::optional<EagleConfig> mEagleConfig;
    std::optional<Tensor> mSkipCrossAttnBlocks;
    std::optional<GuidedDecodingParams> mGuidedDecodingParams;
    std::optional<MillisecondsType> mAllottedTimeMs;
};

} // namespace tensorrt_llm::executor
