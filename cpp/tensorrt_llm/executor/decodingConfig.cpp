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
#include "tensorrt_llm/executor/types.h"

#include <memory>
#include <optional>
#include <utility>

namespace tensorrt_llm::executor
{

// Constructor for ExternalDraftTokensConfig
ExternalDraftTokensConfig::ExternalDraftTokensConfig(VecTokens tokens, std::optional<Tensor> logits,
    std::optional<FloatType> const& acceptanceThreshold, std::optional<bool> const& fastLogits)
    : mTokens(std::move(tokens))
    , mLogits(std::move(logits))
    , mAcceptanceThreshold(acceptanceThreshold)
    , mFastLogits(fastLogits)
{
    TLLM_CHECK(!mTokens.empty());
    if (mLogits)
    {
        TLLM_CHECK(mLogits.value().getShape().size() == 2);
        if (mFastLogits.has_value() && mFastLogits.value())
        {
            // Fast logits path, expected [1, specDecFastLogitsInfo] shape
            TLLM_CHECK(mLogits.value().getShape()[0] == 1);
            TLLM_CHECK(
                mLogits.value().getShape()[1] == (sizeof(SpeculativeDecodingFastLogitsInfo) + 1) / sizeof(float));
        }
        else
        {
            TLLM_CHECK(mLogits.value().getShape()[0] == static_cast<SizeType32>(mTokens.size()));
        }
    }
    if (mAcceptanceThreshold)
    {
        TLLM_CHECK(mAcceptanceThreshold.value() > 0.f);
        TLLM_CHECK(mAcceptanceThreshold.value() <= 1.f);
    }
}

VecTokens ExternalDraftTokensConfig::getTokens() const
{
    return mTokens;
}

std::optional<Tensor> ExternalDraftTokensConfig::getLogits() const
{
    return mLogits;
}

std::optional<FloatType> ExternalDraftTokensConfig::getAcceptanceThreshold() const
{
    return mAcceptanceThreshold;
}

LookaheadDecodingConfig::LookaheadDecodingConfig(
    SizeType32 windowSize, SizeType32 ngramSize, SizeType32 verificationSetSize)
    : mWindowSize(windowSize)
    , mNgramSize(ngramSize)
    , mVerificationSetSize(verificationSetSize)
{
    TLLM_CHECK_WITH_INFO(mNgramSize >= 1, "ngramSize requires >= 1");
    TLLM_CHECK_WITH_INFO(mWindowSize >= 1, "windowSize requires >= 1");
    TLLM_CHECK_WITH_INFO(
        mNgramSize == 1 ? mVerificationSetSize == 0 : true, "ngramSize=1 requires verificationSetSize=0");
    TLLM_CHECK_WITH_INFO(mNgramSize == 1 ? mWindowSize == 1 : true, "ngramSize=1 requires windowSize=1");
    TLLM_CHECK_WITH_INFO(mVerificationSetSize >= 0, "verificationSetSize requires >=0");
}

bool LookaheadDecodingConfig::operator==(LookaheadDecodingConfig const& other) const
{
    return mNgramSize == other.mNgramSize && mWindowSize == other.mWindowSize
        && mVerificationSetSize == other.mVerificationSetSize;
}

std::tuple<SizeType32 const, SizeType32 const, SizeType32 const> LookaheadDecodingConfig::get() const
{
    return std::make_tuple(mWindowSize, mNgramSize, mVerificationSetSize);
}

SizeType32 LookaheadDecodingConfig::getNgramSize() const
{
    return mNgramSize;
}

SizeType32 LookaheadDecodingConfig::getWindowSize() const
{
    return mWindowSize;
}

SizeType32 LookaheadDecodingConfig::getVerificationSetSize() const
{
    return mVerificationSetSize;
}

bool LookaheadDecodingConfig::isLegal(
    SizeType32 windowSize, SizeType32 ngramSize, SizeType32 verificationSetSize) noexcept
{
    bool result = true;
    result &= ngramSize >= 1;
    result &= windowSize >= 1;
    result &= ngramSize == 1 ? windowSize == 1 : true;
    result &= ngramSize == 1 ? verificationSetSize == 0 : true;
    result &= verificationSetSize >= 0;
    return result;
}

std::tuple<SizeType32, SizeType32, SizeType32, SizeType32> LookaheadDecodingConfig::calculateSpeculativeResource() const
{
    SizeType32 maxPathLen = mNgramSize;
    SizeType32 maxDraftTokens =                                        //
        ((mNgramSize == 1) ? 0 : (mNgramSize - 2))                     // lookahead Window first
        + (mWindowSize - 1 + mVerificationSetSize) * (mNgramSize - 1); // lookahead Window rest and guess tokens
    SizeType32 maxDecodingTokens = maxDraftTokens + 1;                 // + golden Token
    SizeType32 maxDraftPathLen = mNgramSize - 1;
    return std::make_tuple(maxDecodingTokens, maxPathLen, maxDraftTokens, maxDraftPathLen);
}

bool LookaheadDecodingConfig::isLE(LookaheadDecodingConfig const& that) const
{
    return mWindowSize <= that.mWindowSize && mNgramSize <= that.mNgramSize
        && mVerificationSetSize <= that.mVerificationSetSize;
}

EagleConfig::EagleConfig(std::optional<EagleChoices> eagleChoices)
    : mEagleChoices(std::move(eagleChoices))
{
}

bool EagleConfig::operator==(EagleConfig const& other) const
{
    return mEagleChoices == other.mEagleChoices;
}

std::optional<EagleChoices> EagleConfig::getEagleChoices() const
{
    return mEagleChoices;
}

DecodingConfig::DecodingConfig(std::optional<DecodingMode> decodingMode,
    std::optional<LookaheadDecodingConfig> lookaheadDecodingConfig, std::optional<MedusaChoices> medusaChoices,
    std::optional<EagleConfig> eagleConfig)
    : mDecodingMode{decodingMode}
    , mLookaheadDecodingConfig{lookaheadDecodingConfig}
    , mMedusaChoices{std::move(medusaChoices)}
    , mEagleConfig{std::move(eagleConfig)}
{
    if (mLookaheadDecodingConfig)
    {
        TLLM_CHECK_WITH_INFO(mDecodingMode, "LookaheadDecodingConfig is set, but DecodingMode is not set");
        TLLM_CHECK_WITH_INFO(mDecodingMode.value().isLookahead(),
            "LookaheadDecodingConfig is set, but DecodingMode is not set to Lookahead");
    }
    if (mMedusaChoices)
    {
        TLLM_CHECK_WITH_INFO(mDecodingMode, "MedusaChoices are set, but DecodingMode is not set");
        TLLM_CHECK_WITH_INFO(
            mDecodingMode.value().isMedusa(), "MedusaChoices are set, but DecodingMode is not set to Medusa");
    }
    if (mEagleConfig)
    {
        TLLM_CHECK_WITH_INFO(mDecodingMode, "EagleConfig is set, but DecodingMode is not set");
        TLLM_CHECK_WITH_INFO(
            mDecodingMode.value().isEagle(), "EagleConfig is set, but DecodingMode is not set to Eagle");
    }
}

bool DecodingConfig::operator==(DecodingConfig const& other) const
{
    return mDecodingMode == other.mDecodingMode && mLookaheadDecodingConfig == other.mLookaheadDecodingConfig
        && mMedusaChoices == other.mMedusaChoices && mEagleConfig == other.mEagleConfig;
}

std::optional<DecodingMode> DecodingConfig::getDecodingMode() const
{
    return mDecodingMode;
}

void DecodingConfig::setDecodingMode(DecodingMode const& decodingMode)
{
    if (decodingMode.isMedusa() || decodingMode.isLookahead() || decodingMode.isExplicitDraftTokens())
    {
        TLLM_THROW(
            "Decoding mode must not be set with `setDecodingMode` for Medusa, Lookahead or explicit draft tokens. "
            "Please, use setters for the respective configs or set decoding mode at the DecodingConfig constructor");
    }
    mDecodingMode = decodingMode;
}

std::optional<LookaheadDecodingConfig> DecodingConfig::getLookaheadDecodingConfig() const
{
    return mLookaheadDecodingConfig;
}

void DecodingConfig::setLookaheadDecoding(LookaheadDecodingConfig const& lookaheadDecodingConfig)
{
    mLookaheadDecodingConfig = lookaheadDecodingConfig;
    mDecodingMode = DecodingMode::Lookahead();
}

std::optional<MedusaChoices> DecodingConfig::getMedusaChoices() const
{
    return mMedusaChoices;
}

void DecodingConfig::setMedusaChoices(MedusaChoices const& medusaChoices)
{
    mMedusaChoices = medusaChoices;
    mDecodingMode = DecodingMode::Medusa();
}

std::optional<EagleConfig> DecodingConfig::getEagleConfig() const
{
    return mEagleConfig;
}

void DecodingConfig::setEagleConfig(EagleConfig const& eagleConfig)
{
    mEagleConfig = eagleConfig;
    mDecodingMode = DecodingMode::Eagle();
}

} // namespace tensorrt_llm::executor
