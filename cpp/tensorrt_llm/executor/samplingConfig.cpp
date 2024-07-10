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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/types.h"

namespace tensorrt_llm::executor
{

SamplingConfig::SamplingConfig(SizeType32 beamWidth, std::optional<SizeType32> const& topK,
    std::optional<FloatType> const& topP, std::optional<FloatType> const& topPMin,
    std::optional<TokenIdType> const& topPResetIds, std::optional<FloatType> const& topPDecay,
    std::optional<RandomSeedType> const& randomSeed, std::optional<FloatType> const& temperature,
    std::optional<SizeType32> const& minLength, std::optional<FloatType> const& beamSearchDiversityRate,
    std::optional<FloatType> const& repetitionPenalty, std::optional<FloatType> const& presencePenalty,
    std::optional<FloatType> const& frequencyPenalty, std::optional<FloatType> const& lengthPenalty,
    std::optional<SizeType32> const& earlyStopping, std::optional<SizeType32> const& noRepeatNgramSize)
    : mBeamWidth(checkBeamWidth(beamWidth))
    , mTopK(checkTopK(topK))
    , mTopP(checkTopP(topP))
    , mTopPMin(checkTopPMin(topPMin))
    , mTopPResetIds(checkTopPResetIds(topPResetIds))
    , mTopPDecay(checkTopPDecay(topPDecay))
    , mRandomSeed(randomSeed)
    , mTemperature(checkTemperature(temperature))
    , mMinLength(checkMinLength(minLength))
    , mBeamSearchDiversityRate(checkBeamSearchDiversityRate(beamSearchDiversityRate))
    , mRepetitionPenalty(checkRepetitionPenalty(repetitionPenalty))
    , mPresencePenalty(presencePenalty)
    , mFrequencyPenalty(frequencyPenalty)
    , mLengthPenalty(lengthPenalty)
    , mEarlyStopping(earlyStopping)
    , mNoRepeatNgramSize(checkNoRepeatNgramSize(noRepeatNgramSize))
{
}

bool SamplingConfig::operator==(SamplingConfig const& other) const
{
    return mBeamWidth == other.mBeamWidth && mTopK == other.mTopK && mTopP == other.mTopP && mTopPMin == other.mTopPMin
               && mTopPResetIds == other.mTopPResetIds && mTopPDecay == other.mTopPDecay
               && mRandomSeed == other.mRandomSeed && mTemperature == other.mTemperature
               && mMinLength == other.mMinLength && mBeamSearchDiversityRate == other.mBeamSearchDiversityRate
               && mRepetitionPenalty == other.mRepetitionPenalty && mPresencePenalty == other.mPresencePenalty
               && mFrequencyPenalty == other.mFrequencyPenalty && mLengthPenalty == other.mLengthPenalty
               && mEarlyStopping == other.mEarlyStopping,
           mNoRepeatNgramSize == other.mNoRepeatNgramSize;
}

SizeType32 SamplingConfig::getBeamWidth() const
{
    return mBeamWidth;
}

std::optional<SizeType32> SamplingConfig::getTopK() const
{
    return mTopK;
}

std::optional<FloatType> SamplingConfig::getTopP() const
{
    return mTopP;
}

std::optional<FloatType> SamplingConfig::getTopPMin() const
{
    return mTopPMin;
}

std::optional<SizeType32> SamplingConfig::getTopPResetIds() const
{
    return mTopPResetIds;
}

std::optional<FloatType> SamplingConfig::getTopPDecay() const
{
    return mTopPDecay;
}

std::optional<RandomSeedType> SamplingConfig::getRandomSeed() const
{
    return mRandomSeed;
}

std::optional<FloatType> SamplingConfig::getTemperature() const
{
    return mTemperature;
}

std::optional<SizeType32> SamplingConfig::getMinLength() const
{
    return mMinLength;
}

std::optional<FloatType> SamplingConfig::getBeamSearchDiversityRate() const
{
    return mBeamSearchDiversityRate;
}

std::optional<FloatType> SamplingConfig::getRepetitionPenalty() const
{
    return mRepetitionPenalty;
}

std::optional<FloatType> SamplingConfig::getPresencePenalty() const
{
    return mPresencePenalty;
}

std::optional<FloatType> SamplingConfig::getFrequencyPenalty() const
{
    return mFrequencyPenalty;
}

std::optional<FloatType> SamplingConfig::getLengthPenalty() const
{
    return mLengthPenalty;
}

std::optional<SizeType32> SamplingConfig::getEarlyStopping() const
{
    return mEarlyStopping;
}

std::optional<SizeType32> SamplingConfig::getNoRepeatNgramSize() const
{
    return mNoRepeatNgramSize;
}

// the setters

void SamplingConfig::setBeamWidth(SizeType32 beamWidth)
{
    mBeamWidth = checkBeamWidth(beamWidth);
}

void SamplingConfig::setTopK(std::optional<SizeType32> const& topK)
{
    mTopK = checkTopK(topK);
}

void SamplingConfig::setTopP(std::optional<FloatType> const& topP)
{
    mTopP = checkTopP(topP);
}

void SamplingConfig::setTopPMin(std::optional<FloatType> const& topPMin)
{
    mTopPMin = checkTopPMin(topPMin);
}

void SamplingConfig::setTopPResetIds(std::optional<TokenIdType> const& topPResetIds)
{
    mTopPResetIds = checkTopPResetIds(topPResetIds);
}

void SamplingConfig::setTopPDecay(std::optional<FloatType> const& topPDecay)
{
    mTopPDecay = checkTopPDecay(topPDecay);
}

void SamplingConfig::setRandomSeed(std::optional<RandomSeedType> const& randomSeed)
{
    mRandomSeed = randomSeed;
}

void SamplingConfig::setTemperature(std::optional<FloatType> const& temperature)
{
    mTemperature = checkTemperature(temperature);
}

void SamplingConfig::setMinLength(std::optional<SizeType32> const& minLength)
{
    mMinLength = checkMinLength(minLength);
}

void SamplingConfig::setBeamSearchDiversityRate(std::optional<FloatType> const& beamSearchDiversityRate)
{
    mBeamSearchDiversityRate = checkBeamSearchDiversityRate(beamSearchDiversityRate);
}

void SamplingConfig::setRepetitionPenalty(std::optional<FloatType> const& repetitionPenalty)
{
    mRepetitionPenalty = checkRepetitionPenalty(repetitionPenalty);
}

void SamplingConfig::setPresencePenalty(std::optional<FloatType> const& presencePenalty)
{
    mPresencePenalty = presencePenalty;
}

void SamplingConfig::setFrequencyPenalty(std::optional<FloatType> const& frequencyPenalty)
{
    mFrequencyPenalty = frequencyPenalty;
}

void SamplingConfig::setLengthPenalty(std::optional<FloatType> const& lengthPenalty)
{
    mLengthPenalty = lengthPenalty;
}

void SamplingConfig::setEarlyStopping(std::optional<SizeType32> const& earlyStopping)
{
    mEarlyStopping = earlyStopping;
}

void SamplingConfig::setNoRepeatNgramSize(std::optional<SizeType32> const& noRepeatNgramSize)
{
    mNoRepeatNgramSize = checkNoRepeatNgramSize(noRepeatNgramSize);
}

SizeType32 SamplingConfig::checkBeamWidth(SizeType32 beamWidth)
{
    TLLM_CHECK(beamWidth > 0);
    return beamWidth;
}

std::optional<FloatType> const& SamplingConfig::checkTopK(std::optional<FloatType> const& topK)
{
    if (topK.has_value())
    {
        TLLM_CHECK(topK.value() >= 0);
    }
    return topK;
}

std::optional<FloatType> const& SamplingConfig::checkTopP(std::optional<FloatType> const& topP)
{
    if (topP.has_value())
    {
        TLLM_CHECK(topP.value() > 0.f);
        TLLM_CHECK(topP.value() <= 1.f);
    }
    return topP;
}

std::optional<FloatType> const& SamplingConfig::checkTopPMin(std::optional<FloatType> const& topPMin)
{
    if (topPMin.has_value())
    {
        TLLM_CHECK(topPMin.value() > 0.f);
        TLLM_CHECK(topPMin.value() <= 1.f);
    }
    return topPMin;
}

std::optional<TokenIdType> const& SamplingConfig::checkTopPResetIds(std::optional<TokenIdType> const& topPResetIds)
{
    if (topPResetIds.has_value())
    {
        TLLM_CHECK(topPResetIds.value() >= 0);
    }
    return topPResetIds;
}

std::optional<FloatType> const& SamplingConfig::checkTopPDecay(std::optional<FloatType> const& topPDecay)
{
    if (topPDecay.has_value())
    {
        TLLM_CHECK(topPDecay.value() > 0.f);
        TLLM_CHECK(topPDecay.value() <= 1.f);
    }
    return topPDecay;
}

std::optional<FloatType> const& SamplingConfig::checkTemperature(std::optional<FloatType> const& temperature)
{
    if (temperature.has_value())
    {
        TLLM_CHECK(temperature.value() >= 0.f);
    }
    return temperature;
}

std::optional<SizeType32> const& SamplingConfig::checkMinLength(std::optional<SizeType32> const& minLength)
{
    if (minLength.has_value())
    {
        TLLM_CHECK(minLength.value() >= 0);
    }
    return minLength;
}

std::optional<FloatType> const& SamplingConfig::checkRepetitionPenalty(std::optional<FloatType> const& penalty)
{
    if (penalty.has_value())
    {
        TLLM_CHECK(penalty.value() > 0.f);
    }
    return penalty;
}

std::optional<SizeType32> const& SamplingConfig::checkNoRepeatNgramSize(
    std::optional<SizeType32> const& noRepeatNgramSize)
{
    if (noRepeatNgramSize.has_value())
    {
        TLLM_CHECK(noRepeatNgramSize.value() > 0);
    }
    return noRepeatNgramSize;
}

std::optional<FloatType> const& SamplingConfig::checkBeamSearchDiversityRate(
    std::optional<FloatType> const& beamSearchDiversityRate)
{
    if (beamSearchDiversityRate.has_value())
    {
        TLLM_CHECK(beamSearchDiversityRate.value() >= 0.f);
    }
    return beamSearchDiversityRate;
}

} // namespace tensorrt_llm::executor
