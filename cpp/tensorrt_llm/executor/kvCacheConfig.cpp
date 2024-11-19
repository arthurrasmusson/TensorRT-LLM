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

namespace tensorrt_llm::executor
{

KvCacheConfig::KvCacheConfig(bool enableBlockReuse, std::optional<SizeType32> const& maxTokens,
    std::optional<std::vector<SizeType32>> const& maxAttentionWindowVec,
    std::optional<SizeType32> const& sinkTokenLength, std::optional<FloatType> const& freeGpuMemoryFraction,
    std::optional<size_t> const& hostCacheSize, bool onboardBlocks,
    std::optional<FloatType> const& crossKvCacheFraction, std::optional<RetentionPriority> secondaryOffloadMinPriority,
    size_t eventBufferMaxSize, std::optional<tensorrt_llm::runtime::RuntimeDefaults> const& runtimeDefaults)
    : mEnableBlockReuse(enableBlockReuse)
    , mHostCacheSize(hostCacheSize)
    , mOnboardBlocks(onboardBlocks)
    , mSecondaryOffloadMinPriority(secondaryOffloadMinPriority)
    , mEventBufferMaxSize{eventBufferMaxSize}
{
    if (maxTokens)
    {
        setMaxTokens(maxTokens.value());
    }
    if (maxAttentionWindowVec)
    {
        setMaxAttentionWindowVec(maxAttentionWindowVec.value());
    }
    if (sinkTokenLength)
    {
        setSinkTokenLength(sinkTokenLength.value());
    }
    if (freeGpuMemoryFraction)
    {
        setFreeGpuMemoryFraction(freeGpuMemoryFraction.value());
    }
    if (crossKvCacheFraction)
    {
        setCrossKvCacheFraction(crossKvCacheFraction.value());
    }
    if (runtimeDefaults)
    {
        fillEmptyFieldsFromRuntimeDefaults(runtimeDefaults.value());
    }
}

bool KvCacheConfig::getEnableBlockReuse() const
{
    return mEnableBlockReuse;
}

std::optional<SizeType32> KvCacheConfig::getMaxTokens() const
{
    return mMaxTokens;
}

std::optional<std::vector<SizeType32>> KvCacheConfig::getMaxAttentionWindowVec() const
{
    return mMaxAttentionWindowVec;
}

std::optional<SizeType32> KvCacheConfig::getSinkTokenLength() const
{
    return mSinkTokenLength;
}

std::optional<FloatType> KvCacheConfig::getFreeGpuMemoryFraction() const
{
    return mFreeGpuMemoryFraction;
}

std::optional<FloatType> KvCacheConfig::getCrossKvCacheFraction() const
{
    return mCrossKvCacheFraction;
}

std::optional<size_t> KvCacheConfig::getHostCacheSize() const
{
    return mHostCacheSize;
}

bool KvCacheConfig::getOnboardBlocks() const
{
    return mOnboardBlocks;
}

std::optional<RetentionPriority> KvCacheConfig::getSecondaryOffloadMinPriority() const
{
    return mSecondaryOffloadMinPriority;
}

size_t KvCacheConfig::getEventBufferMaxSize() const
{
    return mEventBufferMaxSize;
}

void KvCacheConfig::setEnableBlockReuse(bool enableBlockReuse)
{
    mEnableBlockReuse = enableBlockReuse;
}

void KvCacheConfig::setMaxTokens(SizeType32 maxTokens)
{
    TLLM_CHECK(maxTokens > 0);
    mMaxTokens = maxTokens;
}

void KvCacheConfig::setMaxAttentionWindowVec(std::vector<SizeType32> maxAttentionWindowVec)
{
    for (SizeType32 maxAttentionWindow : maxAttentionWindowVec)
    {
        TLLM_CHECK(maxAttentionWindow > 0);
    }
    mMaxAttentionWindowVec = maxAttentionWindowVec;
}

void KvCacheConfig::setSinkTokenLength(SizeType32 sinkTokenLength)
{
    TLLM_CHECK(sinkTokenLength > 0);
    mSinkTokenLength = sinkTokenLength;
}

void KvCacheConfig::setFreeGpuMemoryFraction(FloatType freeGpuMemoryFraction)
{
    TLLM_CHECK(freeGpuMemoryFraction > 0.F);
    TLLM_CHECK(freeGpuMemoryFraction < 1.F);
    mFreeGpuMemoryFraction = freeGpuMemoryFraction;
}

void KvCacheConfig::setCrossKvCacheFraction(FloatType crossKvCacheFraction)

{
    TLLM_CHECK(crossKvCacheFraction > 0.F);
    TLLM_CHECK(crossKvCacheFraction < 1.F);
    mCrossKvCacheFraction = crossKvCacheFraction;
}

void KvCacheConfig::setHostCacheSize(size_t hostCacheSize)
{
    mHostCacheSize = hostCacheSize;
}

void KvCacheConfig::setOnboardBlocks(bool onboardBlocks)
{
    mOnboardBlocks = onboardBlocks;
}

void KvCacheConfig::setSecondaryOffloadMinPriority(std::optional<RetentionPriority> secondaryOffloadMinPriority)
{
    mSecondaryOffloadMinPriority = secondaryOffloadMinPriority;
}

void KvCacheConfig::setEventBufferMaxSize(size_t eventBufferMaxSize)
{
    mEventBufferMaxSize = eventBufferMaxSize;
}

void KvCacheConfig::fillEmptyFieldsFromRuntimeDefaults(tensorrt_llm::runtime::RuntimeDefaults runtimeDefaults)
{
    if (!mMaxAttentionWindowVec && runtimeDefaults.maxAttentionWindowVec)
    {
        setMaxAttentionWindowVec(runtimeDefaults.maxAttentionWindowVec.value());
    }
    if (!mSinkTokenLength && runtimeDefaults.sinkTokenLength)
    {
        setSinkTokenLength(runtimeDefaults.sinkTokenLength.value());
    }
}

} // namespace tensorrt_llm::executor
