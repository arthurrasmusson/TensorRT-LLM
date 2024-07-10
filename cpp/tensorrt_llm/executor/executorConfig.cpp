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

#include <utility>

namespace tensorrt_llm::executor
{

ExecutorConfig::ExecutorConfig(SizeType32 maxBeamWidth, SchedulerConfig const& schedulerConfig,
    KvCacheConfig const& kvCacheConfig, bool enableChunkedContext, bool normalizeLogProbs,
    SizeType32 iterStatsMaxIterations, SizeType32 requestStatsMaxIterations, BatchingType batchingType,
    std::optional<SizeType32> maxBatchSize, std::optional<SizeType32> maxNumTokens,
    std::optional<ParallelConfig> parallelConfig, std::optional<PeftCacheConfig> const& peftCacheConfig,
    std::optional<LogitsPostProcessorMap> logitsPostProcessorMap,
    std::optional<LogitsPostProcessorBatched> logitsPostProcessorBatched, std::optional<DecodingConfig> decodingConfig,
    float gpuWeightPercent)
    : mMaxBeamWidth(maxBeamWidth)
    , mSchedulerConfig(schedulerConfig)
    , mKvCacheConfig(kvCacheConfig)
    , mEnableChunkedContext(enableChunkedContext)
    , mNormalizeLogProbs(normalizeLogProbs)
    , mIterStatsMaxIterations(iterStatsMaxIterations)
    , mRequestStatsMaxIterations(requestStatsMaxIterations)
    , mBatchingType(batchingType)
    , mMaxBatchSize(maxBatchSize)
    , mMaxNumTokens(maxNumTokens)
    , mParallelConfig(std::move(parallelConfig))
    , mPeftCacheConfig(peftCacheConfig)
    , mLogitsPostProcessorMap(std::move(logitsPostProcessorMap))
    , mLogitsPostProcessorBatched(std::move(logitsPostProcessorBatched))
    , mDecodingConfig(std::move(decodingConfig))
    , mGpuWeightsPercent(gpuWeightPercent)
{
    TLLM_CHECK(iterStatsMaxIterations >= 0);
    TLLM_CHECK(requestStatsMaxIterations >= 0);
    TLLM_CHECK(mMaxBeamWidth > 0);
}

SizeType32 ExecutorConfig::getMaxBeamWidth() const
{
    return mMaxBeamWidth;
}

SchedulerConfig ExecutorConfig::getSchedulerConfig() const
{
    return mSchedulerConfig;
}

KvCacheConfig ExecutorConfig::getKvCacheConfig() const
{
    return mKvCacheConfig;
}

bool ExecutorConfig::getEnableChunkedContext() const
{
    return mEnableChunkedContext;
}

bool ExecutorConfig::getNormalizeLogProbs() const
{
    return mNormalizeLogProbs;
}

SizeType32 ExecutorConfig::getIterStatsMaxIterations() const
{
    return mIterStatsMaxIterations;
}

SizeType32 ExecutorConfig::getRequestStatsMaxIterations() const
{
    return mRequestStatsMaxIterations;
}

BatchingType ExecutorConfig::getBatchingType() const
{
    return mBatchingType;
}

std::optional<SizeType32> ExecutorConfig::getMaxBatchSize() const
{
    return mMaxBatchSize;
}

std::optional<SizeType32> ExecutorConfig::getMaxNumTokens() const
{
    return mMaxNumTokens;
}

std::optional<ParallelConfig> ExecutorConfig::getParallelConfig() const
{
    return mParallelConfig;
}

std::optional<PeftCacheConfig> ExecutorConfig::getPeftCacheConfig() const
{
    return mPeftCacheConfig;
}

std::optional<LogitsPostProcessorMap> ExecutorConfig::getLogitsPostProcessorMap() const
{
    return mLogitsPostProcessorMap;
}

std::optional<LogitsPostProcessorBatched> ExecutorConfig::getLogitsPostProcessorBatched() const
{
    return mLogitsPostProcessorBatched;
}

std::optional<DecodingConfig> ExecutorConfig::getDecodingConfig() const
{
    return mDecodingConfig;
}

float ExecutorConfig::getGpuWeightsPercent() const
{
    return mGpuWeightsPercent;
}

void ExecutorConfig::setMaxBeamWidth(SizeType32 maxBeamWidth)
{
    mMaxBeamWidth = maxBeamWidth;
    TLLM_CHECK(mMaxBeamWidth > 0);
}

void ExecutorConfig::setMaxBatchSize(SizeType32 maxBatchSize)
{
    mMaxBatchSize = maxBatchSize;
    TLLM_CHECK(mMaxBatchSize > 0);
}

void ExecutorConfig::setMaxNumTokens(SizeType32 maxNumTokens)
{
    mMaxNumTokens = maxNumTokens;
    TLLM_CHECK(mMaxNumTokens > 0);
}

void ExecutorConfig::setSchedulerConfig(SchedulerConfig const& schedulerConfig)
{
    mSchedulerConfig = schedulerConfig;
}

void ExecutorConfig::setKvCacheConfig(KvCacheConfig const& kvCacheConfig)
{
    mKvCacheConfig = kvCacheConfig;
}

void ExecutorConfig::setEnableChunkedContext(bool enableChunkedContext)
{
    mEnableChunkedContext = enableChunkedContext;
}

void ExecutorConfig::setNormalizeLogProbs(bool normalizeLogProbs)
{
    mNormalizeLogProbs = normalizeLogProbs;
}

void ExecutorConfig::setIterStatsMaxIterations(SizeType32 iterStatsMaxIterations)
{
    mIterStatsMaxIterations = iterStatsMaxIterations;
    TLLM_CHECK(mIterStatsMaxIterations >= 0);
}

void ExecutorConfig::setRequestStatsMaxIterations(SizeType32 requestStatsMaxIterations)
{
    mRequestStatsMaxIterations = requestStatsMaxIterations;
    TLLM_CHECK(mRequestStatsMaxIterations >= 0);
}

void ExecutorConfig::setBatchingType(BatchingType batchingType)
{
    mBatchingType = batchingType;
}

void ExecutorConfig::setParallelConfig(ParallelConfig const& parallelConfig)
{
    mParallelConfig = parallelConfig;
}

void ExecutorConfig::setPeftCacheConfig(PeftCacheConfig const& peftCacheConfig)
{
    mPeftCacheConfig = peftCacheConfig;
}

void ExecutorConfig::setLogitsPostProcessorMap(LogitsPostProcessorMap const& logitsPostProcessorMap)
{
    mLogitsPostProcessorMap = logitsPostProcessorMap;
}

void ExecutorConfig::setLogitsPostProcessorBatched(LogitsPostProcessorBatched const& logitsPostProcessorBatched)
{
    mLogitsPostProcessorBatched = logitsPostProcessorBatched;
}

void ExecutorConfig::setDecodingConfig(DecodingConfig const& decodingConfig)
{
    mDecodingConfig = decodingConfig;
}

void ExecutorConfig::setGpuWeightsPercent(float const& gpuWeightsPercent)
{
    mGpuWeightsPercent = gpuWeightsPercent;
}

} // namespace tensorrt_llm::executor
