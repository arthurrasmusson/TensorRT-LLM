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

#include "tensorrt_llm/executor/dynamicBatchTuner.h"
#include "tensorrt_llm/common/logger.h"

namespace
{
using namespace tensorrt_llm::executor;

void updateStats(SizeType32 value, std::deque<SizeType32>& stats, int64_t& sum, SizeType32 windowSize)
{
    while (static_cast<SizeType32>(stats.size()) >= windowSize)
    {
        sum -= stats.front();
        stats.pop_front();
    }
    stats.push_back(value);
    sum += value;
}
} // namespace

namespace tensorrt_llm::executor
{

DynamicBatchTuner::DynamicBatchTuner(DynamicBatchConfig const& config)
    : mEnableBatchSizeTuning(config.getEnableBatchSizeTuning())
    , mDynamicBatchMovingAverageWindow(config.getDynamicBatchMovingAverageWindow())
    , mBatchSizeTable(config.getBatchSizeTable())
{
    TLLM_CHECK_WITH_INFO(mBatchSizeTable.size() > 0, "Batch size table is empty.");
    for (size_t i = 1; i < mBatchSizeTable.size(); ++i)
    {
        TLLM_CHECK_WITH_INFO(mBatchSizeTable[i - 1].first < mBatchSizeTable[i].first,
            "Batch size table is not sorted in ascending order.");
    }
}

void DynamicBatchTuner::updateStats(SizeType32 inputLength, SizeType32 outputLength)
{
    ::updateStats(inputLength, mInputLengthStats, mInputLengthSum, mDynamicBatchMovingAverageWindow);
    ::updateStats(outputLength, mOutputLengthStats, mOutputLengthSum, mDynamicBatchMovingAverageWindow);
}

double DynamicBatchTuner::getAverageInputLength() const
{
    return mInputLengthStats.empty() ? 0 : static_cast<double>(mInputLengthSum) / mInputLengthStats.size();
}

double DynamicBatchTuner::getAverageOutputLength() const
{
    return mOutputLengthStats.empty() ? 0 : static_cast<double>(mOutputLengthSum) / mOutputLengthStats.size();
}

SizeType32 DynamicBatchTuner::getRuntimeBatchSize(SizeType32 maxCapacityBatchSize) const
{
    for (size_t i = 0; i < mBatchSizeTable.size(); ++i)
    {
        if (maxCapacityBatchSize < mBatchSizeTable[i].first)
        {
            return mBatchSizeTable[i].second;
        }
    }
    SizeType32 threshold = maxCapacityBatchSize / kBatchSizeFallbackGranularity * kBatchSizeFallbackGranularity;
    if (maxCapacityBatchSize < (threshold + kBatchSizeFallbackThreshold))
    {
        return threshold;
    }
    return maxCapacityBatchSize;
}

} // namespace tensorrt_llm::executor
