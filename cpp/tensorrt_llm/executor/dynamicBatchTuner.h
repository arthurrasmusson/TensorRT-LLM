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

#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/types.h"

#include <deque>

namespace tensorrt_llm::executor
{

/// @brief A class that maintains runtime input and output length statistics and computes runtime dynamic batch size.
class DynamicBatchTuner
{
public:
    explicit DynamicBatchTuner(DynamicBatchConfig const& config);

    /// @brief Check if dynamic batch size tuning is enabled.
    [[nodiscard]] bool isBatchSizeTuningEnabled() const
    {
        return mEnableBatchSizeTuning;
    }

    /// @brief Update current stats given the input and output length from a single request.
    void updateStats(SizeType32 inputLen, SizeType32 outputLen);

    /// @brief Get average input length.
    [[nodiscard]] double getAverageInputLength() const;

    /// @brief Get average output length.
    [[nodiscard]] double getAverageOutputLength() const;

    /// @brief Get the dynamic batch size based on the current statistics.
    [[nodiscard]] SizeType32 getRuntimeBatchSize(SizeType32 maxCapacityBatchSize) const;

private:
    bool mEnableBatchSizeTuning = false;

    SizeType32 mDynamicBatchMovingAverageWindow = 0;

    std::vector<std::pair<SizeType32, SizeType32>> mBatchSizeTable;

    int64_t mInputLengthSum = 0;
    std::deque<SizeType32> mInputLengthStats;

    int64_t mOutputLengthSum = 0;
    std::deque<SizeType32> mOutputLengthStats;

    static SizeType32 const kBatchSizeFallbackGranularity = 512;
    static SizeType32 const kBatchSizeFallbackThreshold = 128;
};

} // namespace tensorrt_llm::executor
