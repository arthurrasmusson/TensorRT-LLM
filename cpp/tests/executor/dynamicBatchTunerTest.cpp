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
#include "tensorrt_llm/common/tllmException.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/types.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using ::testing::_;
using ::testing::Invoke;

using namespace tensorrt_llm::executor;
using namespace tensorrt_llm::common;

TEST(DynamicBatchTunerTest, Stats)
{
    // moving average window size is 3
    DynamicBatchConfig dynamicBatchConfig(true, 3);
    DynamicBatchTuner dynamicBatchTuner(dynamicBatchConfig);

    // check no division by zero issue
    EXPECT_EQ(dynamicBatchTuner.getAverageInputLength(), 0);
    EXPECT_EQ(dynamicBatchTuner.getAverageOutputLength(), 0);

    dynamicBatchTuner.updateStats(1, 2);
    EXPECT_EQ(dynamicBatchTuner.getAverageInputLength(), 1);
    EXPECT_EQ(dynamicBatchTuner.getAverageOutputLength(), 2);

    dynamicBatchTuner.updateStats(2, 3);
    EXPECT_EQ(dynamicBatchTuner.getAverageInputLength(), 1.5);
    EXPECT_EQ(dynamicBatchTuner.getAverageOutputLength(), 2.5);

    dynamicBatchTuner.updateStats(3, 4);
    EXPECT_EQ(dynamicBatchTuner.getAverageInputLength(), 2);
    EXPECT_EQ(dynamicBatchTuner.getAverageOutputLength(), 3);

    // check that the first element is removed from the moving average window
    dynamicBatchTuner.updateStats(4, 5);
    EXPECT_EQ(dynamicBatchTuner.getAverageInputLength(), 3);
    EXPECT_EQ(dynamicBatchTuner.getAverageOutputLength(), 4);
}
