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

namespace tensorrt_llm::executor
{

DynamicBatchConfig::DynamicBatchConfig(bool enableBatchSizeTuning, SizeType32 dynamicBatchMovingAverageWindow,
    std::vector<std::pair<SizeType32, SizeType32>> batchSizeTable)
    : mEnableBatchSizeTuning(enableBatchSizeTuning)
    , mDynamicBatchMovingAverageWindow(dynamicBatchMovingAverageWindow)
    , mBatchSizeTable(batchSizeTable)
{
}

bool DynamicBatchConfig::getEnableBatchSizeTuning() const
{
    return mEnableBatchSizeTuning;
}

SizeType32 DynamicBatchConfig::getDynamicBatchMovingAverageWindow() const
{
    return mDynamicBatchMovingAverageWindow;
}

std::vector<std::pair<SizeType32, SizeType32>> DynamicBatchConfig::getBatchSizeTable() const
{
    return mBatchSizeTable;
}

std::vector<std::pair<SizeType32, SizeType32>> const DynamicBatchConfig::kDefaultBatchSizeTable{
    {144, 128},
    {336, 256},
    {672, 512},
    {832, 768},
    {1280, 1024},
    {1664, 1536},
};

} // namespace tensorrt_llm::executor
