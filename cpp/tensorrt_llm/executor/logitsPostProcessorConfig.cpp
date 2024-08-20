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

LogitsPostProcessorConfig::LogitsPostProcessorConfig(std::optional<LogitsPostProcessorMap> processorMap,
    std::optional<LogitsPostProcessorBatched> processorBatched, bool replicate)
    : mProcessorMap(std::move(processorMap))
    , mProcessorBatched(std::move(processorBatched))
    , mReplicate(replicate)
{
}

std::optional<LogitsPostProcessorMap> LogitsPostProcessorConfig::getProcessorMap() const
{
    return mProcessorMap;
}

std::optional<LogitsPostProcessorBatched> LogitsPostProcessorConfig::getProcessorBatched() const
{
    return mProcessorBatched;
}

bool LogitsPostProcessorConfig::getReplicate() const
{
    return mReplicate;
}

void LogitsPostProcessorConfig::setProcessorMap(LogitsPostProcessorMap const& processorMap)
{
    mProcessorMap = processorMap;
}

void LogitsPostProcessorConfig::setProcessorBatched(LogitsPostProcessorBatched const& processorBatched)
{
    mProcessorBatched = processorBatched;
}

void LogitsPostProcessorConfig::setReplicate(bool replicate)
{
    mReplicate = replicate;
}

} // namespace tensorrt_llm::executor
