/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

DebugConfig::DebugConfig(
    bool debugInputTensors, bool debugOutputTensors, StringVec debugTensorNames, SizeType32 debugTensorsMaxIterations)
    : mDebugInputTensors{debugInputTensors}
    , mDebugOutputTensors{debugOutputTensors}
    , mDebugTensorNames{std::move(debugTensorNames)}
    , mDebugTensorsMaxIterations{debugTensorsMaxIterations}
{
}

bool DebugConfig::operator==(DebugConfig const& other) const
{
    return mDebugInputTensors == other.mDebugInputTensors && mDebugOutputTensors == other.mDebugOutputTensors
        && mDebugTensorNames == other.mDebugTensorNames
        && mDebugTensorsMaxIterations == other.mDebugTensorsMaxIterations;
}

[[nodiscard]] bool DebugConfig::getDebugInputTensors() const
{
    return mDebugInputTensors;
}

[[nodiscard]] bool DebugConfig::getDebugOutputTensors() const
{
    return mDebugOutputTensors;
}

[[nodiscard]] DebugConfig::StringVec const& DebugConfig::getDebugTensorNames() const
{
    return mDebugTensorNames;
}

[[nodiscard]] SizeType32 DebugConfig::getDebugTensorsMaxIterations() const
{
    return mDebugTensorsMaxIterations;
}

void DebugConfig::setDebugInputTensors(bool debugInputTensors)
{
    mDebugInputTensors = debugInputTensors;
}

void DebugConfig::setDebugOutputTensors(bool debugOutputTensors)
{
    mDebugOutputTensors = debugOutputTensors;
}

void DebugConfig::setDebugTensorNames(StringVec const& debugTensorNames)
{
    mDebugTensorNames = debugTensorNames;
}

void DebugConfig::setDebugTensorsMaxIterations(SizeType32 debugTensorsMaxIterations)
{
    mDebugTensorsMaxIterations = debugTensorsMaxIterations;
}

} // namespace tensorrt_llm::executor
