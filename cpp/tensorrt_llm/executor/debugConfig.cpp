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

DebugConfig::DebugConfig(bool dumpInputTensors, bool dumpOuputTensors, StringVec debugTensorNames)
    : mDumpInputTensors{dumpInputTensors}
    , mDumpOuputTensors{dumpOuputTensors}
    , mDebugTensorNames{std::move(debugTensorNames)}
{
}

bool DebugConfig::operator==(DebugConfig const& other) const
{
    return mDumpInputTensors == other.mDumpInputTensors && mDumpOuputTensors == other.mDumpOuputTensors
        && mDebugTensorNames == other.mDebugTensorNames;
}

[[nodiscard]] bool DebugConfig::getDumpInputTensors() const
{
    return mDumpInputTensors;
}

[[nodiscard]] bool DebugConfig::getDumpOutputTensors() const
{
    return mDumpOuputTensors;
}

[[nodiscard]] DebugConfig::StringVec const& DebugConfig::getDebugTensorNames() const
{
    return mDebugTensorNames;
}

void DebugConfig::setDumpInputTensors(bool dumpInputTensors)
{
    mDumpInputTensors = dumpInputTensors;
}

void DebugConfig::setDumpOuputTensors(bool dumpOuputTensors)
{
    mDumpOuputTensors = dumpOuputTensors;
}

void DebugConfig::setDebugTensorNames(StringVec const& debugTensorNames)
{
    mDebugTensorNames = debugTensorNames;
}

} // namespace tensorrt_llm::executor
