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

ExtendedRuntimePerfKnobConfig::ExtendedRuntimePerfKnobConfig(bool multiBlockMode, bool enableContextFMHAFP32Acc)
    : mMultiBlockMode(multiBlockMode)
    , mEnableContextFMHAFP32Acc(enableContextFMHAFP32Acc)
{
}

bool ExtendedRuntimePerfKnobConfig::getMultiBlockMode() const
{
    return mMultiBlockMode;
}

bool ExtendedRuntimePerfKnobConfig::getEnableContextFMHAFP32Acc() const
{
    return mEnableContextFMHAFP32Acc;
}

void ExtendedRuntimePerfKnobConfig::setMultiBlockMode(bool multiBlockMode)
{
    mMultiBlockMode = multiBlockMode;
}

void ExtendedRuntimePerfKnobConfig::setEnableContextFMHAFP32Acc(bool enableContextFMHAFP32Acc)
{
    mEnableContextFMHAFP32Acc = enableContextFMHAFP32Acc;
}

} // namespace tensorrt_llm::executor
