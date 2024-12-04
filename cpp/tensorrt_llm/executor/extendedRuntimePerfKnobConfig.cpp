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

namespace
{
void validateCudaGraphCacheSize(bool cudaGraphMode, tensorrt_llm::executor::SizeType32 cudaGraphCacheSize)
{
    TLLM_CHECK_WITH_INFO(cudaGraphCacheSize >= 0, "CUDA graph cache size must be greater or equal to 0.");
    if (!cudaGraphMode && cudaGraphCacheSize > 0)
    {
        TLLM_LOG_WARNING(
            "Setting cudaGraphCacheSize to a value greater than 0 without enabling cudaGraphMode has no effect.");
    }
}

} // namespace

namespace tensorrt_llm::executor
{

ExtendedRuntimePerfKnobConfig::ExtendedRuntimePerfKnobConfig(
    bool multiBlockMode, bool enableContextFMHAFP32Acc, bool cudaGraphMode, SizeType32 cudaGraphCacheSize)
    : mMultiBlockMode(multiBlockMode)
    , mEnableContextFMHAFP32Acc(enableContextFMHAFP32Acc)
    , mCudaGraphMode(cudaGraphMode)
    , mCudaGraphCacheSize(cudaGraphCacheSize)
{
    validateCudaGraphCacheSize(mCudaGraphMode, mCudaGraphCacheSize);
}

bool ExtendedRuntimePerfKnobConfig::getMultiBlockMode() const
{
    return mMultiBlockMode;
}

bool ExtendedRuntimePerfKnobConfig::getEnableContextFMHAFP32Acc() const
{
    return mEnableContextFMHAFP32Acc;
}

bool ExtendedRuntimePerfKnobConfig::getCudaGraphMode() const
{
    return mCudaGraphMode;
}

SizeType32 ExtendedRuntimePerfKnobConfig::getCudaGraphCacheSize() const
{
    return mCudaGraphCacheSize;
}

void ExtendedRuntimePerfKnobConfig::setMultiBlockMode(bool multiBlockMode)
{
    mMultiBlockMode = multiBlockMode;
}

void ExtendedRuntimePerfKnobConfig::setEnableContextFMHAFP32Acc(bool enableContextFMHAFP32Acc)
{
    mEnableContextFMHAFP32Acc = enableContextFMHAFP32Acc;
}

void ExtendedRuntimePerfKnobConfig::setCudaGraphMode(bool cudaGraphMode)
{
    mCudaGraphMode = cudaGraphMode;
}

void ExtendedRuntimePerfKnobConfig::setCudaGraphCacheSize(SizeType32 cudaGraphCacheSize)
{
    mCudaGraphCacheSize = cudaGraphCacheSize;
    validateCudaGraphCacheSize(mCudaGraphMode, mCudaGraphCacheSize);
}

} // namespace tensorrt_llm::executor
