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
#include "tensorrt_llm/executor/types.h"

#include <optional>

namespace tensorrt_llm::executor
{
LoraConfig::LoraConfig(IdType taskId, std::optional<Tensor> weights, std::optional<Tensor> config)
    : mTaskId(taskId)
    , mWeights(std::move(weights))
    , mConfig(std::move(config))
{
    if (mWeights.has_value() || mConfig.has_value())
    {
        TLLM_CHECK_WITH_INFO(mWeights.has_value() && mConfig.has_value(),
            "Request for LoRA inference must have both lora weights and lora config");

        SizeType32 constexpr expectedWeightsDims = 2;
        SizeType32 constexpr expectedConfigDims = 2;

        TLLM_CHECK_WITH_INFO(
            mWeights.value().getShape().size() == expectedWeightsDims, "Expected weights tensor to have 2 dimensions");
        TLLM_CHECK_WITH_INFO(
            mConfig.value().getShape().size() == expectedConfigDims, "Expected config tensor to have 2 dimensions");
        TLLM_CHECK_WITH_INFO(mWeights.value().getMemoryType() != MemoryType::kGPU
                && mWeights.value().getMemoryType() != MemoryType::kUNKNOWN,
            "Expected lora weights to be in CPU memory");
        TLLM_CHECK_WITH_INFO(mConfig.value().getMemoryType() != MemoryType::kGPU
                && mConfig.value().getMemoryType() != MemoryType::kUNKNOWN,
            "Expected lora weights to be in CPU memory");
        TLLM_CHECK_WITH_INFO(
            mConfig.value().getDataType() == DataType::kINT32, "Expected lora config tensor to have type kINT32");

        TLLM_CHECK_WITH_INFO(mConfig.value().getShape()[0] == mWeights.value().getShape()[0],
            "Expected dim 0 of lora weights and lora config to have the same size");
    }
}

IdType LoraConfig::getTaskId() const
{
    return mTaskId;
}

std::optional<Tensor> LoraConfig::getWeights() const
{
    return mWeights;
}

std::optional<Tensor> LoraConfig::getConfig() const
{
    return mConfig;
}

} // namespace tensorrt_llm::executor
