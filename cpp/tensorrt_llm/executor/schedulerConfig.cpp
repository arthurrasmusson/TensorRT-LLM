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

SchedulerConfig::SchedulerConfig(CapacitySchedulerPolicy capacitySchedulerPolicy,
    std::optional<ContextChunkingPolicy> contextChunkingPolicy, std::optional<DynamicBatchConfig> dynamicBatchConfig)
    : mCapacitySchedulerPolicy(capacitySchedulerPolicy)
    , mContextChunkingPolicy(std::move(contextChunkingPolicy))
    , mDynamicBatchConfig(std::move(dynamicBatchConfig))
{
}

bool SchedulerConfig::operator==(SchedulerConfig const& other) const
{
    return mCapacitySchedulerPolicy == other.mCapacitySchedulerPolicy
        && mContextChunkingPolicy == other.mContextChunkingPolicy;
}

[[nodiscard]] CapacitySchedulerPolicy SchedulerConfig::getCapacitySchedulerPolicy() const
{
    return mCapacitySchedulerPolicy;
}

[[nodiscard]] std::optional<ContextChunkingPolicy> SchedulerConfig::getContextChunkingPolicy() const
{
    return mContextChunkingPolicy;
}

[[nodiscard]] std::optional<DynamicBatchConfig> SchedulerConfig::getDynamicBatchConfig() const
{
    return mDynamicBatchConfig;
}

} // namespace tensorrt_llm::executor
