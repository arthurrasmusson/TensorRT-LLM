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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/types.h"
#include <filesystem>

#include <utility>

namespace tensorrt_llm::executor
{
ParallelConfig::ParallelConfig(CommunicationType commType, CommunicationMode commMode,
    std::optional<std::vector<SizeType32>> deviceIds, std::optional<std::vector<SizeType32>> participantIds,
    std::optional<OrchestratorConfig> const& orchestratorConfig)
    : mCommType(commType)
    , mCommMode(commMode)
    , mDeviceIds(std::move(deviceIds))
    , mParticipantIds(std::move(participantIds))
    , mOrchestratorConfig(orchestratorConfig)
{
    if (mDeviceIds)
    {
        TLLM_CHECK(!mDeviceIds.value().empty());
    }
}

CommunicationType ParallelConfig::getCommunicationType() const
{
    return mCommType;
}

CommunicationMode ParallelConfig::getCommunicationMode() const
{
    return mCommMode;
}

std::optional<std::vector<SizeType32>> ParallelConfig::getDeviceIds() const
{
    return mDeviceIds;
}

std::optional<std::vector<SizeType32>> ParallelConfig::getParticipantIds() const
{
    return mParticipantIds;
}

std::optional<OrchestratorConfig> ParallelConfig::getOrchestratorConfig() const
{
    return mOrchestratorConfig;
}

void ParallelConfig::setCommunicationType(CommunicationType type)
{
    mCommType = type;
}

void ParallelConfig::setCommunicationMode(CommunicationMode mode)
{
    mCommMode = mode;
}

void ParallelConfig::setDeviceIds(std::vector<SizeType32> const& deviceIds)
{
    TLLM_CHECK(!deviceIds.empty());
    mDeviceIds = deviceIds;
}

void ParallelConfig::setParticipantIds(std::vector<SizeType32> const& participantIds)
{
    mParticipantIds = participantIds;
}

void ParallelConfig::setOrchestratorConfig(OrchestratorConfig const& orchestratorConfig)
{
    mOrchestratorConfig = orchestratorConfig;
}

} // namespace tensorrt_llm::executor
