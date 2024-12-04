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

#pragma once

#include "dataTransceiver.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/kvCacheUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/cache_transmission/cacheConcatenate.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include <NvInferRuntimeBase.h>
#include <cstddef>
#include <cstdint>
#include <iterator>

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

// Simple cache block copy. Because it does not involve data splitting or merging, it performs best when the
// parallel topology is completely identical, making it the preferred method.
class CacheFormatter final : public IOFormatter
{
public:
    using CacheState = executor::kv_cache::CacheState;

    CacheFormatter(BaseKVCacheManager* cacheManager)
        : mCacheManager{cacheManager}
    {
        TLLM_CHECK(mCacheManager);
    }

    void formatOutput(executor::kv_cache::Communicator const& comm, LlmRequest const& llmRequest,
        std::vector<executor::kv_cache::ProcessInfo> const& processInfos, CacheState const& selfConfig,
        SizeType32 selfIdx, CacheState const& destConfig) override;

    void formatInput(executor::kv_cache::Communicator const& comm, LlmRequest const& llmRequest,
        std::vector<executor::kv_cache::ProcessInfo> const& processInfos, CacheState const& selfConfig,
        SizeType32 selfIdx, CacheState const& destConfig, runtime::BufferManager const& bufferManager) override;

    [[nodiscard]] bool inquireSupport(CacheState const& selfConfig, CacheState const& destConfig) const override;

    [[nodiscard]] std::vector<SizeType32> getCounterparts(
        CacheState const& selfConfig, SizeType32 selfIdx, CacheState const& destConfig) const override
    {
        return executor::kv_cache::targetIRanks(destConfig, selfConfig, selfIdx);
    }

private:
    BaseKVCacheManager* mCacheManager{};
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager
