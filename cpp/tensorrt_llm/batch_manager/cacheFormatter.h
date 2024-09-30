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
#include "tensorrt_llm/batch_manager/kvCacheUtils.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include <iterator>

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

// Simple cache block copy. Because it does not involve data splitting or merging, it performs best when the
// parallel topology is completely identical, making it the preferred method.
template <typename TComm>
class CacheInputFormatter final : public IOFormatter<TComm, executor::kv_cache::CacheState>
{
public:
    CacheInputFormatter(KVCacheManager* cacheManager)
        : mCacheManager{cacheManager}
    {
        TLLM_CHECK(mCacheManager);
    }

    void operator()(LlmRequest const& llmRequest, typename TComm::TPtrContainer const& srcs) override
    {
        TLLM_CHECK(srcs.size() == 1);
        auto const& src = srcs.front();
        TLLM_CHECK_WITH_INFO(llmRequest.mSamplingConfig.beamWidth == 1, "Currently only supports beam width 1.");
        constexpr SizeType32 beam{0};
        auto const numPools = mCacheManager->getBlockManager().getNumPools();
        // TODO(oargov): are we sure the other side has the same number of pools? this might not hold for pp_size>1...
        for (auto poolIdx = 0; poolIdx < numPools; poolIdx++)
        {
            auto const endIt = getBlockEndIt(*mCacheManager, llmRequest, beam, poolIdx);
            for (auto it = getBlockBeginIt(*mCacheManager, llmRequest, beam, poolIdx); it != endIt; ++it)
            {
                src->recvBuffer(*it);
            }
        }
    }

    [[nodiscard]] bool inquireSupport(executor::kv_cache::CacheState const& selfconfig,
        executor::kv_cache::CacheState const& destConfig) const override
    {
        return selfconfig == destConfig;
    }

    [[nodiscard]] std::vector<SizeType32> getCounterparts(executor::kv_cache::CacheState const& selfconfig,
        SizeType32 selfIdx, executor::kv_cache::CacheState const& destConfig) const override
    {
        return {selfIdx};
    }

private:
    KVCacheManager* mCacheManager{};
};

// Simple cache block copy. Because it does not involve data splitting or merging, it performs best when the
// parallel topology is completely identical, making it the preferred method.
template <typename TComm>
class CacheOutputFormatter final : public IOFormatter<TComm, executor::kv_cache::CacheState>
{
public:
    CacheOutputFormatter(KVCacheManager* cacheManager)
        : mCacheManager{cacheManager}
    {
        TLLM_CHECK(mCacheManager);
    }

    void operator()(LlmRequest const& llmRequest, typename TComm::TPtrContainer const& dsts) override
    {
        TLLM_CHECK(dsts.size() == 1);
        auto const& dst = dsts.front();
        TLLM_CHECK_WITH_INFO(llmRequest.mSamplingConfig.beamWidth == 1, "Currently only supports beam width 1.");
        constexpr SizeType32 beam{0};
        auto const numPools = mCacheManager->getBlockManager().getNumPools();
        // TODO(oargov): are we sure the other side has the same number of pools? this might not hold for pp_size>1...
        for (auto poolIdx = 0; poolIdx < numPools; poolIdx++)
        {
            auto const endIt = getBlockEndIt(*mCacheManager, llmRequest, beam, poolIdx);
            for (auto it = getBlockBeginIt(*mCacheManager, llmRequest, beam, poolIdx); it != endIt; ++it)
            {
                dst->sendBuffer(*it);
            }
        }
    }

    [[nodiscard]] bool inquireSupport(executor::kv_cache::CacheState const& selfconfig,
        executor::kv_cache::CacheState const& destConfig) const override
    {
        return selfconfig == destConfig;
    }

    [[nodiscard]] std::vector<SizeType32> getCounterparts(executor::kv_cache::CacheState const& selfconfig,
        SizeType32 selfIdx, executor::kv_cache::CacheState const& destConfig) const override
    {
        return {selfIdx};
    }

private:
    KVCacheManager* mCacheManager{};
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager
