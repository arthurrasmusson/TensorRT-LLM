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
#include <cstddef>
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

    void operator()(LlmRequest const& llmRequest, typename TComm::TPtrContainer const& srcs,
        executor::kv_cache::CacheState const& selfconfig, SizeType32 selfIdx,
        executor::kv_cache::CacheState const& destConfig) override
    {

        TLLM_CHECK_WITH_INFO(llmRequest.mSamplingConfig.beamWidth == 1, "Currently only supports beam width 1.");
        constexpr SizeType32 beam{0};
        std::vector<runtime::ITensor::SharedPtr> recvBufferTmps;
        std::vector<runtime::ITensor::SharedPtr> outputBuffers;
        auto const numPools = mCacheManager->getBlockManager().getNumPools();
        // TODO(oargov): are we sure the other side has the same number of pools? this might not hold for pp_size>1...
        for (auto poolIdx = 0; poolIdx < numPools; poolIdx++)
        {
            auto const endIt = getBlockEndIt(*mCacheManager, llmRequest, beam, poolIdx);
            for (auto it = getBlockBeginIt(*mCacheManager, llmRequest, beam, poolIdx); it != endIt; ++it)
            {
                for (size_t i = 0; i < srcs.size(); i++)
                {
                    recvBufferTmps.emplace_back(mCacheManager->getBlockManager().getBufferManager().gpu(
                        executor::kv_cache::makeShapeFromCacheState(destConfig), it->getDataType()));
                }
            }
        }
        // sync to alloc buffer
        TLLM_CUDA_CHECK(cudaStreamSynchronize(mCacheManager->getBlockManager().getBufferManager().getStream().get()));

        int idx = 0;
        for (auto poolIdx = 0; poolIdx < numPools; poolIdx++)
        {
            auto const endIt = getBlockEndIt(*mCacheManager, llmRequest, beam, poolIdx);
            for (auto it = getBlockBeginIt(*mCacheManager, llmRequest, beam, poolIdx); it != endIt; ++it)
            {
                for (auto&& src : srcs)
                {
                    src->recvBuffer(*recvBufferTmps[idx]);
                    idx++;
                }
                outputBuffers.push_back(it);
            }
        }

        executor::kv_cache::concatenateKVCacheDispatch(recvBufferTmps.data(), recvBufferTmps.size(),
            getCounterparts(selfconfig, selfIdx, destConfig), destConfig, outputBuffers.data(), outputBuffers.size(),
            selfIdx, selfconfig, mCacheManager->getBlockManager().getBufferManager());
        TLLM_CUDA_CHECK(cudaStreamSynchronize(mCacheManager->getBlockManager().getBufferManager().getStream().get()));
    }

    [[nodiscard]] bool inquireSupport(executor::kv_cache::CacheState const& selfconfig,
        executor::kv_cache::CacheState const& destConfig) const override
    {
        std::unordered_set<SizeType32> setVecSelf{selfconfig.getModelConfig().mNbKvHeadsPerLayer.begin(),
            selfconfig.getModelConfig().mNbKvHeadsPerLayer.end()};

        if (setVecSelf.size() != 1)
        {
            return false;
        }
        std::unordered_set<int> setVecDest{destConfig.getModelConfig().mNbKvHeadsPerLayer.begin(),
            destConfig.getModelConfig().mNbKvHeadsPerLayer.end()};

        if (setVecDest.size() != 1)
        {
            return false;
        }
        if (selfconfig.getModelConfig().mTokensPerBlock != destConfig.getModelConfig().mTokensPerBlock
            || selfconfig.getModelConfig().mSizePerHead != destConfig.getModelConfig().mSizePerHead)
        {

            return false;
        }
        if (selfconfig.getModelConfig().mNbKvHeadsPerLayer.size()
            != destConfig.getModelConfig().mNbKvHeadsPerLayer.size())
        {
            return false;
        }
        if (std::getenv("TLLM_USE_UCX_KVCACHE"))
        {
            if ((selfconfig.getParallelConfig().mPipelineParallelism
                    != destConfig.getParallelConfig().mPipelineParallelism)
                || (selfconfig.getParallelConfig().mTensorParallelism
                    != destConfig.getParallelConfig().mTensorParallelism))
            {
                TLLM_LOG_WARNING("Only symmetric parallelism is supported with UCX_KVCACHE_TRANSFER");
                return false;
            }
        }

        int selfNumHeads
            = selfconfig.getModelConfig().mNbKvHeadsPerLayer[0] * selfconfig.getParallelConfig().mTensorParallelism;
        int destNumHeads
            = destConfig.getModelConfig().mNbKvHeadsPerLayer[0] * destConfig.getParallelConfig().mTensorParallelism;
        return selfNumHeads == destNumHeads;
    }

    [[nodiscard]] std::vector<SizeType32> getCounterparts(executor::kv_cache::CacheState const& selfconfig,
        SizeType32 selfIdx, executor::kv_cache::CacheState const& destConfig) const override
    {
        return executor::kv_cache::targetIRanks(destConfig, selfconfig, selfIdx);
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

    void operator()(LlmRequest const& llmRequest, typename TComm::TPtrContainer const& dsts,
        executor::kv_cache::CacheState const& selfconfig, SizeType32 selfIdx,
        executor::kv_cache::CacheState const& destConfig) override
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
        std::unordered_set<SizeType32> setVecSelf{selfconfig.getModelConfig().mNbKvHeadsPerLayer.begin(),
            selfconfig.getModelConfig().mNbKvHeadsPerLayer.end()};

        if (setVecSelf.size() != 1)
        {
            return false;
        }
        std::unordered_set<int> setVecDest{destConfig.getModelConfig().mNbKvHeadsPerLayer.begin(),
            destConfig.getModelConfig().mNbKvHeadsPerLayer.end()};

        if (setVecDest.size() != 1)
        {
            return false;
        }
        if (selfconfig.getModelConfig().mTokensPerBlock != destConfig.getModelConfig().mTokensPerBlock
            || selfconfig.getModelConfig().mSizePerHead != destConfig.getModelConfig().mSizePerHead)
        {

            return false;
        }
        if (selfconfig.getModelConfig().mNbKvHeadsPerLayer.size()
            != destConfig.getModelConfig().mNbKvHeadsPerLayer.size())
        {
            return false;
        }

        if (std::getenv("TLLM_USE_UCX_KVCACHE"))
        {
            if ((selfconfig.getParallelConfig().mPipelineParallelism
                    != destConfig.getParallelConfig().mPipelineParallelism)
                || (selfconfig.getParallelConfig().mTensorParallelism
                    != destConfig.getParallelConfig().mTensorParallelism))
            {
                TLLM_LOG_WARNING("Only symmetric parallelism is supported with UCX_KVCACHE_TRANSFER");
                return false;
            }
        }
        int selfNumHeads
            = selfconfig.getModelConfig().mNbKvHeadsPerLayer[0] * selfconfig.getParallelConfig().mTensorParallelism;
        int destNumHeads
            = destConfig.getModelConfig().mNbKvHeadsPerLayer[0] * destConfig.getParallelConfig().mTensorParallelism;
        return selfNumHeads == destNumHeads;
    }

    [[nodiscard]] std::vector<SizeType32> getCounterparts(executor::kv_cache::CacheState const& selfconfig,
        SizeType32 selfIdx, executor::kv_cache::CacheState const& destConfig) const override
    {
        return executor::kv_cache::targetIRanks(destConfig, selfconfig, selfIdx);
    }

private:
    KVCacheManager* mCacheManager{};
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager
