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
        executor::kv_cache::CacheState const& destConfig, runtime::BufferManager const& bufferManager) override
    {

        TLLM_CHECK_WITH_INFO(llmRequest.mSamplingConfig.beamWidth == 1, "Currently only supports beam width 1.");
        constexpr SizeType32 beam{0};
        std::vector<runtime::ITensor::SharedPtr> recvBufferTmps;
        std::vector<runtime::ITensor::SharedPtr> outputBuffers;
        auto const numPools = mCacheManager->getBlockManager().getNumPools();
        // TODO(oargov): are we sure the other side has the same number of pools? this might not hold for pp_size>1...
        int blockNum = 0;
        for (auto poolIdx = 0; poolIdx < numPools; poolIdx++)
        {
            auto const endIt = getBlockEndIt(*mCacheManager, llmRequest, beam, poolIdx);
            for (auto it = getBlockBeginIt(*mCacheManager, llmRequest, beam, poolIdx); it != endIt; ++it)
            {
                blockNum++;
            }
        }
        auto cacheShape = executor::kv_cache::makeShapeFromCacheState(destConfig);

        size_t bufferNum = blockNum * srcs.size();
        auto dataType = getBlockBeginIt(*mCacheManager, llmRequest, beam, 0)->getDataType();
        runtime::ITensor::SharedPtr recvBufferTemp = bufferManager.gpu(
            runtime::ITensor::makeShape(
                {static_cast<long>(tensorrt_llm::runtime::ITensor::volume(cacheShape) * bufferNum)}),
            dataType);
        recvBufferTmps.resize(bufferNum);
        for (size_t i = 0; i < bufferNum; i++)
        {
            recvBufferTmps[i]
                = runtime::ITensor::slice(recvBufferTemp, i * tensorrt_llm::runtime::ITensor::volume(cacheShape),
                    tensorrt_llm::runtime::ITensor::volume(cacheShape));
        }
        // sync to alloc buffer
        bufferManager.getStream().synchronize();

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
            selfIdx, selfconfig, bufferManager);
        bufferManager.getStream().synchronize();
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
        executor::kv_cache::CacheState const& destConfig, runtime::BufferManager const& /*bufferManager*/) override
    {
        TLLM_CHECK_WITH_INFO(llmRequest.mSamplingConfig.beamWidth == 1, "Currently only supports beam width 1.");
        constexpr SizeType32 beam{0};
        auto const numPools = mCacheManager->getBlockManager().getNumPools();
        // TODO(oargov): are we sure the other side has the same number of pools? this might not hold for pp_size>1...
        for (auto poolIdx = 0; poolIdx < numPools; poolIdx++)
        {
            auto const endIt = getBlockEndIt(*mCacheManager, llmRequest, beam, poolIdx);
            for (auto it = getBlockBeginIt(*mCacheManager, llmRequest, beam, poolIdx); it != endIt; ++it)
            {
                for (auto&& dst : dsts)
                {
                    dst->sendBuffer(*it);
                }
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
