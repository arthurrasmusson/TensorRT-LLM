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

#include "cacheFormatter.h"

#include "tensorrt_llm/batch_manager/contextProgress.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/runtime/cudaEvent.h"
#include <cstddef>
#include <cstdint>
#include <future>

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

void CacheFormatter::formatOutput(executor::kv_cache::Communicator const& comm, LlmRequest const& llmRequest,
    std::vector<executor::kv_cache::ProcessInfo> const& processInfos, CacheState const& selfConfig, SizeType32 selfIdx,
    CacheState const& destConfig)
{
    NVTX3_SCOPED_RANGE(formatOutput);
    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), "Start sending kvCache for request id:%ld ", llmRequest.mRequestId);

    TLLM_CHECK_WITH_INFO(llmRequest.mSamplingConfig.beamWidth == 1, "Currently only supports beam width 1.");
    TLLM_CHECK(!processInfos.empty());
    constexpr SizeType32 beam{0};
    auto const numPools = mCacheManager->getBlockManager().getNumPools();
    // TODO(oargov): are we sure the other side has the same number of pools? this might not hold for pp_size>1...

    bool layerWise = common::getEnvDisaggLayerwise() && numPools == 1;
    if (layerWise)
    {
        auto& progress = llmRequest.getContextProgress();
        SizeType32 const numLayers = mCacheManager->getBlockManager().getNumLayers();
        runtime::ITensor::Shape offset = runtime::ITensor::makeShape({0, 0});
        std::vector<SizeType32> layersInPool(numPools, 0);
        for (SizeType32 layerIdx = 0; layerIdx < numLayers; layerIdx++)
        {
            auto const poolIdx = mCacheManager->getBlockManager().getLayerPoolIdx(layerIdx);
            auto const layerIdxInPool = layersInPool[poolIdx]++;
            offset.d[1] = layerIdxInPool;
            if (progress != nullptr)
            {
                progress->wait(layerIdx);
            }
            auto const endIt = getBlockEndIt(*mCacheManager, llmRequest, beam, poolIdx);
            for (auto it = getBlockBeginIt(*mCacheManager, llmRequest, beam, poolIdx); it != endIt; ++it)
            {
                // Block dim: [1, numLayersInPool, ...], offset = {0, layerIndexInPool}
                auto layer = runtime::ITensor::slice(it, offset, 1);
                if (offset.d[1] == 0)
                {
                    TLLM_LOG_DEBUG("Block %p of pool %d shape = %s", it->data(), poolIdx,
                        runtime::ITensor::toString(it->getShape()).c_str());
                }
                for (auto const& processInfo : processInfos)
                {
                    TLLM_LOG_DEBUG("send layer %d(%d-%d)", layerIdx, poolIdx, layerIdxInPool);
                    comm.sendBuffer(*layer, executor::kv_cache::DataContext{llmRequest.mRequestId}, processInfo);
                }
            }
        }
    }
    else
    {

        auto sendBufferFun = [&](int deviceId, executor::kv_cache::ProcessInfo const& processInfo)
        {
            NVTX3_SCOPED_RANGE(sendBufferFun);

            TLLM_CUDA_CHECK(cudaSetDevice(deviceId));
            for (auto poolIdx = 0; poolIdx < numPools; poolIdx++)
            {
                auto const endIt = getBlockEndIt(*mCacheManager, llmRequest, beam, poolIdx);
                for (auto it = getBlockBeginIt(*mCacheManager, llmRequest, beam, poolIdx); it != endIt; ++it)
                {

                    comm.sendBuffer(*it, executor::kv_cache::DataContext{llmRequest.mRequestId}, processInfo);
                }
            }
            TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), "End sending kvCache for request id:%ld requestRank :%d ",
                llmRequest.mRequestId, processInfo.getRank());
        };

        int deviceId = mCacheManager->getBlockManager().getBufferManager().getStream().getDevice();
        if (processInfos.size() > 1)
        {
            if (common::getEnvDisableReceiveKVCacheParallel())
            {
                TLLM_LOG_INFO("Disable receiving kvCache in parallel");
                for (auto const& processInfo : processInfos)
                {
                    sendBufferFun(deviceId, processInfo);
                }
            }
            else
            {
                std::vector<std::future<void>> futures;
                futures.reserve(processInfos.size());
                for (auto const& processInfo : processInfos)
                {
                    futures.push_back(std::async(std::launch::async, sendBufferFun, deviceId, std::cref(processInfo)));
                }
                for (auto& future : futures)
                {
                    future.get();
                }
            }
        }
        else
        {
            sendBufferFun(deviceId, processInfos[0]);
        }
    }
    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), "End sending kvCache for request id:%ld ", llmRequest.mRequestId);
}

void CacheFormatter::formatInput(executor::kv_cache::Communicator const& comm, LlmRequest const& llmRequest,
    std::vector<executor::kv_cache::ProcessInfo> const& processInfos, CacheState const& selfConfig, SizeType32 selfIdx,
    CacheState const& destConfig, runtime::BufferManager const& bufferManager)
{
    NVTX3_SCOPED_RANGE(formatInput);
    TLLM_CHECK_WITH_INFO(llmRequest.mSamplingConfig.beamWidth == 1, "Currently only supports beam width 1.");
    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
        "Start receiving kvCache for request id:%ld , context request id:%ld", llmRequest.mRequestId,
        llmRequest.getContextPhaseParams().value().getReqId());
    TLLM_CHECK(!processInfos.empty());

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
            outputBuffers.push_back(it);
        }
    }
    // [numLayersInPool, ...]
    auto cacheShape = executor::kv_cache::makeShapeFromCacheState(destConfig);
    auto cacheVolume = runtime::ITensor::volume(cacheShape);
    size_t bufferNum = blockNum * processInfos.size();
    auto dataType = getBlockBeginIt(*mCacheManager, llmRequest, beam, 0)->getDataType();
    runtime::ITensor::SharedPtr recvBufferTemp;
    {
        NVTX3_SCOPED_RANGE(formatInputAllocBuffer);

        recvBufferTemp
            = bufferManager.gpu(runtime::ITensor::makeShape({static_cast<int64_t>(cacheVolume * bufferNum)}), dataType);
        recvBufferTmps.resize(bufferNum);
        for (size_t i = 0; i < bufferNum; i++)
        {
            recvBufferTmps[i] = runtime::ITensor::slice(recvBufferTemp, i * cacheVolume, cacheVolume);
        }
        // sync to alloc buffer
        bufferManager.getStream().synchronize();
    }

    {
        NVTX3_SCOPED_RANGE(formatInputRecvBuffer);

        auto dataContext = executor::kv_cache::DataContext{llmRequest.getContextPhaseParams().value().getReqId()};
        bool layerWise = common::getEnvDisaggLayerwise() && numPools == 1;
        if (layerWise)
        {
            SizeType32 const numLocalLayers = mCacheManager->getBlockManager().getNumLayers();
            SizeType32 const numLayers = cacheShape.d[0];
            TLLM_CHECK(numLayers % numLocalLayers == 0 || numLocalLayers % numLayers == 0);
            auto layerVolume = cacheVolume / cacheShape.d[0];
            // TODO: support numPools > 1, determining layerIdxInPool, since layers are grouped into pools
            for (SizeType32 layerIdx = 0; layerIdx < numLayers; layerIdx++)
            {
                // TODO: only send/recv required layers for ctxPP < genPP (numLayers > numLocalLayers)
                auto const poolIdx = 0;
                auto const layerIdxInPool = layerIdx;
                int idx = 0;
                auto const endIt = getBlockEndIt(*mCacheManager, llmRequest, beam, poolIdx);
                for (auto it = getBlockBeginIt(*mCacheManager, llmRequest, beam, poolIdx); it != endIt; ++it)
                {
                    if (layerIdxInPool == 0)
                    {
                        TLLM_LOG_DEBUG("Buffer %d of pool %d shape = %s", idx, poolIdx,
                            runtime::ITensor::toString(recvBufferTmps[idx]->getShape()).c_str());
                    }
                    for (auto const& processInfo : processInfos)
                    {
                        TLLM_LOG_DEBUG("recv layer %d(%d-%d)", layerIdx, poolIdx, layerIdxInPool);
                        // Buffer dim: [numLayersInPool * layerVolume]
                        auto layer
                            = runtime::ITensor::slice(recvBufferTmps[idx], layerIdxInPool * layerVolume, layerVolume);
                        comm.recvBuffer(*layer, dataContext, processInfo);
                        idx++;
                    }
                }
            }
        }
        else
        {

            auto recvBufferFun
                = [&](int deviceId, int blockIdxOffset, executor::kv_cache::ProcessInfo const& processInfo)
            {
                NVTX3_SCOPED_RANGE(recvBufferFun);
                TLLM_CUDA_CHECK(cudaSetDevice(deviceId));
                int idx = blockIdxOffset;
                for (auto poolIdx = 0; poolIdx < numPools; poolIdx++)
                {
                    auto const endIt = getBlockEndIt(*mCacheManager, llmRequest, beam, poolIdx);
                    for (auto it = getBlockBeginIt(*mCacheManager, llmRequest, beam, poolIdx); it != endIt; ++it)
                    {
                        size_t commIdx = idx / (blockNum);
                        size_t blockIdx = idx % (blockNum);
                        size_t recvBufferIdx = blockIdx * processInfos.size() + commIdx;
                        comm.recvBuffer(*recvBufferTmps[recvBufferIdx], dataContext, processInfo);
                        idx++;
                    }
                }
            };
            int deviceId = bufferManager.getStream().getDevice();
            if (processInfos.size() > 1)
            {
                int idxOffset = 0;
                if (common::getEnvDisableReceiveKVCacheParallel())
                {
                    for (auto const& processInfo : processInfos)
                    {
                        recvBufferFun(deviceId, idxOffset, processInfo);
                        idxOffset += blockNum;
                    }
                }
                else
                {
                    std::vector<std::future<void>> futures;
                    futures.reserve(processInfos.size());

                    for (auto const& processInfo : processInfos)
                    {
                        futures.push_back(
                            std::async(std::launch::async, recvBufferFun, deviceId, idxOffset, std::cref(processInfo)));
                        idxOffset += blockNum;
                    }
                    for (auto& future : futures)
                    {
                        future.get();
                    }
                }
            }
            else
            {
                recvBufferFun(deviceId, 0, processInfos[0]);
            }
        }
    }
    {
        NVTX3_SCOPED_RANGE(formatInputConcatenate);
        executor::kv_cache::concatenateKVCacheDispatch(recvBufferTmps.data(), recvBufferTmps.size(),
            getCounterparts(selfConfig, selfIdx, destConfig), destConfig, outputBuffers.data(), outputBuffers.size(),
            selfIdx, selfConfig, bufferManager);
        bufferManager.getStream().synchronize();
    }
    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), "Endreceiving kvCache for request id:%ld , context request id:%ld",
        llmRequest.mRequestId, llmRequest.getContextPhaseParams().value().getReqId());
}

[[nodiscard]] bool CacheFormatter::inquireSupport(CacheState const& selfConfig, CacheState const& destConfig) const
{
    std::unordered_set<SizeType32> setVecSelf{
        selfConfig.getModelConfig().mNbKvHeadsPerLayer.begin(), selfConfig.getModelConfig().mNbKvHeadsPerLayer.end()};

    if (setVecSelf.size() != 1)
    {
        return false;
    }
    std::unordered_set<int> setVecDest{
        destConfig.getModelConfig().mNbKvHeadsPerLayer.begin(), destConfig.getModelConfig().mNbKvHeadsPerLayer.end()};

    if (setVecDest.size() != 1)
    {
        return false;
    }
    if (selfConfig.getModelConfig().mTokensPerBlock != destConfig.getModelConfig().mTokensPerBlock
        || selfConfig.getModelConfig().mSizePerHead != destConfig.getModelConfig().mSizePerHead)
    {
        return false;
    }
    if (selfConfig.getModelConfig().mNbKvHeadsPerLayer.size() != destConfig.getModelConfig().mNbKvHeadsPerLayer.size())
    {
        return false;
    }

    int selfNumHeads
        = selfConfig.getModelConfig().mNbKvHeadsPerLayer[0] * selfConfig.getParallelConfig().mTensorParallelism;
    int destNumHeads
        = destConfig.getModelConfig().mNbKvHeadsPerLayer[0] * destConfig.getParallelConfig().mTensorParallelism;
    return selfNumHeads == destNumHeads;
}
} // namespace tensorrt_llm::batch_manager::kv_cache_manager
