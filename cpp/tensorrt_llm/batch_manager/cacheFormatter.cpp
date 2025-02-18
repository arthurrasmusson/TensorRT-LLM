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
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/executor/cache_transmission/cacheConcatenate.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"

#include <cstddef>
#include <cstdint>
#include <future>

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

void CacheFormatter::formatOutput(executor::kv_cache::Communicator const& comm, LlmRequest const& llmRequest,
    std::vector<executor::kv_cache::ProcessInfo> const& processInfos, CacheState const& selfConfig, SizeType32 selfIdx,
    CacheState const& destConfig, runtime::BufferManager const& bufferManager)
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

        int blockNum = 0;
        std::vector<runtime::ITensor::SharedPtr> inputKvCacheBlocks;
        for (auto poolIdx = 0; poolIdx < numPools; poolIdx++)
        {
            auto const endIt = getBlockEndIt(*mCacheManager, llmRequest, beam, poolIdx);
            for (auto it = getBlockBeginIt(*mCacheManager, llmRequest, beam, poolIdx); it != endIt; ++it)
            {
                blockNum++;
                inputKvCacheBlocks.push_back(it);
            }
        }
        TLLM_CHECK(blockNum > 0);
        int deviceId = mCacheManager->getBlockManager().getBufferManager().getStream().getDevice();

        if (common::getEnvTryZCopyForKVCacheTransfer()
            && (destConfig.getParallelConfig().mPipelineParallelism
                <= selfConfig.getParallelConfig().mPipelineParallelism)
            && (destConfig.getParallelConfig().mTensorParallelism <= selfConfig.getParallelConfig().mTensorParallelism))
        {
            TLLM_LOG_DEBUG(" try zcopy for  kv cache");
            NVTX3_SCOPED_RANGE(sendBufferFun);

            TLLM_CHECK(processInfos.size() == 1);

            TLLM_CUDA_CHECK(cudaSetDevice(deviceId));
            for (auto const& processInfo : processInfos)
            {
                for (auto const& block : inputKvCacheBlocks)
                {
                    comm.sendBuffer(*block, executor::kv_cache::DataContext{llmRequest.mRequestId}, processInfo);
                }
            }
            TLLM_LOG_DEBUG(
                mpi::MpiComm::world().getRank(), "End sending kvCache for request id:%ld ", llmRequest.mRequestId);

            return;
        }

        auto cacheBlockSize = inputKvCacheBlocks.at(0)->getSize();
        auto dataType = inputKvCacheBlocks.at(0)->getDataType();
        size_t const sendBufferSize = common::getEnvMemSizeForKVCacheTransferBuffer();
        size_t const sendBufferEleSize = sendBufferSize / common::getDTypeSize(dataType);

        bool const onlyUseAsyncBuffer = sendBufferEleSize == 0;
        runtime::ITensor::SharedPtr preAllocSendBuffer;
        auto const maxConcurrenceNum = static_cast<int>(common::getEnvKVCacheSendMaxConcurrenceNum());
        if (!onlyUseAsyncBuffer && (mConcurrenceSendResource.mConcurrence >= maxConcurrenceNum))
        {
            std::unique_lock lk(mConcurrenceSendResource.mSendbuffersMutex);
            mConcurrenceSendResource.mSendbuffersCV.wait(
                lk, [this, maxConcurrenceNum]() { return mConcurrenceSendResource.mConcurrence < maxConcurrenceNum; });
        }
        if (!onlyUseAsyncBuffer)
        {
            int bufferId = mConcurrenceSendResource.mConcurrence++;

            if (!onlyUseAsyncBuffer
                && mConcurrenceSendResource.mSendbuffers.find(bufferId) == mConcurrenceSendResource.mSendbuffers.end())
            {
                if (common::getEnvKVCacheTransferUseAsyncBuffer())
                {
                    mConcurrenceSendResource.mSendbuffers[bufferId] = bufferManager.gpu(
                        runtime::ITensor::makeShape({static_cast<int64_t>(sendBufferEleSize)}), dataType);
                }
                else
                {
                    mConcurrenceSendResource.mSendbuffers[bufferId] = bufferManager.gpuSync(
                        runtime::ITensor::makeShape({static_cast<int64_t>(sendBufferEleSize)}), dataType);
                }
            }
            preAllocSendBuffer = mConcurrenceSendResource.mSendbuffers[bufferId];
        };

        auto targetNum = processInfos.size();
        TLLM_CHECK((cacheBlockSize * blockNum) % targetNum == 0);
        auto const targetBufferSize = (cacheBlockSize * blockNum) / targetNum;
        std::vector<runtime::ITensor::SharedPtr> outputSplitCaches;

        size_t bufferCoverTargetNum = sendBufferEleSize / targetBufferSize;

        runtime::ITensor::SharedPtr SendBufferTemp;
        if (bufferCoverTargetNum < targetNum)
        {
            SendBufferTemp
                = bufferManager.gpu(runtime::ITensor::makeShape(
                                        {static_cast<int64_t>(targetBufferSize * (targetNum - bufferCoverTargetNum))}),
                    dataType);
        }

        for (size_t i = 0; i < targetNum; i++)
        {
            if (i < bufferCoverTargetNum)
            {
                auto slice = runtime::ITensor::slice(preAllocSendBuffer, i * targetBufferSize, targetBufferSize);
                outputSplitCaches.push_back(std::move(slice));
            }
            else
            {
                auto slice = runtime::ITensor::slice(
                    SendBufferTemp, (i - bufferCoverTargetNum) * targetBufferSize, targetBufferSize);
                outputSplitCaches.push_back(std::move(slice));
            }
        }

        tensorrt_llm::executor::kv_cache::splitKVCacheDispatch(
            inputKvCacheBlocks, outputSplitCaches, destConfig, selfConfig, selfIdx, bufferManager);

        bufferManager.getStream().synchronize();
        if (onlyUseAsyncBuffer)
        {
            bufferCoverTargetNum = targetNum;
        }
        auto sendBufferFun = [&](int deviceId, size_t processIdx)
        {
            NVTX3_SCOPED_RANGE(sendBufferFun);

            TLLM_CUDA_CHECK(cudaSetDevice(deviceId));
            if (processIdx < bufferCoverTargetNum)
            {
                comm.sendBuffer(*outputSplitCaches.at(processIdx),
                    executor::kv_cache::DataContext{llmRequest.mRequestId}, processInfos.at(processIdx));
            }
            else if (bufferCoverTargetNum > 0)
            {
                // copy buffer allocated by cudaMallocAsync to buffer allocated by cudaMalloc before sending
                auto sendBufferIdx = processIdx % bufferCoverTargetNum;
                bufferManager.copy(*outputSplitCaches.at(processIdx), *outputSplitCaches.at(sendBufferIdx));
                bufferManager.getStream().synchronize();
                comm.sendBuffer(*outputSplitCaches.at(sendBufferIdx),
                    executor::kv_cache::DataContext{llmRequest.mRequestId}, processInfos.at(processIdx));
            }
            else
            {

                // bufferCoverTargetNum=0, mSendBuffer size < one outputSlice
                // send multiple times
                size_t remainSendSize = targetBufferSize;
                while (remainSendSize > 0)
                {
                    auto sendSize = std::min(remainSendSize, sendBufferEleSize);
                    auto copySlice = runtime::ITensor::slice(
                        outputSplitCaches.at(processIdx), targetBufferSize - remainSendSize, sendSize);
                    auto copyTargetSlice = runtime::ITensor::slice(preAllocSendBuffer, 0, sendSize);
                    bufferManager.copy(*copySlice, *copyTargetSlice);
                    bufferManager.getStream().synchronize();
                    comm.sendBuffer(*copyTargetSlice, executor::kv_cache::DataContext{llmRequest.mRequestId},
                        processInfos.at(processIdx));

                    remainSendSize -= sendSize;
                }
            }

            TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), "End sending kvCache for request id:%ld requestRank :%d ",
                llmRequest.mRequestId, processInfos.at(processIdx).getRank());
        };

        if (processInfos.size() > 1)
        {
            if (common::getEnvDisableReceiveKVCacheParallel())
            {
                TLLM_LOG_DEBUG("Disable receiving kvCache in parallel");

                for (size_t i = 0; i < processInfos.size(); i++)
                {
                    sendBufferFun(deviceId, i);
                }
            }
            else
            {
                // concurrency num
                auto concurrencyNum
                    = std::min(std::max(static_cast<size_t>(1), bufferCoverTargetNum), processInfos.size());

                auto remainSendNum = processInfos.size();

                while (remainSendNum > 0)
                {
                    auto sendConcurrencyNum = std::min(remainSendNum, concurrencyNum);
                    std::vector<std::future<void>> futures;
                    futures.reserve(sendConcurrencyNum);
                    for (size_t i = 0; i < sendConcurrencyNum; i++)
                    {
                        TLLM_CHECK((i + (processInfos.size() - remainSendNum)) < processInfos.size());
                        futures.push_back(std::async(
                            std::launch::async, sendBufferFun, deviceId, i + (processInfos.size() - remainSendNum)));
                    }
                    for (auto& future : futures)
                    {
                        future.get();
                    }
                    remainSendNum -= sendConcurrencyNum;
                }
            }
        }
        else
        {
            sendBufferFun(deviceId, 0);
        }
        if (!onlyUseAsyncBuffer && (mConcurrenceSendResource.mConcurrence--) >= maxConcurrenceNum)
        {
            mConcurrenceSendResource.mSendbuffersCV.notify_one();
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
    size_t blockNum = 0;
    for (auto poolIdx = 0; poolIdx < numPools; poolIdx++)
    {
        auto const endIt = getBlockEndIt(*mCacheManager, llmRequest, beam, poolIdx);
        for (auto it = getBlockBeginIt(*mCacheManager, llmRequest, beam, poolIdx); it != endIt; ++it)
        {
            blockNum++;
            outputBuffers.push_back(it);
        }
    }

    {
        NVTX3_SCOPED_RANGE(formatInputRecvBuffer);

        auto dataContext = executor::kv_cache::DataContext{llmRequest.getContextPhaseParams().value().getReqId()};
        bool layerWise = common::getEnvDisaggLayerwise() && numPools == 1;
        if (layerWise)
        {

            // [numLayersInPool, ...]
            auto cacheShape = executor::kv_cache::makeShapeFromCacheState(destConfig);
            auto cacheVolume = runtime::ITensor::volume(cacheShape);
            size_t bufferNum = blockNum * processInfos.size();
            auto dataType = getBlockBeginIt(*mCacheManager, llmRequest, beam, 0)->getDataType();
            runtime::ITensor::SharedPtr recvBufferTemp;
            {
                NVTX3_SCOPED_RANGE(formatInputAllocBuffer);

                recvBufferTemp = bufferManager.gpu(
                    runtime::ITensor::makeShape({static_cast<int64_t>(cacheVolume * bufferNum)}), dataType);
                recvBufferTmps.resize(bufferNum);
                for (size_t i = 0; i < bufferNum; i++)
                {
                    recvBufferTmps[i] = runtime::ITensor::slice(recvBufferTemp, i * cacheVolume, cacheVolume);
                }
                // sync to alloc buffer
                bufferManager.getStream().synchronize();
            }
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
            {
                NVTX3_SCOPED_RANGE(formatInputConcatenate);
                executor::kv_cache::concatenateKVCacheDispatch(recvBufferTmps.data(), recvBufferTmps.size(),
                    getCounterparts(selfConfig, selfIdx, destConfig), destConfig, outputBuffers.data(),
                    outputBuffers.size(), selfIdx, selfConfig, bufferManager);
                bufferManager.getStream().synchronize();
            }
        }
        else
        {
            // non-layer-wise
            int deviceId = bufferManager.getStream().getDevice();

            if (common::getEnvTryZCopyForKVCacheTransfer() && destConfig == selfConfig)
            {
                TLLM_LOG_DEBUG("try zcopy for KV cache");
                NVTX3_SCOPED_RANGE(recvBufferFun);

                TLLM_CHECK(processInfos.size() == 1);

                TLLM_CUDA_CHECK(cudaSetDevice(deviceId));
                for (auto const& processInfo : processInfos)
                {
                    for (auto const& block : outputBuffers)
                    {
                        comm.recvBuffer(*block, dataContext, processInfo);
                    }
                }
                TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
                    "Endreceiving kvCache for request id:%ld , context request id:%ld", llmRequest.mRequestId,
                    llmRequest.getContextPhaseParams().value().getReqId());
                return;
            }
            // legacyPath: context executor rank only send data to one gen executor rank. it sends multiple cache
            // blocks.
            auto legacyPath = common::getEnvTryZCopyForKVCacheTransfer()
                && (destConfig.getParallelConfig().mPipelineParallelism
                    >= selfConfig.getParallelConfig().mPipelineParallelism)
                && (destConfig.getParallelConfig().mTensorParallelism
                    >= selfConfig.getParallelConfig().mTensorParallelism);

            runtime::ITensor::SharedPtr recvBufferTemp;
            runtime::ITensor::SharedPtr preAllocRecvBufferTemp;
            std::vector<runtime::ITensor::SharedPtr> recvSplitCaches;

            auto cacheBlockSize = outputBuffers.at(0)->getSize();

            auto dataType = outputBuffers.at(0)->getDataType();
            auto const recvBufferSize = common::getEnvMemSizeForKVCacheTransferBuffer();
            auto const recvBufferEleSize = recvBufferSize / common::getDTypeSize(dataType);
            auto targetNum = processInfos.size();
            TLLM_CHECK((cacheBlockSize * blockNum) % targetNum == 0);
            auto targetBufferSize = (cacheBlockSize * blockNum) / targetNum;

            size_t bufferCoverTargetNum = recvBufferEleSize / targetBufferSize;
            size_t remainNoCoverTargetNum = targetNum > bufferCoverTargetNum ? targetNum - bufferCoverTargetNum : 0;
            bool const onlyUseAsyncBuffer = recvBufferEleSize == 0;
            {
                NVTX3_SCOPED_RANGE(formatInputAllocBuffer);

                TLLM_CHECK(blockNum > 0);
                TLLM_CHECK(outputBuffers.size() == blockNum);
                if (legacyPath)
                {

                    TLLM_LOG_DEBUG("formatOutput using legacy path");
                    auto cacheShape = executor::kv_cache::makeShapeFromCacheState(destConfig);
                    auto cacheVolume = runtime::ITensor::volume(cacheShape);

                    size_t bufferNum = blockNum * processInfos.size();
                    recvBufferTemp = bufferManager.gpu(
                        runtime::ITensor::makeShape({static_cast<int64_t>(cacheVolume * bufferNum)}), dataType);
                    recvSplitCaches.resize(bufferNum);
                    for (size_t i = 0; i < bufferNum; i++)
                    {
                        recvSplitCaches[i] = runtime::ITensor::slice(recvBufferTemp, i * cacheVolume, cacheVolume);
                    }
                }
                else
                {
                    if (!onlyUseAsyncBuffer)
                    {
                        std::string processString = llmRequest.getDataTransceiverState().getCommState()->toString();

                        if (common::getEnvRequestKVCacheSerial())
                        {
                            processString = "default";
                        }

                        {
                            std::scoped_lock<std::mutex> lock(mProcessToRecvBufferMutex);
                            if (mProcessToRecvBuffer.find(processString) == mProcessToRecvBuffer.end())
                            {

                                if (common::getEnvKVCacheTransferUseAsyncBuffer())
                                {
                                    mProcessToRecvBuffer[processString] = bufferManager.gpu(
                                        runtime::ITensor::makeShape({static_cast<int64_t>(recvBufferEleSize)}),
                                        dataType);
                                }
                                else
                                {
                                    mProcessToRecvBuffer[processString] = bufferManager.gpuSync(
                                        runtime::ITensor::makeShape({static_cast<int64_t>(recvBufferEleSize)}),
                                        dataType);
                                }
                            }
                            preAllocRecvBufferTemp = mProcessToRecvBuffer[processString];
                        }
                    }

                    if (bufferCoverTargetNum < targetNum)
                    {
                        recvBufferTemp
                            = bufferManager.gpu(runtime::ITensor::makeShape(
                                                    {static_cast<int64_t>(remainNoCoverTargetNum * targetBufferSize)}),
                                dataType);
                    }
                    for (size_t i = 0; i < targetNum; i++)
                    {
                        if (i < remainNoCoverTargetNum)
                        {
                            recvSplitCaches.push_back(
                                runtime::ITensor::slice(recvBufferTemp, i * targetBufferSize, targetBufferSize));
                        }
                        else
                        {

                            recvSplitCaches.push_back(runtime::ITensor::slice(preAllocRecvBufferTemp,
                                (i - remainNoCoverTargetNum) * targetBufferSize, targetBufferSize));
                        }
                    }
                }

                // sync to alloc buffer
                bufferManager.getStream().synchronize();
            }
            if (onlyUseAsyncBuffer)
            {
                remainNoCoverTargetNum = 0;
                bufferCoverTargetNum = targetNum;
            }

            auto recvBufferFun = [&](int deviceId, size_t processIdx)
            {
                NVTX3_SCOPED_RANGE(recvBufferFun);
                TLLM_CUDA_CHECK(cudaSetDevice(deviceId));
                if (legacyPath)
                {
                    size_t idx = processIdx * blockNum;

                    for (size_t i = 0; i < blockNum; i++)
                    {
                        size_t commIdx = idx / (blockNum);
                        size_t blockIdx = idx % (blockNum);
                        size_t recvBufferIdx = blockIdx * processInfos.size() + commIdx;
                        comm.recvBuffer(*recvSplitCaches[recvBufferIdx], dataContext, processInfos.at(processIdx));
                        idx++;
                    }
                }
                else
                {
                    if (processIdx >= remainNoCoverTargetNum)
                    {
                        comm.recvBuffer(*recvSplitCaches.at(processIdx), dataContext, processInfos.at(processIdx));
                    }
                    else if (bufferCoverTargetNum > 0)
                    {
                        auto recvBufferIdx = processIdx % bufferCoverTargetNum
                            + remainNoCoverTargetNum; // caches.at(recvBufferIdx) is allocated by cudaMalloc
                        comm.recvBuffer(*recvSplitCaches.at(recvBufferIdx), dataContext, processInfos.at(processIdx));
                        bufferManager.copy(*recvSplitCaches.at(recvBufferIdx), *recvSplitCaches.at(processIdx));
                        bufferManager.getStream().synchronize();
                    }
                    else
                    {
                        // bufferCoverTargetNum==0
                        size_t remainRecvSize = targetBufferSize;
                        while (remainRecvSize > 0)
                        {
                            auto recvSize = std::min(remainRecvSize, recvBufferEleSize);
                            auto recvSlice = runtime::ITensor::slice(preAllocRecvBufferTemp, 0, recvSize);
                            auto copySlice = runtime::ITensor::slice(
                                recvSplitCaches.at(processIdx), targetBufferSize - remainRecvSize, recvSize);
                            comm.recvBuffer(*recvSlice, dataContext, processInfos.at(processIdx));
                            bufferManager.copy(*recvSlice, *copySlice);
                            bufferManager.getStream().synchronize();
                            remainRecvSize -= recvSize;
                        }
                    }
                }
            };
            if (processInfos.size() > 1)
            {
                if (common::getEnvDisableReceiveKVCacheParallel())
                {

                    for (size_t i = 0; i < processInfos.size(); i++)
                    {
                        recvBufferFun(deviceId, i);
                    }
                }
                else
                {
                    // concurrency num
                    auto concurrencyNum
                        = std::min(std::max(static_cast<size_t>(1), bufferCoverTargetNum), processInfos.size());
                    auto remainRecvNum = processInfos.size();

                    while (remainRecvNum > 0)
                    {

                        auto recvConcurrencyNum = std::min(remainRecvNum, concurrencyNum);

                        if (remainRecvNum > concurrencyNum && remainRecvNum < (2 * concurrencyNum))
                        {
                            recvConcurrencyNum = remainRecvNum - concurrencyNum;
                        }
                        std::vector<std::future<void>> futures;
                        futures.reserve(recvConcurrencyNum);
                        for (size_t i = 0; i < recvConcurrencyNum; i++)
                        {
                            TLLM_CHECK((i + (processInfos.size() - remainRecvNum)) < processInfos.size());
                            futures.push_back(std::async(std::launch::async, recvBufferFun, deviceId,
                                i + (processInfos.size() - remainRecvNum)));
                        }
                        for (auto& future : futures)
                        {
                            future.get();
                        }
                        remainRecvNum -= recvConcurrencyNum;
                    }
                }
            }
            else
            {
                recvBufferFun(deviceId, 0);
            }

            {
                NVTX3_SCOPED_RANGE(formatInputConcatenate);

                if (legacyPath)
                {
                    executor::kv_cache::concatenateKVCacheDispatch(recvSplitCaches.data(), recvSplitCaches.size(),
                        getCounterparts(selfConfig, selfIdx, destConfig), destConfig, outputBuffers.data(),
                        outputBuffers.size(), selfIdx, selfConfig, bufferManager);
                }
                else
                {
                    executor::kv_cache::concatenateKvCacheV2Dispatch(
                        recvSplitCaches, outputBuffers, destConfig, selfConfig, selfIdx, bufferManager);
                }
                bufferManager.getStream().synchronize();
            }
        }
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
