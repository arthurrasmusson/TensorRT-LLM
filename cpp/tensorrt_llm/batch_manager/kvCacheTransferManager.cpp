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

#include "tensorrt_llm/batch_manager/kvCacheTransferManager.h"

#include "tensorrt_llm/batch_manager/kvCacheEventManager.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/cudaEvent.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include <cstdio>

namespace tr = tensorrt_llm::runtime;

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

KVCacheTransferManager::KVCacheTransferManager(tr::BufferManager const& bufferManager)
    : mBufferManager{bufferManager}
    , mOnboardManager(std::make_shared<tr::CudaStream>())
    , mOffloadManager(std::make_shared<tr::CudaStream>())
{
}

tr::ITensor::SharedPtr KVCacheTransferManager::computeBlockPointer(
    BlockPtr const& block, std::vector<KVCacheBlockPool> const& pools, size_t poolIdx)
{
    TLLM_CHECK_WITH_INFO(!pools.empty(), "Pool index %lu is out of bounds", poolIdx);
    auto const& pool = pools.at(poolIdx);
    auto ptr = block->isPrimary() ? pool.primaryPtr : pool.secondaryPtr;
    auto const blockOffset = block->getMemoryPoolBlockIndex();
    tr::ITensor::SharedPtr blockTensor{tr::ITensor::slice(ptr, blockOffset, 1)};
    return blockTensor;
}

void KVCacheTransferManager::copyBlock(
    BlockPtr const& src, BlockPtr const& dst, std::vector<KVCacheBlockPool> const& pools, bool isOffload)
{
    printf("ENTERED COPY BLOCK\n");
    // TODO: Replace computeBlockPointer with getKOrVBlockPointer calls
    // block spans multiple pool - copy in each pool
    auto const numPools = pools.size();
    for (size_t poolIdx = 0; poolIdx < numPools; poolIdx++)
    {
        auto const srcPtr = computeBlockPointer(src, pools, poolIdx);
        auto dstPtr = computeBlockPointer(dst, pools, poolIdx);
        (isOffload ? mOffloadManager : mOnboardManager).copy(*srcPtr, *dstPtr);
    }
}

void KVCacheTransferManager::onboard(
    BlockPtr const& offloadBlock, BlockPtr const& block, std::vector<KVCacheBlockPool> const& pools)
{
    if (mPendingOffloads.find(offloadBlock->getBlockId()) != mPendingOffloads.end())
    {
        mOnboardManager.getStream().wait(mPendingOffloads[offloadBlock->getBlockId()]);
    }
    copyBlock(offloadBlock, block, pools, false);
}

void KVCacheTransferManager::offload(
    BlockPtr const& block, BlockPtr const& offloadBlock, std::vector<KVCacheBlockPool> const& pools)
{
    mPendingOffloads[block->getBlockId()] = tr::CudaEvent();
    copyBlock(block, offloadBlock, pools, true);
    mOffloadManager.getStream().record(mPendingOffloads[block->getBlockId()]);
}

void KVCacheTransferManager::syncTransfers()
{
    tr::CudaEvent offloadEvent;
    mOffloadManager.getStream().record(offloadEvent);

    tr::CudaEvent onboardEvent;
    mOnboardManager.getStream().record(onboardEvent);

    mBufferManager.getStream().wait(offloadEvent);
    mBufferManager.getStream().wait(onboardEvent);

    // Once we synchronize, clear our list of pending thransfers.
    mPendingOffloads.clear();
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager
