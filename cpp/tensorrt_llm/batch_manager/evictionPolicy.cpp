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

#include "tensorrt_llm/batch_manager/evictionPolicy.h"

namespace tensorrt_llm::batch_manager::eviction_policy
{

void LRUEvictionPolicy::initialize(
    std::vector<BlockPtr>& mAllBlocksById, SizeType32 numPrimaryBlocks, SizeType32 numSecondaryBlocks)
{
    for (SizeType32 blockId = 0; blockId < numPrimaryBlocks; blockId++)
    {
        mFreeBlockIterators.push_back(mFreePrimaryBlocks.insert(mFreePrimaryBlocks.end(), mAllBlocksById[blockId]));
    }

    for (SizeType32 blockId = 0; blockId < numSecondaryBlocks; blockId++)
    {
        mFreeBlockIterators.push_back(
            mFreeSecondaryBlocks.insert(mFreeSecondaryBlocks.end(), mAllBlocksById[numPrimaryBlocks + blockId]));
    }

    mFreePrimaryBlocksSize = numPrimaryBlocks;
    mFreeSecondaryBlocksSize = numSecondaryBlocks;
}

BlockPtr LRUEvictionPolicy::getFreePrimaryBlock()
{
    for (auto block : mFreePrimaryBlocks)
    {
        if (block->isPrimary())
        {
            bool keepLooking;
            do
            {
                keepLooking = false;
                NextBlockMap blockMap = block->getNextBlocks();
                for (auto itr = blockMap.begin(); itr != blockMap.end(); ++itr)
                {
                    if (itr->second->isPrimary())
                    {
                        block = itr->second;
                        keepLooking = true;
                        break;
                    }
                }
            } while (keepLooking);
            return block;
        }
    }
    TLLM_CHECK_WITH_INFO(false, "mFreePrimaryBlocks list has no GPU blocks");
}

BlockPtr LRUEvictionPolicy::getFreeSecondaryBlock()
{
    auto block = mFreeSecondaryBlocks.front();
    while (!block->getNextBlocks().empty())
    {
        block = block->getNextBlocks().begin()->second;
    }
    return block;
}

void LRUEvictionPolicy::releaseBlock(BlockPtr block, bool toFront)
{
    if (block->isPrimary())
    {
        if (toFront)
        {
            mFreeBlockIterators[block->getBlockId()] = mFreePrimaryBlocks.insert(mFreePrimaryBlocks.begin(), block);
            ++mFreePrimaryBlocksSize;
        }
        else
        {
            mFreeBlockIterators[block->getBlockId()] = mFreePrimaryBlocks.insert(mFreePrimaryBlocks.end(), block);
            ++mFreePrimaryBlocksSize;
        }
    }
    else
    {
        if (toFront)
        {
            mFreeBlockIterators[block->getBlockId()] = mFreeSecondaryBlocks.insert(mFreeSecondaryBlocks.begin(), block);
            ++mFreeSecondaryBlocksSize;
        }
        else
        {
            mFreeBlockIterators[block->getBlockId()] = mFreeSecondaryBlocks.insert(mFreeSecondaryBlocks.end(), block);
            ++mFreeSecondaryBlocksSize;
        }
    }
}

SizeType32 LRUEvictionPolicy::getNumFreePrimaryBlocks()
{
    return mFreePrimaryBlocksSize;
}

SizeType32 LRUEvictionPolicy::getNumFreeSecondaryBlocks()
{
    return mFreeSecondaryBlocksSize;
}

void LRUEvictionPolicy::claimBlock(KVCacheBlock block)
{
    auto freeBlockIterator = mFreeBlockIterators[block.getBlockId()];
    if (freeBlockIterator)
    {
        if (block.isPrimary())
        {
            mFreePrimaryBlocks.erase(*freeBlockIterator);
            --mFreePrimaryBlocksSize;
        }
        else
        {
            mFreeSecondaryBlocks.erase(*freeBlockIterator);
            --mFreeSecondaryBlocksSize;
        }
        mFreeBlockIterators[block.getBlockId()] = std::nullopt;
    }
}

} // namespace tensorrt_llm::batch_manager::eviction_policy
