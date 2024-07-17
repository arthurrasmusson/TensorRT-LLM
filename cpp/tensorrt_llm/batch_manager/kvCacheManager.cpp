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

#include "tensorrt_llm/batch_manager/kvCacheManager.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/mpiUtils.h"
#include "tensorrt_llm/kernels/kvCacheIndex.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/worldConfig.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace tc = tensorrt_llm::common;
namespace tk = tensorrt_llm::kernels;
using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm::batch_manager::kv_cache_manager;

namespace
{

//! \brief Split vector into list of blocks of given size.
//! \param vec vector to split
//! \param usableSize part of the vector that is processed
//! \param elementsPerBlock desired size of blocks
//! \param allowPartial whether to append a block smaller than `elementsPerBlock` at the end
//! \return list of blocks
template <typename T>
std::list<std::vector<T>> chopVectorIntoBlocks(
    std::vector<T> const& vec, SizeType32 usableSize, SizeType32 elementsPerBlock, bool allowPartial)
{
    TLLM_CHECK(usableSize <= static_cast<SizeType32>(vec.size()));
    std::list<std::vector<T>> blockedVectors;
    auto const vecEnd = vec.begin() + usableSize;
    for (auto begin = vec.begin(); begin < vecEnd; begin += elementsPerBlock)
    {
        auto blockSize = std::min(elementsPerBlock, static_cast<SizeType32>(std::distance(begin, vecEnd)));
        auto end = begin + blockSize;
        if (blockSize == elementsPerBlock || allowPartial)
        {
            blockedVectors.emplace_back(begin, end);
        }
    }
    return blockedVectors;
}

} // namespace

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

KVCacheBlock::KVCacheBlock(IdType blockId, tk::KVCacheIndex blockIdx)
    : mBlockId(blockId)
    , mMemoryPoolBlockIndex{blockIdx}
    , mRefCount(0)
    , mSchedulingRefCount(0)
    , mPrevBlock(nullptr)
    , mFreeBlockIterator(std::nullopt)
    , mIsFull{false}
{
}

void KVCacheBlock::startScheduling()
{
    mSchedulingRefCount = mRefCount;
}

KVCacheBlock::IdType KVCacheBlock::getBlockId() const
{
    return mBlockId;
}

tk::KVCacheIndex::UnderlyingType KVCacheBlock::getMemoryPoolBlockIndex() const
{
    return mMemoryPoolBlockIndex.get();
}

bool KVCacheBlock::isPrimary() const
{
    return mMemoryPoolBlockIndex.isPrimary();
}

void KVCacheBlock::swapMemoryPoolBlockOffset(std::shared_ptr<KVCacheBlock> otherBlock)
{
    std::swap(mMemoryPoolBlockIndex, otherBlock->mMemoryPoolBlockIndex);
}

void KVCacheBlock::incRefCount()
{
    mRefCount++;
}

void KVCacheBlock::decRefCount()
{
    TLLM_CHECK_WITH_INFO(hasRefs(), "Can't remove link from block that is not allocated");
    mRefCount--;
}

void KVCacheBlock::decSchedulingRefCount()
{
    TLLM_CHECK_WITH_INFO(hasSchedulingRefs(), "Can't remove link from block that is not allocated");
    mSchedulingRefCount--;
}

bool KVCacheBlock::hasRefs() const
{
    return mRefCount > 0;
}

bool KVCacheBlock::isShared() const
{
    return mRefCount > 1;
}

bool KVCacheBlock::hasSchedulingRefs() const
{
    return mSchedulingRefCount > 0;
}

void KVCacheBlock::setTokens(VecTokens& tokens, bool isFull)
{
    mTokens = tokens;
    mIsFull = isFull;
}

VecTokens const& KVCacheBlock::getTokens() const
{
    return mTokens;
}

void KVCacheBlock::setFreeBlockIterator(FreeBlocksQueue::iterator freeBlockIterator)
{
    mFreeBlockIterator = freeBlockIterator;
}

void KVCacheBlock::resetFreeBlockIterator()
{
    mFreeBlockIterator = std::nullopt;
}

std::optional<FreeBlocksQueue::iterator> const& KVCacheBlock::getFreeBlockIterator() const
{
    return mFreeBlockIterator;
}

void KVCacheBlock::setPrevBlock(BlockPtr prevBlock)
{
    mPrevBlock = std::move(prevBlock);
}

void KVCacheBlock::addNextBlock(VecTokens const& tokens, BlockPtr block)
{
    if (mNextBlocks.find(tokens) == mNextBlocks.end())
    {
        mNextBlocks[tokens] = std::move(block);
    }
}

BlockPtr KVCacheBlock::findMatchingBlock(VecTokens const& tokens) const
{
    auto itr = mNextBlocks.find(tokens);
    if (itr == mNextBlocks.end())
    {
        return nullptr;
    }
    else
    {
        return itr->second;
    }
}

std::shared_ptr<KVCacheBlock> KVCacheBlock::findBestGPUBlockToFree(std::shared_ptr<KVCacheBlock> searchStart)
{
    auto block = std::move(searchStart);
    bool keepLooking;
    do
    {
        keepLooking = false;
        for (auto itr = block->mNextBlocks.begin(); itr != block->mNextBlocks.end(); ++itr)
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

std::shared_ptr<KVCacheBlock> KVCacheBlock::findLeafBlock(std::shared_ptr<KVCacheBlock> searchStart)
{
    auto block = std::move(searchStart);
    while (!block->mNextBlocks.empty())
    {
        auto itr = block->mNextBlocks.begin();
        block = itr->second;
    }
    return block;
}

void KVCacheBlock::freeLeafBlock()
{
    // assure that this is a leaf block
    TLLM_CHECK(mNextBlocks.empty());

    // free from previous block
    if (mPrevBlock != nullptr)
    {
        mPrevBlock->removeNextBlock(mTokens);
        mPrevBlock = nullptr;
    }
}

void KVCacheBlock::removeNextBlock(VecTokens const& tokens)
{
    mNextBlocks.erase(tokens);
}

bool KVCacheBlock::isFull() const
{
    return mIsFull;
}

BlockManager::BlockManager(SizeType32 numLayers, SizeType32 numKvHeads, SizeType32 sizePerHead,
    SizeType32 tokensPerBlock, SizeType32 blocksInPrimaryPool, SizeType32 blocksInSecondaryPool,
    std::shared_ptr<runtime::CudaStream> stream, bool onboardBlocks, CacheType cacheType)
    : mNumPrimaryBlocks{blocksInPrimaryPool}
    , mNumSecondaryBlocks{blocksInSecondaryPool}
    , mOnboardBlocks(onboardBlocks)
    , mBufferManager{stream}
    , mNumLayers{numLayers}
    , mBlockSize{numKvHeads * sizePerHead * tokensPerBlock}
    , mSchedulingNumFreeBlocks{0}
    , mTokensPerBlock{tokensPerBlock}
    , mCachedBlocksRoot{std::make_shared<KVCacheBlock>(-1, tk::KVCacheIndex{0})}
    , mAllocTotalBlocks{0}
    , mAllocNewBlocks{0}
    , mReusedBlocks{0}
    , mCacheType{cacheType}
{
    // Create free blocks
    mAllBlocksById.reserve(blocksInPrimaryPool + blocksInSecondaryPool);
    for (KVCacheBlock::IdType blockId = 0; blockId < blocksInPrimaryPool; ++blockId)
    {
        mAllBlocksById.emplace_back(std::make_shared<KVCacheBlock>(blockId, tk::KVCacheIndex{blockId, false}));
    }
    for (KVCacheBlock::IdType blockId = 0; blockId < blocksInSecondaryPool; ++blockId)
    {
        mAllBlocksById.emplace_back(
            std::make_shared<KVCacheBlock>(blocksInPrimaryPool + blockId, tk::KVCacheIndex{blockId, true}));
    }
    for (auto const& block : mAllBlocksById)
    {
        releaseBlock(block);
    }
}

BlockManager::~BlockManager()
{
    TLLM_LOG_DEBUG("BlockManager - total allocated blocks: %lu ", mAllocTotalBlocks);
    TLLM_LOG_DEBUG("BlockManager - allocated new blocks:   %lu ", mAllocNewBlocks);
    TLLM_LOG_DEBUG("BlockManager - reused blocks:          %lu ", mReusedBlocks);
}

void BlockManager::allocatePools(nvinfer1::DataType dtype, bool useUvm)
{
    // Allocate memory pool backing the blocks
    auto const cacheShape = ITensor::makeShape({mNumPrimaryBlocks, mNumLayers, 2, mBlockSize});
    if (useUvm)
        mPrimaryPool = BufferManager::managed(cacheShape, dtype);
    else
        mPrimaryPool = BufferManager::gpuSync(cacheShape, dtype);
    if (mNumSecondaryBlocks > 0)
    {
        auto const cacheShapeOffload = ITensor::makeShape({mNumSecondaryBlocks, mNumLayers, 2, mBlockSize});
        mSecondaryPool = BufferManager::pinned(cacheShapeOffload, dtype);
    }
}

void BlockManager::startScheduling()
{
    mSchedulingNumFreeBlocks = mFreePrimaryBlocks.size();
    for (auto& slotAllocatedBlocks : mAllocatedBlocksPerSeq)
    {
        for (auto& allocatedBlock : slotAllocatedBlocks)
        {
            allocatedBlock->startScheduling();
        }
    }
}

void BlockManager::claimBlock(KVCacheBlock& block)
{
    auto freeBlockIterator = block.getFreeBlockIterator();
    if (freeBlockIterator)
    {
        if (block.isPrimary())
        {
            mFreePrimaryBlocks.erase(*freeBlockIterator);
            block.resetFreeBlockIterator();
        }
        else
        {
            mFreeSecondaryBlocks.erase(*freeBlockIterator);
            block.resetFreeBlockIterator();
        }
    }
}

void BlockManager::claimLeafBlock(KVCacheBlock& block)
{
    block.freeLeafBlock();
    claimBlock(block);
}

std::shared_ptr<KVCacheBlock> BlockManager::findBestGPUBlockToFree()
{
    for (auto block : mFreePrimaryBlocks)
    {
        if (block->isPrimary())
        {
            return KVCacheBlock::findBestGPUBlockToFree(block);
        }
    }
    // Code can only reach this point if there are no GPU blocks in mFreePrimaryBlocks, which is a fatal error
    TLLM_CHECK_WITH_INFO(false, "mFreePrimaryBlocks list has no GPU blocks");
}

BlockPtr BlockManager::getFreeBlock()
{
    auto block = BlockManager::findBestGPUBlockToFree();
    if (block->getTokens().empty())
    {
        ++mAllocNewBlocks;
    }
    ++mAllocTotalBlocks;
    if (!block->getTokens().empty() && mFreeSecondaryBlocks.size() > 0)
    {
        claimBlock(*block);
        // Offload block in primary memory before repurposing
        auto offloadBlock = KVCacheBlock::findLeafBlock(mFreeSecondaryBlocks.front());
        claimLeafBlock(*offloadBlock);
        copyBlock(block, offloadBlock);
        // swap linear block offsets (i.e. make block the offload block)
        block->swapMemoryPoolBlockOffset(offloadBlock);
        releaseBlock(block); // append offload block to mFreeSecondaryBlocks queue
        block = offloadBlock;
    }
    else
    {
        claimLeafBlock(*block);
    }
    return block;
}

tk::KVCacheIndex BlockManager::getKOrVBlockIndex(KVCacheBlock::IdType blockId, SizeType32 fieldIdx) const
{
    auto const& block = mAllBlocksById[blockId];
    auto constexpr layerIdx = 0;
    return tk::KVCacheIndex{common::flat_index3(block->getMemoryPoolBlockIndex(), layerIdx, fieldIdx, mNumLayers, 2)};
}

void KVCacheManager::setOffsets(tk::KVCacheIndex* offsetsPtr, nvinfer1::Dims const& offsetsShape, SizeType32 seqSlotIdx,
    SizeType32 beamIdx, SizeType32 blockIdx, KVCacheBlock::IdType blockId) const
{
    auto constexpr kIdx = 0;
    auto constexpr vIdx = 1;

    for (auto xIdx : {kIdx, vIdx})
    {
        auto const offsetIndex
            = tensorrt_llm::common::flat_index(offsetsShape.d, seqSlotIdx * mMaxBeamWidth + beamIdx, xIdx, blockIdx);
        offsetsPtr[offsetIndex] = mBlockManager.getKOrVBlockIndex(blockId, xIdx);
    }
}

ITensor::SharedPtr BlockManager::computeBlockPointer(std::shared_ptr<KVCacheBlock> block) const
{
    auto pool = block->isPrimary() ? mPrimaryPool : mSecondaryPool;
    auto const blockOffset = block->getMemoryPoolBlockIndex();
    ITensor::SharedPtr blockTensor{ITensor::slice(pool, blockOffset, 1)};
    return blockTensor;
}

//! \brief Copy content of src block to dst.
void BlockManager::copyBlock(BlockPtr src, BlockPtr dst)
{
    // TODO: Replace computeBlockPointer with getKOrVBlockPointer calls
    auto const srcPtr = computeBlockPointer(src);
    auto dstPtr = computeBlockPointer(dst);
    mBufferManager.copy(*srcPtr, *dstPtr);
}

void BlockManager::onboardBlock(BlockPtr offloadBlock)
{
    if (mOnboardBlocks && !offloadBlock->isPrimary())
    {
        auto block = getFreeBlock();
        copyBlock(offloadBlock, block);
        // swap linear block offsets (i.e. make block the offload block and vice versa)
        offloadBlock->swapMemoryPoolBlockOffset(block);
        releaseBlock(block); // append block to offload queue
                             // offloadBlock is now in primary memory pool
    }
}

SizeType32 BlockManager::loadOrAllocateBlocks(std::list<VecTokens> const& blockedTokens, GenerationRequest& sequence)
{
    SizeType32 numMatchedTokens{0};
    auto searchRoot = mCachedBlocksRoot;

    for (auto const& blockTokens : blockedTokens)
    {
        auto matchingBlock = searchRoot != nullptr ? searchRoot->findMatchingBlock(blockTokens) : nullptr;
        if (matchingBlock != nullptr)
        {
            TLLM_CHECK_WITH_INFO(matchingBlock->isFull() || !matchingBlock->hasRefs(),
                "Found matching partially filled block, but somebody else is using it. This should not happen.");

            numMatchedTokens += blockTokens.size();
            if (!matchingBlock->isFull())
            {
                // Make block private and reuse
                claimLeafBlock(*matchingBlock);
                TLLM_LOG_DEBUG("BlockManager::loadOrAllocateBlocks - Matched partially filled block %d",
                    matchingBlock->getBlockId());
            }
            else
            {
                // Recover block and reuse
                claimBlock(*matchingBlock);
                TLLM_LOG_DEBUG(
                    "BlockManager::loadOrAllocateBlocks - Matched full block %d", matchingBlock->getBlockId());
            }
            onboardBlock(matchingBlock);
            addBlockToAllBeams(matchingBlock, sequence);
            searchRoot = matchingBlock;
            ++mReusedBlocks;
        }
        else
        {
            auto block = getFreeBlock();
            addBlockToAllBeams(block, sequence);
            TLLM_LOG_DEBUG(
                "BlockManager::loadOrAllocateBlocks - No match, allocated new block %d", block->getBlockId());
            searchRoot = nullptr; // no matching needed for following blocks
        }
    }
    return numMatchedTokens;
}

void BlockManager::addSequence(
    GenerationRequest& sequence, SizeType32 inputLength, std::shared_ptr<LlmRequest> const& llmRequest)
{
    TLLM_CHECK(llmRequest);

    auto const seqSlotIdx = sequence.getSequenceSlotIdx();

    if (static_cast<SizeType32>(mAllocatedBlocksPerSeq.size()) <= seqSlotIdx)
    {
        mAllocatedBlocksPerSeq.resize(seqSlotIdx + 1);
    }

    auto constexpr beamIdx = 0;
    auto const& tokens
        = mCacheType == CacheType::kSELF ? llmRequest->getTokens(beamIdx) : *(llmRequest->getEncoderTokens().value());

    // Ignore last token because it can't be recovered
    auto blockedTokens = chopVectorIntoBlocks<TokenIdType>(tokens, inputLength - 1, mTokensPerBlock, true);
    // Add empty block if last token is separated
    if (inputLength % mTokensPerBlock == 1)
    {
        blockedTokens.emplace_back();
    }

    auto const prepopulatedPromptLen = loadOrAllocateBlocks(blockedTokens, sequence);
    llmRequest->setPrepopulatedPromptLen(prepopulatedPromptLen);
    TLLM_LOG_DEBUG("addSequence: Request %lu, inputLength %d, prepopulatedPromptLen %d", llmRequest->mRequestId,
        inputLength, prepopulatedPromptLen);
}

void BlockManager::addSequence(GenerationRequest& sequence, SizeType32 numBlocks, SizeType32 unsharedBlockIdx)
{
    // Allocate blocks
    for (SizeType32 bi = 0; bi < numBlocks; ++bi)
    {
        bool shareAmongBeams = bi != unsharedBlockIdx;
        allocateBlock(sequence, shareAmongBeams);
    }
}

void BlockManager::releaseBlock(std::shared_ptr<KVCacheBlock> block, bool toFront)
{
    if (block->isPrimary())
    {
        if (toFront)
        {
            block->setFreeBlockIterator(mFreePrimaryBlocks.insert(mFreePrimaryBlocks.begin(), block));
        }
        else
        {
            block->setFreeBlockIterator(mFreePrimaryBlocks.insert(mFreePrimaryBlocks.end(), block));
        }
    }
    else
    {
        if (toFront)
        {
            block->setFreeBlockIterator(mFreeSecondaryBlocks.insert(mFreeSecondaryBlocks.begin(), block));
        }
        else
        {
            block->setFreeBlockIterator(mFreeSecondaryBlocks.insert(mFreeSecondaryBlocks.end(), block));
        }
    }
}

void BlockManager::addBlockToBeam(BlockPtr& block, GenerationRequest& sequence, SizeType32 beamIdx)
{
    auto const seqSlotIdx = sequence.getSequenceSlotIdx();
    block->incRefCount();
    sequence.addCacheBlock(beamIdx, block->getBlockId());
    mAllocatedBlocksPerSeq.at(seqSlotIdx).push_back(block);
}

void BlockManager::addBlockToAllBeams(BlockPtr& block, GenerationRequest& sequence)
{
    auto const beamWidth = sequence.getBeamWidth();

    for (SizeType32 beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
    {
        addBlockToBeam(block, sequence, beamIdx);
    }
}

void BlockManager::allocateBlock(GenerationRequest& sequence, bool shareAmongBeams)
{
    auto const seqSlotIdx = static_cast<std::size_t>(sequence.getSequenceSlotIdx());
    auto const beamWidth = sequence.getBeamWidth();
    auto const requiredBlocks = shareAmongBeams ? 1 : beamWidth;

    TLLM_CHECK_WITH_INFO(hasFreeBlocks(requiredBlocks), "Can't allocate new blocks. No free blocks left.");

    if (mAllocatedBlocksPerSeq.size() <= seqSlotIdx)
    {
        mAllocatedBlocksPerSeq.resize(seqSlotIdx + 1);
    }

    if (shareAmongBeams)
    {
        // add same block to all beams
        auto block = getFreeBlock();
        for (auto beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
        {
            addBlockToBeam(block, sequence, beamIdx);
        }
    }
    else
    {
        // add different block to each beam
        for (auto beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
        {
            auto block = getFreeBlock();
            addBlockToBeam(block, sequence, beamIdx);
        }
    }
}

void BlockManager::storeBlocks(std::list<VecTokens> blockedTokens, std::vector<KVCacheBlock::IdType> const& blockIds)
{
    TLLM_CHECK_WITH_INFO(blockedTokens.size() <= blockIds.size(), "%lu blockedTokens, %lu blockIds",
        blockedTokens.size(), blockIds.size());

    auto searchRoot = mCachedBlocksRoot;
    bool needMatch = true;

    auto numBlocks = blockedTokens.size();
    for (std::size_t ii = 0; ii < numBlocks; ++ii)
    {
        auto const bid = blockIds[ii];
        TLLM_LOG_DEBUG("BlockManager::storeBlocks - Searching match for block %d", bid);
        auto& block = mAllBlocksById[bid];
        TLLM_CHECK(blockedTokens.size() > 0);
        auto blockTokens = blockedTokens.front();
        blockedTokens.pop_front();
        auto matchedBlock = needMatch ? searchRoot->findMatchingBlock(blockTokens) : nullptr;
        if (matchedBlock != nullptr)
        {
            // Found match
            TLLM_LOG_DEBUG("BlockManager::storeBlocks - Found matching block %d, traverse", matchedBlock->getBlockId());
            searchRoot = matchedBlock;
            // TODO possible optimization: if bid != matchedBlock->getBlockId(),
            // block can be freed and inserted at mFreePrimaryBlocks.begin()
        }
        else
        {
            // No match
            TLLM_LOG_DEBUG(
                "BlockManager::storeBlocks - No match, inserting block %d into search structure", block->getBlockId());
            needMatch = false; // no matching needed for following blocks
            block->setTokens(blockTokens, static_cast<SizeType32>(blockTokens.size()) == mTokensPerBlock);
            block->setPrevBlock(searchRoot);
            searchRoot->addNextBlock(blockTokens, block);
            searchRoot = block;
        }
    }
}

void BlockManager::replaceSharedBlock(GenerationRequest& sequence, SizeType32 blockIdx)
{
    auto const seqSlotIdx = sequence.getSequenceSlotIdx();
    auto const beamWidth = sequence.getBeamWidth();
    auto& allocatedBlocks = mAllocatedBlocksPerSeq.at(seqSlotIdx);

    if (!allocatedBlocks.at((blockIdx + 1) * beamWidth - 1)->isShared())
    {
        return;
    }

    // Free shared block
    for (auto beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
    {
        auto block = allocatedBlocks.at(blockIdx * beamWidth + beamIdx);
        block->decRefCount();
        if (!block->hasRefs())
        {
            releaseBlock(block);
        }
    }

    // Allocate new blocks
    TLLM_CHECK_WITH_INFO(hasFreeBlocks(beamWidth), "Can't allocate new blocks. No free blocks left.");
    for (auto beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
    {
        auto block = getFreeBlock();
        block->incRefCount();
        sequence.changeCacheBlock(beamIdx, blockIdx, block->getBlockId());
        allocatedBlocks.at(blockIdx * beamWidth + beamIdx) = block;
    }
}

void BlockManager::releaseLastBlock(GenerationRequest& sequence)
{
    auto const seqSlotIdx = sequence.getSequenceSlotIdx();
    auto& allocatedBlocks = mAllocatedBlocksPerSeq.at(seqSlotIdx);
    auto it = allocatedBlocks.rbegin();
    auto& block = *it;
    // Decrease ref count
    block->decRefCount();
    // If ref count is zero, move block to free blocks
    if (!block->hasRefs())
    {
        releaseBlock(block, true);
    }
    // Remove block from allocated blocks
    allocatedBlocks.pop_back();
    // Remove stored block ids in sequence
    sequence.removeLastBlock();
}

void BlockManager::releaseBlocks(GenerationRequest& sequence, std::shared_ptr<LlmRequest> const& llmRequest)
{
    auto const seqSlotIdx = sequence.getSequenceSlotIdx();

    if (llmRequest)
    {
        // TODO only store blocks for context in case of beamWidth > 1
        auto cacheBlockIds = sequence.getCacheBlockIds();
        auto constexpr beamIdx = 0;
        auto const& tokens = llmRequest->getTokens(beamIdx);
        auto blockedTokens = chopVectorIntoBlocks<TokenIdType>(tokens, tokens.size() - 1, mTokensPerBlock, true);
        storeBlocks(std::move(blockedTokens), cacheBlockIds[beamIdx]);
    }

    auto& allocatedBlocks = mAllocatedBlocksPerSeq.at(seqSlotIdx);
    for (auto it = allocatedBlocks.rbegin(); it != allocatedBlocks.rend(); ++it)
    {
        auto& block = *it;
        // Decrease ref count
        block->decRefCount();
        // If ref count is zero, move block to free blocks
        if (!block->hasRefs())
        {
            releaseBlock(block);
        }
    }
    // Remove all blocks from allocated blocks
    allocatedBlocks.clear();
    // Remove stored block ids in sequence
    sequence.clearCacheBlocks();
}

void BlockManager::schedulingReleaseBlocks(GenerationRequest& sequence)
{
    auto const seqSlotIdx = sequence.getSequenceSlotIdx();

    for (auto& block : mAllocatedBlocksPerSeq.at(seqSlotIdx))
    {
        // Decrease ref count
        block->decSchedulingRefCount();
        // If ref count is zero, move block to free blocks
        if (!block->hasSchedulingRefs())
        {
            mSchedulingNumFreeBlocks++;
        }
    }
}

KVCacheManager::KVCacheManager(SizeType32 numLayers, SizeType32 numKvHeads, SizeType32 sizePerHead,
    SizeType32 tokensPerBlock, SizeType32 blocksInPrimaryPool, SizeType32 blocksInSecondaryPool,
    SizeType32 maxNumSequences, SizeType32 maxBeamWidth, SizeType32 maxAttentionWindow, SizeType32 sinkTokenLength,
    bool useOneMoreBlock, CudaStreamPtr stream, bool enableBlockReuse, bool onboardBlocks, CacheType cacheType)
    : mMaxNumSequences(maxNumSequences)
    , mMaxBeamWidth(maxBeamWidth)
    , mMaxAttentionWindow(maxAttentionWindow)
    , mBlockManager(numLayers, numKvHeads, sizePerHead, tokensPerBlock, blocksInPrimaryPool, blocksInSecondaryPool,
          std::move(stream), onboardBlocks, cacheType)
    , mSequences(maxNumSequences)
    , mEnableBlockReuse{enableBlockReuse}
    , mCacheType{cacheType}
{
    TLLM_CHECK_WITH_INFO(
        mMaxBeamWidth == 1 || !mEnableBlockReuse, "Block reuse is currently not supported with beam width > 1.");

    // The sink tokens are stored in blocks separate from other tokens.
    // If the last block of sink tokens is only partially filled,
    // we fill that block with a "bubble" to reach the number of tokens per block.
    auto const sinkTokensInLastBlock = sinkTokenLength % tokensPerBlock;
    mSinkBubbleLength = sinkTokensInLastBlock == 0 ? 0 : tokensPerBlock - sinkTokensInLastBlock;

    mSinkBlockTokenLength = mSinkBubbleLength + sinkTokenLength;
    TLLM_CHECK(mSinkBlockTokenLength % tokensPerBlock == 0);

    mMaxTokenNum = mMaxAttentionWindow + mSinkBubbleLength;
    if (useOneMoreBlock)
    {
        mMaxTokenNum += tokensPerBlock;
    }

    mMaxBlocksPerSeq = tc::ceilDiv(mMaxTokenNum, tokensPerBlock);

    auto const maxNumTokens = blocksInPrimaryPool * tokensPerBlock;

    TLLM_LOG_INFO("Max KV cache pages per sequence: %d", mMaxBlocksPerSeq);

    // Check that we can at least fit one sequence in kvCache
    TLLM_CHECK_WITH_INFO(maxNumTokens >= mMaxBeamWidth * mMaxBlocksPerSeq * tokensPerBlock,
        "maxTokensInPagedKvCache (%d) must be large enough to process at least 1 sequence to completion "
        "(i.e. must be larger than beam_width (%d) * tokensPerBlock (%d) * maxBlocksPerSeq (%d))",
        maxNumTokens, mMaxBeamWidth, tokensPerBlock, mMaxBlocksPerSeq);

    mSequenceBlockIndices
        = BufferManager::cpu(ITensor::makeShape({maxNumSequences * mMaxBeamWidth, 2, mMaxBlocksPerSeq}),
            TRTDataType<tk::KVCacheIndex>::value);
}

void KVCacheManager::allocatePools(nvinfer1::DataType dtype, bool useUvm)
{
    mBlockManager.allocatePools(dtype, useUvm);

    if (tc::Logger::getLogger()->getLevel() == tc::Logger::INFO)
    {
        auto const cacheShape = mBlockManager.getPrimaryPool()->getShape();
        auto const cacheVolume = ITensor::volume(cacheShape);
        auto const blocksInPrimaryPool = cacheShape.d[0];
        auto const tokensPerBlock = mBlockManager.getTokensPerBlock();
        auto const maxNumTokens = blocksInPrimaryPool * tokensPerBlock;
        auto const cacheSizeBytes = cacheVolume * BufferDataType(dtype).getSize();
        TLLM_LOG_INFO("Number of tokens per block: %d.", tokensPerBlock);
        TLLM_LOG_INFO("[MemUsageChange] Allocated %0.2f GiB for max tokens in paged KV cache (%d).",
            cacheSizeBytes / static_cast<double>(1 << 30), maxNumTokens);
    }
}

void KVCacheManager::startScheduling()
{
    mBlockManager.startScheduling();
}

SizeType32 KVCacheManager::getNeededBlocksOneStep(LlmRequest const& req, bool twoStepsLookAhead) const
{
    SizeType32 numRequiredBlocks = 0;
    SizeType32 const numDraftTokens = req.getNumDraftTokens();
    SizeType32 const generatedTokens = req.getMaxNumGeneratedTokens();
    SizeType32 const maxTokensToAddToKVCache = req.mMaxNewTokens - generatedTokens + 1;
    SizeType32 const numTokensPerStep = std::min(numDraftTokens + 1, maxTokensToAddToKVCache);
    SizeType32 const numDraftTokensPerStep = std::min(numDraftTokens, maxTokensToAddToKVCache);
    if (req.isContextInitState() && req.isFirstContextChunk())
    {
        // Assumes shared among beam = True
        auto const promptCacheLen
            = std::min(
                  (isCrossKv() ? req.getEncoderLen() : req.mPromptLen) + numDraftTokensPerStep, mMaxAttentionWindow)
            + mSinkBubbleLength;
        auto const numSharedBlocks = promptCacheLen / getTokensPerBlock();
        auto const numUnSharedTokens = promptCacheLen % getTokensPerBlock();
        auto const numUnSharedBlocks
            = tc::ceilDiv(numUnSharedTokens, getTokensPerBlock()) * req.mSamplingConfig.beamWidth;
        numRequiredBlocks = numSharedBlocks + numUnSharedBlocks;
    }
    else if (req.isGenerationInProgressState())
    {
        if (isCrossKv())
        {
            return 0;
        }
        // Here we need to check if next token or the one after would require a new block
        // Because the active requests could be in flight, thus will only get their number
        // of generated tokens to be updated after scheduling
        auto const numPastTokens = req.mPromptLen + generatedTokens + mSinkBubbleLength - 1;
        auto const numNextTokens = numPastTokens + (twoStepsLookAhead ? 2 : 1) * numTokensPerStep;

        if (numNextTokens > mMaxTokenNum)
        {
            return 0;
        }

        auto const numPastBlocks = tc::ceilDiv(numPastTokens, getTokensPerBlock());
        auto const numNextBlocks = tc::ceilDiv(numNextTokens, getTokensPerBlock());
        numRequiredBlocks = (numNextBlocks - numPastBlocks) * req.mSamplingConfig.beamWidth;
    }
    return numRequiredBlocks;
}

SizeType32 KVCacheManager::getNeededBlocksToCompletion(LlmRequest const& req) const
{
    if (isCrossKv())
    {
        return req.getEncoderLen() / getTokensPerBlock();
    }
    SizeType32 numContextBlocks
        = (std::min(req.mPromptLen, mMaxAttentionWindow) + mSinkBubbleLength) / getTokensPerBlock();
    SizeType32 remainingTokens = std::min(req.mPromptLen + req.mMaxNewTokens, mMaxAttentionWindow) + mSinkBubbleLength
        - numContextBlocks * getTokensPerBlock();
    auto neededBlocks
        = numContextBlocks + tc::ceilDiv(remainingTokens, getTokensPerBlock()) * req.mSamplingConfig.beamWidth;
    return neededBlocks;
}

void KVCacheManager::resetBlockOffsets(SizeType32 seqSlotIdx, SizeType32 beamWidth)
{
    auto* offsetsPtr = bufferCast<tk::KVCacheIndex>(*mSequenceBlockIndices);
    auto const& offsetsShape = mSequenceBlockIndices->getShape();
    auto const begin = tc::flat_index(offsetsShape.d, seqSlotIdx * mMaxBeamWidth, 0, 0);
    auto const end = begin + beamWidth * offsetsShape.d[1] * offsetsShape.d[2];
    std::fill(offsetsPtr + begin, offsetsPtr + end,
        tk::KVCacheIndex{std::numeric_limits<tk::KVCacheIndex::UnderlyingType>::max()});
}

void KVCacheManager::cacheBlockOffsets(GenerationRequest const& seq, SizeType32 seqSlotIdx)
{
    auto const& cacheBlocks = seq.getCacheBlockIds();
    auto const beamWidth = seq.getBeamWidth();

    auto* offsetsPtr = bufferCast<tk::KVCacheIndex>(*mSequenceBlockIndices);
    auto const& offsetsShape = mSequenceBlockIndices->getShape();

    for (SizeType32 beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
    {
        auto const& beamCacheBlock = cacheBlocks[beamIdx];
        for (SizeType32 blockIdx = 0; blockIdx < static_cast<SizeType32>(beamCacheBlock.size()); ++blockIdx)
        {
            auto const blockId = beamCacheBlock.at(blockIdx);
            setOffsets(offsetsPtr, offsetsShape, seqSlotIdx, beamIdx, blockIdx, blockId);
        }
    }
}

void KVCacheManager::cacheNewBlockOffsets(GenerationRequest const& seq, SizeType32 seqSlotIdx)
{
    auto const& cacheBlocks = seq.getCacheBlockIds();
    auto const beamWidth = seq.getBeamWidth();

    auto* offsetsPtr = bufferCast<tk::KVCacheIndex>(*mSequenceBlockIndices);
    auto const& offsetsShape = mSequenceBlockIndices->getShape();

    for (SizeType32 beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
    {
        auto const& beamCacheBlock = cacheBlocks[beamIdx];
        auto const blockId = beamCacheBlock.back();
        auto const blockIdx = static_cast<SizeType32>(beamCacheBlock.size() - 1);
        setOffsets(offsetsPtr, offsetsShape, seqSlotIdx, beamIdx, blockIdx, blockId);
    }
}

void KVCacheManager::updateNewBlockPointer(GenerationRequest const& seq, SizeType32 seqSlotIdx, SizeType32 blockIdx)
{
    auto const& cacheBlocks = seq.getCacheBlockIds();
    auto const beamWidth = seq.getBeamWidth();

    auto* offsetsPtr = bufferCast<tk::KVCacheIndex>(*mSequenceBlockIndices);
    auto const& offsetsShape = mSequenceBlockIndices->getShape();

    for (SizeType32 beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
    {
        auto const& beamCacheBlock = cacheBlocks[beamIdx];
        auto const blockId = beamCacheBlock.at(blockIdx);
        setOffsets(offsetsPtr, offsetsShape, seqSlotIdx, beamIdx, blockIdx, blockId);
    }
}

void KVCacheManager::addContextTokens(SizeType32 seqSlotIdx, SizeType32 numTokens)
{
    auto seq = mSequences.at(seqSlotIdx);
    seq->addNewTokens(numTokens);
    SizeType32 numContextOnlyBlocks = numTokens / getTokensPerBlock();
    for (SizeType32 i = 0; i < numContextOnlyBlocks; ++i)
    {
        mBlockManager.allocateBlock(*seq, true);
        cacheNewBlockOffsets(*seq, seqSlotIdx);
    }
    if (numContextOnlyBlocks * getTokensPerBlock() < numTokens)
    {
        mBlockManager.allocateBlock(*seq);
        cacheNewBlockOffsets(*seq, seqSlotIdx);
    }
}

void KVCacheManager::updateToken(SizeType32 seqSlotIdx, bool addToken)
{
    auto& seq = *mSequences.at(seqSlotIdx);
    auto currNumTokens = seq.getNumTokens();

    if (addToken)
    {
        seq.addNewTokens(1);
    }
    else
    {
        seq.removeTokens(1);
    }

    auto newNumTokens = seq.getNumTokens();

    if (!addToken)
    {
        std::swap(currNumTokens, newNumTokens);
    }

    SizeType32 const cyclicTokenNum = mMaxTokenNum - mSinkBlockTokenLength;
    SizeType32 const nextTokenIdxInCycle = (currNumTokens - mSinkBlockTokenLength) % cyclicTokenNum;
    SizeType32 const nextTokenIdxInCache = mSinkBlockTokenLength + nextTokenIdxInCycle;

    // (nextTokenIdxInCache - mSinkBlockTokenLength) % cyclicTokenNum == 0)
    // <=> nextTokenIdxInCycle == 0
    // <=> nextTokenIdxInCache == mSinkBlockTokenLength
    // => nextTokenIdxInCache % getTokensPerBlock() == 0

    // Check if require a new block
    if (nextTokenIdxInCache % getTokensPerBlock() == 0)
    {
        if (newNumTokens <= mMaxTokenNum)
        {
            if (addToken)
            {
                mBlockManager.allocateBlock(seq);
                cacheNewBlockOffsets(seq, seqSlotIdx);
            }
            else
            {
                mBlockManager.releaseLastBlock(seq);
            }
        }
        else if (seq.getBeamWidth() > 1)
        {
            TLLM_CHECK_WITH_INFO(addToken, "Remove token is not supported with beam search");
            // Get next block index
            SizeType32 nextBlockIdx = nextTokenIdxInCache / getTokensPerBlock();
            // Replace the shared block with the unshared ones
            mBlockManager.replaceSharedBlock(seq, nextBlockIdx);
            updateNewBlockPointer(seq, seqSlotIdx, nextBlockIdx);
        }
    }
}

void KVCacheManager::addToken(SizeType32 seqSlotIdx)
{
    updateToken(seqSlotIdx, true);
}

void KVCacheManager::addSequence(
    SizeType32 seqSlotIdx, SizeType32 inputLength, SizeType32 beamWidth, std::shared_ptr<LlmRequest> const& llmRequest)
{
    TLLM_CHECK(seqSlotIdx < mMaxNumSequences);
    resetBlockOffsets(seqSlotIdx, beamWidth);

    // Need to add the bubble after the sink tokens to use even block size
    inputLength += mSinkBubbleLength;
    auto sequence = std::make_shared<GenerationRequest>(seqSlotIdx, inputLength, beamWidth);

    // Enable cyclic kv cache when inputLength exceeds maxAttentionWindow.
    // Note that currently cyclic kv cache doesn't work with shared kv cache of different beams.
    bool const enableCyclicKvCache = inputLength >= mMaxTokenNum;

    // Get the final token index in kv cache
    SizeType32 finalTokenKVIdx
        = mSinkBlockTokenLength + (inputLength - 1 - mSinkBlockTokenLength) % (mMaxTokenNum - mSinkBlockTokenLength);

    // Get block index that with shareAmongBeams=False.
    SizeType32 unsharedBlockIdx = -1;
    if (!enableCyclicKvCache || beamWidth > 1 || finalTokenKVIdx % getTokensPerBlock() > 0)
    {
        unsharedBlockIdx = ((finalTokenKVIdx + 1) % getTokensPerBlock() == 0)
            ? finalTokenKVIdx / getTokensPerBlock() + 1
            : finalTokenKVIdx / getTokensPerBlock();
    }

    inputLength = std::min(inputLength, mMaxTokenNum);
    auto const numContextBlocks = tc::ceilDiv(inputLength, getTokensPerBlock());

    if (!enableCyclicKvCache && mEnableBlockReuse)
    {
        mBlockManager.addSequence(*sequence, inputLength, llmRequest);
    }
    else
    {
        mBlockManager.addSequence(*sequence, numContextBlocks, unsharedBlockIdx);
    }
    cacheBlockOffsets(*sequence, seqSlotIdx);
    mSequences[sequence->getSequenceSlotIdx()] = std::move(sequence);
}

void KVCacheManager::removeSequence(SizeType32 seqSlotIdx, std::shared_ptr<LlmRequest> const& llmRequest)
{
    auto& seq = mSequences.at(seqSlotIdx);
    if (seq)
    {
        // Free all blocks for this sequence
        if (mEnableBlockReuse)
        {
            mBlockManager.releaseBlocks(*seq, llmRequest);
        }
        else
        {
            mBlockManager.releaseBlocks(*seq, {});
        }
    }
    // Release sequence
    seq.reset();
}

void KVCacheManager::schedulingRemoveSequence(SizeType32 seqSlotIdx)
{
    auto& seq = mSequences.at(seqSlotIdx);
    // Mimic Free all blocks for this sequence
    mBlockManager.schedulingReleaseBlocks(*seq);
}

ITensor::UniquePtr KVCacheManager::getBlockPoolPointers() const
{
    auto poolPtrs = BufferManager::cpu(ITensor::makeShape({2}), TRTDataType<void*>::value);
    auto poolPtrsRange = BufferRange<void*>(*poolPtrs);
    poolPtrsRange[0] = mBlockManager.getPrimaryPool()->data();
    auto secondaryPool = mBlockManager.getSecondaryPool();
    poolPtrsRange[1] = secondaryPool ? secondaryPool->data() : nullptr;
    return poolPtrs;
}

SizeType32 KVCacheManager::copyBlockOffsets(
    ITensor& output, SizeType32 outputSlotOffset, SizeType32 seqSlotIdx, SizeType32 beamWidth) const
{
    auto* dstPtr = bufferCast<tk::KVCacheIndex>(output);
    auto* srcPtr = bufferCast<tk::KVCacheIndex>(*mSequenceBlockIndices);
    auto const& dstShape = output.getShape();
    auto const& srcShape = mSequenceBlockIndices->getShape();

    SizeType32 constexpr kIdx = 0;
    SizeType32 constexpr vIdx = 1;

    auto const& sequence = mSequences[seqSlotIdx];
    SizeType32 maxBlockCount{0};
    // Get page table for each KV cache pool
    for (SizeType32 beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
    {
        auto const beamBlockCount = sequence->getCacheBlockIds()[beamIdx].size();
        auto const copyChunkSize = beamBlockCount * sizeof(tk::KVCacheIndex);

        for (auto xIdx : {kIdx, vIdx})
        {
            auto const srcIndex = tc::flat_index(srcShape.d, seqSlotIdx * mMaxBeamWidth + beamIdx, xIdx, 0);
            auto const dstIndex = tc::flat_index(dstShape.d, outputSlotOffset + beamIdx, xIdx, 0);
            std::memcpy(dstPtr + dstIndex, srcPtr + srcIndex, copyChunkSize);
        }
        maxBlockCount = std::max<SizeType32>(maxBlockCount, static_cast<SizeType32>(beamBlockCount));
    }
    return maxBlockCount;
}

void KVCacheManager::getBlockOffsetsOfBatch(
    ITensor& output, SizeType32 firstBatchSlotIdx, SizeType32 batchSize, SizeType32 beamWidth) const
{
    // Get page table for each KV cache pool
    for (auto batchSlotIdx = 0; batchSlotIdx < batchSize; ++batchSlotIdx)
    {
        copyBlockOffsets(output, batchSlotIdx * beamWidth, firstBatchSlotIdx + batchSlotIdx, beamWidth);
    }
}

std::tuple<SizeType32, SizeType32> const KVCacheManager::calculateMaxNumBlocks(KvCacheConfig const& config,
    nvinfer1::DataType dtype, ModelConfig const& modelConfig, WorldConfig const& worldConfig,
    runtime::BufferManager const& bufferManager)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const freeMemFraction = config.freeGpuMemoryFraction.value_or(KvCacheConfig::kDefaultGpuMemFraction);
    TLLM_CHECK_WITH_INFO(freeMemFraction < 1.0f,
        "Invalid freeMemFraction, freeMemFraction (%f) must be smaller than 1.0f", freeMemFraction);
    auto const cacheSizePerToken
        = kv_cache_manager::KVCacheManager::calculateCacheSizePerToken(modelConfig, worldConfig);
    auto const cacheSizeBytesPerToken = cacheSizePerToken * BufferDataType(dtype).getSize();
    auto const [freeMem, totalMem] = tc::getDeviceMemoryInfo(config.useUvm);
    auto maxTokens = static_cast<SizeType32>(freeMemFraction
        * static_cast<double>(freeMem + bufferManager.memoryPoolFree()) / static_cast<double>(cacheSizeBytesPerToken));
    TLLM_LOG_INFO("Memory usage when calculating max tokens in paged kv cache: total: %0.2f GiB, available: %0.2f GiB",
        (totalMem / static_cast<double>(1 << 30)),
        ((freeMem + bufferManager.memoryPoolFree()) / static_cast<double>(1 << 30)));

    // If user specified a number of tokens
    if (config.maxTokens.has_value())
    {
        // If user also specified a free gpu memory fraction, take the min
        if (config.freeGpuMemoryFraction.has_value())
        {
            maxTokens = std::min(config.maxTokens.value(), maxTokens);
            //  else use the number of tokens specified by user
            TLLM_LOG_WARNING(
                "Both freeGpuMemoryFraction (aka kv_cache_free_gpu_mem_fraction) "
                "and maxTokens (aka max_tokens_in_paged_kv_cache) "
                "are set (to %f and %ld, respectively). The smaller value will be used.",
                freeMemFraction, (int64_t) config.maxTokens.value());
        }
        else
        {
            maxTokens = config.maxTokens.value();
        }
    }
    if (worldConfig.getSize() > 1)
    {
        TLLM_CHECK(worldConfig.validMpiConfig());
        // make sure all ranks use same value for maxTokens
        int64_t __maxTokensRank{maxTokens};
        int64_t __maxTokensWorld{0};
        COMM_SESSION.allreduce(&__maxTokensRank, &__maxTokensWorld, 1, mpi::MpiType::kINT64, mpi::MpiOp::MIN);
        maxTokens = static_cast<SizeType32>(__maxTokensWorld);
    }

    auto const tokensPerBlock = modelConfig.getTokensPerBlock();
    auto const blocksInPrimaryPool = tc::ceilDiv(maxTokens, tokensPerBlock);
    TLLM_LOG_INFO("Number of blocks in KV cache primary pool: %d", blocksInPrimaryPool);

    auto maxTokensSecondary = static_cast<SizeType32>(config.hostCacheSize.value_or(0) / cacheSizeBytesPerToken);
    auto const blocksInSecondaryPool = std::max(0, maxTokensSecondary / tokensPerBlock);
    TLLM_LOG_INFO("Number of blocks in KV cache secondary pool: %d, onboard blocks to primary memory before reuse: %s",
        blocksInSecondaryPool, config.onboardBlocks ? "true" : "false");

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return std::make_tuple(blocksInPrimaryPool, blocksInSecondaryPool);
}

void KVCacheManager::removeToken(SizeType32 seqSlotIdx)
{
    auto& seq = *mSequences.at(seqSlotIdx);
    auto const beamWidth = seq.getBeamWidth();

    TLLM_CHECK_WITH_INFO(beamWidth == 1, "removeToken does not support beamWidth > 1");
    if (seq.getNumTokens() == 0)
    {
        return;
    }
    updateToken(seqSlotIdx, false);
}

void KVCacheManager::rewindKVCache(SizeType32 seqSlotIdx, SizeType32 rewindLengths)
{
    for (SizeType32 si = 0; si < rewindLengths; ++si)
    {
        removeToken(seqSlotIdx);
    }
}

GenerationRequest const& KVCacheManager::getSequence(SizeType32 seqSlotIdx) const
{
    auto reqPtr = mSequences.at(seqSlotIdx);
    TLLM_CHECK(reqPtr);
    return *reqPtr;
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager
