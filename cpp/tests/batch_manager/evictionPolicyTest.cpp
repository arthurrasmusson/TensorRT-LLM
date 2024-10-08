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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <vector>

namespace tc = tensorrt_llm::common;
namespace tk = tensorrt_llm::kernels;
namespace tr = tensorrt_llm::runtime;

using namespace tensorrt_llm::batch_manager::eviction_policy;
using namespace tensorrt_llm::batch_manager::kv_cache_manager;

#define NUM_PRIMARY_BLOCKS 8
#define NUM_SECONDARY_BLOCKS 4

class LRUPolicyTest : public ::testing::Test
{
public:
    void SetUp() override
    {
        policy = std::make_shared<LRUEvictionPolicy>();
        std::vector<BlockPtr> allBlocksById;

        for (KVCacheBlock::IdType blockId = 0; blockId < NUM_PRIMARY_BLOCKS; ++blockId)
        {
            allBlocksById.push_back(std::make_shared<KVCacheBlock>(blockId, tk::KVCacheIndex{blockId, false}));
        }

        for (KVCacheBlock::IdType blockId = 0; blockId < NUM_SECONDARY_BLOCKS; ++blockId)
        {
            allBlocksById.push_back(
                std::make_shared<KVCacheBlock>(NUM_PRIMARY_BLOCKS + blockId, tk::KVCacheIndex{blockId, true}));
        }
        policy->initialize(allBlocksById, NUM_PRIMARY_BLOCKS, NUM_SECONDARY_BLOCKS);
    }

    void TearDown() override {}

    std::shared_ptr<BaseEvictionPolicy> policy;
};

TEST_F(LRUPolicyTest, NumFreeBlocksTest)
{
    EXPECT_EQ(NUM_PRIMARY_BLOCKS, policy->getNumFreePrimaryBlocks());
    EXPECT_EQ(NUM_SECONDARY_BLOCKS, policy->getNumFreeSecondaryBlocks());

    auto primaryBlock = policy->getFreePrimaryBlock();
    policy->claimBlock(*primaryBlock);
    EXPECT_EQ(NUM_PRIMARY_BLOCKS - 1, policy->getNumFreePrimaryBlocks());
    EXPECT_EQ(NUM_SECONDARY_BLOCKS, policy->getNumFreeSecondaryBlocks());

    auto secondaryBlock = policy->getFreeSecondaryBlock();
    policy->claimBlock(*secondaryBlock);
    EXPECT_EQ(NUM_PRIMARY_BLOCKS - 1, policy->getNumFreePrimaryBlocks());
    EXPECT_EQ(NUM_SECONDARY_BLOCKS - 1, policy->getNumFreeSecondaryBlocks());
}

TEST_F(LRUPolicyTest, GetFreeBlockTest)
{
    auto primaryBlock = policy->getFreePrimaryBlock();
    EXPECT_FALSE(primaryBlock->hasRefs());
    EXPECT_TRUE(primaryBlock->isPrimary());

    auto secondaryBlock = policy->getFreeSecondaryBlock();
    EXPECT_FALSE(secondaryBlock->hasRefs());
    EXPECT_FALSE(secondaryBlock->isPrimary());
}

TEST_F(LRUPolicyTest, ReleaseBlockTest)
{
    auto origPrimaryBlock = policy->getFreePrimaryBlock();
    policy->claimBlock(*origPrimaryBlock);

    EXPECT_NE(origPrimaryBlock->getBlockId(), policy->getFreePrimaryBlock()->getBlockId());

    policy->releaseBlock(origPrimaryBlock, true);
    EXPECT_EQ(origPrimaryBlock->getBlockId(), policy->getFreePrimaryBlock()->getBlockId());

    policy->claimBlock(*origPrimaryBlock);
    policy->releaseBlock(origPrimaryBlock);

    EXPECT_NE(origPrimaryBlock->getBlockId(), policy->getFreePrimaryBlock()->getBlockId());
}

TEST_F(LRUPolicyTest, LRUTest)
{
    auto block1 = policy->getFreePrimaryBlock();
    policy->claimBlock(*block1);

    auto block2 = policy->getFreePrimaryBlock();
    policy->claimBlock(*block2);

    policy->releaseBlock(block2);
    policy->releaseBlock(block1);

    for (int i = 0; i < NUM_PRIMARY_BLOCKS - 2; i++)
    {
        auto block = policy->getFreePrimaryBlock();
        policy->claimBlock(*block);
    }
    ASSERT_EQ(policy->getFreePrimaryBlock()->getBlockId(), block2->getBlockId());

    policy->claimBlock(*policy->getFreePrimaryBlock());

    ASSERT_EQ(policy->getFreePrimaryBlock()->getBlockId(), block1->getBlockId());
}
