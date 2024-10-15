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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/runtime/iBuffer.h"

#include <algorithm>
#include <memory>
#include <set>

using namespace tensorrt_llm::batch_manager;
using namespace tensorrt_llm::batch_manager::kv_cache_manager;
namespace tc = tensorrt_llm::common;
namespace tk = tensorrt_llm::kernels;
namespace tr = tensorrt_llm::runtime;

using ParamType = bool;

std::string generateTestName(testing::TestParamInfo<ParamType> const& info)
{
    auto const homogeneousLayers = info.param;
    std::string name = "KVCacheManagerTest";
    if (homogeneousLayers)
    {
        name += "Homogeneous";
    }
    else
    {
        name += "Heterogeneous";
    }
    return name;
}

class KVCacheManagerTest
    : public ::testing::Test,
      public ::testing::WithParamInterface<ParamType> // NOLINT(cppcoreguidelines-pro-type-member-init)
{
protected:
    void SetUp() override
    {
        auto const deviceCount = tc::getDeviceCount();
        if (deviceCount == 0)
        {
            GTEST_SKIP();
        }
    }

    void TearDown() override {}
};

namespace
{
void allocateBlocks(BlockManager& manager, GenerationRequest& sequence, std::size_t numBlocks, bool shareAmongBeams)
{
    for (std::size_t i = 0; i < numBlocks; ++i)
    {
        manager.allocateBlock(sequence, shareAmongBeams);
    }
}
} // namespace

TEST_F(KVCacheManagerTest, BlockManagerTest)
{
    auto constexpr numLayers = 12;
    auto constexpr numKvHeads = 6;
    auto constexpr sizePerHead = 128;
    auto constexpr tokensPerBlock = 64;
    auto constexpr blocksInPrimaryPool = 24;
    auto constexpr blocksInSecondaryPool = 0;
    auto constexpr maxNumSequences = 8;
    auto const stream = std::make_shared<tr::CudaStream>();
    auto constexpr onboardBlocks = true;

    BlockManager blockManager(std::vector<BlockManager::SizeType32>(numLayers, numKvHeads), sizePerHead, tokensPerBlock,
        blocksInPrimaryPool, blocksInSecondaryPool, maxNumSequences, stream, onboardBlocks);

    EXPECT_EQ(blockManager.getTokensPerBlock(), tokensPerBlock);
    EXPECT_EQ(blockManager.getMaxNumBlocks(), blocksInPrimaryPool);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    auto constexpr requestId = 42;
    auto constexpr beamWidth = 8;
    auto constexpr numBlocksPerBeam = blocksInPrimaryPool / beamWidth;
    auto constexpr numTokens = tokensPerBlock * numBlocksPerBeam;
    GenerationRequest seq0{requestId, numTokens, beamWidth, numBlocksPerBeam};
    blockManager.addSequence(seq0, numBlocksPerBeam, numBlocksPerBeam - 1);
    auto constexpr occupiedBlocks = (numBlocksPerBeam - 1) + beamWidth;
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - occupiedBlocks);
    auto const& ids = seq0.getCacheBlockIds();
    std::set<std::int32_t> idSet{};
    EXPECT_EQ(ids.size(), beamWidth);
    for (auto const& beam : ids)
    {
        EXPECT_EQ(beam.size(), blocksInPrimaryPool / beamWidth);
        idSet.insert(beam.begin(), beam.end());
    }
    EXPECT_EQ(idSet.size(), occupiedBlocks);
    blockManager.releaseBlocks(seq0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    blockManager.addSequence(seq0, numBlocksPerBeam, -1);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocksPerBeam);
    EXPECT_EQ(ids.size(), beamWidth);
    for (std::size_t i = 0u; i < ids.front().size(); ++i)
    {
        for (std::size_t beam = 1u; beam < ids.size(); ++beam)
        {
            EXPECT_EQ(ids.at(beam).at(i), ids.at(0).at(i));
        }
    }
    blockManager.releaseBlocks(seq0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    // occupy 22/24 blocks
    EXPECT_NO_THROW(blockManager.addSequence(seq0, numBlocksPerBeam, numBlocksPerBeam - 1));
    GenerationRequest seq1{requestId + 1, numTokens, beamWidth, numBlocksPerBeam};
    EXPECT_NO_THROW(blockManager.addSequence(seq1, numBlocksPerBeam, numBlocksPerBeam - 1));
    // same requestId not allowed
    GenerationRequest seq2{requestId, numTokens, beamWidth, numBlocksPerBeam};
    EXPECT_THROW(blockManager.addSequence(seq2, numBlocksPerBeam, numBlocksPerBeam - 1), std::runtime_error);
    // no more blocks
    GenerationRequest seq3{requestId + 2, numTokens, beamWidth, numBlocksPerBeam};
    EXPECT_THROW(blockManager.addSequence(seq3, numBlocksPerBeam, numBlocksPerBeam - 1), std::runtime_error);
}

TEST_F(KVCacheManagerTest, BlockManagerReuseTest)
{
    // tc::Logger::getLogger()->setLevel(tc::Logger::Level::DEBUG);

    auto constexpr numLayers = 12;
    auto constexpr numKvHeads = 6;
    auto constexpr sizePerHead = 16;
    auto constexpr tokensPerBlock = 4;
    auto constexpr maxBlocksPerSeq = 4;
    auto constexpr blocksInPrimaryPool = 8;
    auto constexpr blocksInSecondaryPool = 0;
    auto constexpr maxNumSequences = 8;
    auto const stream = std::make_shared<tr::CudaStream>();
    auto constexpr onboardBlocks = true;

    BlockManager blockManager(std::vector<BlockManager::SizeType32>(numLayers, numKvHeads), sizePerHead, tokensPerBlock,
        blocksInPrimaryPool, blocksInSecondaryPool, maxNumSequences, stream, onboardBlocks);

    EXPECT_EQ(blockManager.getTokensPerBlock(), tokensPerBlock);
    EXPECT_EQ(blockManager.getMaxNumBlocks(), blocksInPrimaryPool);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    SizeType32 constexpr maxNewTokens{0};
    auto constexpr beamWidth = 1;
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    auto inputTokens = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8});
    auto const inputLength = static_cast<SizeType32>(inputTokens->size());
    LlmRequest::RequestIdType requestId{0};
    auto llmRequest0 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming);

    GenerationRequest seq0{requestId, inputLength, beamWidth, maxBlocksPerSeq};

    ///////////////////////////////////////////////////////////////////////////
    // add request and then remove it
    auto constexpr beamIdx = 0;
    auto promptLen0 = llmRequest0->getNumTokens(beamIdx);
    auto numContextBlocks0 = tc::ceilDiv(promptLen0, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq0, promptLen0, numContextBlocks0, llmRequest0);
    EXPECT_EQ(llmRequest0->getContextCurrentPosition(), 0);
    EXPECT_THAT(seq0.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({0, 1, 2}));
    llmRequest0->addNewToken(9, beamIdx);
    llmRequest0->addNewToken(10, beamIdx);
    auto numTokens = llmRequest0->getNumTokens(beamIdx);
    auto numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(numBlocks, 3);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    // blocks 0, 1, 2 are stored for reuse (block 2 contains [8, 9])
    blockManager.releaseBlocks(seq0, llmRequest0);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // new request with same tokens and then remove it
    requestId = 1;
    auto llmRequest1 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming);
    GenerationRequest seq1{requestId, inputLength, beamWidth, maxBlocksPerSeq};

    // reuse blocks 0, 1 and get new block 3
    auto promptLen1 = llmRequest1->getNumTokens(beamIdx);
    auto numContextBlocks1 = tc::ceilDiv(promptLen1, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq1, promptLen1, numContextBlocks1, llmRequest1);
    EXPECT_EQ(llmRequest1->getContextCurrentPosition(), 2 * tokensPerBlock);
    EXPECT_THAT(seq1.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({0, 1, 3}));
    llmRequest1->addNewToken(9, beamIdx);
    llmRequest1->addNewToken(10, beamIdx);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    // block 3 matches block 2 and will be freed
    blockManager.releaseBlocks(seq1, llmRequest1);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // add both requests again and then remove them
    // reuse blocks 0, 1 and get new block 4
    auto inputTokens0 = std::make_shared<VecTokens>(*inputTokens);
    inputTokens0->emplace_back(9);
    llmRequest0 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens0, samplingConfig, isStreaming);
    promptLen0 = llmRequest0->getNumTokens(beamIdx);
    numContextBlocks0 = tc::ceilDiv(promptLen0, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq0, promptLen0, numContextBlocks0, llmRequest0);
    EXPECT_EQ(llmRequest0->getContextCurrentPosition(), 2 * tokensPerBlock);
    EXPECT_THAT(seq0.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({0, 1, 4}));
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    // reuse blocks 0, 1 and reuse block 2
    auto inputTokens1 = std::make_shared<VecTokens>(llmRequest1->getTokens(0));
    llmRequest1 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens1, samplingConfig, isStreaming);
    promptLen1 = llmRequest1->getNumTokens(beamIdx);
    numContextBlocks1 = tc::ceilDiv(promptLen1, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq1, promptLen1, numContextBlocks1, llmRequest1);
    EXPECT_EQ(llmRequest1->getContextCurrentPosition(), llmRequest1->getNumTokens(beamIdx) - 1);
    EXPECT_THAT(seq1.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({0, 1, 2}));
    llmRequest1->addNewToken(10, beamIdx);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks + 1);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks - 1);

    blockManager.releaseBlocks(seq0, llmRequest0);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);
    // blocks 2 is stored for reuse (block contains [8, 9, 10])
    blockManager.releaseBlocks(seq1, llmRequest1);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // add request with less tokens
    auto inputLength2 = tokensPerBlock + 1;
    auto inputTokens2
        = std::make_shared<VecTokens>(VecTokens{inputTokens->begin(), inputTokens->begin() + inputLength2});
    requestId = 2;
    auto llmRequest2 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens2, samplingConfig, isStreaming);

    numTokens = llmRequest2->getNumTokens(beamIdx);
    GenerationRequest seq2{requestId, numTokens, beamWidth, maxBlocksPerSeq};
    // reuse block 0, get new block 5
    auto promptLen2 = llmRequest2->getNumTokens(beamIdx);
    auto numContextBlocks2 = tc::ceilDiv(promptLen2, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq2, promptLen2, numContextBlocks2, llmRequest2);
    EXPECT_EQ(llmRequest2->getContextCurrentPosition(), tokensPerBlock);
    EXPECT_THAT(seq2.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({0, 5}));
    llmRequest2->addNewToken(5, beamIdx);
    numTokens = llmRequest2->getNumTokens(beamIdx);
    numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    ///////////////////////////////////////////////////////////////////////////
    // add request with more tokens
    auto inputLength3 = 11;
    auto inputTokens3 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11});
    requestId = 3;
    auto llmRequest3 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens3, samplingConfig, isStreaming);

    numTokens = llmRequest3->getNumTokens(beamIdx);
    GenerationRequest seq3{requestId, numTokens, beamWidth, maxBlocksPerSeq};
    // reuse blocks 0, 1, get new block 6
    auto promptLen3 = llmRequest3->getNumTokens(beamIdx);
    auto numContextBlocks3 = tc::ceilDiv(promptLen3, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq3, promptLen3, numContextBlocks3, llmRequest3);
    EXPECT_EQ(llmRequest3->getContextCurrentPosition(), 2 * tokensPerBlock);
    EXPECT_THAT(seq3.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({0, 1, 6}));
    llmRequest3->addNewToken(11, beamIdx);
    numTokens = llmRequest3->getNumTokens(beamIdx);
    // one block used by both seq2 and seq3
    numBlocks += tc::ceilDiv(numTokens, tokensPerBlock) - 1;
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    blockManager.releaseBlocks(seq2, llmRequest2);
    blockManager.releaseBlocks(seq3, llmRequest3);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // add request with 11 tokens, then discard few tokens from request and release a shorter one
    auto inputTokens4 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12});
    auto inputTokens4Short = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8});
    requestId = 4;
    auto llmRequest4 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens4, samplingConfig, isStreaming);

    numTokens = llmRequest4->getNumTokens(beamIdx);
    GenerationRequest seq4{requestId, numTokens, beamWidth, maxBlocksPerSeq};

    // reuse blocks 0, 1 and add new block 7
    auto promptLen4 = llmRequest4->getNumTokens(beamIdx);
    auto numContextBlocks4 = tc::ceilDiv(promptLen4, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq4, promptLen4, numContextBlocks4, llmRequest4);
    EXPECT_EQ(llmRequest4->getContextCurrentPosition(), 2 * tokensPerBlock);
    EXPECT_THAT(seq4.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({0, 1, 7}));
    numTokens = llmRequest4->getNumTokens(beamIdx);
    numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    auto llmRequest4Short
        = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens4Short, samplingConfig, isStreaming);

    blockManager.releaseBlocks(seq4, llmRequest4Short);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // add request with 11 tokens again and make sure no discarded tokens reuse happens
    // reuse blocks 0, 1 and add block 3
    promptLen4 = llmRequest4->getNumTokens(beamIdx);
    numContextBlocks4 = tc::ceilDiv(promptLen4, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq4, promptLen4, numContextBlocks4, llmRequest4);
    EXPECT_EQ(llmRequest4->getContextCurrentPosition(), 2 * tokensPerBlock);
    EXPECT_THAT(seq4.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({0, 1, 3}));
    numTokens = llmRequest4->getNumTokens(beamIdx);
    numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    blockManager.releaseBlocks(seq4, llmRequest4);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // add request with max size that doesn't reuse blocks
    auto inputLength5 = blocksInPrimaryPool * tokensPerBlock - 1;
    auto inputTokens5 = std::make_shared<VecTokens>(VecTokens(inputLength5, 0));
    requestId = 5;
    auto llmRequest5 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens5, samplingConfig, isStreaming);

    numTokens = llmRequest5->getNumTokens(beamIdx);
    GenerationRequest seq5{requestId, numTokens, beamWidth, maxBlocksPerSeq};
    // no reuse, all blocks need to be freed
    auto promptLen5 = llmRequest5->getNumTokens(beamIdx);
    auto numContextBlocks5 = tc::ceilDiv(promptLen5, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq5, promptLen5, numContextBlocks5, llmRequest5);
    llmRequest5->addNewToken(0, beamIdx);
    EXPECT_EQ(llmRequest5->getContextCurrentPosition(), 0);

    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), blocksInPrimaryPool);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), 0);

    blockManager.releaseBlocks(seq5, llmRequest5);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // add request with min size that doesn't reuse blocks
    auto inputLength6 = 1;
    auto inputTokens6 = std::make_shared<VecTokens>(VecTokens(inputLength6, 0));
    requestId = 6;
    auto llmRequest6 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens6, samplingConfig, isStreaming);

    numTokens = llmRequest6->getNumTokens(beamIdx);
    GenerationRequest seq6{requestId, numTokens, beamWidth, maxBlocksPerSeq};
    // no reuse, all blocks need to be freed
    auto promptLen6 = llmRequest6->getNumTokens(beamIdx);
    auto numContextBlocks6 = tc::ceilDiv(promptLen6, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq6, promptLen6, numContextBlocks6, llmRequest6);
    llmRequest6->addNewToken(0, beamIdx);
    EXPECT_EQ(llmRequest6->getContextCurrentPosition(), 0);

    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 1);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - 1);

    blockManager.releaseBlocks(seq6, llmRequest6);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);
}

TEST_F(KVCacheManagerTest, BlockManagerReuseWithExtraIdTest)
{
    // tc::Logger::getLogger()->setLevel(tc::Logger::Level::DEBUG);
    using VecTokenExtraIds = LlmRequest::VecTokenExtraIds;

    auto constexpr numLayers = 12;
    auto constexpr numKvHeads = 6;
    auto constexpr sizePerHead = 16;
    auto constexpr tokensPerBlock = 4;
    auto constexpr maxBlocksPerSeq = 4;
    auto constexpr blocksInPrimaryPool = 16;
    auto constexpr blocksInSecondaryPool = 0;
    auto constexpr maxNumSequences = 8;
    auto const stream = std::make_shared<tr::CudaStream>();
    auto constexpr onboardBlocks = true;

    BlockManager blockManager(std::vector(numLayers, numKvHeads), sizePerHead, tokensPerBlock, blocksInPrimaryPool,
        blocksInSecondaryPool, maxNumSequences, stream, onboardBlocks);

    EXPECT_EQ(blockManager.getTokensPerBlock(), tokensPerBlock);
    EXPECT_EQ(blockManager.getMaxNumBlocks(), blocksInPrimaryPool);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    SizeType32 constexpr maxNewTokens{0};
    auto constexpr beamWidth = 1;
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    // assume prompt id starts from 100
    auto inputTokens = std::make_shared<VecTokens>(VecTokens{100, 101, 102, 103, 104, 105, 0, 1, 2});
    auto inputTokenExtraIds = std::make_shared<VecTokenExtraIds>(VecTokenExtraIds{1, 1, 2, 2, 3, 3, 0, 0, 0});
    auto const inputLength = static_cast<SizeType32>(inputTokens->size());
    LlmRequest::RequestIdType requestId{0};
    auto llmRequest0 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt, std::nullopt, false,
        std::nullopt, false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt,
        LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, inputTokenExtraIds);

    GenerationRequest seq0{requestId, inputLength, beamWidth, maxBlocksPerSeq};

    ///////////////////////////////////////////////////////////////////////////
    // add request and then remove it
    auto constexpr beamIdx = 0;
    auto promptLen0 = llmRequest0->getNumTokens(beamIdx);
    auto numContextBlocks0 = tc::ceilDiv(promptLen0, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq0, promptLen0, numContextBlocks0, llmRequest0);
    EXPECT_EQ(llmRequest0->getContextCurrentPosition(), 0);
    EXPECT_THAT(seq0.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({0, 1, 2}));
    llmRequest0->addNewToken(3, beamIdx);
    llmRequest0->addNewToken(4, beamIdx);
    auto numTokens = llmRequest0->getNumTokens(beamIdx);
    auto numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(numBlocks, 3);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    // blocks 0, 1, 2 are stored for reuse (block 2 contains [(2, 0), (3, 0)])
    blockManager.releaseBlocks(seq0, llmRequest0);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // new request with same tokens and then remove it
    requestId = 1;
    auto llmRequest1 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt, std::nullopt, false,
        std::nullopt, false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt,
        LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, inputTokenExtraIds);
    GenerationRequest seq1{requestId, inputLength, beamWidth, maxBlocksPerSeq};

    // reuse blocks 0, 1 and get new block 3
    auto promptLen1 = llmRequest1->getNumTokens(beamIdx);
    auto numContextBlocks1 = tc::ceilDiv(promptLen1, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq1, promptLen1, numContextBlocks1, llmRequest1);
    EXPECT_EQ(llmRequest1->getContextCurrentPosition(), 2 * tokensPerBlock);
    EXPECT_THAT(seq1.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({0, 1, 3}));
    llmRequest1->addNewToken(3, beamIdx);
    llmRequest1->addNewToken(4, beamIdx);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    // block 3 matches block 2 and will be freed
    blockManager.releaseBlocks(seq1, llmRequest1);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // add both requests again and then remove them
    // reuse blocks 0, 1 and get new block 4
    llmRequest0 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt, std::nullopt, false,
        std::nullopt, false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt,
        LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, inputTokenExtraIds);
    llmRequest0->addNewToken(3, beamIdx);
    promptLen0 = llmRequest0->getNumTokens(beamIdx);
    numContextBlocks0 = tc::ceilDiv(promptLen0, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq0, promptLen0, numContextBlocks0, llmRequest0);
    EXPECT_EQ(llmRequest0->getContextCurrentPosition(), 2 * tokensPerBlock);
    EXPECT_THAT(seq0.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({0, 1, 4}));
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    // reuse blocks 0, 1 and reuse block 2
    auto inputTokens1 = std::make_shared<VecTokens>(llmRequest1->getTokens(0));
    auto inputTokenExtraIds1 = std::make_shared<VecTokenExtraIds>(*inputTokenExtraIds);
    inputTokenExtraIds1->push_back(0);
    inputTokenExtraIds1->push_back(0);
    llmRequest1 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens1, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt, std::nullopt, false,
        std::nullopt, false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt,
        LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, inputTokenExtraIds1);
    promptLen1 = llmRequest1->getNumTokens(beamIdx);
    numContextBlocks1 = tc::ceilDiv(promptLen1, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq1, promptLen1, numContextBlocks1, llmRequest1);
    EXPECT_EQ(llmRequest1->getContextCurrentPosition(), llmRequest1->getNumTokens(beamIdx) - 1);
    EXPECT_THAT(seq1.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({0, 1, 2}));
    llmRequest1->addNewToken(5, beamIdx);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks + 1);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks - 1);

    blockManager.releaseBlocks(seq0, llmRequest0);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);
    // blocks 2 is stored for reuse (block contains [(2, 0), (3, 0), (4, 0)])
    blockManager.releaseBlocks(seq1, llmRequest1);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // add request with totally different extra ids
    auto inputTokenExtraIds2 = std::make_shared<VecTokenExtraIds>(VecTokenExtraIds{4, 4, 5, 5, 6, 6, 0, 0, 0});
    requestId = 2;
    auto llmRequest2 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt, std::nullopt, false,
        std::nullopt, false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt,
        LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, inputTokenExtraIds2);

    numTokens = llmRequest2->getNumTokens(beamIdx);
    GenerationRequest seq2{requestId, numTokens, beamWidth, maxBlocksPerSeq};
    // no reuse, get new block 5, 6, 7
    auto promptLen2 = llmRequest2->getNumTokens(beamIdx);
    auto numContextBlocks2 = tc::ceilDiv(promptLen2, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq2, promptLen2, numContextBlocks2, llmRequest2);
    EXPECT_EQ(llmRequest2->getContextCurrentPosition(), 0);
    EXPECT_THAT(seq2.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({5, 6, 7}));
    llmRequest2->addNewToken(3, beamIdx);
    numTokens = llmRequest2->getNumTokens(beamIdx);
    numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    ///////////////////////////////////////////////////////////////////////////
    // add request with partial different extra ids
    auto inputTokenExtraIds3 = std::make_shared<VecTokenExtraIds>(VecTokenExtraIds{1, 1, 2, 2, 4, 4, 0, 0, 0});
    requestId = 3;
    auto llmRequest3 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt, std::nullopt, false,
        std::nullopt, false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt,
        LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, inputTokenExtraIds3);

    numTokens = llmRequest3->getNumTokens(beamIdx);
    GenerationRequest seq3{requestId, numTokens, beamWidth, maxBlocksPerSeq};
    // reuse block 0, get new block 8, 9
    auto promptLen3 = llmRequest3->getNumTokens(beamIdx);
    auto numContextBlocks3 = tc::ceilDiv(promptLen3, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq3, promptLen3, numContextBlocks3, llmRequest3);
    EXPECT_EQ(llmRequest3->getContextCurrentPosition(), tokensPerBlock);
    EXPECT_THAT(seq3.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({0, 8, 9}));
    llmRequest3->addNewToken(3, beamIdx);
    numTokens = llmRequest3->getNumTokens(beamIdx);
    numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks * 2);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks * 2);

    blockManager.releaseBlocks(seq2, llmRequest2);
    blockManager.releaseBlocks(seq3, llmRequest3);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);
}

TEST_F(KVCacheManagerTest, BlockManagerReuseWithLoraTaskIdTest)
{
    // tc::Logger::getLogger()->setLevel(tc::Logger::Level::DEBUG);
    using VecTokenExtraIds = LlmRequest::VecTokenExtraIds;

    auto constexpr numLayers = 12;
    auto constexpr numKvHeads = 6;
    auto constexpr sizePerHead = 16;
    auto constexpr tokensPerBlock = 4;
    auto constexpr maxBlocksPerSeq = 4;
    auto constexpr blocksInPrimaryPool = 16;
    auto constexpr blocksInSecondaryPool = 0;
    auto constexpr maxNumSequences = 8;
    auto const stream = std::make_shared<tr::CudaStream>();
    auto constexpr onboardBlocks = true;

    BlockManager blockManager(std::vector(numLayers, numKvHeads), sizePerHead, tokensPerBlock, blocksInPrimaryPool,
        blocksInSecondaryPool, maxNumSequences, stream, onboardBlocks);

    EXPECT_EQ(blockManager.getTokensPerBlock(), tokensPerBlock);
    EXPECT_EQ(blockManager.getMaxNumBlocks(), blocksInPrimaryPool);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    SizeType32 constexpr maxNewTokens{0};
    auto constexpr beamWidth = 1;
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    // loraTaskId is 1 for common cases
    LlmRequest::LoraTaskIdType loraTaskId{1};
    auto inputTokens = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8});
    auto const inputLength = static_cast<SizeType32>(inputTokens->size());
    LlmRequest::RequestIdType requestId{0};
    auto llmRequest0 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        loraTaskId);
    GenerationRequest seq0{requestId, inputLength, beamWidth, maxBlocksPerSeq};

    ///////////////////////////////////////////////////////////////////////////
    // add request and then remove it
    auto constexpr beamIdx = 0;
    auto promptLen0 = llmRequest0->getNumTokens(beamIdx);
    auto numContextBlocks0 = tc::ceilDiv(promptLen0, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq0, promptLen0, numContextBlocks0, llmRequest0);
    EXPECT_EQ(llmRequest0->getContextCurrentPosition(), 0);
    EXPECT_THAT(seq0.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({0, 1, 2}));
    llmRequest0->addNewToken(9, beamIdx);
    llmRequest0->addNewToken(10, beamIdx);
    auto numTokens = llmRequest0->getNumTokens(beamIdx);
    auto numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(numBlocks, 3);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    // blocks 0, 1, 2 are stored for reuse (block 2 contains [8, 9])
    blockManager.releaseBlocks(seq0, llmRequest0);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // new request with same tokens and loraTaskId, then remove it
    requestId = 1;
    auto llmRequest1 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        loraTaskId);
    GenerationRequest seq1{requestId, inputLength, beamWidth, maxBlocksPerSeq};

    // reuse blocks 0, 1 and get new block 3
    auto promptLen1 = llmRequest1->getNumTokens(beamIdx);
    auto numContextBlocks1 = tc::ceilDiv(promptLen1, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq1, promptLen1, numContextBlocks1, llmRequest1);
    EXPECT_EQ(llmRequest1->getContextCurrentPosition(), 2 * tokensPerBlock);
    EXPECT_THAT(seq1.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({0, 1, 3}));
    llmRequest1->addNewToken(9, beamIdx);
    llmRequest1->addNewToken(10, beamIdx);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    // block 3 matches block 2 and will be freed
    blockManager.releaseBlocks(seq1, llmRequest1);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // add both requests again and then remove them
    // reuse blocks 0, 1 and get new block 4
    llmRequest0 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        loraTaskId);
    llmRequest0->addNewToken(9, beamIdx);
    promptLen0 = llmRequest0->getNumTokens(beamIdx);
    numContextBlocks0 = tc::ceilDiv(promptLen0, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq0, promptLen0, numContextBlocks0, llmRequest0);
    EXPECT_EQ(llmRequest0->getContextCurrentPosition(), 2 * tokensPerBlock);
    EXPECT_THAT(seq0.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({0, 1, 4}));
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    // reuse blocks 0, 1 and reuse block 2
    auto inputTokens1 = std::make_shared<VecTokens>(llmRequest1->getTokens(0));
    llmRequest1 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens1, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        loraTaskId);
    promptLen1 = llmRequest1->getNumTokens(beamIdx);
    numContextBlocks1 = tc::ceilDiv(promptLen1, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq1, promptLen1, numContextBlocks1, llmRequest1);
    EXPECT_EQ(llmRequest1->getContextCurrentPosition(), llmRequest1->getNumTokens(beamIdx) - 1);
    EXPECT_THAT(seq1.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({0, 1, 2}));
    llmRequest1->addNewToken(10, beamIdx);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks + 1);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks - 1);

    blockManager.releaseBlocks(seq0, llmRequest0);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);
    // blocks 2 is stored for reuse (block contains [8, 9, 10])
    blockManager.releaseBlocks(seq1, llmRequest1);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // add request with loraTaskId 2
    loraTaskId = static_cast<LlmRequest::LoraTaskIdType>(2);
    requestId = 2;
    auto llmRequest2 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        loraTaskId);

    numTokens = llmRequest2->getNumTokens(beamIdx);
    GenerationRequest seq2{requestId, numTokens, beamWidth, maxBlocksPerSeq};
    // no reuse, get new block 5, 6, 7
    auto promptLen2 = llmRequest2->getNumTokens(beamIdx);
    auto numContextBlocks2 = tc::ceilDiv(promptLen2, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq2, promptLen2, numContextBlocks2, llmRequest2);
    EXPECT_EQ(llmRequest2->getContextCurrentPosition(), 0);
    EXPECT_THAT(seq2.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({5, 6, 7}));
    llmRequest2->addNewToken(9, beamIdx);
    numTokens = llmRequest2->getNumTokens(beamIdx);
    numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);
    // blocks 5, 6, 7 are stored with loraTaskId 2
    blockManager.releaseBlocks(seq2, llmRequest2);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // add request with loraTaskId 2 and more tokens
    auto inputLength3 = 11;
    auto inputTokens3 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11});
    requestId = 3;
    auto llmRequest3 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens3, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        loraTaskId);

    numTokens = llmRequest3->getNumTokens(beamIdx);
    GenerationRequest seq3{requestId, numTokens, beamWidth, maxBlocksPerSeq};
    // reuse blocks 5, 6, get new block 8
    auto promptLen3 = llmRequest3->getNumTokens(beamIdx);
    auto numContextBlocks3 = tc::ceilDiv(promptLen3, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq3, promptLen3, numContextBlocks3, llmRequest3);
    EXPECT_EQ(llmRequest3->getContextCurrentPosition(), 2 * tokensPerBlock);
    EXPECT_THAT(seq3.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({5, 6, 8}));
    llmRequest3->addNewToken(11, beamIdx);
    numTokens = llmRequest3->getNumTokens(beamIdx);
    numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);
    // blocks 8 is stored with [8, 9, 11] and loraTaskId 2
    blockManager.releaseBlocks(seq3, llmRequest3);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // add request with loraTaskId 1 again but with less tokens
    loraTaskId = static_cast<LlmRequest::LoraTaskIdType>(1);
    auto inputLength4 = 5;
    auto inputTokens4 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4});
    requestId = 4;
    auto llmRequest4 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens4, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        loraTaskId);

    numTokens = llmRequest4->getNumTokens(beamIdx);
    GenerationRequest seq4{requestId, numTokens, beamWidth, maxBlocksPerSeq};
    // reuse blocks 0, get new block 9
    auto promptLen4 = llmRequest4->getNumTokens(beamIdx);
    auto numContextBlocks4 = tc::ceilDiv(promptLen4, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq4, promptLen4, numContextBlocks4, llmRequest4);
    EXPECT_EQ(llmRequest4->getContextCurrentPosition(), tokensPerBlock);
    EXPECT_THAT(seq4.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({0, 9}));
    llmRequest4->addNewToken(5, beamIdx);
    numTokens = llmRequest4->getNumTokens(beamIdx);
    numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);
    // blocks 9 is stored with [4] and loraTaskId 1
    blockManager.releaseBlocks(seq4, llmRequest4);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);
}

TEST_F(KVCacheManagerTest, BlockManagerReuseWithExtraIdAndLoraTaskIdTest)
{
    // tc::Logger::getLogger()->setLevel(tc::Logger::Level::DEBUG);
    using VecTokenExtraIds = LlmRequest::VecTokenExtraIds;

    auto constexpr numLayers = 12;
    auto constexpr numKvHeads = 6;
    auto constexpr sizePerHead = 16;
    auto constexpr tokensPerBlock = 4;
    auto constexpr maxBlocksPerSeq = 4;
    auto constexpr blocksInPrimaryPool = 16;
    auto constexpr blocksInSecondaryPool = 0;
    auto constexpr maxNumSequences = 8;
    auto const stream = std::make_shared<tr::CudaStream>();
    auto constexpr onboardBlocks = true;

    BlockManager blockManager(std::vector(numLayers, numKvHeads), sizePerHead, tokensPerBlock, blocksInPrimaryPool,
        blocksInSecondaryPool, maxNumSequences, stream, onboardBlocks);

    EXPECT_EQ(blockManager.getTokensPerBlock(), tokensPerBlock);
    EXPECT_EQ(blockManager.getMaxNumBlocks(), blocksInPrimaryPool);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    SizeType32 constexpr maxNewTokens{0};
    auto constexpr beamWidth = 1;
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    // assume prompt id starts from 100
    auto inputTokens = std::make_shared<VecTokens>(VecTokens{100, 101, 102, 103, 104, 105, 0, 1, 2});
    auto inputTokenExtraIds = std::make_shared<VecTokenExtraIds>(VecTokenExtraIds{1, 1, 2, 2, 3, 3, 0, 0, 0});
    auto const inputLength = static_cast<SizeType32>(inputTokens->size());
    LlmRequest::LoraTaskIdType loraTaskId1{1};
    LlmRequest::RequestIdType requestId{0};
    auto llmRequest0 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        loraTaskId1, std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt, std::nullopt, false,
        std::nullopt, false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt,
        LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, inputTokenExtraIds);

    GenerationRequest seq0{requestId, inputLength, beamWidth, maxBlocksPerSeq};

    ///////////////////////////////////////////////////////////////////////////
    // add request with loraTaskId 1 and then remove it
    auto constexpr beamIdx = 0;
    auto promptLen0 = llmRequest0->getNumTokens(beamIdx);
    auto numContextBlocks0 = tc::ceilDiv(promptLen0, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq0, promptLen0, numContextBlocks0, llmRequest0);
    EXPECT_EQ(llmRequest0->getContextCurrentPosition(), 0);
    EXPECT_THAT(seq0.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({0, 1, 2}));
    llmRequest0->addNewToken(3, beamIdx);
    llmRequest0->addNewToken(4, beamIdx);
    auto numTokens = llmRequest0->getNumTokens(beamIdx);
    auto numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(numBlocks, 3);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    // blocks 0, 1, 2 are stored for reuse (block 2 contains [(2, 0), (3, 0)] with loraTaskId 1)
    blockManager.releaseBlocks(seq0, llmRequest0);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // new request with same tokens but different loraTaskId and then remove it
    requestId = 1;
    LlmRequest::LoraTaskIdType loraTaskId2 = static_cast<LlmRequest::LoraTaskIdType>(2);
    auto llmRequest1 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        loraTaskId2, std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt, std::nullopt, false,
        std::nullopt, false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt,
        LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, inputTokenExtraIds);
    GenerationRequest seq1{requestId, inputLength, beamWidth, maxBlocksPerSeq};

    // no reuse, get new block 3, 4, 5
    auto promptLen1 = llmRequest1->getNumTokens(beamIdx);
    auto numContextBlocks1 = tc::ceilDiv(promptLen1, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq1, promptLen1, numContextBlocks1, llmRequest1);
    EXPECT_EQ(llmRequest1->getContextCurrentPosition(), 0);
    EXPECT_THAT(seq1.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({3, 4, 5}));
    llmRequest1->addNewToken(3, beamIdx);
    llmRequest1->addNewToken(4, beamIdx);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    // blocks 3, 4, 5 are stored for reuse (block 5 contains [(2, 0), (3, 0)] with loraTaskId 2)
    blockManager.releaseBlocks(seq1, llmRequest1);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // add both requests again and then remove them
    llmRequest0 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        loraTaskId1, std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt, std::nullopt, false,
        std::nullopt, false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt,
        LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, inputTokenExtraIds);
    llmRequest0->addNewToken(3, beamIdx);
    promptLen0 = llmRequest0->getNumTokens(beamIdx);
    numContextBlocks0 = tc::ceilDiv(promptLen0, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq0, promptLen0, numContextBlocks0, llmRequest0);
    EXPECT_EQ(llmRequest0->getContextCurrentPosition(), 2 * tokensPerBlock);
    // reuse blocks 0, 1 and get new block 6
    EXPECT_THAT(seq0.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({0, 1, 6}));
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    // reuse blocks 3, 4 and reuse block 5
    auto inputTokens1 = std::make_shared<VecTokens>(llmRequest1->getTokens(0));
    auto inputTokenExtraIds1 = std::make_shared<VecTokenExtraIds>(*inputTokenExtraIds);
    inputTokenExtraIds1->push_back(0);
    inputTokenExtraIds1->push_back(0);
    llmRequest1 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens1, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        loraTaskId2, std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt, std::nullopt, false,
        std::nullopt, false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt,
        LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, inputTokenExtraIds1);
    promptLen1 = llmRequest1->getNumTokens(beamIdx);
    numContextBlocks1 = tc::ceilDiv(promptLen1, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq1, promptLen1, numContextBlocks1, llmRequest1);
    EXPECT_EQ(llmRequest1->getContextCurrentPosition(), llmRequest1->getNumTokens(beamIdx) - 1);
    EXPECT_THAT(seq1.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({3, 4, 5}));
    llmRequest1->addNewToken(5, beamIdx);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks * 2);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks * 2);

    blockManager.releaseBlocks(seq0, llmRequest0);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);
    blockManager.releaseBlocks(seq1, llmRequest1);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // add request with totally different extra ids and loraTaskId 1
    auto inputTokenExtraIds2 = std::make_shared<VecTokenExtraIds>(VecTokenExtraIds{4, 4, 5, 5, 6, 6, 0, 0, 0});
    requestId = 2;
    auto llmRequest2 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        loraTaskId1, std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt, std::nullopt, false,
        std::nullopt, false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt,
        LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, inputTokenExtraIds2);

    numTokens = llmRequest2->getNumTokens(beamIdx);
    GenerationRequest seq2{requestId, numTokens, beamWidth, maxBlocksPerSeq};
    // no reuse, get new block 7, 8, 9
    auto promptLen2 = llmRequest2->getNumTokens(beamIdx);
    auto numContextBlocks2 = tc::ceilDiv(promptLen2, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq2, promptLen2, numContextBlocks2, llmRequest2);
    EXPECT_EQ(llmRequest2->getContextCurrentPosition(), 0);
    EXPECT_THAT(seq2.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({7, 8, 9}));
    llmRequest2->addNewToken(3, beamIdx);
    numTokens = llmRequest2->getNumTokens(beamIdx);
    numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    ///////////////////////////////////////////////////////////////////////////
    // add request with partial different extra ids and loraTaskId 1
    auto inputTokenExtraIds3 = std::make_shared<VecTokenExtraIds>(VecTokenExtraIds{1, 1, 2, 2, 4, 4, 0, 0, 0});
    requestId = 3;
    auto llmRequest3 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        loraTaskId1, std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt, std::nullopt, false,
        std::nullopt, false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt,
        LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, inputTokenExtraIds3);

    numTokens = llmRequest3->getNumTokens(beamIdx);
    GenerationRequest seq3{requestId, numTokens, beamWidth, maxBlocksPerSeq};
    // reuse block 0, get new block 10, 11
    auto promptLen3 = llmRequest3->getNumTokens(beamIdx);
    auto numContextBlocks3 = tc::ceilDiv(promptLen3, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq3, promptLen3, numContextBlocks3, llmRequest3);
    EXPECT_EQ(llmRequest3->getContextCurrentPosition(), tokensPerBlock);
    EXPECT_THAT(seq3.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({0, 10, 11}));
    llmRequest3->addNewToken(3, beamIdx);
    numTokens = llmRequest3->getNumTokens(beamIdx);
    numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks * 2);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks * 2);

    ///////////////////////////////////////////////////////////////////////////
    // add request with partial different extra ids and loraTaskId 2
    requestId = 4;
    auto llmRequest4 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        loraTaskId2, std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt, std::nullopt, false,
        std::nullopt, false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt,
        LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, inputTokenExtraIds3);

    numTokens = llmRequest4->getNumTokens(beamIdx);
    GenerationRequest seq4{requestId, numTokens, beamWidth, maxBlocksPerSeq};
    // reuse block 3, get new block 12, 13
    auto promptLen4 = llmRequest4->getNumTokens(beamIdx);
    auto numContextBlocks4 = tc::ceilDiv(promptLen4, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq4, promptLen4, numContextBlocks4, llmRequest4);
    EXPECT_EQ(llmRequest4->getContextCurrentPosition(), tokensPerBlock);
    EXPECT_THAT(seq4.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({3, 12, 13}));
    llmRequest4->addNewToken(3, beamIdx);
    numTokens = llmRequest4->getNumTokens(beamIdx);
    numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks * 3);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks * 3);

    blockManager.releaseBlocks(seq2, llmRequest2);
    blockManager.releaseBlocks(seq3, llmRequest3);
    blockManager.releaseBlocks(seq4, llmRequest4);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);
}

TEST_F(KVCacheManagerTest, KVCacheManagerPerRequestStatsTest)
{
    // tc::Logger::getLogger()->setLevel(tc::Logger::Level::DEBUG);

    auto constexpr numLayers = 12;
    auto constexpr numHeads = 6;
    auto constexpr sizePerHead = 16;
    auto constexpr tokensPerBlock = 4;
    auto constexpr maxBlocksPerSeq = 4;
    auto constexpr maxNumSequences = 8;
    auto constexpr blocksInPrimaryPool = 16;
    auto constexpr blocksInSecondaryPool = 0;
    auto constexpr onboardBlocks = true;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr beamWidth = 1;
    SizeType32 constexpr maxNewTokens{0};
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksInPrimaryPool,
        blocksInSecondaryPool, maxNumSequences, beamWidth, tokensPerBlock * maxBlocksPerSeq, 0, false, stream, true,
        onboardBlocks);

    auto inputTokens = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8});
    auto const inputLength = static_cast<SizeType32>(inputTokens->size());

    LlmRequest::RequestIdType requestId{0};
    auto llmRequest0 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming);

    // Add the sequence to req0
    EXPECT_NO_THROW(kvCacheManager.addSequence(requestId, inputLength, beamWidth, llmRequest0));

    // After first addition, check allocations and reuses
    auto numBlocks = tc::ceilDiv(inputLength, tokensPerBlock);
    EXPECT_EQ(llmRequest0->getReusedBlocksPerRequest(), 0);
    EXPECT_EQ(llmRequest0->getAllocTotalBlocksPerRequest(), numBlocks);
    EXPECT_EQ(llmRequest0->getAllocNewBlocksPerRequest(), numBlocks);

    EXPECT_NO_THROW(kvCacheManager.removeSequence(requestId, llmRequest0));

    requestId = 1;
    auto llmRequest1 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming);

    EXPECT_NO_THROW(kvCacheManager.addSequence(requestId, inputLength, beamWidth, llmRequest1));

    auto const numSharedBlocks = inputLength / tokensPerBlock;
    EXPECT_EQ(llmRequest1->getReusedBlocksPerRequest(), numSharedBlocks);
    EXPECT_EQ(llmRequest1->getAllocTotalBlocksPerRequest(), numBlocks - numSharedBlocks);
    EXPECT_EQ(llmRequest1->getAllocNewBlocksPerRequest(), numBlocks - numSharedBlocks);
}

TEST_P(KVCacheManagerTest, KVCacheManagerAllocationTest)
{
    using DType = half;

    auto constexpr numLayers = 2;
    auto constexpr numHeads = 2;
    auto constexpr sizePerHead = 64;
    auto constexpr tokensPerBlock = 64;
    auto constexpr maxBlocksPerSeq = 10;
    auto constexpr maxNumSequences = 8;
    auto constexpr maxBeamWidth = 4;
    auto constexpr sinkTokenLength = 0;
    auto constexpr useOneMoreBlock = false;
    auto constexpr dtype = nvinfer1::DataType::kHALF;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr maxNumTokens = tokensPerBlock * maxBlocksPerSeq;
    auto constexpr maxAttentionWindow = maxNumTokens;
    auto constexpr inputLength = maxNumTokens - tokensPerBlock - 1;
    auto constexpr numSharedBlocks = inputLength / tokensPerBlock;
    auto constexpr numBlocksPerSeq = numSharedBlocks + (maxBlocksPerSeq - numSharedBlocks) * maxBeamWidth;

    auto constexpr totalNumBlocks = maxNumSequences * numBlocksPerSeq;
    auto constexpr blocksInSecondaryPool = 0;

    auto constexpr enableBlockReuse = false;
    auto constexpr useUvm = false;
    auto constexpr onboardBlocks = true;
    auto const homogeneousLayers = GetParam();

    // determine global memory page size
    auto [free1, total1] = tc::getDeviceMemoryInfo(false);
    float* ptr;
    cudaMalloc(&ptr, 1);
    auto [free2, total2] = tc::getDeviceMemoryInfo(false);
    cudaFree(ptr);
    int64_t cudaPageSize = static_cast<int64_t>(free1) - free2;

    KVCacheManager kvCacheManager = homogeneousLayers
        ? KVCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, totalNumBlocks, blocksInSecondaryPool,
            maxNumSequences, maxBeamWidth, maxAttentionWindow, sinkTokenLength, useOneMoreBlock, stream,
            enableBlockReuse, onboardBlocks)
        : KVCacheManager(std::vector<KVCacheManager::SizeType32>(numLayers, numHeads), sizePerHead, tokensPerBlock,
            totalNumBlocks, blocksInSecondaryPool, maxNumSequences, maxBeamWidth, maxAttentionWindow, sinkTokenLength,
            useOneMoreBlock, stream, enableBlockReuse, onboardBlocks);

    auto [freeDeviceMemoryBefore, totalDeviceMemoryBefore] = tc::getDeviceMemoryInfo(useUvm);
    kvCacheManager.allocatePools(dtype, useUvm);
    auto [freeDeviceMemoryAfter, totalDeviceMemoryAfter] = tc::getDeviceMemoryInfo(useUvm);

    int64_t freeDeviceMemoryDiff = freeDeviceMemoryBefore - freeDeviceMemoryAfter;
    int64_t expectedDeviceMemoryDiff = sizeof(DType) * static_cast<int64_t>(totalNumBlocks) * numLayers * 2 * numHeads
        * tokensPerBlock * sizePerHead;
    expectedDeviceMemoryDiff = ((expectedDeviceMemoryDiff + cudaPageSize - 1) / cudaPageSize) * cudaPageSize;
    EXPECT_EQ(freeDeviceMemoryDiff, expectedDeviceMemoryDiff);
}

TEST_P(KVCacheManagerTest, KVCacheManagerTest)
{
    using DType = half;
    using SizeType32 = KVCacheManager::SizeType32;

    auto constexpr numLayers = 4;
    auto constexpr numHeads = 2;
    // heterogeneous layers
    std::vector<SizeType32> const numHeadsPerLayer({3, 2, 1, 2});
    std::map<SizeType32, SizeType32> const expectedHeadsPerPool({{0, 1}, {1, 2}, {2, 3}});
    std::map<SizeType32, SizeType32> const expectedLayersPerPool({{0, 1}, {1, 2}, {2, 1}});
    auto constexpr sizePerHead = 64;
    auto constexpr hiddenSize = numHeads * sizePerHead;
    auto constexpr tokensPerBlock = 64;
    auto constexpr maxBlocksPerSeq = 10;
    auto constexpr maxNumSequences = 8;
    auto constexpr maxBeamWidth = 4;
    auto constexpr sinkTokenLength = 0;
    auto constexpr useOneMoreBlock = false;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr requestId = 7;
    auto constexpr maxNumTokens = tokensPerBlock * maxBlocksPerSeq;
    auto constexpr maxAttentionWindow = maxNumTokens;
    auto constexpr inputLength = maxNumTokens - tokensPerBlock - 1;
    auto constexpr numSharedBlocks = inputLength / tokensPerBlock;
    auto constexpr numBlocksPerSeq = numSharedBlocks + (maxBlocksPerSeq - numSharedBlocks) * maxBeamWidth;

    auto constexpr totalNumBlocks = maxNumSequences * numBlocksPerSeq;
    auto constexpr blocksInSecondaryPool = 0;

    auto constexpr enableBlockReuse = false;
    auto constexpr onboardBlocks = true;
    auto const homogeneousLayers = GetParam();
    auto const expectedNumPools = homogeneousLayers ? 1 : static_cast<SizeType32>(expectedHeadsPerPool.size());

    KVCacheManager kvCacheManager = homogeneousLayers
        ? KVCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, totalNumBlocks, blocksInSecondaryPool,
            maxNumSequences, maxBeamWidth, maxAttentionWindow, sinkTokenLength, useOneMoreBlock, stream,
            enableBlockReuse, onboardBlocks)
        : KVCacheManager(numHeadsPerLayer, sizePerHead, tokensPerBlock, totalNumBlocks, blocksInSecondaryPool,
            maxNumSequences, maxBeamWidth, maxAttentionWindow, sinkTokenLength, useOneMoreBlock, stream,
            enableBlockReuse, onboardBlocks);

    EXPECT_EQ(kvCacheManager.getMaxBlocksPerSeq(), maxBlocksPerSeq);
    EXPECT_EQ(kvCacheManager.getTokensPerBlock(), tokensPerBlock);
    EXPECT_EQ(kvCacheManager.getMaxNumBlocks(), totalNumBlocks);

    auto const& blockManager = kvCacheManager.getBlockManager();
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks);

    EXPECT_NO_THROW(kvCacheManager.addSequence(requestId, inputLength, maxBeamWidth));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks - numSharedBlocks - maxBeamWidth);

    tr::ITensor::SharedPtr kvCacheBlockOffsets = tr::BufferManager::cpu(
        tr::ITensor::makeShape({expectedNumPools, maxNumSequences * maxBeamWidth, 2, maxBlocksPerSeq}),
        tr::TRTDataType<tk::KVCacheIndex>::value);

    auto kvCacheBlockOffsetsRange = tensorrt_llm::runtime::BufferRange<tk::KVCacheIndex>(*kvCacheBlockOffsets);
    std::fill(kvCacheBlockOffsetsRange.begin(), kvCacheBlockOffsetsRange.end(),
        tk::KVCacheIndex{std::numeric_limits<tk::KVCacheIndex::UnderlyingType>::max()});

    kvCacheManager.copyBlockOffsets(*kvCacheBlockOffsets, 0, requestId);

    EXPECT_EQ(blockManager.getNumPools(), expectedNumPools);
    if (homogeneousLayers)
    {
        EXPECT_EQ(blockManager.getBlockSize(0), numHeads * sizePerHead * tokensPerBlock);
    }
    else
    {
        for (auto layerIdx = 0; layerIdx < numLayers; layerIdx++)
        {
            EXPECT_EQ(blockManager.getBlockSize(blockManager.getLayerPoolIdx(layerIdx)),
                numHeadsPerLayer.at(layerIdx) * sizePerHead * tokensPerBlock);
        }
    }

    {
        for (auto poolIdx = 0; poolIdx < blockManager.getNumPools(); poolIdx++)
        {
            auto const numLayersInPool = homogeneousLayers ? numLayers : expectedLayersPerPool.at(poolIdx);
            auto const offsetBetweenBlocks = numLayersInPool * 2;
            tr::ITensor::SharedPtr blockOffsetsSlice
                = tr::ITensor::slice(tr::ITensor::at(kvCacheBlockOffsets, {poolIdx}), 0, maxBeamWidth);
            auto blockOffsetsShape = blockOffsetsSlice->getShape();
            auto* const blockOffsetsPtr = tr::bufferCast<tk::KVCacheIndex>(*blockOffsetsSlice);

            tk::KVCacheIndex::UnderlyingType runningSum{0};
            for (auto block = 0; block < numSharedBlocks; ++block)
            {
                for (auto beam = 0; beam < maxBeamWidth; ++beam)
                {
                    auto const kOffsetIdx = tc::flat_index(blockOffsetsShape.d, beam, 0, block);
                    auto const vOffsetIdx = tc::flat_index(blockOffsetsShape.d, beam, 1, block);
                    auto const kOffset = blockOffsetsPtr[kOffsetIdx];
                    auto const vOffset = blockOffsetsPtr[vOffsetIdx];
                    EXPECT_EQ(kOffset.get(), runningSum) << "beam:" << beam << " block:" << block;
                    ASSERT_EQ(vOffset.get(), runningSum + 1) << "beam:" << beam << " block:" << block;
                }
                runningSum += offsetBetweenBlocks;
            }
            {
                auto const block = numSharedBlocks;
                for (auto beam = 0; beam < maxBeamWidth; ++beam)
                {
                    auto const kOffsetIdx = tc::flat_index(blockOffsetsShape.d, beam, 0, block);
                    auto const vOffsetIdx = tc::flat_index(blockOffsetsShape.d, beam, 1, block);
                    auto const kOffset = blockOffsetsPtr[kOffsetIdx];
                    auto const vOffset = blockOffsetsPtr[vOffsetIdx];
                    EXPECT_EQ(kOffset.get(), runningSum) << "beam:" << beam << " block:" << block;
                    ASSERT_EQ(vOffset.get(), runningSum + 1) << "beam:" << beam << " block:" << block;
                    runningSum += offsetBetweenBlocks;
                }
            }
            {
                auto const block = numSharedBlocks + 1;
                auto const expected = std::numeric_limits<tk::KVCacheIndex::UnderlyingType>::max();
                for (auto beam = 0; beam < maxBeamWidth; ++beam)
                {
                    auto const kOffsetIdx = tc::flat_index(blockOffsetsShape.d, beam, 0, block);
                    auto const vOffsetIdx = tc::flat_index(blockOffsetsShape.d, beam, 1, block);
                    auto const kOffset = blockOffsetsPtr[kOffsetIdx];
                    auto const vOffset = blockOffsetsPtr[vOffsetIdx];
                    EXPECT_EQ(kOffset.get(), expected) << "beam:" << beam << " block:" << block;
                    ASSERT_EQ(vOffset.get(), expected) << "beam:" << beam << " block:" << block;
                }
            }
        }
    }

    EXPECT_NO_THROW(kvCacheManager.addToken(requestId));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks - numSharedBlocks - maxBeamWidth);
    EXPECT_NO_THROW(kvCacheManager.addToken(requestId));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks - numBlocksPerSeq);
    EXPECT_NO_THROW(kvCacheManager.removeSequence(requestId));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks);

    auto currentNumBlocks = totalNumBlocks;
    for (auto requestId = 0; requestId < maxNumSequences; ++requestId)
    {
        EXPECT_NO_THROW(kvCacheManager.addSequence(requestId, inputLength, maxBeamWidth));
        currentNumBlocks -= numSharedBlocks + maxBeamWidth;
        EXPECT_EQ(blockManager.getNumFreeBlocks(), currentNumBlocks);
        EXPECT_NO_THROW(kvCacheManager.addToken(requestId));
        EXPECT_NO_THROW(kvCacheManager.addToken(requestId));
        currentNumBlocks -= maxBeamWidth;
        EXPECT_EQ(blockManager.getNumFreeBlocks(), currentNumBlocks);
    }
    EXPECT_EQ(blockManager.getNumFreeBlocks(), 0);
}

TEST_P(KVCacheManagerTest, KVCacheManagerRewindTokensTest)
{
    using DType = half;

    auto constexpr numLayers = 2;
    auto constexpr numHeads = 2;
    auto constexpr sizePerHead = 64;
    auto constexpr hiddenSize = numHeads * sizePerHead;
    auto constexpr tokensPerBlock = 64;
    auto constexpr maxBlocksPerSeq = 10;
    auto constexpr maxNumSequences = 8;
    auto constexpr maxBeamWidth = 1;
    auto constexpr sinkTokenLength = 0;
    auto constexpr useOneMoreBlock = false;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr requestId = 7;
    auto constexpr maxNumTokens = tokensPerBlock * maxBlocksPerSeq;
    auto constexpr maxAttentionWindow = maxNumTokens;
    auto constexpr inputLength = maxNumTokens - tokensPerBlock - 1;
    auto constexpr numSharedBlocks = inputLength / tokensPerBlock;
    auto constexpr numBlocksPerSeq = numSharedBlocks + (maxBlocksPerSeq - numSharedBlocks) * maxBeamWidth;

    auto constexpr totalNumBlocks = maxNumSequences * numBlocksPerSeq;
    auto constexpr blocksInSecondaryPool = 0;

    auto constexpr enableBlockReuse = false;
    auto constexpr onboardBlocks = true;
    auto const homogeneousLayers = GetParam();

    KVCacheManager kvCacheManager = homogeneousLayers
        ? KVCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, totalNumBlocks, blocksInSecondaryPool,
            maxNumSequences, maxBeamWidth, maxAttentionWindow, sinkTokenLength, useOneMoreBlock, stream,
            enableBlockReuse, onboardBlocks)
        : KVCacheManager(std::vector<KVCacheManager::SizeType32>(numLayers, numHeads), sizePerHead, tokensPerBlock,
            totalNumBlocks, blocksInSecondaryPool, maxNumSequences, maxBeamWidth, maxAttentionWindow, sinkTokenLength,
            useOneMoreBlock, stream, enableBlockReuse, onboardBlocks);

    EXPECT_EQ(kvCacheManager.getTokensPerBlock(), tokensPerBlock);
    EXPECT_EQ(kvCacheManager.getMaxNumBlocks(), totalNumBlocks);

    auto const& blockManager = kvCacheManager.getBlockManager();
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks);

    EXPECT_NO_THROW(kvCacheManager.addSequence(requestId, inputLength, maxBeamWidth));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks - numSharedBlocks - maxBeamWidth);

    EXPECT_NO_THROW(kvCacheManager.addToken(requestId));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks - numSharedBlocks - maxBeamWidth);
    EXPECT_NO_THROW(kvCacheManager.addToken(requestId));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks - numBlocksPerSeq);
    EXPECT_NO_THROW(kvCacheManager.addToken(requestId));
    EXPECT_NO_THROW(kvCacheManager.addToken(requestId));
    EXPECT_NO_THROW(kvCacheManager.addToken(requestId));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks - numBlocksPerSeq);
    EXPECT_NO_THROW(kvCacheManager.rewindKVCache(requestId, 4));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks - numSharedBlocks - maxBeamWidth);
    EXPECT_NO_THROW(kvCacheManager.removeSequence(requestId));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks);

    auto currentNumBlocks = totalNumBlocks;
    for (auto requestId = 0; requestId < maxNumSequences; ++requestId)
    {
        EXPECT_NO_THROW(kvCacheManager.addSequence(requestId, inputLength, maxBeamWidth));
        currentNumBlocks -= numSharedBlocks + maxBeamWidth;
        EXPECT_EQ(blockManager.getNumFreeBlocks(), currentNumBlocks);
        EXPECT_NO_THROW(kvCacheManager.addToken(requestId));
        EXPECT_NO_THROW(kvCacheManager.addToken(requestId));
        currentNumBlocks -= maxBeamWidth;
        EXPECT_EQ(blockManager.getNumFreeBlocks(), currentNumBlocks);
        EXPECT_NO_THROW(kvCacheManager.rewindKVCache(requestId, 2));
        currentNumBlocks += maxBeamWidth;
        EXPECT_EQ(blockManager.getNumFreeBlocks(), currentNumBlocks);
    }
}

TEST_P(KVCacheManagerTest, KVCacheManagerMaxAttentionWindowTest)
{
    using DType = half;
    using SizeType32 = KVCacheManager::SizeType32;

    auto constexpr numLayers = 4;
    auto constexpr numHeads = 2;
    // heterogeneous layers
    std::vector<SizeType32> const numHeadsPerLayer({3, 2, 1, 2});
    std::map<SizeType32, SizeType32> const expectedHeadsPerPool({{0, 1}, {1, 2}, {2, 3}});
    std::map<SizeType32, SizeType32> const expectedLayersPerPool({{0, 1}, {1, 2}, {2, 1}});
    auto constexpr sizePerHead = 64;
    auto constexpr hiddenSize = numHeads * sizePerHead;
    auto constexpr tokensPerBlock = 64;
    auto constexpr blockLengthPerSeq = 10;
    auto constexpr maxNumSequences = 8;
    auto constexpr maxBeamWidth = 1;
    auto constexpr sinkTokenLength = 0;
    auto constexpr useOneMoreBlock = false;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr requestId = 7;
    auto constexpr maxNumTokens = tokensPerBlock * blockLengthPerSeq;

    auto constexpr inputLength = maxNumTokens - tokensPerBlock - 1;
    // Enable cyclic kv cache for all new generated tokens.
    auto constexpr maxAttentionWindow = inputLength;
    auto constexpr numSharedBlocks = std::min(inputLength, maxAttentionWindow) / tokensPerBlock;
    auto constexpr numBlocksPerSeq = numSharedBlocks + (blockLengthPerSeq - numSharedBlocks) * maxBeamWidth;
    auto constexpr maxBlocksPerSeq = tc::ceilDiv(maxAttentionWindow, tokensPerBlock);

    auto constexpr totalNumBlocks = maxNumSequences * numBlocksPerSeq;
    auto constexpr blocksInSecondaryPool = 0;

    auto constexpr enableBlockReuse = false;
    auto constexpr onboardBlocks = true;
    auto const homogeneousLayers = GetParam();
    auto const expectedNumPools = homogeneousLayers ? 1 : static_cast<SizeType32>(expectedHeadsPerPool.size());

    KVCacheManager kvCacheManager = homogeneousLayers
        ? KVCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, totalNumBlocks, blocksInSecondaryPool,
            maxNumSequences, maxBeamWidth, maxAttentionWindow, sinkTokenLength, useOneMoreBlock, stream,
            enableBlockReuse, onboardBlocks)
        : KVCacheManager(numHeadsPerLayer, sizePerHead, tokensPerBlock, totalNumBlocks, blocksInSecondaryPool,
            maxNumSequences, maxBeamWidth, maxAttentionWindow, sinkTokenLength, useOneMoreBlock, stream,
            enableBlockReuse, onboardBlocks);

    EXPECT_EQ(kvCacheManager.getMaxBlocksPerSeq(), maxBlocksPerSeq);
    EXPECT_EQ(kvCacheManager.getTokensPerBlock(), tokensPerBlock);
    EXPECT_EQ(kvCacheManager.getMaxNumBlocks(), totalNumBlocks);

    auto const& blockManager = kvCacheManager.getBlockManager();
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks);

    EXPECT_NO_THROW(kvCacheManager.addSequence(requestId, inputLength, maxBeamWidth));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks - numSharedBlocks - maxBeamWidth);

    tr::ITensor::SharedPtr kvCacheBlockOffsets = tr::BufferManager::cpu(
        tr::ITensor::makeShape({expectedNumPools, maxNumSequences * maxBeamWidth, 2, maxBlocksPerSeq}),
        tr::TRTDataType<tk::KVCacheIndex>::value);

    kvCacheManager.copyBlockOffsets(*kvCacheBlockOffsets, 0, requestId);

    EXPECT_EQ(blockManager.getNumPools(), expectedNumPools);
    if (homogeneousLayers)
    {
        EXPECT_EQ(blockManager.getBlockSize(0), numHeads * sizePerHead * tokensPerBlock);
    }
    else
    {
        for (auto layerIdx = 0; layerIdx < numLayers; layerIdx++)
        {
            EXPECT_EQ(blockManager.getBlockSize(blockManager.getLayerPoolIdx(layerIdx)),
                numHeadsPerLayer.at(layerIdx) * sizePerHead * tokensPerBlock);
        }
    }

    {
        for (auto poolIdx = 0; poolIdx < blockManager.getNumPools(); poolIdx++)
        {
            auto const numLayersInPool = homogeneousLayers ? numLayers : expectedLayersPerPool.at(poolIdx);
            auto const offsetBetweenBlocks = numLayersInPool * 2;
            tr::ITensor::SharedPtr blockOffsetsSlice
                = tr::ITensor::slice(tr::ITensor::at(kvCacheBlockOffsets, {poolIdx}), 0, maxBeamWidth);
            auto blockOffsetsShape = blockOffsetsSlice->getShape();
            auto const blockOffsetsPtr = tr::bufferCast<tk::KVCacheIndex>(*blockOffsetsSlice);

            tk::KVCacheIndex::UnderlyingType runningSum{0};
            for (auto block = 0; block < numSharedBlocks; ++block)
            {
                for (auto beam = 0; beam < maxBeamWidth; ++beam)
                {
                    auto const kOffsetIdx = tc::flat_index(blockOffsetsShape.d, beam, 0, block);
                    auto const vOffsetIdx = tc::flat_index(blockOffsetsShape.d, beam, 1, block);
                    auto const kOffset = blockOffsetsPtr[kOffsetIdx];
                    auto const vOffset = blockOffsetsPtr[vOffsetIdx];
                    EXPECT_EQ(kOffset.get(), runningSum) << "beam:" << beam << " block:" << block;
                    ASSERT_EQ(vOffset.get(), runningSum + 1) << "beam:" << beam << " block:" << block;
                }
                runningSum += offsetBetweenBlocks;
            }
            {
                auto const block = numSharedBlocks;
                for (auto beam = 0; beam < maxBeamWidth; ++beam)
                {
                    auto const kOffsetIdx = tc::flat_index(blockOffsetsShape.d, beam, 0, block);
                    auto const vOffsetIdx = tc::flat_index(blockOffsetsShape.d, beam, 1, block);
                    auto const kOffset = blockOffsetsPtr[kOffsetIdx];
                    auto const vOffset = blockOffsetsPtr[vOffsetIdx];
                    EXPECT_EQ(kOffset.get(), runningSum) << "beam:" << beam << " block:" << block;
                    ASSERT_EQ(vOffset.get(), runningSum + 1) << "beam:" << beam << " block:" << block;
                    runningSum += offsetBetweenBlocks;
                }
            }
        }
    }

    EXPECT_NO_THROW(kvCacheManager.addToken(requestId));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks - numBlocksPerSeq + 1);
    EXPECT_NO_THROW(kvCacheManager.addToken(requestId));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks - numBlocksPerSeq + 1);
    EXPECT_NO_THROW(kvCacheManager.removeSequence(requestId));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks);

    auto currentNumBlocks = totalNumBlocks;
    for (auto requestId = 0; requestId < maxNumSequences; ++requestId)
    {
        EXPECT_NO_THROW(kvCacheManager.addSequence(requestId, inputLength, maxBeamWidth));
        currentNumBlocks -= numSharedBlocks + maxBeamWidth;
        EXPECT_EQ(blockManager.getNumFreeBlocks(), currentNumBlocks);
        EXPECT_NO_THROW(kvCacheManager.addToken(requestId));
        EXPECT_EQ(blockManager.getNumFreeBlocks(), currentNumBlocks);
    }
    EXPECT_EQ(blockManager.getNumFreeBlocks(), maxNumSequences);
}

TEST_F(KVCacheManagerTest, KVCacheManagerMaxAttentionWindowWithReuseTest)
{
    auto constexpr numLayers = 2;
    auto constexpr numHeads = 2;
    auto constexpr sizePerHead = 64;
    auto constexpr tokensPerBlock = 4;
    auto constexpr maxNumSequences = 8;
    auto constexpr maxBeamWidth = 1;
    auto constexpr sinkTokenLength = 0;
    auto constexpr useOneMoreBlock = false;
    auto const stream = std::make_shared<tr::CudaStream>();

    // Enable cyclic kv cache for long input tokens.
    auto constexpr maxAttentionWindow = 16;
    auto constexpr maxBlocksPerSeq = tc::ceilDiv(maxAttentionWindow, tokensPerBlock);

    auto constexpr blocksInPrimaryPool = 16;
    auto constexpr blocksInSecondaryPool = 0;

    auto constexpr enableBlockReuse = true;
    auto constexpr onboardBlocks = true;

    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksInPrimaryPool,
        blocksInSecondaryPool, maxNumSequences, maxBeamWidth, maxAttentionWindow, sinkTokenLength, useOneMoreBlock,
        stream, enableBlockReuse, onboardBlocks);

    auto const& blockManager = kvCacheManager.getBlockManager();

    SizeType32 constexpr maxNewTokens = 4;

    // prepare tokens with token[i] = 1000 + i
    TokenIdType constexpr firstToken = 1000;

    auto constexpr beamWidth = maxBeamWidth;
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    SizeType32 requestId = 0;
    int inputLength = 16;
    auto inputTokens = std::make_shared<VecTokens>(inputLength);
    std::iota(inputTokens->begin(), inputTokens->end(), firstToken);
    auto llmRequest = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming);
    auto constexpr beamIdx = 0;

    ///////////////////////////////////////////////////////////////////////////
    // add a long request and then remove it
    kvCacheManager.addSequence(requestId, inputLength, beamWidth, llmRequest);
    GenerationRequest const& seq0 = kvCacheManager.getSequence(requestId);
    EXPECT_EQ(llmRequest->getContextCurrentPosition(), 0);
    EXPECT_THAT(seq0.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({0, 1, 2, 3}));

    // add tokens to enable cyclic kv cache
    llmRequest->addNewToken(1016, beamIdx);
    kvCacheManager.addToken(requestId);
    llmRequest->addNewToken(1017, beamIdx);
    kvCacheManager.addToken(requestId);
    auto numTokens = llmRequest->getNumTokens(beamIdx);
    auto numBlocks = seq0.getCacheBlockIds()[beamIdx].size();
    EXPECT_EQ(numBlocks, maxBlocksPerSeq);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);
    EXPECT_NO_THROW(kvCacheManager.removeSequence(requestId, llmRequest));

    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // add a short request and then remove it
    requestId = 1;
    inputLength = 7;
    inputTokens->resize(inputLength);
    llmRequest = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming);
    kvCacheManager.addSequence(requestId, inputLength, beamWidth, llmRequest);
    GenerationRequest const& seq1 = kvCacheManager.getSequence(requestId);
    EXPECT_EQ(llmRequest->getContextCurrentPosition(), 0);
    EXPECT_THAT(seq1.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({4, 5}));

    llmRequest->addNewToken(1007, beamIdx);
    kvCacheManager.addToken(requestId);
    llmRequest->addNewToken(1008, beamIdx);
    kvCacheManager.addToken(requestId);
    numTokens = llmRequest->getNumTokens(beamIdx);
    numBlocks = seq1.getCacheBlockIds()[beamIdx].size();
    EXPECT_EQ(numBlocks, 3);
    EXPECT_NO_THROW(kvCacheManager.removeSequence(requestId, llmRequest));

    ///////////////////////////////////////////////////////////////////////////
    // add a medium request and then remove it
    // reuse first 2 blocks {4, 5} in previous request, and get new block 7
    requestId = 2;
    inputLength = 10;
    inputTokens->resize(inputLength);
    std::iota(inputTokens->begin(), inputTokens->end(), firstToken);
    llmRequest = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming);
    kvCacheManager.addSequence(requestId, inputLength, beamWidth, llmRequest);
    GenerationRequest const& seq2 = kvCacheManager.getSequence(requestId);
    EXPECT_EQ(llmRequest->getContextCurrentPosition(), 2 * tokensPerBlock);
    EXPECT_THAT(seq2.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({4, 5, 7}));

    numTokens = llmRequest->getNumTokens(beamIdx);
    numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(numBlocks, 3);
    EXPECT_NO_THROW(kvCacheManager.removeSequence(requestId, llmRequest));

    ///////////////////////////////////////////////////////////////////////////
    // add a longer request within attention window and try to reuse
    // reuse first 2 blocks {4, 5} in previous request, and get new blocks 8, 9
    // upon reached attention window, shared block 4 is replaced with unshared block 10
    requestId = 3;
    inputLength = 15;
    inputTokens->resize(inputLength);
    std::iota(inputTokens->begin(), inputTokens->end(), firstToken);
    llmRequest = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming);
    kvCacheManager.addSequence(requestId, inputLength, beamWidth, llmRequest);
    GenerationRequest const& seq3 = kvCacheManager.getSequence(requestId);
    EXPECT_EQ(llmRequest->getContextCurrentPosition(), 2 * tokensPerBlock);
    EXPECT_THAT(seq3.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({4, 5, 8, 9}));

    llmRequest->addNewToken(1015, beamIdx);
    kvCacheManager.addToken(requestId);
    llmRequest->addNewToken(1016, beamIdx);
    kvCacheManager.addToken(requestId);
    EXPECT_THAT(seq3.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({10, 5, 8, 9}));
    EXPECT_NO_THROW(kvCacheManager.removeSequence(requestId, llmRequest));

    ///////////////////////////////////////////////////////////////////////////
    // add a long request that exceeded attention window, no reuse
    requestId = 4;
    inputLength = 20;
    inputTokens->resize(inputLength);
    std::iota(inputTokens->begin(), inputTokens->end(), firstToken);
    llmRequest = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming);
    kvCacheManager.addSequence(requestId, inputLength, beamWidth, llmRequest);
    EXPECT_EQ(llmRequest->getContextCurrentPosition(), 0);
    GenerationRequest const& seq4 = kvCacheManager.getSequence(requestId);
    EXPECT_THAT(seq4.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({11, 12, 13, 14}));
}

TEST_P(KVCacheManagerTest, KVCacheManagerSinkTokenLengthTest)
{
    using DType = half;
    using SizeType32 = KVCacheManager::SizeType32;

    auto constexpr numLayers = 4;
    auto constexpr numHeads = 2;
    // heterogeneous layers
    std::vector<SizeType32> const numHeadsPerLayer({3, 2, 1, 2});
    std::map<SizeType32, SizeType32> const expectedHeadsPerPool({{0, 1}, {1, 2}, {2, 3}});
    std::map<SizeType32, SizeType32> const expectedLayersPerPool({{0, 1}, {1, 2}, {2, 1}});
    auto constexpr sizePerHead = 64;
    auto constexpr hiddenSize = numHeads * sizePerHead;
    auto constexpr tokensPerBlock = 64;
    auto constexpr maxBlocksPerSeq = 10;
    auto constexpr maxNumSequences = 8;
    auto constexpr maxBeamWidth = 4;
    auto constexpr sinkTokenLength = 4;
    auto constexpr useOneMoreBlock = true;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr requestId = static_cast<RequestIdType>(7);
    auto constexpr sinkTokensInLastBlock = sinkTokenLength % tokensPerBlock;
    auto constexpr bubbleLength = (sinkTokensInLastBlock) ? tokensPerBlock - sinkTokensInLastBlock : 0;
    auto constexpr inputLength = tokensPerBlock * maxBlocksPerSeq - bubbleLength - 1;
    auto constexpr maxAttentionWindow = inputLength - tokensPerBlock;

    auto constexpr numSharedBlocks = (sinkTokenLength + bubbleLength) / tokensPerBlock;
    auto constexpr numBlocksPerSeq = numSharedBlocks + (maxBlocksPerSeq - numSharedBlocks) * maxBeamWidth;
    auto constexpr totalNumBlocks = maxNumSequences * numBlocksPerSeq;
    auto constexpr numSharedBlocksCtx = (inputLength + bubbleLength) / tokensPerBlock;

    auto constexpr blocksInSecondaryPool = 0;

    auto constexpr enableBlockReuse = false;
    auto constexpr onboardBlocks = true;
    auto const homogeneousLayers = GetParam();
    auto const expectedNumPools = homogeneousLayers ? 1 : static_cast<SizeType32>(expectedHeadsPerPool.size());

    KVCacheManager kvCacheManager = homogeneousLayers
        ? KVCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, totalNumBlocks, blocksInSecondaryPool,
            maxNumSequences, maxBeamWidth, maxAttentionWindow, sinkTokenLength, useOneMoreBlock, stream,
            enableBlockReuse, onboardBlocks)
        : KVCacheManager(numHeadsPerLayer, sizePerHead, tokensPerBlock, totalNumBlocks, blocksInSecondaryPool,
            maxNumSequences, maxBeamWidth, maxAttentionWindow, sinkTokenLength, useOneMoreBlock, stream,
            enableBlockReuse, onboardBlocks);

    EXPECT_EQ(kvCacheManager.getMaxBlocksPerSeq(), maxBlocksPerSeq);
    EXPECT_EQ(kvCacheManager.getTokensPerBlock(), tokensPerBlock);
    EXPECT_EQ(kvCacheManager.getMaxNumBlocks(), totalNumBlocks);

    auto const& blockManager = kvCacheManager.getBlockManager();
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks);

    EXPECT_NO_THROW(kvCacheManager.addSequence(requestId, inputLength, maxBeamWidth));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks - numSharedBlocksCtx - maxBeamWidth);

    tr::ITensor::SharedPtr kvCacheBlockOffsets = tr::BufferManager::cpu(
        tr::ITensor::makeShape({expectedNumPools, maxNumSequences * maxBeamWidth, 2, maxBlocksPerSeq}),
        tr::TRTDataType<tk::KVCacheIndex>::value);

    kvCacheManager.copyBlockOffsets(*kvCacheBlockOffsets, 0, requestId);

    EXPECT_EQ(blockManager.getNumPools(), expectedNumPools);
    if (homogeneousLayers)
    {
        EXPECT_EQ(blockManager.getBlockSize(0), numHeads * sizePerHead * tokensPerBlock);
    }
    else
    {
        for (auto layerIdx = 0; layerIdx < numLayers; layerIdx++)
        {
            EXPECT_EQ(blockManager.getBlockSize(blockManager.getLayerPoolIdx(layerIdx)),
                numHeadsPerLayer.at(layerIdx) * sizePerHead * tokensPerBlock);
        }
    }

    {
        for (auto poolIdx = 0; poolIdx < blockManager.getNumPools(); poolIdx++)
        {
            auto const numLayersInPool = homogeneousLayers ? numLayers : expectedLayersPerPool.at(poolIdx);
            auto const offsetBetweenBlocks = numLayersInPool * 2;
            tr::ITensor::SharedPtr blockOffsetsSlice
                = tr::ITensor::slice(tr::ITensor::at(kvCacheBlockOffsets, {poolIdx}), 0, maxBeamWidth);
            auto blockOffsetsShape = blockOffsetsSlice->getShape();
            auto const blockOffsetsPtr = tr::bufferCast<tk::KVCacheIndex>(*blockOffsetsSlice);

            tk::KVCacheIndex::UnderlyingType runningSum{0};
            for (auto block = 0; block < numSharedBlocksCtx; ++block)
            {
                for (auto beam = 0; beam < maxBeamWidth; ++beam)
                {
                    auto const kOffsetIdx = tc::flat_index(blockOffsetsShape.d, beam, 0, block);
                    auto const vOffsetIdx = tc::flat_index(blockOffsetsShape.d, beam, 1, block);
                    auto const kOffset = blockOffsetsPtr[kOffsetIdx];
                    auto const vOffset = blockOffsetsPtr[vOffsetIdx];
                    EXPECT_EQ(kOffset.get(), runningSum) << "beam:" << beam << " block:" << block;
                    ASSERT_EQ(vOffset.get(), runningSum + 1) << "beam:" << beam << " block:" << block;
                }
                runningSum += offsetBetweenBlocks;
            }
            {
                auto const block = numSharedBlocksCtx;
                for (auto beam = 0; beam < maxBeamWidth; ++beam)
                {
                    auto const kOffsetIdx = tc::flat_index(blockOffsetsShape.d, beam, 0, block);
                    auto const vOffsetIdx = tc::flat_index(blockOffsetsShape.d, beam, 1, block);
                    auto const kOffset = blockOffsetsPtr[kOffsetIdx];
                    auto const vOffset = blockOffsetsPtr[vOffsetIdx];
                    EXPECT_EQ(kOffset.get(), runningSum) << "beam:" << beam << " block:" << block;
                    ASSERT_EQ(vOffset.get(), runningSum + 1) << "beam:" << beam << " block:" << block;
                    runningSum += offsetBetweenBlocks;
                }
            }
        }
    }

    // replace the shared block with unshared blocks
    EXPECT_NO_THROW(kvCacheManager.addToken(requestId));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks - numSharedBlocksCtx - maxBeamWidth * 2 + 1);
    EXPECT_NO_THROW(kvCacheManager.addToken(requestId));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks - numSharedBlocksCtx - maxBeamWidth * 2 + 1);
    EXPECT_NO_THROW(kvCacheManager.removeSequence(requestId));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks);

    auto currentNumBlocks = totalNumBlocks;
    for (auto requestCounter = 0; requestCounter < maxNumSequences; ++requestCounter)
    {
        auto const nextRequestId = static_cast<RequestIdType>(requestId + requestCounter);
        EXPECT_NO_THROW(kvCacheManager.addSequence(nextRequestId, inputLength, maxBeamWidth));
        currentNumBlocks -= numSharedBlocksCtx + maxBeamWidth;
        EXPECT_EQ(blockManager.getNumFreeBlocks(), currentNumBlocks);
        EXPECT_NO_THROW(kvCacheManager.addToken(nextRequestId));
        currentNumBlocks -= maxBeamWidth - 1;
        EXPECT_EQ(blockManager.getNumFreeBlocks(), currentNumBlocks);
    }
    auto numUsedBlocks = maxNumSequences * (numSharedBlocksCtx + maxBeamWidth * 2 - 1);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks - numUsedBlocks);
}

TEST_P(KVCacheManagerTest, KVCacheManagerBatchTest)
{
    using DType = half;
    using SizeType32 = KVCacheManager::SizeType32;

    auto constexpr numLayers = 4;
    auto constexpr numHeads = 2;
    // heterogeneous layers
    std::vector<SizeType32> const numHeadsPerLayer({3, 2, 1, 2});
    std::map<SizeType32, SizeType32> const expectedHeadsPerPool({{0, 1}, {1, 2}, {2, 3}});
    std::map<SizeType32, SizeType32> const expectedLayersPerPool({{0, 1}, {1, 2}, {2, 1}});
    auto constexpr sizePerHead = 64;
    auto constexpr hiddenSize = numHeads * sizePerHead;
    auto constexpr tokensPerBlock = 64;
    auto constexpr maxBlocksPerSeq = 32;
    auto constexpr maxNumSequences = 8;
    auto constexpr maxBeamWidth = 4;
    auto constexpr sinkTokenLength = 0;
    auto constexpr useOneMoreBlock = false;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr maxNumTokens = tokensPerBlock * maxBlocksPerSeq;
    auto constexpr maxAttentionWindow = maxNumTokens;
    auto constexpr inputLength = maxNumTokens - 2;
    auto constexpr numBlocksPerSeq = maxBlocksPerSeq - 1 + maxBeamWidth;

    auto constexpr totalNumBlocks = maxNumSequences * numBlocksPerSeq;
    auto constexpr blocksInSecondaryPool = 0;

    auto constexpr enableBlockReuse = false;
    auto constexpr onboardBlocks = true;
    auto const homogeneousLayers = GetParam();
    auto const expectedNumPools = homogeneousLayers ? 1 : static_cast<SizeType32>(expectedHeadsPerPool.size());

    KVCacheManager kvCacheManager = homogeneousLayers
        ? KVCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, totalNumBlocks, blocksInSecondaryPool,
            maxNumSequences, maxBeamWidth, maxAttentionWindow, sinkTokenLength, useOneMoreBlock, stream,
            enableBlockReuse, onboardBlocks)
        : KVCacheManager(numHeadsPerLayer, sizePerHead, tokensPerBlock, totalNumBlocks, blocksInSecondaryPool,
            maxNumSequences, maxBeamWidth, maxAttentionWindow, sinkTokenLength, useOneMoreBlock, stream,
            enableBlockReuse, onboardBlocks);

    EXPECT_EQ(kvCacheManager.getMaxBlocksPerSeq(), maxBlocksPerSeq);
    EXPECT_EQ(kvCacheManager.getTokensPerBlock(), tokensPerBlock);
    EXPECT_EQ(kvCacheManager.getMaxNumBlocks(), totalNumBlocks);

    auto const& blockManager = kvCacheManager.getBlockManager();
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks);

    for (auto requestId = 0; requestId < maxNumSequences; ++requestId)
    {
        EXPECT_NO_THROW(kvCacheManager.addSequence(requestId, inputLength, maxBeamWidth));
        auto const currentNumBlocks = totalNumBlocks - (requestId + 1) * numBlocksPerSeq;
        EXPECT_EQ(blockManager.getNumFreeBlocks(), currentNumBlocks);
    }

    tr::ITensor::SharedPtr kvCacheBlockOffsets = tr::BufferManager::cpu(
        tr::ITensor::makeShape({expectedNumPools, maxNumSequences * maxBeamWidth, 2, maxBlocksPerSeq}),
        tr::TRTDataType<tk::KVCacheIndex>::value);

    kvCacheManager.getBlockOffsetsOfBatch(*kvCacheBlockOffsets, 0, maxNumSequences, maxBeamWidth);

    EXPECT_EQ(blockManager.getNumPools(), expectedNumPools);
    if (homogeneousLayers)
    {
        EXPECT_EQ(blockManager.getBlockSize(0), numHeads * sizePerHead * tokensPerBlock);
    }
    else
    {
        for (auto layerIdx = 0; layerIdx < numLayers; layerIdx++)
        {

            EXPECT_EQ(blockManager.getBlockSize(blockManager.getLayerPoolIdx(layerIdx)),
                numHeadsPerLayer.at(layerIdx) * sizePerHead * tokensPerBlock);
        }
    }

    {
        for (auto poolIdx = 0; poolIdx < blockManager.getNumPools(); poolIdx++)
        {
            auto const numLayersInPool = homogeneousLayers ? numLayers : expectedLayersPerPool.at(poolIdx);
            auto const offsetBetweenBlocks = numLayersInPool * 2;
            tr::ITensor::SharedPtr blockOffsetsSlice = tr::ITensor::slice(
                tr::ITensor::at(kvCacheBlockOffsets, {poolIdx}), 0, maxNumSequences * maxBeamWidth);
            auto blockOffsetsShape = blockOffsetsSlice->getShape();
            auto const blockOffsetsPtr = tr::bufferCast<tk::KVCacheIndex>(*blockOffsetsSlice);

            tk::KVCacheIndex::UnderlyingType runningSum{0};
            for (auto requestId = 0; requestId < maxNumSequences; ++requestId)
            {
                for (auto block = 0; block < maxBlocksPerSeq - 1; ++block)
                {
                    for (auto beam = 0; beam < maxBeamWidth; ++beam)
                    {
                        auto const kOffsetIdx
                            = tc::flat_index(blockOffsetsShape.d, requestId * maxBeamWidth + beam, 0, block);
                        auto const vOffsetIdx
                            = tc::flat_index(blockOffsetsShape.d, requestId * maxBeamWidth + beam, 1, block);
                        auto const kOffset = blockOffsetsPtr[kOffsetIdx];
                        auto const vOffset = blockOffsetsPtr[vOffsetIdx];
                        EXPECT_EQ(kOffset.get(), runningSum) << "beam:" << beam << " block:" << block;
                        ASSERT_EQ(vOffset.get(), runningSum + 1) << "beam:" << beam << " block:" << block;
                    }
                    runningSum += offsetBetweenBlocks;
                }
                auto const block = maxBlocksPerSeq - 1;
                {
                    for (auto beam = 0; beam < maxBeamWidth; ++beam)
                    {
                        auto const kOffsetIdx
                            = tc::flat_index(blockOffsetsShape.d, requestId * maxBeamWidth + beam, 0, block);
                        auto const vOffsetIdx
                            = tc::flat_index(blockOffsetsShape.d, requestId * maxBeamWidth + beam, 1, block);
                        auto const kOffset = blockOffsetsPtr[kOffsetIdx];
                        auto const vOffset = blockOffsetsPtr[vOffsetIdx];
                        EXPECT_EQ(kOffset.get(), runningSum) << "beam:" << beam << " block:" << block;
                        ASSERT_EQ(vOffset.get(), runningSum + 1) << "beam:" << beam << " block:" << block;
                        runningSum += offsetBetweenBlocks;
                    }
                }
            }
        }
    }
}

void testNeededBlocksOneStep(bool kv_cache_block_reuse, int beamWidth, int draftLen, bool homogeneousLayers)
{
    using DType = half;
    using SizeType32 = KVCacheManager::SizeType32;

    auto constexpr numLayers = 4;
    auto constexpr numHeads = 2;
    // heterogeneous layers
    std::vector<SizeType32> const numHeadsPerLayer({3, 2, 1, 2});
    auto constexpr sizePerHead = 64;
    auto constexpr hiddenSize = numHeads * sizePerHead;
    auto constexpr tokensPerBlock = 8;
    auto constexpr maxBlocksPerSeq = 10;
    auto constexpr maxNumSequences = 8;
    auto constexpr sinkTokenLength = 0;
    auto constexpr useOneMoreBlock = false;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr totalNumBlocks = maxNumSequences * maxBlocksPerSeq;

    TLLM_CHECK(draftLen == 0 || beamWidth == 1);

    // Deal with one sequence for now
    auto constexpr requestId = static_cast<RequestIdType>(7);
    SizeType32 maxNewTokens = 20;
    bool isStreaming = false;

    SizeType32 maxInputLength{65};
    SizeType32 maxMaxBeamWidth{beamWidth};

    for (int maxBeamWidth = 1; maxBeamWidth <= maxMaxBeamWidth; ++maxBeamWidth)
    {
        tr::SamplingConfig const samplingConfig{maxBeamWidth};
        for (int inputLength = 1; inputLength < maxInputLength; ++inputLength)
        {
            auto constexpr maxNumTokens = tokensPerBlock * maxBlocksPerSeq;
            // auto constexpr maxAttentionWindow = maxNumTokens / 2;
            auto constexpr maxAttentionWindow = 46;
            auto constexpr totalNumBlocks = maxNumSequences * maxBlocksPerSeq;
            auto constexpr blocksInSecondaryPool = 0;
            auto constexpr onboardBlocks = true;

            KVCacheManager kvCacheManager = homogeneousLayers
                ? KVCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, totalNumBlocks,
                    blocksInSecondaryPool, maxNumSequences, maxBeamWidth, maxAttentionWindow, sinkTokenLength,
                    useOneMoreBlock, stream, kv_cache_block_reuse, onboardBlocks)
                : KVCacheManager(numHeadsPerLayer, sizePerHead, tokensPerBlock, totalNumBlocks, blocksInSecondaryPool,
                    maxNumSequences, maxBeamWidth, maxAttentionWindow, sinkTokenLength, useOneMoreBlock, stream,
                    kv_cache_block_reuse, onboardBlocks);

            EXPECT_EQ(kvCacheManager.getMaxBlocksPerSeq(), tc::ceilDiv(maxAttentionWindow, tokensPerBlock));

            auto inputTokens = std::make_shared<VecTokens>(VecTokens(inputLength, 0));

            auto draftTokens = std::make_shared<std::vector<SizeType32>>(draftLen);
            auto llmRequest
                = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming);
            llmRequest->setDraftTokens(draftTokens);

            auto remainingBlocksToCompletion = kvCacheManager.getRemainingBlocksToCompletion(*llmRequest);
            auto neededBlocksOneStep = kvCacheManager.getNeededBlocksOneStep(*llmRequest, false);

            EXPECT_NO_THROW(kvCacheManager.addSequence(requestId, inputLength, maxBeamWidth, llmRequest));
            for (int di = 0; di < draftLen && di < maxNewTokens && (inputLength + di) < maxAttentionWindow; ++di)
            {
                for (int beam = 0; beam < maxBeamWidth; beam++)
                {
                    llmRequest->addNewToken(1, beam);
                }
                EXPECT_NO_THROW(kvCacheManager.addToken(requestId));
            }

            auto numUsedBlocksThisStep = kvCacheManager.getUsedNumBlocks();
            EXPECT_EQ(numUsedBlocksThisStep, neededBlocksOneStep);

            // Simulate adding new tokens during generation
            llmRequest->mState = LlmRequestState::kGENERATION_IN_PROGRESS;
            for (int i = draftLen; i < maxNewTokens && (inputLength + i) < maxAttentionWindow; i += (draftLen + 1))
            {
                auto numCurrentlyUsedBlocks = kvCacheManager.getUsedNumBlocks();
                for (int beam = 0; beam < maxBeamWidth; beam++)
                {
                    llmRequest->addNewToken(1, beam);
                }

                neededBlocksOneStep = kvCacheManager.getNeededBlocksOneStep(*llmRequest, false);

                for (int beam = 0; beam < maxBeamWidth; beam++)
                {
                    for (int di = 0;
                         di < draftLen && (i + di) < maxNewTokens && (inputLength + i + di) < maxAttentionWindow; ++di)
                    {
                        llmRequest->addNewToken(1, beam);
                    }
                }

                for (int di = 0;
                     di < draftLen + 1 && (i + di) < maxNewTokens && (inputLength + i + di) < maxAttentionWindow; ++di)
                {
                    EXPECT_NO_THROW(kvCacheManager.addToken(requestId));
                }
                numUsedBlocksThisStep = kvCacheManager.getUsedNumBlocks() - numCurrentlyUsedBlocks;

                EXPECT_EQ(numUsedBlocksThisStep, neededBlocksOneStep);
            }

            // After adding all tokens, we should match remainingBlocksToCompletion
            EXPECT_EQ(remainingBlocksToCompletion, kvCacheManager.getUsedNumBlocks());
            EXPECT_EQ(kvCacheManager.getRemainingBlocksToCompletion(*llmRequest), 0);
        }
    }
}

TEST_P(KVCacheManagerTest, neededBlocksOneStepKvCacheBlockReuse)
{
    testNeededBlocksOneStep(true, 1, 0, GetParam()); // maxBeamWidth is 1 when kv cache reuse is enabled
}

TEST_P(KVCacheManagerTest, neededBlocksOneStep)
{
    testNeededBlocksOneStep(false, 4, 0, GetParam());
}

TEST_P(KVCacheManagerTest, neededBlocksOneStepKvCacheBlockReuseDraftTokens)
{
    testNeededBlocksOneStep(true, 1, 5, GetParam());
}

INSTANTIATE_TEST_SUITE_P(KVCacheManagerTest, KVCacheManagerTest, testing::Values(true, false), // homogeneousLayers
    generateTestName);
