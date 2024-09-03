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

#include "tensorrt_llm/batch_manager/cacheTransceiver.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/common/mpiUtils.h"
#include "tensorrt_llm/executor/contextPhaseState.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/runtime/common.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace tr = tensorrt_llm::runtime;
using SizeType32 = tensorrt_llm::runtime::SizeType32;
using LlmRequest = tensorrt_llm::batch_manager::LlmRequest;
using namespace tensorrt_llm::batch_manager::kv_cache_manager;
using namespace tensorrt_llm::batch_manager;
namespace texec = tensorrt_llm::executor;

using testing::Return;
using testing::ReturnRef;

// ---------------------------------------
//            CacheConfigTest
// ---------------------------------------

class CacheConfigTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
public:
    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(CacheConfigTest, EqualTo)
{
    constexpr SizeType32 vocabSize{25};
    constexpr SizeType32 nbAttentionLayers{10};
    constexpr SizeType32 nbRnnLayers{2};
    constexpr SizeType32 nbHeads{12};
    constexpr SizeType32 hiddenSize{768};
    constexpr nvinfer1::DataType dtype{nvinfer1::DataType::kFLOAT};
    constexpr SizeType32 tokensPerBlock{64};
    constexpr SizeType32 tensorParallelism{8};
    constexpr SizeType32 pipelineParallelism{2};
    constexpr SizeType32 sizePerHead{hiddenSize / nbHeads};

    tr::ModelConfig modelConfig{vocabSize, nbAttentionLayers, nbRnnLayers, nbHeads, hiddenSize, dtype};
    modelConfig.setTokensPerBlock(tokensPerBlock);
    tr::WorldConfig worldConfig{tensorParallelism, pipelineParallelism};

    CacheConfig config0{modelConfig, worldConfig, dtype};
    CacheConfig config1{
        nbAttentionLayers, nbHeads, sizePerHead, tokensPerBlock, tensorParallelism, pipelineParallelism, dtype};
    EXPECT_EQ(config0, config1);
}

// TODO: Restore gmock and multi-rank tests.

// ---------------------------------------
//          RealTransceiverTest
// ---------------------------------------

class MpiSymmetricalCacheTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
protected:
    void SetUp() override {}

    void TearDown() override {}

    SizeType32 setUpCommunicator()
    {
        tensorrt_llm::mpi::initialize(tensorrt_llm::mpi::MpiThreadSupport::THREAD_MULTIPLE);
        mComm = std::addressof(tensorrt_llm::mpi::MpiComm::session());
        mWorldSize = mComm->getSize();
        isSender = mComm->getRank() % 2 == 0;
        return mWorldSize;
    }

    void setUpCacheManager()
    {
        auto constexpr numLayers = 2;
        auto constexpr numHeads = 2;
        auto constexpr sizePerHead = 64;
        auto constexpr hiddenSize = numHeads * sizePerHead;
        auto constexpr tokensPerBlock = 64;
        auto constexpr maxBlocksPerSeq = 10;
        auto constexpr maxBeamWidth = 4;
        auto constexpr sinkTokenLength = 0;
        auto constexpr useOneMoreBlock = false;
        mMaxNumSequences = 8;
        auto const stream = std::make_shared<tr::CudaStream>();

        auto constexpr maxNumTokens = tokensPerBlock * maxBlocksPerSeq;
        auto constexpr maxAttentionWindow = maxNumTokens;
        auto constexpr inputLength = maxNumTokens - tokensPerBlock - 1;
        auto constexpr numSharedBlocks = inputLength / tokensPerBlock;
        auto constexpr numBlocksPerSeq = numSharedBlocks + (maxBlocksPerSeq - numSharedBlocks) * maxBeamWidth;

        auto totalNumBlocks = mMaxNumSequences * numBlocksPerSeq;
        auto constexpr blocksInSecondaryPool = 0;

        auto constexpr enableBlockReuse = false;
        auto constexpr onboardBlocks = true;

        mManager = std::make_unique<KVCacheManager>(numLayers, numHeads, sizePerHead, tokensPerBlock, totalNumBlocks,
            blocksInSecondaryPool, mMaxNumSequences, maxBeamWidth, maxAttentionWindow, sinkTokenLength, useOneMoreBlock,
            stream, enableBlockReuse, onboardBlocks);

        // UVM seems to be incompatible with MPI, and it is continuing to investigate.
        bool constexpr useUvm = false;
        mManager->allocatePools(nvinfer1::DataType::kFLOAT, useUvm);
    }

    void setUpCacheTransceiver()
    {
        if (isSender)
        {
            mResponder = makeMpiCacheResponder(*mComm, mManager.get());
        }
        else
        {
            mRequester = makeMpiCacheRequester(*mComm, mManager.get());
        }
    }

    auto makeLlmRequest(SizeType32 length, std::vector<texec::SizeType32> contextRanks)
    {
        constexpr SizeType32 maxNewTokens{1};
        texec::Request request{VecTokens(length), maxNewTokens};
        auto state = std::make_unique<texec::ContextPhaseState>(mRequestId, std::move(contextRanks));
        auto stats = texec::ContextPhaseParams({}, state.release());
        request.setContextPhaseParams(std::move(stats));
        return std::make_unique<LlmRequest>(mRequestId++, std::move(request));
    }

    void addRequestAndTransportCache(std::shared_ptr<LlmRequest> const& llmRequest)
    {
        auto constexpr beamIdx{0};
        auto constexpr beamWidth{1};
        TLLM_CHECK(mSeqSlotIdx < mMaxNumSequences);
        llmRequest->mSeqSlot = mSeqSlotIdx++;
        mManager->addSequence(llmRequest->mSeqSlot.value(), llmRequest->getNumTokens(beamIdx), beamWidth, llmRequest);
        if (isSender)
        {
            auto blockEndIt = getBlockEndIt(*mManager, *llmRequest, beamIdx);
            for (auto it = getBlockBeginIt(*mManager, *llmRequest, beamIdx); it != blockEndIt; ++it)
            {
                TLLM_CUDA_CHECK(cudaMemsetAsync(it->data(), llmRequest->mRequestId, it->getSizeInBytes()));
            }
            auto future = mResponder->respondAndSendAsync(*llmRequest);
            future.get();
        }
        else
        {
            auto future = mRequester->requestAndReceiveAsync(*llmRequest);
            future.get();
            auto blockEndIt = getBlockEndIt(*mManager, *llmRequest, beamIdx);
            for (auto it = getBlockBeginIt(*mManager, *llmRequest, beamIdx); it != blockEndIt; ++it)
            {
                std::vector<char> bytes(it->getSizeInBytes());
                TLLM_CUDA_CHECK(cudaMemcpy(bytes.data(), it->data(), it->getSizeInBytes(), cudaMemcpyDeviceToHost));
                EXPECT_TRUE(std::all_of(
                    bytes.begin(), bytes.end(), [llmRequest](int i) { return i == llmRequest->mRequestId; }));
            }
        }
    }

    bool isSender{false};
    tensorrt_llm::mpi::MpiComm const* mComm;
    SizeType32 mSeqSlotIdx{0};
    SizeType32 mWorldSize{0};
    LlmRequest::RequestIdType mRequestId{0};
    SizeType32 mMaxNumSequences{};
    std::unique_ptr<KVCacheManager> mManager;
    std::unique_ptr<DataResponder> mResponder;
    std::unique_ptr<DataRequester> mRequester;
};

TEST_F(MpiSymmetricalCacheTest, SimpleTest)
{
    auto worldSize = setUpCommunicator();
    if (worldSize % 2)
    {
        GTEST_SKIP() << "An even number of processes is required to run this test.";
    }
    setUpCacheManager();
    setUpCacheTransceiver();
    for (auto len : {10, 20, 30})
    {
        addRequestAndTransportCache(makeLlmRequest(len, {0}));
    }
}
