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

// ---------------------------------------
//          MockTransceiverTest
// ---------------------------------------

class MockMpiComm : public MpiComm
{
public:
    MockMpiComm()
        : MpiComm{tensorrt_llm::mpi::MpiComm::session()}
    {
        ON_CALL(*this, getRank).WillByDefault(Return(0));
        ON_CALL(*this, getSize).WillByDefault(Return(2));
        ON_CALL(*this, recvRequestId(testing::An<SizeType32>())).WillByDefault(Return(0));
        ON_CALL(*this, recvRequestId()).WillByDefault(Return(std::pair<int, LlmRequest::RequestIdType>{1, 0}));
    }

    MOCK_METHOD(void, sendRequestId, (const LlmRequest::RequestIdType, const SizeType32), (const, override));
    MOCK_METHOD(LlmRequest::RequestIdType, recvRequestId, (const SizeType32), (const, override));
    MOCK_METHOD((std::pair<int, LlmRequest::RequestIdType>), recvRequestId, (), (const, override));
    MOCK_METHOD(void, sendBuffer, (tr::IBuffer const&, int), (const, override));
    MOCK_METHOD(void, recvBuffer, (tr::IBuffer&, int), (const, override));
    MOCK_METHOD(void, setCudaDevice, (), (const, override));
    MOCK_METHOD(int, getRank, (), (const, override));
    MOCK_METHOD(int, getSize, (), (const, override));
};

class MockDataSender : public DataSender
{
public:
    MOCK_METHOD(bool, inquireSupport, (DataContext const*), (override));
    MOCK_METHOD(void, send, (LlmRequest const&, DataContext const&), (override));
};

class MockDataReceiver : public DataReceiver
{
public:
    MOCK_METHOD(bool, inquireSupport, (DataContext const*), (override));
    MOCK_METHOD(void, receive, (LlmRequest const&, DataContext const&), (override));
};

class MockTransceiverTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
public:
    void SetUp() override {}

    void TearDown() override {}

    struct Params
    {
        Params(int numTransceiver)
            : mComm{std::make_unique<MockMpiComm>()}
            , mSelfContext{std::make_unique<DataContext>(std::vector<SizeType32>{0})}
        {
            std::generate_n(
                std::back_inserter(mSenders), numTransceiver, [] { return std::make_unique<MockDataSender>(); });
            std::generate_n(
                std::back_inserter(mReceivers), numTransceiver, [] { return std::make_unique<MockDataReceiver>(); });
        }

        std::unique_ptr<MockMpiComm> mComm;
        std::unique_ptr<DataContext> mSelfContext;
        std::vector<std::unique_ptr<MockDataSender>> mSenders;
        std::vector<std::unique_ptr<MockDataReceiver>> mReceivers;
    };

    template <typename TBase, typename TDerived>
    static auto move(std::vector<std::unique_ptr<TDerived>>& param)
    {
        static_assert(std::is_base_of<TBase, TDerived>());
        return std::vector<std::unique_ptr<TBase>>{
            std::make_move_iterator(param.begin()), std::make_move_iterator(param.end())};
    }

    static auto makeResponder(Params&& params)
    {
        return MpiResponder{
            move<DataSender>(params.mSenders), *params.mComm, std::make_unique<DataContext>(*params.mSelfContext)};
    }

    static auto makeRequester(Params&& params)
    {
        return MpiRequester{
            move<DataReceiver>(params.mReceivers), *params.mComm, std::make_unique<DataContext>(*params.mSelfContext)};
    }

    static auto makeLlmRequest(
        LlmRequest::RequestIdType requestId = 0, SizeType32 maxNewTokens = 1, VecTokens inputTokens = {-1})
    {
        texec::Request request{std::move(inputTokens), maxNewTokens};
        auto state = std::make_unique<texec::ContextPhaseState>(requestId, std::vector<texec::SizeType32>{});
        auto stats = texec::ContextPhaseParams({}, state.release());
        request.setContextPhaseParams(std::move(stats));
        return std::make_unique<LlmRequest>(requestId, std::move(request));
    }
};

TEST_F(MockTransceiverTest, MpiResponderBasic)
{
    Params params{1};
    ON_CALL(*params.mSenders.front(), inquireSupport).WillByDefault(Return(true));
    EXPECT_CALL(*params.mSenders.front(), inquireSupport).WillOnce(Return(true));
    EXPECT_CALL(*params.mSenders.front(), send).WillOnce(Return());
    EXPECT_CALL(*params.mComm, setCudaDevice).WillOnce(Return());
    EXPECT_CALL(*params.mComm, recvRequestId()).WillOnce(Return(std::pair<int, LlmRequest::RequestIdType>{1, 0}));
    auto responder = makeResponder(std::move(params));
    auto request = makeLlmRequest(0);
    auto future = responder.respondAndSendAsync(*request);
    future.get();
}

TEST_F(MockTransceiverTest, MpiRequesterBasic)
{
    Params params{1};
    ON_CALL(*params.mReceivers.front(), inquireSupport).WillByDefault(Return(true));
    EXPECT_CALL(*params.mReceivers.front(), inquireSupport).WillOnce(Return(true));
    EXPECT_CALL(*params.mReceivers.front(), receive).WillOnce(Return());
    EXPECT_CALL(*params.mComm, setCudaDevice).WillOnce(Return());
    EXPECT_CALL(*params.mComm, sendRequestId).WillOnce(Return());
    auto requester = makeRequester(std::move(params));
    auto request = makeLlmRequest(0);
    auto future = requester.requestAndReceiveAsync(*request, DataContext{{1}, 0});
    future.get();
}

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
        mComm = std::make_unique<MpiComm>(tensorrt_llm::mpi::MpiComm::session());
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
        mCacheConfig = std::make_unique<CacheConfig>(
            numLayers, numHeads, sizePerHead, tokensPerBlock, 1, 1, nvinfer1::DataType::kFLOAT);

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
        for (int i = 0; i < mWorldSize / 2; ++i)
        {
            mSenderRanks.insert(mSenderRanks.end(), i * 2);
            mReceiverRanks.insert(mReceiverRanks.end(), i * 2 + 1);
        }

        TLLM_CHECK(mCacheConfig);
        auto selfRank = mComm->getRank();
        if (isSender)
        {
            SizeType32 selfIdx
                = std::distance(mSenderRanks.begin(), std::find(mSenderRanks.begin(), mSenderRanks.end(), selfRank));
            CacheContext selfContext{*mCacheConfig, mSenderRanks, selfIdx};
            std::vector<std::unique_ptr<DataSender>> senders;
            senders.emplace_back(std::make_unique<CacheBlockSender>(mManager.get(), *mComm, selfContext));
            mResponder = std::make_unique<MpiResponder>(
                std::move(senders), *mComm, std::make_unique<CacheContext>(selfContext));
        }
        else
        {
            SizeType32 selfIdx = std::distance(
                mReceiverRanks.begin(), std::find(mReceiverRanks.begin(), mReceiverRanks.end(), selfRank));
            CacheContext selfContext{*mCacheConfig, mReceiverRanks, selfIdx};
            std::vector<std::unique_ptr<DataReceiver>> receivers;
            receivers.emplace_back(std::make_unique<CacheBlockReceiver>(mManager.get(), *mComm, selfContext));
            mRequester = std::make_unique<MpiRequester>(
                std::move(receivers), *mComm, std::make_unique<CacheContext>(selfContext));
        }
    }

    auto makeLlmRequest(SizeType32 length)
    {
        constexpr SizeType32 maxNewTokens{1};
        texec::Request request{VecTokens(length), maxNewTokens};
        auto state = std::make_unique<texec::ContextPhaseState>(mRequestId, std::vector<texec::SizeType32>{});
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
            DataContext context{mSenderRanks, 0};
            auto future = mRequester->requestAndReceiveAsync(*llmRequest, context);
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
    std::unique_ptr<MpiComm> mComm;
    SizeType32 mSeqSlotIdx{0};
    SizeType32 mWorldSize{0};
    LlmRequest::RequestIdType mRequestId{0};
    SizeType32 mMaxNumSequences{};
    std::unique_ptr<KVCacheManager> mManager;
    std::unique_ptr<DataResponder> mResponder;
    std::unique_ptr<DataRequester> mRequester;
    std::unique_ptr<CacheConfig> mCacheConfig;
    std::vector<SizeType32> mSenderRanks, mReceiverRanks;
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
        addRequestAndTransportCache(makeLlmRequest(len));
    }
}
