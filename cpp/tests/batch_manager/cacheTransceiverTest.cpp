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
#include "tensorrt_llm/executor/dataTransceiverState.h"
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
//            RequestInfoTest
// ---------------------------------------

template <typename T>
T serializeDeserialize(T val)
{
    auto size = T::serializedSize(val);
    std::ostringstream oss;
    T::serialize(val, oss);
    EXPECT_EQ(oss.str().size(), size);

    std::istringstream iss(oss.str());
    return T::deserialize(iss);
}

class RequestInfoTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
public:
    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(RequestInfoTest, Basic)
{
    auto state = std::make_unique<texec::DataTransceiverState>();
    state->setCommState(texec::kv_cache::CommState{12, "127.0.0.1"});
    state->setCacheState(texec::kv_cache::CacheState{10, 12, 128, 128, 8, 8, nvinfer1::DataType::kFLOAT});
    RequestInfo info{1, *state};
    auto info2 = serializeDeserialize(info);
    EXPECT_EQ(info, info2);
}

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

    tr::ModelConfig modelConfig{
        vocabSize, nbAttentionLayers + nbRnnLayers, nbAttentionLayers, nbRnnLayers, nbHeads, hiddenSize, dtype};
    modelConfig.setTokensPerBlock(tokensPerBlock);
    tr::WorldConfig worldConfig{tensorParallelism, pipelineParallelism};

    texec::kv_cache::CacheState state0{modelConfig, worldConfig};
    texec::kv_cache::CacheState state1{
        nbAttentionLayers, nbHeads, sizePerHead, tokensPerBlock, tensorParallelism, pipelineParallelism, dtype};
    EXPECT_EQ(state0, state1);
}

// ---------------------------------------
//          MockTransceiverTest
// ---------------------------------------

class MockDataSender : public DataSender
{
public:
    MockDataSender()
    {
        ON_CALL(*this, getCommState).WillByDefault(ReturnRef(mState));
        ON_CALL(*this, recvRequestInfo)
            .WillByDefault(Return(RequestInfo{0,
                texec::DataTransceiverState{
                    texec::kv_cache::CacheState{10, 12, 128, 128, 8, 8, nvinfer1::DataType::kFLOAT},
                    texec::kv_cache::CommState{std::vector<SizeType32>{0}, 0}}}));
        ON_CALL(*this, availableRelease).WillByDefault(Return(true));
    }

    MOCK_METHOD(RequestInfo, recvRequestInfo, (), (override));
    MOCK_METHOD(void, sendSync, (LlmRequest const&), (override));
    MOCK_METHOD(texec::kv_cache::CommState const&, getCommState, (), (const override));
    MOCK_METHOD(void, setCommState, (texec::kv_cache::CommState const&), (override));
    MOCK_METHOD(bool, availableRelease, (LlmRequest const&), (override));

private:
    static texec::kv_cache::CommState mState;
};

texec::kv_cache::CommState MockDataSender::mState;

class MockDataReceiver : public DataReceiver
{
public:
    MOCK_METHOD(void, sendRequestInfo, (LlmRequest const&), (override));
    MOCK_METHOD(void, receiveSync, (LlmRequest const&), (override));
};

class MockTransceiverTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
public:
    void SetUp() override {}

    void TearDown() override {}

    static auto makeLlmRequest(
        LlmRequest::RequestIdType requestId = 0, SizeType32 maxNewTokens = 1, VecTokens inputTokens = {-1})
    {
        texec::Request request{std::move(inputTokens), maxNewTokens};
        auto state = std::make_unique<texec::DataTransceiverState>();
        auto stats = texec::ContextPhaseParams({}, requestId, state.release());
        request.setContextPhaseParams(std::move(stats));
        return std::make_unique<LlmRequest>(requestId, std::move(request));
    }
};

TEST_F(MockTransceiverTest, MpiResponderBasic)
{
    auto sender = std::make_unique<MockDataSender>();
    EXPECT_CALL(*sender, recvRequestInfo)
        .WillOnce(Return(RequestInfo{0,
            texec::DataTransceiverState{texec::kv_cache::CacheState{10, 12, 128, 128, 8, 8, nvinfer1::DataType::kFLOAT},
                texec::kv_cache::CommState{std::vector<SizeType32>{0}, 0}}}));
    EXPECT_CALL(*sender, sendSync).WillOnce(Return());
    EXPECT_CALL(*sender, availableRelease).WillOnce(Return(true));

    DataResponder responder{std::move(sender)};
    auto request = makeLlmRequest(0);
    auto future = responder.respondAndSendAsync(*request);
    future.get();
}

TEST_F(MockTransceiverTest, MpiRequesterBasic)
{
    auto receiver = std::make_unique<MockDataReceiver>();
    EXPECT_CALL(*receiver, sendRequestInfo).WillOnce(Return());
    EXPECT_CALL(*receiver, receiveSync).WillOnce(Return());

    DataRequester requester{std::move(receiver)};
    auto request = makeLlmRequest(0);
    auto future = requester.requestAndReceiveAsync(*request);
    future.get();
}

// TODO: Restore multi-rank tests.

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
        mComm = std::addressof(tensorrt_llm::mpi::MpiComm::world());
        mWorldSize = mComm->getSize();
        mlocalRank = mComm->getRank() / 2;
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
        auto constexpr dataType = nvinfer1::DataType::kFLOAT;

        mManager = std::make_unique<KVCacheManager>(numLayers, numHeads, sizePerHead, tokensPerBlock, totalNumBlocks,
            blocksInSecondaryPool, mMaxNumSequences, maxBeamWidth, maxAttentionWindow, sinkTokenLength, useOneMoreBlock,
            stream, enableBlockReuse, onboardBlocks);
        mCacheState = std::make_unique<texec::kv_cache::CacheState>(
            numLayers, numHeads, sizePerHead, tokensPerBlock, 1, 1, dataType);

        // UVM seems to be incompatible with MPI, and it is continuing to investigate.
        bool constexpr useUvm = false;
        mManager->allocatePools(dataType, useUvm);
    }

    void setUpCacheTransceiver()
    {
        if (isSender)
        {
            mResponder = std::make_unique<DataResponder>(std::make_unique<MpiDataSender<texec::kv_cache::CacheState>>(
                *mComm, *mCacheState, mlocalRank, std::make_unique<CacheOutputFormatter<MpiComm>>(mManager.get())));
        }
        else
        {
            mRequester = std::make_unique<DataRequester>(std::make_unique<MpiDataReceiver<texec::kv_cache::CacheState>>(
                *mComm, *mCacheState, mlocalRank, std::make_unique<CacheInputFormatter<MpiComm>>(mManager.get())));
        }
    }

    auto makeLlmRequest(SizeType32 length)
    {
        constexpr SizeType32 maxNewTokens{1};
        texec::Request request{VecTokens(length), maxNewTokens};
        auto state = std::make_unique<texec::DataTransceiverState>();
        state->setCommState(texec::kv_cache::CommState{std::vector<int>{0}});
        state->setCacheState(*mCacheState);
        auto stats = texec::ContextPhaseParams({}, mRequestId, state.release());
        request.setContextPhaseParams(std::move(stats));
        return std::make_unique<LlmRequest>(mRequestId++, std::move(request));
    }

    void addRequestAndTransportCache(std::shared_ptr<LlmRequest> const& llmRequest)
    {
        auto constexpr beamIdx{0};
        auto constexpr beamWidth{1};
        mManager->addSequence(llmRequest->mRequestId, llmRequest->getNumTokens(beamIdx), beamWidth, llmRequest);
        if (isSender)
        {
            auto blockEndIt = getBlockEndIt(*mManager, *llmRequest, beamIdx, 0);
            for (auto it = getBlockBeginIt(*mManager, *llmRequest, beamIdx, 0); it != blockEndIt; ++it)
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
            auto blockEndIt = getBlockEndIt(*mManager, *llmRequest, beamIdx, 0);
            for (auto it = getBlockBeginIt(*mManager, *llmRequest, beamIdx, 0); it != blockEndIt; ++it)
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
    SizeType32 mWorldSize{0}, mlocalRank{0};
    LlmRequest::RequestIdType mRequestId{0};
    SizeType32 mMaxNumSequences{};
    std::unique_ptr<KVCacheManager> mManager;
    std::unique_ptr<DataResponder> mResponder;
    std::unique_ptr<DataRequester> mRequester;
    std::unique_ptr<texec::kv_cache::CacheState> mCacheState;
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
