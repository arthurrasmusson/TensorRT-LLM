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
#include <cstdio>
#include <memory>
#if ENABLE_UCX
#include "tensorrt_llm/batch_manager/ucxDataTransceiver.h"
#endif
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/mpiUtils.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/utils/multiDeviceUtils.h"
#include "gtest/gtest.h"
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <random>
#include <thread>

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
    if (tensorrt_llm::mpi::MpiComm::world().getSize() > 2)
    {
        GTEST_SKIP() << "mpirun with procs<=2 is required to run this test.";
    }
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
    if (tensorrt_llm::mpi::MpiComm::world().getSize() > 2)
    {
        GTEST_SKIP() << "mpirun with procs<=2 is required to run this test.";
    }
    constexpr SizeType32 vocabSize{25};
    constexpr SizeType32 nbAttentionLayers{10};
    constexpr SizeType32 nbRnnLayers{2};
    constexpr SizeType32 nbHeads{12};
    constexpr SizeType32 hiddenSize{768};
    constexpr nvinfer1::DataType dtype{nvinfer1::DataType::kFLOAT};
    constexpr SizeType32 tokensPerBlock{64};
    constexpr SizeType32 tensorParallelism{8};
    constexpr SizeType32 pipelineParallelism{2};
    constexpr SizeType32 contextParallelism{1};
    constexpr SizeType32 sizePerHead{hiddenSize / nbHeads};

    tr::ModelConfig modelConfig{
        vocabSize, nbAttentionLayers + nbRnnLayers, nbAttentionLayers, nbRnnLayers, nbHeads, hiddenSize, dtype};
    modelConfig.setTokensPerBlock(tokensPerBlock);
    tr::WorldConfig worldConfig{tensorParallelism, pipelineParallelism, contextParallelism};

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
    MOCK_METHOD(texec::kv_cache::CommState const&, getCommState, (), (const));
    MOCK_METHOD(void, setCommState, (texec::kv_cache::CommState), (override));
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
    if (tensorrt_llm::mpi::MpiComm::world().getSize() > 2)
    {
        GTEST_SKIP() << "mpirun with procs<=2 is required to run this test.";
    }
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

    if (tensorrt_llm::mpi::MpiComm::world().getSize() > 2)
    {
        GTEST_SKIP() << "mpirun with procs<=2 is required to run this test.";
    }
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
        tensorrt_llm::mpi::MpiComm::setSession(mComm->split(static_cast<int>(isSender), mlocalRank));
        return mWorldSize;
    }

    void setUpCacheManager()
    {
        auto constexpr numLayers = 4;
        auto constexpr numHeads = 2;
        auto constexpr sizePerHead = 64;
        auto constexpr hiddenSize = numHeads * sizePerHead;
        auto constexpr tokensPerBlock = 64;
        auto constexpr maxBlocksPerSeq = 10;
        auto constexpr maxBeamWidth = 4;
        auto constexpr sinkTokenLength = 0;
        mMaxNumSequences = 8;
        auto const stream = std::make_shared<tr::CudaStream>();

        auto constexpr maxNumTokens = tokensPerBlock * maxBlocksPerSeq;
        auto constexpr maxAttentionWindow = maxNumTokens;
        auto constexpr temporaryAttentionWindow = 0;
        auto constexpr inputLength = maxNumTokens - tokensPerBlock - 1;
        auto constexpr numSharedBlocks = inputLength / tokensPerBlock;
        auto constexpr numBlocksPerSeq = numSharedBlocks + (maxBlocksPerSeq - numSharedBlocks) * maxBeamWidth;

        auto totalNumBlocks = mMaxNumSequences * numBlocksPerSeq;
        auto constexpr blocksInSecondaryPool = 0;

        auto constexpr enableBlockReuse = false;
        auto constexpr onboardBlocks = true;
        auto constexpr dataType = nvinfer1::DataType::kFLOAT;

        mManager = std::make_unique<KVCacheManager>(numLayers, numHeads, sizePerHead, tokensPerBlock, totalNumBlocks,
            blocksInSecondaryPool, mMaxNumSequences, maxBeamWidth, maxAttentionWindow, temporaryAttentionWindow,
            sinkTokenLength, stream, std::nullopt, enableBlockReuse, onboardBlocks);
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
            mResponder = std::make_unique<DataResponder>(std::make_unique<MpiDataSender>(
                *mComm, *mCacheState, mlocalRank, std::make_unique<CacheFormatter>(mManager.get())));
        }
        else
        {
            mRequester = std::make_unique<DataRequester>(std::make_unique<MpiDataReceiver>(
                *mComm, *mCacheState, mlocalRank, std::make_unique<CacheFormatter>(mManager.get())));
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
                TLLM_CUDA_CHECK(cudaMemset(it->data(), llmRequest->mRequestId, it->getSizeInBytes()));
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
    if (worldSize != 2)
    {
        GTEST_SKIP() << "mpirun 2 processes is required to run this test.";
    }
    setUpCacheManager();
    setUpCacheTransceiver();
    for (auto len : {10, 20, 30})
    {
        addRequestAndTransportCache(makeLlmRequest(len));
    }
}

#if ENABLE_MULTI_DEVICE

using AsymmetricTestParam = std::tuple<int, int, int, int, int, int, int, int, nvinfer1::DataType>;

class MpiAsymmetricalCacheTest : public ::testing::TestWithParam<AsymmetricTestParam>
{

protected:
    void SetUp() override {}

    void TearDown() override {}

    void setUpCommunicator(int contextTp, int contextPp, int genTp, int genPp)
    {
#if ENABLE_MULTI_DEVICE
        tensorrt_llm::mpi::initialize(tensorrt_llm::mpi::MpiThreadSupport::THREAD_MULTIPLE);

        if (tensorrt_llm::mpi::MpiComm::world().getSize() != 8)
        {
            GTEST_SKIP() << "mpirun with procs=8  is required to run this test.";
        }
        int worldSize = tensorrt_llm::mpi::MpiComm::world().getSize();
        int worldRank = tensorrt_llm::mpi::MpiComm::world().getRank();
        tensorrt_llm::mpi::MpiComm::world().barrier();
        int contextRanks = contextTp * contextPp;
        int genRanks = genTp * genPp;
        int nprocs = (contextRanks + genRanks);

        mIsContext = false;
        mIsGeneration = false;
        mParticipatingComm = tensorrt_llm::mpi::MpiComm::world().split(static_cast<int>(worldRank < nprocs), worldRank);
        tensorrt_llm::mpi::MpiComm::setSession(
            tensorrt_llm::mpi::MpiComm::world().split(static_cast<int>(worldRank < nprocs), worldRank));

        mIsContext = worldRank < contextRanks;
        mIsGeneration = (worldRank >= contextRanks && worldRank < (contextRanks + genRanks));
        if (worldRank >= nprocs)
        {
            return;
        }
        TLLM_LOG_INFO("Run cacheTransceiverTest for ContextTp: %d, ContextPp: %d, GenTp: %d, GenPp:%d", contextTp,
            contextPp, genTp, genPp);
        mComm = std::addressof(mParticipatingComm);

        mWorldSize = mComm->getSize();
        mRank = mComm->getRank();

        {
            mIsContext = mRank < contextRanks;
            mIsGeneration = (mRank >= contextRanks && mRank < (contextRanks + genRanks));
            mRankInInstance = mIsContext ? mRank : (mRank - contextRanks);
            mSizeInInstance = mIsContext ? (contextTp * contextPp) : (genTp * genPp);
            int color = 0;
            if (mIsGeneration)
            {
                color = 1;
            }
            if (mIsContext)
            {
                color = 2;
            }
            auto sessionComm = mComm->split(static_cast<int>(color), mComm->getRank());

            if (mIsContext)
            {
                mTpSize = contextTp;
                mPpSize = contextPp;
            }
            if (mIsGeneration)
            {
                mTpSize = genTp;
                mPpSize = genPp;
            }

            mTpRank = mRankInInstance % mTpSize;
            mPpRank = mRankInInstance / mTpSize;
            mContextRankSize = contextRanks;
            mGenRankSize = genRanks;
            mContextTpSize = contextTp;
            mContextPpSize = contextPp;

            EXPECT_EQ((sessionComm.getRank()), mRankInInstance);
            EXPECT_EQ(sessionComm.getSize(), mSizeInInstance);
            tensorrt_llm::mpi::MpiComm::setSession(std::move(sessionComm));
        }
#else
        GTEST_SKIP() << "ENABLE_MULTI_DEVICE  is required to run this test.";

#endif
    }

    void setUpCacheManager(
        int numLayers, int numHeads, int sizePerHead, int tokensPerBlock, nvinfer1::DataType dataType)
    {

        if (!(mIsContext || mIsGeneration))
        {
            return;
        }

        ASSERT_EQ(numLayers % mPpSize, 0);
        ASSERT_EQ(numHeads % mTpSize, 0);
        int numHeadsPerRank = numHeads / mTpSize;
        auto hiddenSize = numHeadsPerRank * sizePerHead;
        auto maxBlocksPerSeq = 10;
        auto maxBeamWidth = 1;
        auto constexpr sinkTokenLength = 0;
        mMaxNumSequences = 8;
        auto const stream = std::make_shared<tr::CudaStream>();

        auto maxNumTokens = tokensPerBlock * maxBlocksPerSeq;
        auto maxAttentionWindow = maxNumTokens;
        auto constexpr temporaryAttentionWindow = 0;
        auto inputLength = maxNumTokens - tokensPerBlock - 1;
        auto numSharedBlocks = inputLength / tokensPerBlock;
        auto numBlocksPerSeq = numSharedBlocks + (maxBlocksPerSeq - numSharedBlocks) * maxBeamWidth;

        auto totalNumBlocks = mMaxNumSequences * numBlocksPerSeq;
        auto constexpr blocksInSecondaryPool = 0;

        auto constexpr enableBlockReuse = false;
        auto constexpr onboardBlocks = true;

        mManager = std::make_unique<KVCacheManager>(numLayers / mPpSize, numHeadsPerRank, sizePerHead, tokensPerBlock,
            totalNumBlocks, blocksInSecondaryPool, mMaxNumSequences, maxBeamWidth, maxAttentionWindow,
            temporaryAttentionWindow, sinkTokenLength, stream, std::nullopt, enableBlockReuse, onboardBlocks);
        mCacheState = std::make_unique<texec::kv_cache::CacheState>(
            numLayers, numHeadsPerRank, sizePerHead, tokensPerBlock, mTpSize, mPpSize, dataType);

        mContextCacheState = std::make_unique<texec::kv_cache::CacheState>(numLayers, numHeads / mContextTpSize,
            sizePerHead, tokensPerBlock, mContextTpSize, mContextPpSize, dataType);

        // UVM seems to be incompatible with MPI, and it is continuing to investigate.
        bool constexpr useUvm = false;
        mManager->allocatePools(dataType, useUvm);
    }

    void setUpCacheTransceiver()
    {
        if (!(mIsContext || mIsGeneration))
        {
            return;
        }

        if (tensorrt_llm::common::getEnvUseUCXKvCache())
        {
#if ENABLE_UCX

            TLLM_LOG_INFO("Enable UCX KV cache transport.");
            namespace su = tensorrt_llm::executor::serialize_utils;

            if (mIsContext)
            {

                mResponder = makeUcxCacheResponder(*mCacheState, mRankInInstance, mManager.get());

                COMM_SESSION.barrier();
                tensorrt_llm::executor::kv_cache::CommState commState = mResponder->getCommState();
                std::ostringstream oStream;
                su::serialize(commState, oStream);
                auto str = oStream.str();
                std::vector<char> buffer(str.begin(), str.end());
                std::vector<SizeType32> sizeofBuffer(COMM_SESSION.getSize());
                SizeType32 bufferSize = buffer.size();
                COMM_SESSION.allgather(&bufferSize, sizeofBuffer.data(), 1, tensorrt_llm::mpi::MpiType::kINT32);
                SizeType32 recvBufferSize = std::accumulate(sizeofBuffer.begin(), sizeofBuffer.end(), 0);
                std::vector<char> recvBuffer(recvBufferSize);
                std::vector<int> displs(COMM_SESSION.getSize());
                for (int r = 0; r < COMM_SESSION.getSize(); r++)
                {
                    displs[r] = (r == 0) ? 0 : (displs[r - 1] + sizeofBuffer[r - 1]);
                }
                COMM_SESSION.allgatherv(buffer.data(), bufferSize, tensorrt_llm::mpi::MpiType::kCHAR, recvBuffer.data(),
                    sizeofBuffer, displs, tensorrt_llm::mpi::MpiType::kCHAR);

                // deserialize
                std::vector<tensorrt_llm::executor::kv_cache::CommState> commSessionCommState(COMM_SESSION.getSize());
                std::vector<tensorrt_llm::executor::kv_cache::SocketState> socketStates;
                for (int i = 0; i < COMM_SESSION.getSize(); i++)
                {
                    std::vector<char> serBuffer(
                        recvBuffer.begin() + displs[i], recvBuffer.begin() + (displs[i] + sizeofBuffer[i]));
                    su::VectorWrapBuf<char> strbuf(serBuffer);
                    std::istream is(&strbuf);
                    commSessionCommState[i] = su::deserialize<tensorrt_llm::executor::kv_cache::CommState>(is);
                    TLLM_CHECK_WITH_INFO(
                        commSessionCommState[i].getSocketState().size() == 1, "getSocketState size should be 1");
                    socketStates.push_back(commSessionCommState[i].getSocketState()[0]);
                }
                tensorrt_llm::executor::kv_cache::CommState allCommState{socketStates, mRankInInstance};
                mResponder->setCommState(std::move(allCommState));
                {
                    mContextCommState
                        = std::make_unique<tensorrt_llm::executor::kv_cache::CommState>(mResponder->getCommState());
                    std::ostringstream oStream2;
                    su::serialize(*mContextCommState, oStream2);
                    auto str2 = oStream2.str();
                    std::vector<char> buffer2(str2.begin(), str2.end());
                    mComm->bcast(buffer2, 0);
                }
            }
            else
            {
                mRequester = makeUcxCacheRequester(*mCacheState, mRankInInstance, mManager.get());
                std::vector<char> buffer{};
                mComm->bcast(buffer, 0);
                su::VectorWrapBuf<char> strbuf(buffer);
                std::istream is(&strbuf);
                mContextCommState = std::make_unique<tensorrt_llm::executor::kv_cache::CommState>(
                    su::deserialize<tensorrt_llm::executor::kv_cache::CommState>(is));
            }
#else
            TLLM_THROW("Responder based UCX need to compiled with ENABLE_UCX");
#endif
        }
        else
        {

            TLLM_LOG_INFO("Enable MPI KV cache transport.");

            if (mIsContext)
            {
                mResponder = std::make_unique<DataResponder>(std::make_unique<MpiDataSender>(
                    *mComm, *mCacheState, mRankInInstance, std::make_unique<CacheFormatter>(mManager.get())));
            }
            else
            {
                mRequester = std::make_unique<DataRequester>(std::make_unique<MpiDataReceiver>(
                    *mComm, *mCacheState, mRankInInstance, std::make_unique<CacheFormatter>(mManager.get())));
            }

            std::vector<int> contextRankVec(mContextRankSize);
            for (int i = 0; i < contextRankVec.size(); i++)
            {
                contextRankVec[i] = i;
            }
            mContextCommState = std::make_unique<tensorrt_llm::executor::kv_cache::CommState>(contextRankVec);
        }
    }

    auto makeLlmRequest(SizeType32 length)
    {

        constexpr SizeType32 maxNewTokens{1};
        texec::Request request{VecTokens(length), maxNewTokens};

        auto state = std::make_unique<texec::DataTransceiverState>();

        state->setCommState(texec::kv_cache::CommState{*mContextCommState});
        state->setCacheState(*mContextCacheState);
        auto stats = texec::ContextPhaseParams({}, mRequestId, state.release());
        request.setContextPhaseParams(std::move(stats));
        return std::make_unique<LlmRequest>(mRequestId++, std::move(request));
    }

    std::future<void> addRequestAndTransportCacheForContext(std::shared_ptr<LlmRequest> const& llmRequest)
    {
        auto constexpr beamIdx{0};
        auto constexpr beamWidth{1};
        mManager->addSequence(llmRequest->mRequestId, llmRequest->getNumTokens(beamIdx), beamWidth, llmRequest);
        auto blockEndIt = getBlockEndIt(*mManager, *llmRequest, beamIdx, 0);
        int blockIdx = 0;
        for (auto it = getBlockBeginIt(*mManager, *llmRequest, beamIdx, 0); it != blockEndIt; ++it)
        {

            fillBlockData(*it, blockIdx, llmRequest->mRequestId);
            blockIdx++;
        }
        mManager->getBlockManager().getBufferManager().getStream().synchronize();
        auto future = mResponder->respondAndSendAsync(*llmRequest);
        // future.get();
        return future;
    }

    void addRequestAndTransportCacheForGeneration(std::shared_ptr<LlmRequest> const& llmRequest)
    {
        auto constexpr beamIdx{0};
        auto constexpr beamWidth{1};
        mManager->addSequence(llmRequest->mRequestId, llmRequest->getNumTokens(beamIdx), beamWidth, llmRequest);

        auto future = mRequester->requestAndReceiveAsync(*llmRequest);
        future.get();
        int blockIdx = 0;

        auto blockEndIt = getBlockEndIt(*mManager, *llmRequest, beamIdx, 0);
        for (auto it = getBlockBeginIt(*mManager, *llmRequest, beamIdx, 0); it != blockEndIt; ++it)
        {

            verifyBlockData(*it, blockIdx, llmRequest->mRequestId);
            blockIdx++;
        }
    }

    void fillBlockData(tensorrt_llm::runtime::ITensor& blockData, int blockId, LlmRequest::RequestIdType requestId)
    {
        auto hostTensor
            = mManager->getBlockManager().getBufferManager().cpu(blockData.getShape(), blockData.getDataType());
        int layerSizePerRank = mCacheState->getModelConfig().mNbKvHeadsPerLayer.size() / mPpSize;
        int startLayerId = layerSizePerRank * mPpRank;
        int headSizePerRank = mCacheState->getModelConfig().mNbKvHeadsPerLayer.at(0);
        int startHeadId = headSizePerRank * mTpRank;
        int tokensPerBlock = mCacheState->getModelConfig().mTokensPerBlock;
        int startTokenId = blockId * tokensPerBlock;
        int sizePerHead = mCacheState->getModelConfig().mSizePerHead;
        auto dataTypeSize = tensorrt_llm::common::getDTypeSize(blockData.getDataType());

        for (int layerId = 0; layerId < layerSizePerRank; layerId++)
        {
            for (int headId = 0; headId < headSizePerRank; headId++)
            {
                for (int tokenId = 0; tokenId < tokensPerBlock; tokenId++)
                {
                    for (int hiddenId = 0; hiddenId < sizePerHead; hiddenId++)
                    {
                        size_t keyIndex = layerId * (2 * headSizePerRank * tokensPerBlock * sizePerHead)
                            + headId * (tokensPerBlock * sizePerHead) + tokenId * sizePerHead + hiddenId;
                        size_t valueIndex
                            = keyIndex + static_cast<size_t>(headSizePerRank * tokensPerBlock * sizePerHead);

                        std::visit(
                            [&](auto generateValue)
                            {
                                using ValueType = decltype(generateValue);
                                auto* dataPtr = static_cast<ValueType*>(hostTensor->data(keyIndex));
                                *dataPtr = generateValue;
                            },
                            generateExpectedValue(requestId, tokenId + startTokenId, layerId + startLayerId,
                                headId + startHeadId, hiddenId, true, blockData.getDataType()));

                        std::visit(
                            [&](auto generateValue)
                            {
                                using ValueType = decltype(generateValue);
                                auto* dataPtr = static_cast<ValueType*>(hostTensor->data(valueIndex));
                                *dataPtr = generateValue;
                            },
                            generateExpectedValue(requestId, tokenId + startTokenId, layerId + startLayerId,
                                headId + startHeadId, hiddenId, false, blockData.getDataType()));
                    }
                }
            }
        }
        mManager->getBlockManager().getBufferManager().copy(*hostTensor, blockData);
    }

    void verifyBlockData(tensorrt_llm::runtime::ITensor& blockData, int blockId, LlmRequest::RequestIdType requestId)
    {
        auto hostTensor
            = mManager->getBlockManager().getBufferManager().cpu(blockData.getShape(), blockData.getDataType());
        int layerSizePerRank = mCacheState->getModelConfig().mNbKvHeadsPerLayer.size() / mPpSize;
        int startLayerId = layerSizePerRank * mPpRank;
        int headSizePerRank = mCacheState->getModelConfig().mNbKvHeadsPerLayer.at(0);
        int startHeadId = headSizePerRank * mTpRank;
        int tokensPerBlock = mCacheState->getModelConfig().mTokensPerBlock;
        int startTokenId = blockId * tokensPerBlock;
        int sizePerHead = mCacheState->getModelConfig().mSizePerHead;

        mManager->getBlockManager().getBufferManager().copy(blockData, *hostTensor);
        mManager->getBlockManager().getBufferManager().getStream().synchronize();

        for (int layerId = 0; layerId < layerSizePerRank; layerId++)
        {
            for (int headId = 0; headId < headSizePerRank; headId++)
            {
                for (int tokenId = 0; tokenId < tokensPerBlock; tokenId++)
                {
                    for (int hiddenId = 0; hiddenId < sizePerHead; hiddenId++)
                    {
                        size_t keyIndex = layerId * (2 * headSizePerRank * tokensPerBlock * sizePerHead)
                            + headId * (tokensPerBlock * sizePerHead) + tokenId * sizePerHead + hiddenId;
                        size_t valueIndex
                            = keyIndex + static_cast<size_t>(headSizePerRank * tokensPerBlock * sizePerHead);

                        std::visit(
                            [&](auto generateValue)
                            {
                                using ValueType = decltype(generateValue);
                                auto* dataPtr = static_cast<ValueType*>(hostTensor->data(keyIndex));
                                EXPECT_EQ(*dataPtr, generateValue);
                            },
                            generateExpectedValue(requestId, tokenId + startTokenId, layerId + startLayerId,
                                headId + startHeadId, hiddenId, true, blockData.getDataType()));

                        std::visit(
                            [&](auto generateValue)
                            {
                                using ValueType = decltype(generateValue);
                                auto* dataPtr = static_cast<ValueType*>(hostTensor->data(valueIndex));
                                EXPECT_EQ(*dataPtr, generateValue);
                            },
                            generateExpectedValue(requestId, tokenId + startTokenId, layerId + startLayerId,
                                headId + startHeadId, hiddenId, false, blockData.getDataType()));
                    }
                }
            }
        }
    }

    std::variant<double, float, int16_t, int8_t> generateExpectedValue(LlmRequest::RequestIdType requestId, int tokenId,
        int layerId, int headId, int hiddenId, bool key, nvinfer1::DataType dataType)
    {

        size_t seed = 0;
        std::size_t hashValue = std::hash<LlmRequest::RequestIdType>{}(requestId);
        std::hash<int> hasher{};
        seed ^= hashValue + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= hasher(tokenId) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= hasher(layerId) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= hasher(headId) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= hasher(hiddenId) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed += key;
        generator.seed(seed);
        std::uniform_real_distribution<double> dis(-100.0f, 100.0f);
        double value = dis(generator);
        auto dataTypeSize = tensorrt_llm::common::getDTypeSize(dataType);
        switch (dataTypeSize)
        {
        case 8: return value; break;
        case 4: return static_cast<float>(value); break;
        case 2: return static_cast<int16_t>(value); break;
        case 1: return static_cast<int8_t>(value); break;
        default: TLLM_CHECK_WITH_INFO(false, "generateExpectedValue only support dataTypeSize in [8,4,2,1]"); break;
        };
        return 0.F;
    }

    bool mIsContext{false};
    bool mIsGeneration{false};
    tensorrt_llm::mpi::MpiComm const* mComm;
    tensorrt_llm::mpi::MpiComm mParticipatingComm{nullptr, false};
    SizeType32 mWorldSize{0}, mRank{0}, mRankInInstance{0};
    SizeType32 mSizeInInstance{0}, mTpRank{0}, mPpRank{0}, mTpSize{0}, mPpSize{0}, mContextRankSize{0}, mGenRankSize{0},
        mContextTpSize{0}, mContextPpSize{0};
    LlmRequest::RequestIdType mRequestId{0};
    SizeType32 mMaxNumSequences{};
    std::unique_ptr<KVCacheManager> mManager;
    std::unique_ptr<DataResponder> mResponder;
    std::unique_ptr<DataRequester> mRequester;
    std::unique_ptr<texec::kv_cache::CacheState> mCacheState;
    std::unique_ptr<texec::kv_cache::CacheState> mContextCacheState;
    std::unique_ptr<texec::kv_cache::CommState> mContextCommState;
    std::mt19937 generator;
};

TEST_P(MpiAsymmetricalCacheTest, TestCase)
{
    if (!(tensorrt_llm::common::getEnvUseUCXKvCache()))
    {
        setenv("UCX_TLS", "^cuda_ipc", 1); // disable cuda_ipc for testing for mpi
    }
    AsymmetricTestParam param = GetParam();
    int contextTp = std::get<0>(param);
    int contextPp = std::get<1>(param);
    int genTp = std::get<2>(param);
    int genPp = std::get<3>(param);
    int numLayers = std::get<4>(param);
    int numHeads = std::get<5>(param);
    int sizePerHead = std::get<6>(param);
    int tokensPerBlock = std::get<7>(param);
    nvinfer1::DataType dataType = std::get<8>(param);

    setUpCommunicator(contextTp, contextPp, genTp, genPp);

    if (mIsContext || mIsGeneration)
    {
        setUpCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, dataType);
        setUpCacheTransceiver();
        std::vector<std::shared_ptr<tensorrt_llm::batch_manager::LlmRequest>> requests;

        for (auto len : {10, 20, 30})
        {
            requests.emplace_back(makeLlmRequest(len));
        }
        std::vector<std::future<void>> contextFutures;

        if (mIsContext)
        {
            for (auto&& request : requests)
            {
                contextFutures.push_back(std::move(addRequestAndTransportCacheForContext(request)));
            }
            mComm->barrier();
        }
        else
        {
            mComm->barrier();
            for (auto&& request : requests)
            {
                addRequestAndTransportCacheForGeneration(request);
            }
        }
        if (mIsContext)
        {
            for (auto&& cfuture : contextFutures)
            {
                cfuture.get();
            }
        }
        mComm->barrier();
    }
    tensorrt_llm::mpi::MpiComm::world().barrier();
}

INSTANTIATE_TEST_CASE_P(MpiAsymmetricCaseTest0, MpiAsymmetricalCacheTest,
    testing::Combine(testing::Values(1, 2), testing::Values(1, 2), testing::Values(1, 2), testing::Values(1, 2),
        testing::Values(4), testing::Values(4), testing::Values(4), testing::Values(16),
        testing::Values(nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kINT8)));

INSTANTIATE_TEST_CASE_P(MpiAsymmetricCaseTest1, MpiAsymmetricalCacheTest,
    testing::Combine(testing::Values(4), testing::Values(1), testing::Values(1), testing::Values(4), testing::Values(4),
        testing::Values(4), testing::Values(4), testing::Values(8),
        testing::Values(nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kINT8)));

INSTANTIATE_TEST_CASE_P(MpiAsymmetricCaseTest2, MpiAsymmetricalCacheTest,
    testing::Combine(testing::Values(1), testing::Values(2), testing::Values(1), testing::Values(1, 4),
        testing::Values(16), testing::Values(16), testing::Values(4), testing::Values(8),
        testing::Values(nvinfer1::DataType::kFLOAT)));

#endif
