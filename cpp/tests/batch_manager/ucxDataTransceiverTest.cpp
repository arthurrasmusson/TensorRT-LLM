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

#include "tensorrt_llm/batch_manager/ucxDataTransceiver.h"
#include "tensorrt_llm/batch_manager/cacheFormatter.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/common/mpiUtils.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/tllmBuffers.h"
#include "tensorrt_llm/runtime/worldConfig.h"
#include <NvInferRuntimeBase.h>
#include <chrono>
#include <cuda.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <thread>

namespace tr = tensorrt_llm::runtime;
namespace tle = tensorrt_llm::executor;
using SizeType32 = tensorrt_llm::runtime::SizeType32;
using LlmRequest = tensorrt_llm::batch_manager::LlmRequest;
using namespace tensorrt_llm::batch_manager::kv_cache_manager;
using namespace tensorrt_llm::batch_manager;
namespace texec = tensorrt_llm::executor;

using testing::Return;
using testing::ReturnRef;

// ---------------------------------------
//          UcxCommTest
// ---------------------------------------

class UcxCommTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
public:
    void SetUp() override
    {
        TLLM_CUDA_CHECK(cudaSetDevice(0));
        mSelfContext = ucxx::createContext({}, ucxx::Context::defaultFeatureFlags);
        mSelfWorker = mSelfContext->createWorker();
        mSelfWorker->setProgressThreadStartCallback(cudaFree, nullptr);
        mSelfWorker->startProgressThread();

        mPeerContext = ucxx::createContext({}, ucxx::Context::defaultFeatureFlags);
        mPeerWorker = mPeerContext->createWorker();
        mPeerWorker->setProgressThreadStartCallback(cudaFree, nullptr);
        mPeerWorker->startProgressThread();
    }

    void TearDown() override {}

protected:
    // Two sets of ucxx objects to simulate two nodes
    std::shared_ptr<ucxx::Context> mSelfContext{nullptr};
    std::shared_ptr<ucxx::Worker> mSelfWorker{nullptr};

    std::shared_ptr<ucxx::Context> mPeerContext{nullptr};
    std::shared_ptr<ucxx::Worker> mPeerWorker{nullptr};
    int mDeviceId{0};
};

TEST_F(UcxCommTest, Basic)
{
    auto selfEndpoint = mSelfWorker->createEndpointFromWorkerAddress(mPeerWorker->getAddress());
    auto peerEndpoint = mPeerWorker->createEndpointFromWorkerAddress(mSelfWorker->getAddress());

    UcxComm selfComm(selfEndpoint), peerComm(peerEndpoint);

    auto state = std::make_unique<tle::DataTransceiverState>();
    state->setCommState(tle::kv_cache::CommState{12, "127.0.0.1"});
    state->setCacheState(tle::kv_cache::CacheState{10, 12, 128, 128, 8, 8, nvinfer1::DataType::kFLOAT});
    RequestInfo info{1, *state};
    auto sendFuture = std::async(std::launch::async, [&]() { selfComm.sendRequestInfo(info); });
    auto recvFuture = std::async(std::launch::async, [&]() { return peerComm.recvRequestInfo(); });

    ASSERT_EQ(sendFuture.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    ASSERT_EQ(recvFuture.wait_for(std::chrono::seconds(5)), std::future_status::ready);

    sendFuture.get();
    EXPECT_EQ(info, recvFuture.get());
}

TEST_F(UcxCommTest, ListenerConnection)
{
    using ContextPair = std::pair<std::shared_ptr<ucxx::Listener>, std::promise<std::shared_ptr<ucxx::Endpoint>>>;
    ContextPair context;
    auto& listener = context.first;
    auto& endpointPromise = context.second;

    listener = mSelfWorker->createListener(
        0,
        [](ucp_conn_request_h conn_request, void* data)
        {
            auto context = reinterpret_cast<ContextPair*>(data);
            context->second.set_value(context->first->createEndpointFromConnRequest(conn_request));
        },
        &context);

    auto peerEndpoint = mPeerWorker->createEndpointFromHostname(listener->getIp(), listener->getPort());

    auto endpointFuture = endpointPromise.get_future();
    ASSERT_EQ(endpointFuture.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    auto selfEndpoint = endpointFuture.get();
    ASSERT_TRUE(selfEndpoint);

    UcxComm selfComm(selfEndpoint), peerComm(peerEndpoint);

    LlmRequest::RequestIdType sendId{123};
    auto state = std::make_unique<tle::DataTransceiverState>();
    state->setCommState(tle::kv_cache::CommState{12, "127.0.0.1"});
    state->setCacheState(tle::kv_cache::CacheState{10, 12, 128, 128, 8, 8, nvinfer1::DataType::kFLOAT});
    RequestInfo info{sendId, *state};
    auto sendFuture = std::async(std::launch::async, [&]() { selfComm.sendRequestInfo(info); });
    auto recvFuture = std::async(std::launch::async, [&]() { return peerComm.recvRequestInfo(); });

    ASSERT_EQ(sendFuture.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    ASSERT_EQ(recvFuture.wait_for(std::chrono::seconds(5)), std::future_status::ready);

    sendFuture.get();
    EXPECT_EQ(info, recvFuture.get());
}

TEST_F(UcxCommTest, HostBufferSync)
{
    auto selfEndpoint = mSelfWorker->createEndpointFromWorkerAddress(mPeerWorker->getAddress());
    auto peerEndpoint = mPeerWorker->createEndpointFromWorkerAddress(mSelfWorker->getAddress());

    UcxComm selfComm(selfEndpoint), peerComm(peerEndpoint);

    int32_t expectedValue = 234;

    tr::HostBuffer sendBuffer{sizeof(int32_t), nvinfer1::DataType::kINT32},
        recvBuffer{sizeof(int32_t), nvinfer1::DataType::kINT32};
    *reinterpret_cast<int32_t*>(sendBuffer.data()) = expectedValue;
    auto sendFuture = std::async(std::launch::async, [&selfComm, &sendBuffer]() { selfComm.sendBuffer(sendBuffer); });
    auto recvFuture = std::async(std::launch::async, [&peerComm, &recvBuffer]() { peerComm.recvBuffer(recvBuffer); });
    ASSERT_EQ(sendFuture.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    ASSERT_EQ(recvFuture.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    sendFuture.get();
    recvFuture.get();

    EXPECT_EQ(*reinterpret_cast<int32_t*>(recvBuffer.data()), expectedValue);
}

TEST_F(UcxCommTest, DeviceBufferSync)
{
    auto selfEndpoint = mSelfWorker->createEndpointFromWorkerAddress(mPeerWorker->getAddress());
    auto peerEndpoint = mPeerWorker->createEndpointFromWorkerAddress(mSelfWorker->getAddress());

    UcxComm selfComm(selfEndpoint), peerComm(peerEndpoint);

    int32_t expectedValue = 234;
    tr::StaticDeviceBuffer sendBuffer{sizeof(int32_t), nvinfer1::DataType::kINT32},
        recvBuffer{sizeof(int32_t), nvinfer1::DataType::kINT32};

    TLLM_CUDA_CHECK(
        cudaMemcpy(sendBuffer.data(), &expectedValue, sizeof(decltype(expectedValue)), cudaMemcpyHostToDevice));
    auto sendFuture = std::async(std::launch::async, [&selfComm, &sendBuffer]() { selfComm.sendBuffer(sendBuffer); });
    auto recvFuture = std::async(std::launch::async, [&peerComm, &recvBuffer]() { peerComm.recvBuffer(recvBuffer); });
    ASSERT_EQ(sendFuture.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    ASSERT_EQ(recvFuture.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    sendFuture.get();
    recvFuture.get();

    int32_t gotValue = 0;
    TLLM_CUDA_CHECK(cudaMemcpy(&gotValue, recvBuffer.data(), sizeof(decltype(expectedValue)), cudaMemcpyDeviceToHost));
    EXPECT_EQ(gotValue, expectedValue);
}

// ---------------------------------------
//          MockTransceiverTest
// ---------------------------------------
class MockUcxComm : public UcxComm
{
public:
    MockUcxComm()
        : UcxComm{nullptr}
    {
        ON_CALL(*this, sendRequestInfo).WillByDefault(Return());
        ON_CALL(*this, recvRequestInfo)
            .WillByDefault(Return(RequestInfo{123,
                tle::DataTransceiverState{tle::kv_cache::CacheState{10, 12, 128, 128, 8, 8, nvinfer1::DataType::kFLOAT},
                    tle::kv_cache::CommState{12, "127.0.0.1"}}}));
    }

    MOCK_METHOD(void, sendRequestInfo, (RequestInfo const&), (const, override));
    MOCK_METHOD(RequestInfo, recvRequestInfo, (), (const, override));
    MOCK_METHOD(void, sendBuffer, (tr::IBuffer const&), (const, override));
    MOCK_METHOD(void, recvBuffer, (tr::IBuffer&), (const, override));
};

class MockUcxCommFactory : public UcxCommFactory
{
public:
    MockUcxCommFactory()
        : UcxCommFactory{}
    {
        ON_CALL(*this, create).WillByDefault(testing::InvokeWithoutArgs(this, &MockUcxCommFactory::popMockComm));
    }

    void addMockComm(std::unique_ptr<MockUcxComm>&& comm)
    {
        mComms.emplace_back(std::move(comm));
    }

    MOCK_METHOD(std::unique_ptr<UcxComm>, create, (std::shared_ptr<ucxx::Endpoint> const& endpoint), (override));

private:
    std::unique_ptr<UcxComm> popMockComm()
    {
        auto comm = std::move(mComms.front());
        mComms.pop_front();
        return comm;
    }

    std::deque<std::unique_ptr<MockUcxComm>> mComms;
};

template <typename TComm>
class MockIOFormatter final : public IOFormatter<TComm, tle::kv_cache::CacheState>
{
public:
    MockIOFormatter() = default;

    void operator()(LlmRequest const& llmRequest, typename TComm::TPtrContainer const& comm,
        tensorrt_llm::executor::kv_cache::CacheState const& selfconfig, SizeType32 selfIdx,
        tensorrt_llm::executor::kv_cache::CacheState const& destConfig) override
    {
        mockTransfer(llmRequest, comm);
    }

    MOCK_METHOD(void, mockTransfer, (LlmRequest const&, std::vector<TComm const*> const&) );

    MOCK_METHOD(
        bool, inquireSupport, (tle::kv_cache::CacheState const&, tle::kv_cache::CacheState const&), (const, override));

    MOCK_METHOD(std::vector<SizeType32>, getCounterparts,
        (tle::kv_cache::CacheState const& selfconfig, SizeType32 selfIdx, tle::kv_cache::CacheState const& destConfig),
        (const, override));
};

class MockTransceiverTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
public:
    void SetUp() override
    {
        TLLM_CUDA_CHECK(cudaSetDevice(0));
        mPeerContext = ucxx::createContext({}, ucxx::Context::defaultFeatureFlags);
        mPeerWorker = mPeerContext->createWorker();
        mPeerWorker->setProgressThreadStartCallback(cudaFree, nullptr);
        mPeerWorker->startProgressThread();
    }

    void TearDown() override {}

    static auto makeLlmRequest(LlmRequest::RequestIdType requestId)
    {
        return LlmRequest{requestId, tle::Request({10}, 1)};
    }

    static auto makeLlmRequest(LlmRequest::RequestIdType requestId, LlmRequest::RequestIdType contextRequestId,
        std::string const& contextIp, std::uint16_t contextPort)
    {
        auto request = tle::Request({10}, 1);
        auto dataTransceiverState = std::make_unique<tle::DataTransceiverState>();
        dataTransceiverState->setCommState(tle::kv_cache::CommState{contextPort, contextIp});
        tensorrt_llm::executor::kv_cache::CacheState fakeCacheState{10, 12, 128, 128, 8, 8, nvinfer1::DataType::kFLOAT};

        dataTransceiverState->setCacheState(fakeCacheState);
        request.setContextPhaseParams(tle::ContextPhaseParams({}, contextRequestId, dataTransceiverState.release()));
        return LlmRequest{requestId, request};
    }

    std::shared_ptr<ucxx::Context> mPeerContext{nullptr};
    std::shared_ptr<ucxx::Worker> mPeerWorker{nullptr};
};

TEST_F(MockTransceiverTest, UcxSenderBasic)
{
    auto formatter = std::make_unique<MockIOFormatter<UcxComm>>();

    // setup mock formatter
    // [NOTE] Test expects dynamic support inquire is not implemented
    EXPECT_CALL(*formatter, inquireSupport).WillOnce(Return(true));
    // Make sure correct request are selected and send
    EXPECT_CALL(*formatter, mockTransfer)
        .WillOnce(
            testing::DoAll(testing::WithArgs<1>(
                               [](auto comm) {
                                   comm[0]->sendBuffer(tr::HostBuffer{sizeof(int32_t), nvinfer1::DataType::kINT32});
                               }),
                Return()));

    // setup mock data communicator
    auto factory = std::make_unique<MockUcxCommFactory>();
    {
        auto mockComm = std::make_unique<MockUcxComm>();
        EXPECT_CALL(*mockComm, recvRequestInfo)
            .WillOnce(Return(RequestInfo{123,
                tle::DataTransceiverState{tle::kv_cache::CacheState{10, 12, 128, 128, 8, 8, nvinfer1::DataType::kFLOAT},
                    tle::kv_cache::CommState{12, "127.0.0.1"}}}));
        // Ensure correct comm is passed for all operations of the same
        // request.
        EXPECT_CALL(*mockComm, sendBuffer).Times(1);
        factory->addMockComm(std::move(mockComm));
    }
    EXPECT_CALL(*factory, create).Times(1);
    tensorrt_llm::executor::kv_cache::CacheState fakeCacheState{0, 0, 0, 0, 0, 0, nvinfer1::DataType::kFLOAT};
    auto sender = UcxDataSender<tle::kv_cache::CacheState>(std::move(factory), fakeCacheState, 0, std::move(formatter));

    // Expect recvRequestInfo will be blocking until a connection has been initialized
    auto requestId = std::async(std::launch::async, [&sender]() { return sender.recvRequestInfo(); });
    EXPECT_EQ(requestId.wait_for(std::chrono::seconds(3)), std::future_status::timeout);

    // Start connection to respond and send
    auto const& sendComm = sender.getCommState().getSocketState().at(0);
    auto peerEndpoint = mPeerWorker->createEndpointFromHostname(sendComm.mIp, sendComm.mPort);
    ASSERT_EQ(requestId.wait_for(std::chrono::seconds(5)), std::future_status::ready);

    EXPECT_EQ(requestId.get().getRequestId(), 123);
    sender.sendSync(makeLlmRequest(123));
}

TEST_F(MockTransceiverTest, UcxReceiverBasic)
{
    auto formatter = std::make_unique<MockIOFormatter<UcxComm>>();

    EXPECT_CALL(*formatter, inquireSupport).WillOnce(Return(true));
    EXPECT_CALL(*formatter, mockTransfer)
        .WillOnce(testing::DoAll(testing::WithArgs<1>(
                                     [](auto comm)
                                     {
                                         auto buffer = tr::HostBuffer{sizeof(int32_t), nvinfer1::DataType::kINT32};
                                         comm[0]->recvBuffer(buffer);
                                     }),
            Return()));
    EXPECT_CALL(*formatter, getCounterparts).Times(1).WillOnce(Return(std::vector<SizeType32>{0}));
    // setup mock data communicator
    auto factory = std::make_unique<MockUcxCommFactory>();
    {
        auto mockComm = std::make_unique<MockUcxComm>();
        EXPECT_CALL(*mockComm, sendRequestInfo).Times(1);
        // Ensure correct comm is passed for all operations of the same
        // request.
        EXPECT_CALL(*mockComm, recvBuffer).Times(1);
        factory->addMockComm(std::move(mockComm));
    }
    EXPECT_CALL(*factory, create).Times(1);

    // Only need listener so requester can establish an endpoint
    auto listener = mPeerWorker->createListener(
        0, [](ucp_conn_request_h conn_request, void* data) {}, nullptr);
    tensorrt_llm::executor::kv_cache::CacheState fakeCacheState{10, 12, 128, 128, 8, 8, nvinfer1::DataType::kFLOAT};

    auto receiver
        = UcxDataReceiver<tle::kv_cache::CacheState>(std::move(factory), fakeCacheState, 0, std::move(formatter));

    // Request containing context phase info, i.e. 123 is the request id in context executor
    auto llmRequest = makeLlmRequest(0, 123, listener->getIp(), listener->getPort());
    receiver.sendRequestInfo(llmRequest);
    receiver.receiveSync(llmRequest);
}

// ---------------------------------------
//          RealTransceiverTest
// ---------------------------------------

// [FIXME] add multi processes test

class UcxSymmetricalCacheTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
protected:
    void SetUp() override {}

    void TearDown() override {}

    SizeType32 setUpController()
    {
        // MpiComm is used as controller to pass UCX communicator info between
        // processes
        tensorrt_llm::mpi::initialize(tensorrt_llm::mpi::MpiThreadSupport::THREAD_MULTIPLE);
        mController = &tensorrt_llm::mpi::MpiComm::session();
        mWorldSize = mController->getSize();
        mlocalRank = mController->getRank() / 2;

        isSender = mController->getRank() % 2 == 0;
        return mWorldSize;
    }

    class Node
    {
    public:
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
            mCacheState = std::make_unique<tle::kv_cache::CacheState>(
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
            auto constexpr dataType = nvinfer1::DataType::kFLOAT;

            mManager = std::make_unique<KVCacheManager>(numLayers, numHeads, sizePerHead, tokensPerBlock,
                totalNumBlocks, blocksInSecondaryPool, mMaxNumSequences, maxBeamWidth, maxAttentionWindow,
                sinkTokenLength, useOneMoreBlock, stream, enableBlockReuse, onboardBlocks);

            // UVM seems to be incompatible with MPI, and it is continuing to investigate.
            bool constexpr useUvm = false;
            mManager->allocatePools(nvinfer1::DataType::kFLOAT, useUvm);
        }

        std::unique_ptr<tle::DataTransceiverState> makeDataTransceiverState()
        {
            auto state = std::make_unique<tle::DataTransceiverState>();
            state->setCommState(tle::kv_cache::CommState{mSocketComm.mPort, mSocketComm.mIp});
            state->setCacheState(*mCacheState);
            return state;
        }

        std::shared_ptr<LlmRequest> makeLlmRequest(
            SizeType32 length, std::unique_ptr<tle::DataTransceiverState> dataTransceiverState = nullptr)
        {
            SizeType32 constexpr maxNewTokens{1};
            auto request = tle::Request(VecTokens(length), 1);
            if (dataTransceiverState)
            {
                request.setContextPhaseParams(tle::ContextPhaseParams({}, mRequestId, dataTransceiverState.release()));
            }
            return std::make_shared<LlmRequest>(mRequestId++, request);
        }

        void setUpCacheTransceiver()
        {
            mResponder = makeUcxCacheResponder(*mCacheState, mRank, mManager.get());
            mSocketComm = mResponder->getCommState().getSocketState().at(0);
            mRequester = makeUcxCacheRequester(*mCacheState, mRank, mManager.get());
        }

        void setUpRank(int rank)
        {
            mRank = rank;
        }

        [[nodiscard]] std::future<void> addRequestAndSendCache(std::shared_ptr<LlmRequest> const& llmRequest) const
        {
            auto constexpr beamIdx{0};
            auto constexpr beamWidth{1};
            mManager->addSequence(llmRequest->mRequestId, llmRequest->getNumTokens(beamIdx), beamWidth, llmRequest);

            // send
            auto blockEndIt = getBlockEndIt(*mManager, *llmRequest, beamIdx, 0);
            for (auto it = getBlockBeginIt(*mManager, *llmRequest, beamIdx, 0); it != blockEndIt; ++it)
            {
                TLLM_CUDA_CHECK(cudaMemsetAsync(it->data(), llmRequest->mRequestId, it->getSizeInBytes()));
            }
            return mResponder->respondAndSendAsync(*llmRequest);
        }

        void addRequestAndReceiveCache(std::shared_ptr<LlmRequest> const& llmRequest) const
        {
            auto constexpr beamIdx{0};
            auto constexpr beamWidth{1};
            mManager->addSequence(llmRequest->mRequestId, llmRequest->getNumTokens(beamIdx), beamWidth, llmRequest);

            // receive
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

        tle::kv_cache::SocketState mSocketComm;
        SizeType32 mMaxNumSequences{};
        std::unique_ptr<KVCacheManager> mManager;
        std::unique_ptr<DataResponder> mResponder;
        std::unique_ptr<DataRequester> mRequester;
        std::unique_ptr<tle::kv_cache::CacheState> mCacheState;
        SizeType32 mRank{0};
        LlmRequest::RequestIdType mRequestId{0};
    };

    bool isSender{false};
    tensorrt_llm::mpi::MpiComm const* mController;
    SizeType32 mWorldSize{0}, mlocalRank{0};
};

TEST_F(UcxSymmetricalCacheTest, SameProcessTest)
{

    Node source, destination;
    source.setUpCacheManager();
    source.setUpCacheTransceiver();
    destination.setUpCacheManager();
    destination.setUpCacheTransceiver();

    // Process request one by one.. Can't batch send/recv in case
    // the requester tries to ask for a request that has not been added
    // to responder.
    for (auto len : {10, 20, 30})
    {
        auto sourceRequest = source.makeLlmRequest(len, source.makeDataTransceiverState());
        auto sourceFuture = source.addRequestAndSendCache(sourceRequest);

        // pass source data context to destination to start requesting
        auto sourceContext = source.makeDataTransceiverState();
        auto destinationRequest = destination.makeLlmRequest(len, std::move(sourceContext));
        destination.addRequestAndReceiveCache(destinationRequest);

        sourceFuture.get();
    }
}

TEST_F(UcxSymmetricalCacheTest, MultiProcessTest)
{
    auto worldSize = setUpController();
    if (worldSize % 2)
    {
        GTEST_SKIP() << "An even number of processes is required to run this test.";
    }

    Node self;
    self.setUpCacheManager();
    self.setUpCacheTransceiver();
    self.setUpRank(mlocalRank);
    // Send sender context to receiver
    if (isSender)
    {
        // Test setup, send consistent context phase state member first
        auto sourceContext = self.makeDataTransceiverState();
        auto const& socketState = sourceContext->getCommState().value().getSocketState().at(0);
        auto& address = socketState.mIp;
        auto& port = socketState.mPort;
        uint64_t addressLength = address.length();
        mController->send(std::addressof(addressLength), 1, tensorrt_llm::mpi::MpiType::kUINT64, 1, 0);
        mController->send(address.data(), addressLength, tensorrt_llm::mpi::MpiType::kBYTE, 1, 0);
        mController->send(std::addressof(port), 1, tensorrt_llm::mpi::MpiType::kBF16, 1, 0);

        for (auto len : {10, 20, 30})
        {
            auto dataTransceiverState = self.makeDataTransceiverState();
            auto request = self.makeLlmRequest(len, std::move(dataTransceiverState));
            auto future = self.addRequestAndSendCache(request);
            // Simulate workflow of controller sending context phase param
            // to trigger data transfer.
            mController->send(std::addressof(request->mRequestId), 1, tensorrt_llm::mpi::MpiType::kUINT64, 1, 0);
            future.get();
        }
    }
    else
    {
        tle::kv_cache::SocketState sourceSocket;
        // Test setup, receive consistent context phase state member first
        uint64_t addressLength{0};
        mController->recv(std::addressof(addressLength), 1, tensorrt_llm::mpi::MpiType::kUINT64, 0, 0);
        sourceSocket.mIp = std::string(addressLength, '0');
        mController->recv(std::addressof(sourceSocket.mIp[0]), addressLength, tensorrt_llm::mpi::MpiType::kBYTE, 0, 0);
        mController->recv(std::addressof(sourceSocket.mPort), 1, tensorrt_llm::mpi::MpiType::kBF16, 0, 0);

        for (auto len : {10, 20, 30})
        {
            // Simulate workflow of controller receiving context phase param.
            LlmRequest::RequestIdType contextRequestId;
            mController->recv(std::addressof(contextRequestId), 1, tensorrt_llm::mpi::MpiType::kUINT64, 0, 0);

            auto dataTransceiverState = self.makeDataTransceiverState();
            dataTransceiverState->setCommState(tle::kv_cache::CommState{sourceSocket.mPort, sourceSocket.mIp});
            auto request = self.makeLlmRequest(len, std::move(dataTransceiverState));
            self.addRequestAndReceiveCache(request);
        }
    }
}
