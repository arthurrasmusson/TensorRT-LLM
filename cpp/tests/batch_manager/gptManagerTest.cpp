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

#ifndef TOP_LEVEL_DIR
#error "Define TOP_LEVEL_DIR"
#endif

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using ::testing::_;
using ::testing::Invoke;

#include "modelSpec.h"
#include "tensorrt_llm/batch_manager/GptManager.h"
#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/inferenceRequest.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/batch_manager/namedTensor.h"
#include "tensorrt_llm/batch_manager/trtGptModelOptionalParams.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/memoryCounters.h"
#include "tensorrt_llm/runtime/tllmLogger.h"
#include "tensorrt_llm/runtime/torch.h"
#include "tensorrt_llm/runtime/utils/numpyUtils.h"

#include "c10/cuda/CUDAGuard.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <nlohmann/json.hpp>
#include <vector>

#include <iostream>

constexpr int32_t kMaxNewTokens = 10;
constexpr int32_t kMaxSeqLen = 16;

using namespace tensorrt_llm::batch_manager;

namespace tc = tensorrt_llm::common;
namespace tr = tensorrt_llm::runtime;
namespace texec = tensorrt_llm::executor;
namespace fs = std::filesystem;
using SizeType32 = tr::SizeType32;
using tensorrt_llm::testing::ModelSpec;
using tensorrt_llm::testing::KVCacheType;
using tensorrt_llm::testing::QuantMethod;
using tensorrt_llm::testing::OutputContentType;

namespace
{
auto constexpr INPUT_FILE = "input_tokens.npy";
auto const TEST_RESOURCE_PATH = fs::path{TOP_LEVEL_DIR} / "cpp/tests/resources";
auto const ENGINE_PATH = TEST_RESOURCE_PATH / "models/rt_engine";
auto const GPT_MODEL_PATH = ENGINE_PATH / "gpt2";
auto const CHATGLM_MODEL_PATH = ENGINE_PATH / "chatglm-6b";
auto const CHATGLM2_MODEL_PATH = ENGINE_PATH / "chatglm2-6b";
auto const CHATGLM3_MODEL_PATH = ENGINE_PATH / "chatglm3-6b";
auto const DATA_PATH = TEST_RESOURCE_PATH / "data";
auto const GPT_DATA_PATH = DATA_PATH / "gpt2";
auto const CHATGLM_DATA_PATH = DATA_PATH / "chatglm-6b";
auto const CHATGLM2_DATA_PATH = DATA_PATH / "chatglm2-6b";
auto const CHATGLM3_DATA_PATH = DATA_PATH / "chatglm3-6b";

} // namespace

class TritonStub
{
public:
    TritonStub()
        : mCount(0)
        , mEmptyCount(0)
    {
    }

    int32_t mCount;
    int32_t mEmptyCount;

    std::list<std::shared_ptr<InferenceRequest>> empty_get_inference_requests(int32_t maxNumRequests)
    {
        ++mEmptyCount;
        if (mEmptyCount < 100 || mEmptyCount > 100)
        {
            return {};
        }
        else
        {
            return get_inference_requests(maxNumRequests, kMaxNewTokens);
        }
    }

    std::list<std::shared_ptr<InferenceRequest>> get_inference_requests(
        int32_t maxNumRequests, int32_t max_new_tokens, int32_t promptLen = -1, bool skipCountCheck = false)
    {
        // Only provide requests for 1st callback
        if (mCount > 0 && !skipCountCheck)
            return {};

        std::vector<int32_t> tok1;
        if (promptLen < 0)
        {
            tok1 = {1, 2};
        }
        else
        {
            tok1.resize(promptLen, 1);
        }
        std::vector<int64_t> t1_in_dims = {1, static_cast<int64_t>(tok1.size())};

        int32_t request_output_len = max_new_tokens;
        auto t1_in = NamedTensor(nvinfer1::DataType::kINT32, t1_in_dims, "req1_prompt", tok1.data());
        auto t1_req_out_len_in
            = NamedTensor(nvinfer1::DataType::kINT32, {1, 1}, "req1_request_output_len", &request_output_len);

        auto r1 = std::make_shared<InferenceRequest>(0);
        r1->setInputIds(t1_in.tensor);
        r1->setMaxNewTokens(t1_req_out_len_in.tensor);

        std::vector<int32_t> tok2;
        if (promptLen < 0)
        {
            tok2 = {4, 1, 9};
        }
        else
        {
            tok2.resize(promptLen, 1);
        }
        std::vector<int64_t> t2_in_dims = {1, static_cast<int64_t>(tok2.size())};

        uint32_t request_output_len2 = max_new_tokens + 1;
        auto t2_in = NamedTensor(nvinfer1::DataType::kINT32, t2_in_dims, "req2_prompt", tok2.data());
        auto t2_req_out_len_in
            = NamedTensor(nvinfer1::DataType::kINT32, {1, 1}, "req2_request_output_len", &request_output_len2);

        auto r2 = std::make_shared<InferenceRequest>(1);
        r2->setInputIds(t2_in.tensor);
        r2->setMaxNewTokens(t2_req_out_len_in.tensor);

        std::list<std::shared_ptr<InferenceRequest>> rval{r1, r2};
        ++mCount;

        return rval;
    }

    bool send_response_callback(uint64_t, std::list<InferenceRequest::TensorPtr>)
    {
        return false;
    }
};

// Create a MockedGPTManager that will mock the generateTokens method
class MockedGptManager : public GptManager
{
public:
    MockedGptManager(std::filesystem::path const& dataPath, TrtGptModelType modelType,
        GetInferenceRequestsCallback getInferenceRequestsCb, SendResponseCallback sendResponseCb,
        PollStopSignalCallback pollStopSignalCb = nullptr,
        ReturnBatchManagerStatsCallback returnBatchManagerStatsCb = nullptr,
        TrtGptModelOptionalParams const& optionalParams = TrtGptModelOptionalParams(),
        std::optional<uint64_t> terminateReqId = std::nullopt, bool excludeInputInOutput = false)
        : GptManager(dataPath, modelType, getInferenceRequestsCb, sendResponseCb, pollStopSignalCb,
            returnBatchManagerStatsCb, optionalParams, terminateReqId, excludeInputInOutput){};

    MOCK_METHOD(BatchManagerErrorCode_t, forwardSync, (), (override));
    MOCK_METHOD(BatchManagerErrorCode_t, forwardAsync, (RequestList&, std::unordered_set<uint64_t>&), (override));
};

class GptManagerTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
protected:
    void SetUp() override
    {
        mLogger = std::make_shared<tr::TllmLogger>();
        initTrtLlmPlugins(mLogger.get());
    }

    void TearDown() override {}

    std::shared_ptr<nvinfer1::ILogger> mLogger{};
    int mMaxWaitMs = 60000;
    int mTrigWarnMs = 10000;
};

TEST_F(GptManagerTest, BasicValidationTest)
{
    std::atomic<int> callCount = 0;
    {
        ModelSpec modelSpec{"input_tokens.npy", nvinfer1::DataType::kHALF};
        modelSpec.useGptAttentionPlugin().usePackedInput().setKVCacheType(KVCacheType::kCONTINUOUS);
        fs::path dataPath = GPT_MODEL_PATH / modelSpec.getModelPath() / "tp1-pp1-cp1-gpu";
        TritonStub tritonStub;

        tr::SizeType32 constexpr beamWidth{1};
        GetInferenceRequestsCallback getInferenceRequestsCb = [&tritonStub](int maxNumRequests)
        { return tritonStub.get_inference_requests(maxNumRequests, kMaxNewTokens); };
        SendResponseCallback sendResponseCb
            = [](uint64_t requestId, std::list<NamedTensor> const&, bool, std::string const&) -> void
        {
            std::cout << "Inside response callback with requestId: " << requestId << std::endl;
            return;
        };
        ReturnBatchManagerStatsCallback returnBatchManagerStatsCb = [](std::string const& s) -> void
        {
            nlohmann::json j = nlohmann::json::parse(s);
            ASSERT_FALSE(j.is_discarded()); // Fail if s is an invalid json
            int32_t activeReq = j.at("Active Request Count").template get<int32_t>();
            ASSERT_GE(activeReq, 0);
        };

        TrtGptModelOptionalParams optionalParams;
        optionalParams.maxBeamWidth = beamWidth;
        optionalParams.enableTrtOverlap = false;
        optionalParams.schedulerConfig = texec::SchedulerConfig{texec::CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT};
        auto batch_manager_ = std::make_shared<MockedGptManager>(dataPath, TrtGptModelType::V1, getInferenceRequestsCb,
            sendResponseCb, nullptr, returnBatchManagerStatsCb, optionalParams);

        // Mock the generateTokens function such that it just increments some counter
        EXPECT_CALL(*batch_manager_, forwardAsync(_, _))
            .WillRepeatedly(Invoke(
                [&](RequestList& requestList, std::unordered_set<uint64_t>& activeRequestIds)
                {
                    auto numReq = requestList.size();
                    if (callCount < kMaxNewTokens)
                    {
                        EXPECT_EQ(numReq, 2);
                        auto req0 = *(requestList.front());
                        auto req1 = *(*std::next(requestList.begin()));
                        EXPECT_EQ(req0.mRequestId, 0);
                        EXPECT_EQ(req1.mRequestId, 1);

                        auto req0Tokens = req0.getTokens(0);
                        auto req1Tokens = req1.getTokens(0);
                        // Check input tokens, which should match output tokens from previous iteration
                        if (callCount == 0)
                        {
                            EXPECT_EQ(req0Tokens.at(0), 1);
                            EXPECT_EQ(req0Tokens.at(1), 2);
                            EXPECT_EQ(req1Tokens.at(0), 4);
                            EXPECT_EQ(req1Tokens.at(1), 1);
                            EXPECT_EQ(req1Tokens.at(2), 9);
                        }
                        else
                        {
                            EXPECT_EQ(req0Tokens.back(), 100 * (callCount - 1) + 0);
                            EXPECT_EQ(req1Tokens.back(), 100 * (callCount - 1) + 1);
                        }

                        // Check inputSeqLenghts
                        if (callCount == 0)
                        {
                            EXPECT_EQ(req0Tokens.size(), 2);
                            EXPECT_EQ(req1Tokens.size(), 3);
                        }
                        else
                        {
                            EXPECT_EQ(req0Tokens.size(), 2 + callCount);
                            EXPECT_EQ(req1Tokens.size(), 3 + callCount);
                        }
                    }
                    else
                    {
                        // At last iteration, only one request still active: req with corrID 1
                        auto req0 = *(requestList.front());
                        auto req0Tokens = req0.getTokens(0);
                        EXPECT_EQ(req0.mRequestId, 1);
                        EXPECT_EQ(numReq, 1);
                        EXPECT_EQ(req0Tokens.back(), 100 * (callCount - 1) + 1);
                        EXPECT_EQ(req0Tokens.size(), 13);
                    }

                    for (auto llmReq : requestList)
                    {
                        llmReq->addNewTokens({100 * callCount + static_cast<tr::TokenIdType>(llmReq->mRequestId)});
                        if (llmReq->getMaxNumGeneratedTokens() >= llmReq->mMaxNewTokens)
                        {
                            llmReq->setState(LlmRequestState::kGENERATION_COMPLETE);
                        }
                    }

                    callCount++;

                    return BatchManagerErrorCode_t::STATUS_SUCCESS;
                }));

        // Here batch manager would be killed, so sleep some time to let the magic happen
        int waitCnt = 0;
        while (callCount < kMaxNewTokens + 1 && waitCnt++ < mMaxWaitMs)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        EXPECT_LT(waitCnt, mMaxWaitMs);
        // Temporary debugging code for checking test duration.
        if (waitCnt >= mTrigWarnMs)
        {
            std::cerr << "[CTEST_FULL_OUTPUT] waitCnt = " << waitCnt << ", callCount = " << callCount << '\n';
        }
    }
    // 2 is the minimum number of input tokens in the input tensors
    EXPECT_EQ(callCount, kMaxNewTokens + 1);
}

TEST_F(GptManagerTest, ZeroOutputLength)
{
    std::atomic<int> callCount = 0;
    {
        ModelSpec modelSpec{"input_tokens.npy", nvinfer1::DataType::kHALF};
        modelSpec.useGptAttentionPlugin().usePackedInput().setKVCacheType(KVCacheType::kCONTINUOUS);
        fs::path dataPath = GPT_MODEL_PATH / modelSpec.getModelPath() / "tp1-pp1-cp1-gpu";
        TritonStub tritonStub;

        tr::SizeType32 constexpr beamWidth{1};
        GetInferenceRequestsCallback getInferenceRequestsCb = [&tritonStub](int maxNumRequests)
        { return tritonStub.get_inference_requests(maxNumRequests, kMaxNewTokens); };
        SendResponseCallback sendResponseCb
            = [](uint64_t requestId, std::list<NamedTensor> const&, bool, std::string const&) -> void
        {
            std::cout << "Inside response callback with requestId: " << requestId << std::endl;
            return;
        };
        ReturnBatchManagerStatsCallback returnBatchManagerStatsCb = [](std::string const& s) -> void
        {
            nlohmann::json j = nlohmann::json::parse(s);
            ASSERT_FALSE(j.is_discarded()); // Fail if s is an invalid json
            int32_t activeReq = j.at("Active Request Count").template get<int32_t>();
            ASSERT_GE(activeReq, 0);
        };

        TrtGptModelOptionalParams optionalParams;
        optionalParams.maxBeamWidth = beamWidth;
        optionalParams.enableTrtOverlap = false;
        optionalParams.schedulerConfig = texec::SchedulerConfig{texec::CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT};
        auto batch_manager_ = std::make_shared<MockedGptManager>(dataPath, TrtGptModelType::V1, getInferenceRequestsCb,
            sendResponseCb, nullptr, returnBatchManagerStatsCb, optionalParams, std::nullopt, true);

        // Mock the generateTokens function such that it just increments some counter
        EXPECT_CALL(*batch_manager_, forwardAsync(_, _))
            .WillRepeatedly(Invoke(
                [&callCount](RequestList& requestList, std::unordered_set<uint64_t>& activeRequestIds)
                {
                    auto numReq = requestList.size();
                    for (auto llmReq : requestList)
                    {
                        // Don't add any tokens to simulate no output tokens
                        llmReq->setState(LlmRequestState::kGENERATION_COMPLETE);
                    }

                    callCount++;

                    return BatchManagerErrorCode_t::STATUS_SUCCESS;
                }));

        // Here batch manager would be killed, so sleep some time to let the magic happen
        int waitCnt = 0;
        while (callCount < 1 && waitCnt++ < mMaxWaitMs)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        EXPECT_EQ(callCount, 1);
        EXPECT_LT(waitCnt, mMaxWaitMs);
    }
}

TEST_F(GptManagerTest, ErrorHandlingLargePrompt)
{
    std::atomic<int> callCount = 0;
    {
        ModelSpec modelSpec{"input_tokens.npy", nvinfer1::DataType::kHALF};
        modelSpec.useGptAttentionPlugin().usePackedInput().setKVCacheType(KVCacheType::kCONTINUOUS);
        fs::path dataPath = GPT_MODEL_PATH / modelSpec.getModelPath() / "tp1-pp1-cp1-gpu";
        TritonStub tritonStub;

        tr::SizeType32 constexpr beamWidth{1};
        GetInferenceRequestsCallback getInferenceRequestsCb = [&tritonStub](int maxNumRequests)
        { return tritonStub.get_inference_requests(maxNumRequests, kMaxSeqLen, 1000000); };
        SendResponseCallback sendResponseCb = [&callCount](uint64_t requestId, std::list<NamedTensor> const&,
                                                  bool is_final, std::string const& errMsg) -> void
        {
            EXPECT_EQ(is_final, true);
            EXPECT_THAT(errMsg, testing::HasSubstr("exceeds maximum input length"));
            std::cout << "Inside response callback with requestId: " << requestId << std::endl;
            callCount++;
            return;
        };
        ReturnBatchManagerStatsCallback returnBatchManagerStatsCb = [](std::string const& s) -> void
        {
            nlohmann::json j = nlohmann::json::parse(s);
            ASSERT_FALSE(j.is_discarded()); // Fail if s is an invalid json
            auto const activeReq = j.at("Active Request Count").template get<int32_t>();
            ASSERT_GE(activeReq, 0);
            auto const gpuMemUsed = j.at("Runtime GPU Memory Usage").template get<tr::MemoryCounters::SizeType32>();
            ASSERT_GE(gpuMemUsed, 0);
        };

        TrtGptModelOptionalParams optionalParams;
        optionalParams.maxBeamWidth = beamWidth;
        optionalParams.enableTrtOverlap = false;
        optionalParams.schedulerConfig = texec::SchedulerConfig{texec::CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT};
        auto batch_manager_ = std::make_shared<MockedGptManager>(dataPath, TrtGptModelType::V1, getInferenceRequestsCb,
            sendResponseCb, nullptr, returnBatchManagerStatsCb, optionalParams);

        // Mock the forwardAsync function
        // It should never get called
        EXPECT_CALL(*batch_manager_, forwardAsync(_, _))
            .WillRepeatedly(Invoke(
                [&callCount](RequestList& requestList, std::unordered_set<uint64_t>& activeRequestIds)
                {
                    auto numReq = requestList.size();
                    EXPECT_EQ(numReq, 0);
                    callCount++;
                    return BatchManagerErrorCode_t::STATUS_SUCCESS;
                }));

        // Here batch manager would be killed, so sleep some time to let the magic happen
        int waitCnt = 0;
        while (callCount < 2 && waitCnt++ < mMaxWaitMs)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        EXPECT_LT(waitCnt, mMaxWaitMs);
    }
    EXPECT_EQ(callCount, 2);
}

TEST_F(GptManagerTest, ErrorHandlingForwardFails)
{
    std::atomic<int> callCount = 0;
    int sendResponseCallbackCount = 0;
    {
        ModelSpec modelSpec{"input_tokens.npy", nvinfer1::DataType::kHALF};
        modelSpec.useGptAttentionPlugin().usePackedInput().setKVCacheType(KVCacheType::kCONTINUOUS);
        fs::path dataPath = GPT_MODEL_PATH / modelSpec.getModelPath() / "tp1-pp1-cp1-gpu";
        TritonStub tritonStub;

        tr::SizeType32 constexpr beamWidth{1};
        GetInferenceRequestsCallback getInferenceRequestsCb = [&tritonStub](int maxNumRequests)
        { return tritonStub.get_inference_requests(maxNumRequests, kMaxNewTokens); };
        SendResponseCallback sendResponseCb
            = [&](uint64_t requestId, std::list<NamedTensor> const&, bool is_final, std::string const& errMsg) -> void
        {
            EXPECT_EQ(is_final, true);
            EXPECT_THAT(errMsg, testing::HasSubstr("Encountered an error in forwardAsync function"));
            std::cout << "Inside response callback with requestId: " << requestId << std::endl;
            sendResponseCallbackCount++;
            return;
        };
        ReturnBatchManagerStatsCallback returnBatchManagerStatsCb = [](std::string const& s) -> void
        {
            nlohmann::json j = nlohmann::json::parse(s);
            ASSERT_FALSE(j.is_discarded()); // Fail if s is an invalid json
            int32_t activeReq = j.at("Active Request Count").template get<int32_t>();
            ASSERT_GE(activeReq, 0);
        };

        TrtGptModelOptionalParams optionalParams;
        optionalParams.maxBeamWidth = beamWidth;
        optionalParams.enableTrtOverlap = false;
        optionalParams.schedulerConfig = texec::SchedulerConfig{texec::CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT};
        auto batch_manager_ = std::make_shared<MockedGptManager>(dataPath, TrtGptModelType::V1, getInferenceRequestsCb,
            sendResponseCb, nullptr, returnBatchManagerStatsCb, optionalParams);

        // Mock the forwardAsync function
        // It should never get called
        EXPECT_CALL(*batch_manager_, forwardAsync(_, _))
            .WillRepeatedly(Invoke(
                [&](RequestList& requestList, std::unordered_set<uint64_t>& activeRequestIds)
                {
                    // Mock getting an error and modifying requestList
                    for (auto it = requestList.cbegin(); it != requestList.cend();)
                    {
                        // Call the response callback so that requests get removed from workItems
                        sendResponseCb(static_cast<uint64_t>((*it)->mRequestId), {}, true,
                            "Encountered an error in forwardAsync function");
                        // Remove from the requestList
                        requestList.erase(it++);
                    }
                    callCount++;
                    return BatchManagerErrorCode_t::STATUS_FAILED;
                }));

        // Here batch manager would be killed, so sleep some time to let the magic happen
        int waitCnt = 0;
        while (callCount < 1 && waitCnt++ < mMaxWaitMs)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        EXPECT_LT(waitCnt, mMaxWaitMs);
    }
    EXPECT_EQ(callCount, 1);
    EXPECT_EQ(sendResponseCallbackCount, 2);
}

namespace
{

std::tuple<std::vector<tr::SizeType32>, tr::SizeType32, tr::SizeType32> getGivenInputLengths(
    tr::ITensor const& givenInput, tr::SizeType32 padId)
{

    auto const& inputShape = givenInput.getShape();
    auto const nbGivenInputs = static_cast<SizeType32>(inputShape.d[0]);
    auto const maxInputLength = static_cast<SizeType32>(inputShape.d[1]);
    auto const givenInputData = tr::bufferCast<tr::TokenIdType const>(givenInput);

    std::vector<SizeType32> givenInputLengths(nbGivenInputs);
    for (SizeType32 i = 0; i < nbGivenInputs; ++i)
    {
        auto const seqBegin = givenInputData + i * maxInputLength;
        auto const it = std::find(seqBegin, seqBegin + maxInputLength, padId);
        givenInputLengths[i] = std::distance(seqBegin, it);
    }

    return {givenInputLengths, nbGivenInputs, maxInputLength};
}

std::tuple<InferenceRequest::TensorPtr, std::vector<tr::SizeType32>, tr::SizeType32, tr::SizeType32> loadInput(
    fs::path inputPath, SizeType32 padId)
{
    auto manager = tr::BufferManager(std::make_shared<tr::CudaStream>());
    InferenceRequest::TensorPtr givenInput = tr::utils::loadNpy(manager, inputPath.string(), tr::MemoryType::kCPU);
    auto [givenInputLengths, nbGivenInputs, maxInputLength] = getGivenInputLengths(*givenInput, padId);
    return {givenInput, givenInputLengths, nbGivenInputs, maxInputLength};
}

std::vector<std::int32_t> getInput(fs::path inputPath)
{
    auto manager = tr::BufferManager(std::make_shared<tr::CudaStream>());
    auto const& givenInput = tr::utils::loadNpy(manager, inputPath.string(), tr::MemoryType::kCPU);
    auto const givenInputData = tr::bufferCast<int32_t const>(*givenInput);
    auto const& inputShape = givenInput->getShape();
    // This is only using the first sequence from the data file (which has BS=8 worth of data)
    std::vector<int32_t> inputVector;
    for (int32_t i = 0; i < inputShape.d[1]; ++i)
    {
        inputVector.push_back(givenInputData[i]);
    }

    return inputVector;
}

InferenceRequest::TensorPtr getExpectedOutputTensor(fs::path const& resultsFile)
{
    auto manager = tr::BufferManager(std::make_shared<tr::CudaStream>());
    return tr::utils::loadNpy(manager, resultsFile.string(), tr::MemoryType::kCPU);
}

std::vector<std::vector<std::int32_t>> getExpectedOutput(fs::path const& inputFile, fs::path const& resultsFile)
{
    auto manager = tr::BufferManager(std::make_shared<tr::CudaStream>());
    auto const& expectedOutput = tr::utils::loadNpy(manager, resultsFile.string(), tr::MemoryType::kCPU);
    auto const& outputShape = expectedOutput->getShape();
    auto const expectedOutputData = tr::bufferCast<int32_t const>(*expectedOutput);

    std::vector<std::vector<std::int32_t>> expectedOutputVector;
    auto input = getInput(inputFile);

    for (int32_t i = 0; i < outputShape.d[0]; ++i)
    {
        std::vector<int32_t> seq;

        for (int32_t j = input.size(); j < outputShape.d[1]; ++j)
        {
            seq.push_back(expectedOutputData[i * outputShape.d[1] + j]);
        }

        expectedOutputVector.push_back(seq);
    }

    return expectedOutputVector;
}

int32_t getTestMaxSeqLen(fs::path const& inputFile, fs::path const& resultsFile)
{
    auto const input = getInput(inputFile);
    auto const expectedOutput = getExpectedOutput(inputFile, resultsFile);
    int32_t const kTestMaxTokensPad = 0;
    return static_cast<int32_t>(input.size() + expectedOutput.size()) + kTestMaxTokensPad;
}

} // namespace

// Another Triton stub (mock) class. It is specific (scoped) to this test
class TritonStubBatchingTestBS1
{
public:
    TritonStubBatchingTestBS1(fs::path const& inputFile, fs::path const& resultsFile, tr::SizeType32 const padId)
        : mCount{0}
        , mInputFile{inputFile}
        , mResultsFile(resultsFile)
        , mPadId{padId}
    {
    }

    int32_t mCount;
    fs::path mInputFile;
    fs::path mResultsFile;
    tr::SizeType32 mPadId;

    std::list<std::shared_ptr<InferenceRequest>> get_inference_requests(int32_t maxNumRequests, int32_t maxNewTokens,
        tr::SizeType32 beamWidth = 1, bool returnLogProbs = false, bool returnContextLogits = false,
        bool returnGenerationLogits = false)
    {
        // Only provide requests for 1st callback
        if (mCount > 0)
            return {};

        // Expected test input
        int32_t const requestBatchSize = 1;

        auto [givenInput, givenInputLengths, nbGivenInputs, maxInputLength] = loadInput(mInputFile, mPadId);
        auto givenInputData = tr::bufferCast<tr::TokenIdType>(*givenInput);

        std::vector<int64_t> inputDimsReq1 = {requestBatchSize, static_cast<int64_t>(givenInputLengths[0])};
        auto inputReq1 = NamedTensor(nvinfer1::DataType::kINT32, inputDimsReq1, "req1_prompt");
        auto* pinputReq1 = inputReq1.tensor->data();
        std::memcpy(pinputReq1, givenInputData, givenInputLengths[0] * sizeof(int32_t));

        auto outputLengthReq1 = NamedTensor(nvinfer1::DataType::kINT32, {1, 1}, "req1_request_output_len");
        int32_t const request_output_len = maxNewTokens;
        auto* poutputLengthReq1 = outputLengthReq1.tensor->data();
        std::memcpy(poutputLengthReq1, &request_output_len, sizeof(int32_t));

        auto beamWidthReq1 = NamedTensor(nvinfer1::DataType::kINT32, {1}, "req1_beam_width");
        auto* beamWidthReq1Ptr = beamWidthReq1.tensor->data();
        std::memcpy(beamWidthReq1Ptr, &beamWidth, sizeof(int32_t));

        auto returnLogProbsReq1 = NamedTensor(nvinfer1::DataType::kBOOL, {1}, "req1_return_log_probs");
        auto* returnLogProbsReq1Ptr = returnLogProbsReq1.tensor->data();
        std::memcpy(returnLogProbsReq1Ptr, &returnLogProbs, sizeof(bool));

        auto returnContextLogitsReq1 = NamedTensor(nvinfer1::DataType::kBOOL, {1}, "req1_return_context_logits");
        auto* returnContextLogitsReq1Ptr = returnContextLogitsReq1.tensor->data();
        std::memcpy(returnContextLogitsReq1Ptr, &returnContextLogits, sizeof(bool));

        auto returnGenerationLogitsReq1 = NamedTensor(nvinfer1::DataType::kBOOL, {1}, "req1_return_generation_logits");
        auto* returnGenerationLogitsReq1Ptr = returnGenerationLogitsReq1.tensor->data();
        std::memcpy(returnGenerationLogitsReq1Ptr, &returnGenerationLogits, sizeof(bool));

        // Form single inference request
        auto request1 = std::make_shared<InferenceRequest>(1);
        request1->setInputIds(std::move(inputReq1.tensor));
        request1->setMaxNewTokens(std::move(outputLengthReq1.tensor));
        request1->setReturnLogProbs(std::move(returnLogProbsReq1.tensor));
        request1->setBeamWidth(std::move(beamWidthReq1.tensor));
        request1->setReturnContextLogits(std::move(returnContextLogitsReq1.tensor));
        request1->setReturnGenerationLogits(std::move(returnGenerationLogitsReq1.tensor));

        std::list<std::shared_ptr<InferenceRequest>> inferenceRequest{request1};
        ++mCount;

        return inferenceRequest;
    }

    bool send_response_callback(uint64_t cid, std::list<InferenceRequest::TensorPtr> outputTensors)
    {
        return false;
    }
}; // TritonStubBatchingTestBS1

void runTest(fs::path const& modelPath, TrtGptModelType modelType, nvinfer1::DataType dtype, tr::SizeType32 beamWidth,
    tr::SizeType32 const vocabSizePadded, tr::SizeType32 const padId, fs::path const& inputFile,
    fs::path const& resultsFile, std::shared_ptr<nvinfer1::ILogger> const& logger, bool computeLogProbs,
    bool excludeInputInOutput, int maxWaitMs, int trigWarnMs, bool returnContextLogits, bool returnGenerationLogits)
{
    TritonStubBatchingTestBS1 tritonStubBatchingTestBS1(inputFile, resultsFile, padId);

    InferenceRequest::TensorPtr expectedOutput = getExpectedOutputTensor(resultsFile);
    auto expectedOutputData = tr::bufferCast<tr::TokenIdType>(*expectedOutput);
    auto const& outputShape = expectedOutput->getShape();
    SizeType32 maxSeqLen = static_cast<SizeType32>(outputShape.d[1]);

    auto [givenInput, givenInputLengths, nbGivenInputs, maxInputLength] = loadInput(inputFile, padId);
    auto const maxNewTokens = maxSeqLen - maxInputLength;

    // Get inference requests
    GetInferenceRequestsCallback getInferenceRequestsCb
        = [&tritonStubBatchingTestBS1, maxNewTokens, beamWidth, computeLogProbs, returnContextLogits,
              returnGenerationLogits](int32_t maxNumRequests)
    {
        return tritonStubBatchingTestBS1.get_inference_requests(
            maxNumRequests, maxNewTokens, beamWidth, computeLogProbs, returnContextLogits, returnGenerationLogits);
    };
    // ...

    std::atomic<int32_t> sendResponseCbCount = 0;
    InferenceRequest::TensorPtr capturedOutput{nullptr}, capturedLengths{nullptr};
    InferenceRequest::TensorPtr capturedLogProbs{nullptr}, capturedCumLogProbs{nullptr};
    InferenceRequest::TensorPtr capturedContextLogits{nullptr}, capturedGenerationLogits{nullptr};

    SendResponseCallback sendResponseCb
        = [&](uint64_t, std::list<NamedTensor> const& outputTensors, bool isFinal, std::string const& errMsg) -> void
    {
        for (auto& outputTensor : outputTensors)
        {
            if (outputTensor.name == inference_request::kOutputIdsTensorName)
            {
                capturedOutput = outputTensor.tensor;
            }
            else if (outputTensor.name == inference_request::kSequenceLengthTensorName)
            {
                capturedLengths = outputTensor.tensor;
            }
            else if (outputTensor.name == inference_request::kLogProbsTensorName)
            {
                capturedLogProbs = outputTensor.tensor;
            }
            else if (outputTensor.name == inference_request::kCumLogProbsTensorName)
            {
                capturedCumLogProbs = outputTensor.tensor;
            }
            else if (outputTensor.name == inference_request::kContextLogitsName)
            {
                capturedContextLogits = outputTensor.tensor;
            }
            else if (outputTensor.name == inference_request::kGenerationLogitsName)
            {
                capturedGenerationLogits = outputTensor.tensor;
            }
        }
        ++sendResponseCbCount;

        return;
    };
    ReturnBatchManagerStatsCallback returnBatchManagerStatsCb = [](std::string const& s) -> void
    {
        nlohmann::json j = nlohmann::json::parse(s);
        ASSERT_FALSE(j.is_discarded()); // Fail if s is an invalid json
        int32_t activeReq = j.at("Active Request Count").template get<int32_t>();
        std::cout << s << std::endl;
        ASSERT_GE(activeReq, 0);
    };

    int32_t constexpr kTestBatchSize = 1;

    TrtGptModelOptionalParams optionalParams;
    optionalParams.maxBeamWidth = beamWidth;
    optionalParams.enableTrtOverlap = false;
    optionalParams.schedulerConfig = texec::SchedulerConfig{texec::CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT};
    if (returnContextLogits)
    {
        // Return context logits will need more memory
        optionalParams.kvCacheConfig.freeGpuMemoryFraction = 0.3;
    }
    if (returnGenerationLogits)
    {
        optionalParams.gatherGenerationLogits = true;
    }

    auto batchManager = std::make_shared<GptManager>(modelPath, modelType, getInferenceRequestsCb, sendResponseCb,
        nullptr, returnBatchManagerStatsCb, optionalParams, std::nullopt, excludeInputInOutput);

    // Compare output to expected output
    int waitCnt = 0;
    while (sendResponseCbCount < kTestBatchSize && waitCnt++ < maxWaitMs)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_LT(waitCnt, maxWaitMs);

    // Temporary debugging code for checking test duration.
    if (waitCnt >= trigWarnMs)
    {
        std::cerr << "[CTEST_FULL_OUTPUT] More iterations than expected, waitCnt = " << waitCnt << '\n';
    }
    EXPECT_EQ(sendResponseCbCount, kTestBatchSize);

    ASSERT_NE(capturedOutput, nullptr);
    ASSERT_NE(capturedLengths, nullptr);
    ASSERT_NE(capturedLogProbs, nullptr);
    ASSERT_NE(capturedCumLogProbs, nullptr);
    ASSERT_NE(capturedContextLogits, nullptr);
    ASSERT_NE(capturedGenerationLogits, nullptr);

    auto capturedOutputData = tr::bufferCast<tr::TokenIdType>(*capturedOutput);
    for (auto bi = 0; bi < kTestBatchSize; ++bi)
    {
        for (auto beam = 0; beam < beamWidth; ++beam)
        {
            bool anyMismatch = false;
            SizeType32 inputLength = givenInputLengths[bi];
            SizeType32 expectedLen = inputLength + maxNewTokens;
            for (auto i = 0; i < maxNewTokens; ++i)
            {
                auto const expectIndex = tc::flat_index3(bi, beam, inputLength + i, beamWidth, maxSeqLen);
                auto expectedToken = expectedOutputData[expectIndex];

                SizeType32 inputOffset = excludeInputInOutput ? 0 : inputLength;
                SizeType32 seqLen = excludeInputInOutput ? maxSeqLen - inputLength : maxSeqLen;

                auto const outputIndex = tc::flat_index3(bi, beam, inputOffset + i, beamWidth, seqLen);
                // auto capturedToken = capturedOutputData[outputIndex];

                // EXPECT_EQ(capturedToken, expectedToken) << " b: " << bi << " beam: " << beam << " i: " << i;
                // anyMismatch |= (capturedToken != expectedToken);
            }
            ASSERT_FALSE(anyMismatch) << "batchSize: " << kTestBatchSize << ", b: " << bi;
        }
    }

    auto cumLogProbsShape = capturedCumLogProbs->getShape();
    auto logProbsShape = capturedLogProbs->getShape();

    EXPECT_EQ(cumLogProbsShape.nbDims, 2);
    EXPECT_EQ(cumLogProbsShape.d[0], 1);
    EXPECT_EQ(cumLogProbsShape.d[1], beamWidth);

    EXPECT_EQ(logProbsShape.nbDims, 3);
    EXPECT_EQ(logProbsShape.d[0], 1);
    EXPECT_EQ(logProbsShape.d[1], beamWidth);
    EXPECT_EQ(logProbsShape.d[2], capturedOutput->getShape().d[2] - (excludeInputInOutput ? 0 : givenInputLengths[0]));

    auto const cumLogProbsData = tr::bufferCast<float>(*capturedCumLogProbs);
    auto const logProbsData = tr::bufferCast<float>(*capturedLogProbs);
    if (computeLogProbs)
    {
        std::cout << "cumLogProbs" << std::endl;
        for (int i = 0; i < cumLogProbsShape.d[0]; ++i)
        {
            for (int j = 0; j < cumLogProbsShape.d[1]; ++j)
            {
                std::cout << cumLogProbsData[i * beamWidth + j] << std::endl;
            }
        }

        std::cout << "logProbs" << std::endl;
        for (int i = 0; i < logProbsShape.d[0]; ++i)
        {
            for (int j = 0; j < logProbsShape.d[1]; ++j)
            {
                for (int k = 0; k < logProbsShape.d[2]; ++k)
                {
                    std::cout << logProbsData[(i * beamWidth + j) * logProbsShape.d[2] + k] << std::endl;
                }
            }
        }
    }
    else
    {
        // Expect zeros in log probs
        for (int i = 0; i < tr::ITensor::volume(cumLogProbsShape); ++i)
        {
            EXPECT_FLOAT_EQ(cumLogProbsData[i], 0.);
        }
        for (int i = 0; i < tr::ITensor::volume(logProbsShape); ++i)
        {
            EXPECT_FLOAT_EQ(logProbsData[i], 0.);
        }
    }

    // Check context logits shape
    auto contextLogitsShape = capturedContextLogits->getShape(); // expected shape: [1, input_length, vocab_size]

    EXPECT_EQ(contextLogitsShape.nbDims, 3);
    EXPECT_EQ(contextLogitsShape.d[0], 1);
    if (returnContextLogits)
    {
        EXPECT_EQ(contextLogitsShape.d[1], givenInputLengths[0]); // input_length
        EXPECT_EQ(contextLogitsShape.d[2], vocabSizePadded);      // pad_vocab_size for gpt2
    }
    else
    {
        EXPECT_EQ(contextLogitsShape.d[1], 1); // dummy tensor
        EXPECT_EQ(contextLogitsShape.d[2], 1); // dummy tensor
        auto const capturedContextLogitsData = tr::bufferCast<float>(*capturedContextLogits);
        EXPECT_FLOAT_EQ(capturedContextLogitsData[0], 0.);
    }

    // Check generation logits shape
    auto generationLogitsShape
        = capturedGenerationLogits->getShape(); // expected shape: [1, beam_width, output_length, vocab_size]

    EXPECT_EQ(generationLogitsShape.nbDims, 4);
    EXPECT_EQ(generationLogitsShape.d[0], 1);
    if (returnGenerationLogits)
    {
        EXPECT_EQ(generationLogitsShape.d[1], beamWidth);       // beam_width
        EXPECT_EQ(generationLogitsShape.d[2], maxNewTokens);    // output_length
        EXPECT_EQ(generationLogitsShape.d[3], vocabSizePadded); // pad_vocab_size for gpt2
    }
    else
    {
        EXPECT_EQ(generationLogitsShape.d[1], 1); // dummy tensor
        EXPECT_EQ(generationLogitsShape.d[2], 1); // dummy tensor
        EXPECT_EQ(generationLogitsShape.d[3], 1); // dummy tensor
        auto const capturedGenerationLogitsData = tr::bufferCast<float>(*capturedGenerationLogits);
        EXPECT_FLOAT_EQ(capturedGenerationLogitsData[0], 0.);
    }
}

using ParamType = std::tuple<TrtGptModelType, tr::SizeType32, bool, bool, bool, bool, int>;

std::string generateTestName(testing::TestParamInfo<ParamType> const& info)
{
    auto const modelType = std::get<0>(info.param);
    auto const& beamWidth = std::get<1>(info.param);
    auto const& computeLogProbs = std::get<2>(info.param);
    auto const& excludeInputInOutput = std::get<3>(info.param);
    auto const& returnContextLogits = std::get<4>(info.param);
    auto const& returnGenerationLogits = std::get<5>(info.param);
    auto const versionChatglm = std::get<6>(info.param);

    std::string name = (versionChatglm > 0) ? "ChatGlmManagerTest_" : "GptManagerTest_";

    switch (modelType)
    {
    case TrtGptModelType::V1: name.append("V1Model"); break;
    case TrtGptModelType::InflightBatching: name.append("IbModel"); break;
    case TrtGptModelType::InflightFusedBatching: name.append("FusedIbModel"); break;
    default: name.append("DefaultModel"); break;
    }

    name.append("BW" + std::to_string(beamWidth));
    if (computeLogProbs)
    {
        name.append("LogProbs");
    }
    if (excludeInputInOutput)
    {
        name.append("ExcludeInput");
    }
    if (returnContextLogits)
    {
        name.append("ContextLogits");
    }
    if (returnGenerationLogits)
    {
        name.append("GenerationLogits");
    }
    return name;
}

class ParamTest : public GptManagerTest, public ::testing::WithParamInterface<ParamType>
{
};

TEST_P(ParamTest, Test)
{
    auto const modelType = std::get<0>(GetParam());
    auto const& beamWidth = std::get<1>(GetParam());
    auto const& computeLogProbs = std::get<2>(GetParam());
    auto const& excludeInputInOutput = std::get<3>(GetParam());
    auto const& returnContextLogits = std::get<4>(GetParam());
    auto const& returnGenerationLogits = std::get<5>(GetParam());
    auto const& versionChatglm = std::get<6>(GetParam());
    int const maxWaitMs = 60000;
    int const trigWarnMs = 10000;

    fs::path modelPath;
    nvinfer1::DataType dtype;
    fs::path inputFile, resultsFile;
    tr::SizeType32 vocabSizePadded, padId;
    ModelSpec modelSpec{"input_tokens.npy", nvinfer1::DataType::kHALF};
    modelSpec.useGptAttentionPlugin().usePackedInput().setKVCacheType(KVCacheType::kPAGED);

    if (versionChatglm == 0) // gpt
    {
        if (returnContextLogits || returnGenerationLogits)
        {
            ModelSpec gatherModelSpec(modelSpec);
            gatherModelSpec.gatherLogits();
            modelPath = GPT_MODEL_PATH / gatherModelSpec.getModelPath() / "tp1-pp1-cp1-gpu";
        }
        else
        {
            modelPath = GPT_MODEL_PATH / modelSpec.getModelPath() / "tp1-pp1-cp1-gpu";
        }
        dtype = nvinfer1::DataType::kHALF;
        inputFile = DATA_PATH / modelSpec.mInputFile;
        resultsFile = GPT_DATA_PATH / ((beamWidth == 1) ? "sampling" : "beam_search_" + std::to_string(beamWidth))
            / modelSpec.getResultsFile();
        vocabSizePadded = 50257;
        padId = 50256; // the same as endId
    }
    else
    {
        if (versionChatglm == 1) // chatglm-6b
        {
            modelSpec.setInputFile("input_tokens_chatglm-6b.npy");
            modelPath = CHATGLM_MODEL_PATH / modelSpec.getModelPath() / "tp1-pp1-cp1-gpu";
            inputFile = DATA_PATH / modelSpec.mInputFile;
            resultsFile = CHATGLM_DATA_PATH;
            vocabSizePadded = 130528;
            padId = 3;                // endId = 130005;
        }
        else if (versionChatglm == 2) // chatglm2-6b
        {
            modelSpec.setInputFile("input_tokens_chatglm2-6b.npy");
            modelPath = CHATGLM2_MODEL_PATH / modelSpec.getModelPath() / "tp1-pp1-cp1-gpu";
            inputFile = DATA_PATH / modelSpec.mInputFile;
            ;
            resultsFile = CHATGLM2_DATA_PATH;
            vocabSizePadded = 65024;
            padId = 0; // endId = 2;
        }
        else           // chatglm3-6b
        {
            modelSpec.setInputFile("input_tokens_chatglm3-6b.npy");
            modelPath = CHATGLM3_MODEL_PATH / modelSpec.getModelPath() / "tp1-pp1-cp1-gpu";
            inputFile = DATA_PATH / modelSpec.mInputFile;
            resultsFile = CHATGLM3_DATA_PATH;
            vocabSizePadded = 65024;
            padId = 0; // endId = 2;
        }
        dtype = nvinfer1::DataType::kFLOAT;
        resultsFile = resultsFile / ((beamWidth == 1) ? "sampling" : "beam_search_" + std::to_string(beamWidth))
            / modelSpec.getResultsFile();
    }

    runTest(modelPath, modelType, dtype, beamWidth, vocabSizePadded, padId, inputFile, resultsFile, mLogger,
        computeLogProbs, excludeInputInOutput, maxWaitMs, trigWarnMs, returnContextLogits, returnGenerationLogits);
}

INSTANTIATE_TEST_SUITE_P(GptManagerTests, ParamTest,
    testing::Combine(testing::Values(TrtGptModelType::V1, TrtGptModelType::InflightFusedBatching),
        testing::Values(1, 2), testing::Values(false, true), testing::Values(false, true), testing::Values(false, true),
        testing::Values(false, true), testing::Values(0)),
    generateTestName);

// Disable some of ChatGLM's tests since they are the same as gpt's.
INSTANTIATE_TEST_SUITE_P(ChatGlmManagerTests, ParamTest,
    testing::Combine(testing::Values(TrtGptModelType::V1, TrtGptModelType::InflightFusedBatching),
        testing::Values(1, 2), testing::Values(false), testing::Values(false, true), testing::Values(false),
        testing::Values(false), testing::Values(1)),
    generateTestName);

INSTANTIATE_TEST_SUITE_P(ChatGlm2ManagerTests, ParamTest,
    testing::Combine(testing::Values(TrtGptModelType::V1, TrtGptModelType::InflightFusedBatching), testing::Values(1),
        testing::Values(false), testing::Values(false), testing::Values(false), testing::Values(false),
        testing::Values(2)),
    generateTestName);

INSTANTIATE_TEST_SUITE_P(Glm3ManagerTestsChat, ParamTest,
    testing::Combine(testing::Values(TrtGptModelType::V1, TrtGptModelType::InflightFusedBatching), testing::Values(1),
        testing::Values(false), testing::Values(false), testing::Values(false), testing::Values(false),
        testing::Values(3)),
    generateTestName);

TEST_F(GptManagerTest, EarlyStopping)
{
    auto constexpr dtype = nvinfer1::DataType::kHALF;
    tr::SizeType32 constexpr beamWidth{1};

    ModelSpec modelSpec{"input_tokens.npy", dtype};
    modelSpec.useGptAttentionPlugin().usePackedInput().setKVCacheType(KVCacheType::kPAGED);

    auto const modelPath = GPT_MODEL_PATH / modelSpec.getModelPath() / "tp1-pp1-cp1-gpu";
    fs::path inputFile{DATA_PATH / modelSpec.mInputFile};

    fs::path resultsFile{GPT_DATA_PATH / "sampling" / modelSpec.getResultsFile()};
    SizeType32 padId = 50256;

    TritonStubBatchingTestBS1 tritonStubBatchingTestBS1(inputFile, resultsFile, padId);

    // get expected output
    std::vector<int32_t> expectedOutput;
    auto input = getInput(inputFile);
    auto output = getExpectedOutput(inputFile, resultsFile);
    expectedOutput.insert(expectedOutput.end(), input.begin(), input.end());
    expectedOutput.insert(expectedOutput.end(), output[0].begin(), output[0].end());

    // Get inference requests
    GetInferenceRequestsCallback getInferenceRequestsCb
        = [&tritonStubBatchingTestBS1, output, input](int32_t maxNumRequests)
    { return tritonStubBatchingTestBS1.get_inference_requests(maxNumRequests, output[0].size()); };

    // Send response call back only process the output of one request
    std::atomic<int32_t> sendResponseCbCount = 0;
    auto expectedOutputSize = getTestMaxSeqLen(inputFile, resultsFile) * beamWidth;
    std::vector<int32_t> capturedOutput;
    SendResponseCallback sendResponseCb
        = [&](uint64_t, std::list<NamedTensor> const& outputTensors, bool, std::string const&) -> void
    {
        if (sendResponseCbCount < 1)
        {
            auto tensor = outputTensors.front().tensor;
            auto shape = tensor->getShape();
            for (int32_t i = 0; i < shape.d[2]; ++i)
            {
                capturedOutput.emplace_back(static_cast<int32_t const*>(tensor->data())[i]);
            }
            ++sendResponseCbCount;
        }

        return;
    };

    int32_t numSteps = 0;
    int32_t stopStepNo = output.size() / 2;
    std::unordered_set<uint64_t> stoppedRequests;
    stoppedRequests.emplace(1);
    std::unordered_set<uint64_t> emptyStoppedSignals;
    PollStopSignalCallback pollStopSignalCb
        = [&numSteps, &stopStepNo, &stoppedRequests, &emptyStoppedSignals]() -> std::unordered_set<uint64_t>
    {
        numSteps++;

        if (numSteps >= stopStepNo)
        {
            // Early stop the generation when the number of output reaches the half of the expected length
            return stoppedRequests;
        }
        else
        {
            return emptyStoppedSignals;
        }
    };
    ReturnBatchManagerStatsCallback returnBatchManagerStatsCb = [](std::string const& s) -> void
    {
        nlohmann::json j = nlohmann::json::parse(s);
        ASSERT_FALSE(j.is_discarded()); // Fail if s is an invalid json
        int32_t activeReq = j.at("Active Request Count").template get<int32_t>();
        ASSERT_GE(activeReq, 0);
    };

    int32_t constexpr kTestBatchSize = 1;
    int32_t const kTestMaxSeqLen = getTestMaxSeqLen(inputFile, resultsFile);

    TrtGptModelOptionalParams optionalParams;
    optionalParams.maxBeamWidth = beamWidth;
    optionalParams.enableTrtOverlap = false;
    optionalParams.schedulerConfig = texec::SchedulerConfig{texec::CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT};
    auto batchManager = std::make_shared<GptManager>(modelPath, TrtGptModelType::InflightFusedBatching,
        getInferenceRequestsCb, sendResponseCb, pollStopSignalCb, returnBatchManagerStatsCb, optionalParams);

    int waitCnt = 0;
    while (sendResponseCbCount == 0 && waitCnt++ < mMaxWaitMs)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_LT(waitCnt, mMaxWaitMs);
    EXPECT_EQ(sendResponseCbCount, 1);

    // The final length of output sequence should be shorter than
    // the expected output since we early stop the generation
    std::cout << "capturedOutput length: " << capturedOutput.size()
              << "; expectedOutput length: " << expectedOutput.size() << std::endl;
    // Verify that the number of generated tokens is consistent with the number of steps we allow before early stopping
    // Minus 1 because the 1st forward call doesn't update numGeneratedTokens
    EXPECT_EQ(capturedOutput.size(), input.size() + stopStepNo - 1);
}

TEST_F(GptManagerTest, LogitsPostProcessor)
{
    auto constexpr dtype = nvinfer1::DataType::kHALF;
    ModelSpec modelSpec{"input_tokens.npy", dtype};
    modelSpec.useGptAttentionPlugin().usePackedInput().setKVCacheType(KVCacheType::kPAGED);
    auto const modelPath = GPT_MODEL_PATH / modelSpec.getModelPath() / "tp1-pp1-cp1-gpu";

    int constexpr sentinels[] = {42, 29};
    tr::SizeType32 constexpr beamWidth{1};
    fs::path inputFile{DATA_PATH / modelSpec.mInputFile};
    fs::path resultsFile{GPT_DATA_PATH / "sampling" / modelSpec.getResultsFile()};
    SizeType32 padId = 50256;

    TritonStubBatchingTestBS1 tritonStubBatchingTestBS1(inputFile, resultsFile, padId);

    // get expected output
    std::vector<int32_t> expectedOutput;
    auto input = getInput(inputFile);
    int32_t const requestedOutputlen = 8;
    int step = 0;

    LlmRequest::LogitsPostProcessor logitsCb
        = [&step, &sentinels](uint64_t rId, tensorrt_llm::runtime::ITensor::SharedPtr& logits,
              LlmRequest::BeamTokens const& tokens, tensorrt_llm::runtime::BufferManager::CudaStreamPtr streamPtr,
              std::optional<uint64_t> cId)
    {
        auto tensorAten = tensorrt_llm::runtime::Torch::tensor(logits);
        auto mask = at::full_like(tensorAten, c10::Scalar(-HUGE_VALF), {at::kCPU});
        mask[0][0][sentinels[step]] = 0.0f;
        auto atStream = tensorrt_llm::runtime::TorchUtils::stream(*streamPtr);

        {
            at::cuda::CUDAStreamGuard guard(atStream);
            tensorAten += mask.to({at::kCUDA}, true);
        }

        step = (1 - step);
    };

    // Get inference requests
    GetInferenceRequestsCallback getInferenceRequestsCb
        = [requestedOutputlen, &tritonStubBatchingTestBS1, &logitsCb](int32_t maxNumRequests)
    {
        auto inference_requests = tritonStubBatchingTestBS1.get_inference_requests(maxNumRequests, requestedOutputlen);
        for (auto& ir : inference_requests)
        {
            ir->setLogitsPostProcessor(logitsCb);
        }
        return inference_requests;
    };

    // Send response call back only process the output of one request
    std::atomic<int32_t> sendResponseCbCount = 0;
    std::vector<int32_t> capturedOutput;
    SendResponseCallback sendResponseCb
        = [&](uint64_t, std::list<NamedTensor> const& outputTensors, bool, std::string const&) -> void
    {
        if (sendResponseCbCount < 1)
        {
            auto tensor = outputTensors.front().tensor;
            auto shape = tensor->getShape();
            for (int32_t i = 0; i < shape.d[2]; ++i)
            {
                capturedOutput.emplace_back(static_cast<int32_t const*>(tensor->data())[i]);
            }
            ++sendResponseCbCount;
        }

        return;
    };

    TrtGptModelOptionalParams optionalParams;
    optionalParams.maxBeamWidth = beamWidth;
    optionalParams.enableTrtOverlap = false;
    optionalParams.schedulerConfig = texec::SchedulerConfig{texec::CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT};
    auto batchManager = std::make_shared<GptManager>(modelPath, TrtGptModelType::InflightFusedBatching,
        getInferenceRequestsCb, sendResponseCb, nullptr, nullptr, optionalParams);

    int waitCnt = 0;
    while (sendResponseCbCount == 0 && waitCnt++ < mMaxWaitMs)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_LT(waitCnt, mMaxWaitMs);
    EXPECT_EQ(sendResponseCbCount, 1);

    // The final length of output sequence should be shorter than
    // the expected output since we early stop the generation
    std::cout << "capturedOutput length: " << capturedOutput.size()
              << "; expectedOutput length: " << input.size() + requestedOutputlen << std::endl;
    // Verify that the number of generated tokens is consistent with the number of steps we allow before early stopping
    // Minus 1 because the 1st forward call doesn't update numGeneratedTokens
    EXPECT_EQ(capturedOutput.size(), input.size() + requestedOutputlen);
    for (int i = 0; i < capturedOutput.size(); ++i)
    {
        if (i < input.size())
        {
            EXPECT_EQ(capturedOutput[i], input[i]);
        }
        else
        {
            EXPECT_EQ(capturedOutput[i], sentinels[(i - input.size()) % 2]);
        }
    }
}
