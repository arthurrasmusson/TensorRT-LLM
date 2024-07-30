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

#include "tensorrt_llm/executor/executor.h"
#include "modelSpec.h"
#include "tensorrt_llm/batch_manager/trtGptModel.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/mpiUtils.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/executor/version.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/tllmLogger.h"
#include "tensorrt_llm/runtime/utils/numpyUtils.h"
#include "tests/utils/common.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <functional>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

using ::testing::_;
using ::testing::Invoke;

namespace tr = tensorrt_llm::runtime;
namespace tb = tensorrt_llm::batch_manager;
namespace tc = tensorrt_llm::common;

using namespace tensorrt_llm::testing;
using namespace tensorrt_llm::executor;
using namespace std::chrono_literals;
namespace fs = std::filesystem;
using tensorrt_llm::testing::ModelSpec;
using tensorrt_llm::testing::KVCacheType;
using tensorrt_llm::testing::QuantMethod;
using tensorrt_llm::testing::OutputContentType;

namespace
{

auto const TEST_RESOURCE_PATH = std::filesystem::path{TOP_LEVEL_DIR} / "cpp/tests/resources";
auto const ENGINE_PATH = TEST_RESOURCE_PATH / "models/rt_engine";
auto const GPT_MODEL_PATH = ENGINE_PATH / "gpt2";
auto const LLAMA_MODEL_PATH = ENGINE_PATH / "llama-7b-hf";
auto const MEDUSA_MODEL_PATH = ENGINE_PATH / "vicuna-7b-v1.3";
auto const CHATGLM_MODEL_PATH = ENGINE_PATH / "chatglm-6b";
auto const CHATGLM2_MODEL_PATH = ENGINE_PATH / "chatglm2-6b";
auto const CHATGLM3_MODEL_PATH = ENGINE_PATH / "chatglm3-6b";
auto const GLM_MODEL_PATH = ENGINE_PATH / "glm-10b";
auto const DATA_PATH = TEST_RESOURCE_PATH / "data";
auto const GPT_DATA_PATH = DATA_PATH / "gpt2";
auto const LLAMA_DATA_PATH = DATA_PATH / "llama-7b-hf";
auto const MEDUSA_DATA_PATH = DATA_PATH / "vicuna-7b-v1.3";
auto const CHATGLM_DATA_PATH = DATA_PATH / "chatglm-6b";
auto const CHATGLM2_DATA_PATH = DATA_PATH / "chatglm2-6b";
auto const CHATGLM3_DATA_PATH = DATA_PATH / "chatglm3-6b";
auto const GLM_DATA_PATH = DATA_PATH / "glm-10b";
auto const ENC_DEC_DATA_BASE = DATA_PATH / "enc_dec";
auto const ENC_DEC_ENGINE_BASE = TEST_RESOURCE_PATH / "models/enc_dec/trt_engines";

auto constexpr T5_NAME = "t5-small";
auto constexpr BART_NAME = "bart-large-cnn";

ModelSpec getDefaultModelSpec()
{
    static ModelSpec modelSpec{"input_tokens.npy", nvinfer1::DataType::kHALF};
    modelSpec.useGptAttentionPlugin().setKVCacheType(KVCacheType::kPAGED).usePackedInput();

    return modelSpec;
}

auto const FP16_GPT_ATTENTION_PACKED_DIR
    = getDefaultModelSpec().setKVCacheType(KVCacheType::kCONTINUOUS).getModelPath();
auto const FP16_GPT_ATTENTION_PACKED_PAGED_DIR = getDefaultModelSpec().getModelPath();
auto const FP16_GPT_LORA_DIR = getDefaultModelSpec().useLoraPlugin().getModelPath();
auto const FP16_GPT_ATTENTION_PACKED_PAGED_RETURN_ACCEPTED_TOKENS_LOGITS_DIR
    = getDefaultModelSpec().useDraftTokensExternalDecoding().gatherLogits().returnAcceptedTokensLogits().getModelPath();
auto const FP16_GPT_ATTENTION_PACKED_PAGED_GATHER_DIR = getDefaultModelSpec().gatherLogits().getModelPath();
auto const FP16_PLUGIN_PACKED_PAGED_RESULT_FILE = getDefaultModelSpec().getResultsFile();
auto const FP16_PLUGIN_PACKED_PAGED_LONG_RESULT_FILE
    = getDefaultModelSpec().setInputFile("input_tokens_long.npy").getResultsFile();
auto const FP16_PLUGIN_PACKED_PAGED_GATHER_RESULT_FILE = getDefaultModelSpec().gatherLogits().getResultsFile();
auto const FP16_PLUGIN_PACKED_PAGED_GENERATION_LOGITS_FILE
    = getDefaultModelSpec().gatherLogits().getGenerationLogitsFile();
auto const FP16_PLUGIN_PACKED_PAGED_CONTEXT_LOGITS_FILE = getDefaultModelSpec().gatherLogits().getContextLogitsFile();
auto const FP16_PLUGIN_PACKED_PAGED_CUM_LOG_PROBS_FILE = getDefaultModelSpec().getCumLogProbsFile();
auto const FP16_PLUGIN_PACKED_PAGED_GATHER_CUM_LOG_PROBS_FILE
    = getDefaultModelSpec().gatherLogits().getCumLogProbsFile();
auto const FP16_PLUGIN_PACKED_PAGED_LOG_PROBS_FILE = getDefaultModelSpec().getLogProbsFile();
auto const FP16_PLUGIN_PACKED_PAGED_GATHER_LOG_PROBS_FILE = getDefaultModelSpec().gatherLogits().getLogProbsFile();
auto const FP16_PLUGIN_PACKED_PAGED_RESULT_TP1_PP1_FILE = getDefaultModelSpec().getResultsFile();
auto const FP16_PLUGIN_PACKED_PAGED_RESULT_TP4_PP1_FILE
    = getDefaultModelSpec().useTensorParallelism(4).getResultsFile();
auto const FP16_PLUGIN_PACKED_PAGED_RESULT_TP2_PP2_FILE
    = getDefaultModelSpec().useTensorParallelism(2).usePipelineParallelism(2).getResultsFile();
auto const FP16_PLUGIN_PACKED_PAGED_RESULT_TP1_PP4_FILE
    = getDefaultModelSpec().usePipelineParallelism(4).getResultsFile();

auto const LORA_DATA_PATH = DATA_PATH / "lora-test-weights-gpt2-tp1";
auto const LORA_WEIGHTS_FILE = LORA_DATA_PATH / "source.npy";
auto const LORA_CONFIG_FILE = LORA_DATA_PATH / "config.npy";
auto const EXECUTOR_WORKER_PATH
    = std::filesystem::path{TOP_LEVEL_DIR} / "cpp/build/tensorrt_llm/executor_worker/executorWorker";

std::string getEncDecEnginePath(std::string const& modelName, SizeType32 tp, SizeType32 pp)
{
    return modelName + '/' + std::to_string(tp * pp) + "-gpu/float16";
}

TokenIdType getDecTokenFromJsonConfig(std::filesystem::path decEnginePath, std::string const& token_name)
{
    TokenIdType tokenId = 0;
    try
    {
        std::ifstream decoderJsonConfigPath(decEnginePath / "config.json");
        auto const decoderPretrainedConfig
            = nlohmann::json::parse(decoderJsonConfigPath, nullptr, true, true).at("pretrained_config");
        tokenId = decoderPretrainedConfig.at(token_name).template get<int32_t>();
    }
    catch (nlohmann::json::out_of_range& e)
    {
        TLLM_LOG_ERROR(
            "Parameter %s cannot be found from decoder config.json in pretrained_config. Using default id 0.",
            token_name.c_str());
    }
    catch (nlohmann::json::type_error const& e)
    {
        TLLM_LOG_ERROR(
            "Parameter %s has a different type from decoder config.json in pretrained_config. Using default id 0.",
            token_name.c_str());
    }
    return tokenId;
}

TestData loadTestData(BeamResult const& beamResults, ITensor const& givenInput, SizeType32 const maxBeamWidth,
    tr::BufferManager& manager, OutputConfig const& outConfig, ModelIds const& modelIds)
{
    auto const [givenInputLengths, nbGivenInputs, maxInputLength] = getGivenInputLengths(givenInput, modelIds.padId);
    auto const& [beamWidth, resultsFile, contextLogitsFile, genLogitsFile, cumLogProbsFile, logProbsFile] = beamResults;

    TestData testData{nbGivenInputs, beamWidth};
    testData.expectedOutputIds = tr::utils::loadNpy(manager, resultsFile.string(), tr::MemoryType::kCPU);

    auto const& outputShape = testData.expectedOutputIds->getShape();
    EXPECT_EQ(outputShape.nbDims, 2);
    EXPECT_EQ(nbGivenInputs * beamWidth, outputShape.d[0]);
    testData.maxSeqLen = static_cast<SizeType32>(outputShape.d[1]);
    EXPECT_LE(maxInputLength, testData.maxSeqLen);
    EXPECT_LE(beamWidth, maxBeamWidth);

    auto const maxNewTokens = testData.maxSeqLen - maxInputLength;

    testData.endIds.insert(testData.endIds.end(), nbGivenInputs, modelIds.endId);

    if (outConfig.returnContextLogits && beamWidth == 1)
    {
        testData.loadContextLogits(contextLogitsFile, givenInputLengths, manager);
    }
    if (outConfig.returnGenerationLogits && beamWidth == 1)
    {
        testData.loadGenerationLogits(genLogitsFile, manager);
    }
    if (outConfig.returnLogProbs && beamWidth == 1)
    {
        testData.loadLogProbs(cumLogProbsFile, logProbsFile, manager);
    }

    for (SizeType32 inputIdx = 0; inputIdx < nbGivenInputs; ++inputIdx)
    {
        for (SizeType32 beam = 0; beam < beamWidth; ++beam)
        {
            SizeType32 expectedLen = givenInputLengths[inputIdx] + maxNewTokens;
            testData.expectedOutputLengths[inputIdx * beamWidth + beam] = expectedLen;
        }
    }

    return testData;
}

} // namespace

class GptExecutorTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
protected:
    void SetUp() override
    {
        mLogger = std::make_shared<tr::TllmLogger>();
        initTrtLlmPlugins(mLogger.get());
    }

    void TearDown() override {}

    std::shared_ptr<nvinfer1::ILogger> mLogger{};
    SizeType32 mMaxWaitMs = 300000;
    SizeType32 mTrigWarnMs = 10000;
};

void testInvalidCtor(std::filesystem::path const& enginePath, ModelType modelType, ExecutorConfig executorConfig,
    std::string expectedErrMsg = "")
{
    try
    {
        auto executor = Executor(enginePath, modelType, executorConfig);

        FAIL() << "Expected TllmException";
    }
    catch (std::exception const& e)
    {
        EXPECT_THAT(e.what(), testing::HasSubstr(expectedErrMsg));
    }
}

TEST_F(GptExecutorTest, version)
{
    EXPECT_STRNE(kTensorRtLlmVersion, "@TRTLLM_VERSION@");
    EXPECT_STREQ(kTensorRtLlmVersion, version());
}

TEST_F(GptExecutorTest, validCtor)
{
    SizeType32 beamWidth = 1;
    auto executorConfig = ExecutorConfig(beamWidth);
    auto trtEnginePath = (GPT_MODEL_PATH / FP16_GPT_ATTENTION_PACKED_PAGED_DIR / "tp1-pp1-gpu");
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);
}

TEST_F(GptExecutorTest, invalidCtor)
{
    SizeType32 beamWidth = 1;
    auto executorConfig = ExecutorConfig(beamWidth);
    auto trtEnginePath = GPT_MODEL_PATH / FP16_GPT_ATTENTION_PACKED_PAGED_DIR / "tp1-pp1-gpu";
    std::filesystem::path invalidPath{"Bla"};

    // Invalid path
    {
        testInvalidCtor(invalidPath, ModelType::kDECODER_ONLY, executorConfig, "File does not exist");
    }
}

TEST_F(GptExecutorTest, enqueueAfterShutdown)
{
    SizeType32 beamWidth = 1;
    auto executorConfig = ExecutorConfig(beamWidth);
    auto trtEnginePath = (GPT_MODEL_PATH / FP16_GPT_ATTENTION_PACKED_PAGED_DIR / "tp1-pp1-gpu");
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    SizeType32 maxNewTokens = 5;
    VecTokens inputTokens{1, 2, 3, 4};
    auto request = Request(inputTokens, maxNewTokens, false, tensorrt_llm::executor::SamplingConfig(beamWidth));
    auto requestId = executor.enqueueRequest(request);

    bool done = false;
    int iter = 0;
    while (!done && iter < mMaxWaitMs)
    {
        std::chrono::milliseconds waitTime(1);
        auto responses = executor.awaitResponses(requestId, waitTime);
        for (auto& response : responses)
        {
            if (response.hasError())
            {
                FAIL();
            }
            else
            {
                done = response.getResult().isFinal;
            }
        }
        ++iter;
    }
    EXPECT_LT(iter, mMaxWaitMs);

    executor.shutdown();

    EXPECT_FALSE(executor.canEnqueueRequests());

    std::string expErrMsg{"Shutdown called"};
    EXPECT_THAT([&]() { auto reqId = executor.enqueueRequest(request); },
        testing::Throws<tensorrt_llm::common::TllmException>(
            testing::Property(&tensorrt_llm::common::TllmException::what, testing::HasSubstr(expErrMsg))));
    EXPECT_THAT([&]() { auto resp = executor.awaitResponses(); },
        testing::Throws<tensorrt_llm::common::TllmException>(
            testing::Property(&tensorrt_llm::common::TllmException::what, testing::HasSubstr(expErrMsg))));
    EXPECT_THAT([&]() { auto stats = executor.getLatestIterationStats(); },
        testing::Throws<tensorrt_llm::common::TllmException>(
            testing::Property(&tensorrt_llm::common::TllmException::what, testing::HasSubstr(expErrMsg))));
    EXPECT_THAT([&]() { auto stats = executor.getLatestRequestStats(); },
        testing::Throws<tensorrt_llm::common::TllmException>(
            testing::Property(&tensorrt_llm::common::TllmException::what, testing::HasSubstr(expErrMsg))));
    EXPECT_THAT([&]() { executor.cancelRequest(requestId); },
        testing::Throws<tensorrt_llm::common::TllmException>(
            testing::Property(&tensorrt_llm::common::TllmException::what, testing::HasSubstr(expErrMsg))));
}

TEST_F(GptExecutorTest, missingPeftTask)
{
    SizeType32 beamWidth = 1;
    auto executorConfig = ExecutorConfig(beamWidth);
    auto trtEnginePath = GPT_MODEL_PATH / FP16_GPT_LORA_DIR / "tp1-pp1-gpu";
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    // Create the request
    SizeType32 maxNewTokens = 5;
    VecTokens inputTokens{1, 2, 3, 4};
    auto request = Request(inputTokens, maxNewTokens, false, tensorrt_llm::executor::SamplingConfig(beamWidth));
    auto loraConfig = LoraConfig{10};
    request.setLoraConfig(loraConfig);

    auto requestId = executor.enqueueRequest(request);

    bool done = false;
    std::chrono::milliseconds waitTime(mMaxWaitMs);
    auto responses = executor.awaitResponses(requestId, waitTime);
    for (auto& response : responses)
    {
        if (response.hasError())
        {
            auto err = response.getErrorMsg();
            EXPECT_EQ(err, std::string("LoRA task 10 not found in cache. Please send LoRA weights with request"));
            done = true;
        }
        else
        {
            FAIL() << "Expects error due to missing Lora weights";
        }
    }
    EXPECT_TRUE(done);
}

TEST_F(GptExecutorTest, ReturnAcceptedTokenLogits)
{
    SizeType32 constexpr beamWidth{1};
    SizeType32 constexpr vocabSizePadded{50257}; // gpt vocabSizePadded

    // Create executor config
    auto executorConfig = ExecutorConfig(beamWidth);

    // Enable kv cache reuse of executorConfig
    bool enableBlockReuse = true;
    FloatType freeGpuMemoryFraction = 0.5;
    auto kvCacheConfig
        = KvCacheConfig(enableBlockReuse, std::nullopt, std::nullopt, std::nullopt, freeGpuMemoryFraction);
    executorConfig.setKvCacheConfig(kvCacheConfig);

    // Create executor
    auto trtEnginePath
        = (GPT_MODEL_PATH / FP16_GPT_ATTENTION_PACKED_PAGED_RETURN_ACCEPTED_TOKENS_LOGITS_DIR / "tp1-pp1-gpu");
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    // Create request
    SizeType32 maxNewTokens = 5;
    VecTokens inputTokens{1, 2, 3, 4, 5, 6, 7, 8};
    auto request = Request(inputTokens, maxNewTokens, false, tensorrt_llm::executor::SamplingConfig(beamWidth));

    // Set draft tokens
    auto draftTokens = VecTokens{9, 10, 11, 12, 13}; // draft tokens
    auto draftLength = draftTokens.size();
    FloatType const acceptanceThreshold = 0.00001f;  // Ensure the draft token can be accepted
    auto externalDraftTokensConfig = ExternalDraftTokensConfig(draftTokens, std::nullopt, acceptanceThreshold);
    request.setExternalDraftTokensConfig(externalDraftTokensConfig);

    // Set return accepted token logits for this request
    OutputConfig outConfig;
    outConfig.returnGenerationLogits = true;
    request.setOutputConfig(outConfig);

    // Enqueue this request
    auto requestId = executor.enqueueRequest(request);

    bool done = false;
    int iter = 0;
    while (!done && iter < 5000)
    {
        std::chrono::milliseconds waitTime(mMaxWaitMs);
        auto responses = executor.awaitResponses(requestId, waitTime);
        for (auto& response : responses)
        {
            if (response.hasError())
            {
                FAIL();
            }
            else
            {
                auto result = response.getResult();
                done = result.isFinal;
                auto& genLogits = result.generationLogits;
                EXPECT_TRUE(genLogits.has_value());

                // Expected shape: (1, numAcceptedDraftToken, vocabSizePadded)
                auto const& acceptedTokenLogitsShape = genLogits->getShape();
                EXPECT_EQ(acceptedTokenLogitsShape.size(), 3);
                EXPECT_EQ(acceptedTokenLogitsShape[0], 1);
                EXPECT_LE(acceptedTokenLogitsShape[1], draftLength);     // number of accepted tokens
                EXPECT_EQ(acceptedTokenLogitsShape[2], vocabSizePadded); // vocabSizePadded
            }
        }
        ++iter;
    }
}

TEST_F(GptExecutorTest, GenerationLogitsEarlyStop)
{
    SizeType32 constexpr beamWidth{1};
    SizeType32 constexpr vocabSizePadded{50257}; // gpt vocabSizePadded
    auto constexpr streaming = false;

    // Create executor config
    auto executorConfig = ExecutorConfig(beamWidth);

    // Create executor
    auto trtEnginePath = (GPT_MODEL_PATH / FP16_GPT_ATTENTION_PACKED_PAGED_GATHER_DIR / "tp1-pp1-gpu");
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    auto const inputPath = DATA_PATH / "input_tokens.npy";
    ModelIds modelIds{50256, 50256};

    auto manager = tr::BufferManager(std::make_shared<tr::CudaStream>());
    auto const& givenInput = tr::utils::loadNpy(manager, inputPath.string(), tr::MemoryType::kCPU);
    auto [givenInputLengths, nbGivenInputs, maxInputLength] = getGivenInputLengths(*givenInput, modelIds.padId);
    auto const* const givenInputData = tr::bufferCast<TokenIdType const>(*givenInput);

    auto const& inputShape = givenInput->getShape();
    ASSERT_EQ(inputShape.nbDims, 2);
    ASSERT_GT(inputShape.d[0], 0);

    BeamResult beamResult{beamWidth};
    auto const resultsPath
        = GPT_DATA_PATH / ((beamWidth == 1) ? "sampling" : "beam_search_" + std::to_string(beamWidth));
    beamResult.resultsFile = resultsPath / FP16_PLUGIN_PACKED_PAGED_GATHER_RESULT_FILE;
    beamResult.contextLogitsFile = resultsPath / FP16_PLUGIN_PACKED_PAGED_CONTEXT_LOGITS_FILE;
    beamResult.genLogitsFile = resultsPath / FP16_PLUGIN_PACKED_PAGED_GENERATION_LOGITS_FILE;

    // Set return generation logits for this request
    OutputConfig outConfig;
    outConfig.returnGenerationLogits = true;
    outConfig.excludeInputFromOutput = true;

    // Load expected outputs for each beam width value
    auto testData = loadTestData(beamResult, *givenInput, beamWidth, manager, outConfig, modelIds);
    auto const maxSeqLen = testData.maxSeqLen;

    // Load expected outputs and inputs
    std::vector<Request> requests;
    std::vector<SizeType32> reqMaxNewTokens;

    auto constexpr reqIdx = 0;
    SizeType32 inputLen = givenInputLengths.at(reqIdx);
    auto maxNewTokens = maxSeqLen - maxInputLength;
    reqMaxNewTokens.push_back(maxNewTokens);
    auto const* const seqBegin = givenInputData + reqIdx * maxInputLength;

    auto request = Request(VecTokens(seqBegin, seqBegin + inputLen), maxNewTokens, streaming,
        tensorrt_llm::executor::SamplingConfig(beamWidth), outConfig, modelIds.endId);
    // copy request
    auto request2 = request;

    auto const expectedOutputData = tr::BufferRange<TokenIdType const>(*testData.expectedOutputIds);
    auto const expectedOutputLengths = testData.expectedOutputLengths;
    auto const endPos = expectedOutputLengths[reqIdx] - 3;
    auto const endIndex = tc::flat_index3(reqIdx, beamWidth - 1, endPos, beamWidth, maxSeqLen);
    auto const endToken = expectedOutputData[endIndex];

    // Set end id to stop early
    request.setEndId(endToken);
    requests.emplace_back(std::move(request));

    // Set stop words to stop early
    request2.setStopWords({{endToken}});
    requests.emplace_back(std::move(request2));

    // Enqueue requests
    auto requestIds = executor.enqueueRequests(requests);

    std::map<IdType, SizeType32> expectedNewTokens;
    expectedNewTokens[requestIds.at(0)] = endPos - inputLen;
    expectedNewTokens[requestIds.at(1)] = endPos - inputLen + 1;

    std::map<IdType, bool> done;
    std::for_each(requestIds.begin(), requestIds.end(), [&done](auto id) { done[id] = false; });
    int iter = 0;
    while (!(std::all_of(done.begin(), done.end(), [](auto x) { return x.second; })) && iter < 5000)
    {
        std::chrono::milliseconds waitTime(mMaxWaitMs);
        auto responses = executor.awaitResponses(waitTime);
        for (auto& response : responses)
        {
            if (response.hasError())
            {
                FAIL();
            }
            else
            {
                auto const reqId = response.getRequestId();
                auto const& result = response.getResult();
                EXPECT_TRUE(result.isFinal);
                done.at(reqId) = result.isFinal;

                // only 1 beam
                auto const& outputIds = result.outputTokenIds.at(0);
                EXPECT_EQ(outputIds.size(), expectedNewTokens.at(reqId)) << "req " << reqId;

                auto const& genLogits = result.generationLogits;
                EXPECT_TRUE(genLogits.has_value());

                // Expected shape: (1, numAcceptedDraftToken, vocabSizePadded)
                auto const& generationLogitsShape = genLogits->getShape();
                EXPECT_EQ(generationLogitsShape.size(), 3);
                EXPECT_EQ(generationLogitsShape[0], 1);
                EXPECT_LE(generationLogitsShape[1], maxNewTokens);
                EXPECT_EQ(generationLogitsShape[2], vocabSizePadded);

                auto const genLogitsTensor = detail::toITensor(*genLogits);
                genLogitsTensor->squeeze(0); // only 1 beam

                for (size_t outputIdx = 0; outputIdx < expectedNewTokens.at(reqId); ++outputIdx)
                {
                    // logits argmax should be equal to tokenId
                    auto const genLogitsSlice = tr::ITensor::slice(genLogitsTensor, outputIdx, 1);
                    auto const genLogitsRange = tr::BufferRange<float>(*genLogitsSlice);
                    auto const* maxPos = std::max_element(genLogitsRange.begin(), genLogitsRange.end());
                    auto const maxIdx = std::distance(genLogitsRange.begin(), maxPos);

                    auto const tokenId = outputIds.at(outputIdx);
                    EXPECT_EQ(tokenId, maxIdx) << "req " << reqId << " outputIdx " << outputIdx;
                }
            }
        }
        ++iter;
    }
}

using ParamType = std::tuple<bool, bool, int>;
using ParamCancelReqType = std::tuple<bool, bool, int, std::string>;
using ParamStatsType = std::tuple<int, bool>;
using AllParamsType = std::tuple<BatchingType, bool, int, bool, bool, bool, bool, std::string, bool>;
using EncDecParamsType = std::tuple<std::string, SizeType32, SizeType32, SizeType32, SizeType32>;

// requestSize, beam_width

std::string generateTestName(testing::TestParamInfo<ParamType> const& info)
{
    auto const streaming = std::get<0>(info.param);
    auto const excludeInputFromOutput = std::get<1>(info.param);
    auto const beamWidth = std::get<2>(info.param);
    std::string name = "ExecutorTest";
    if (streaming)
    {
        name += "Streaming";
    }
    if (excludeInputFromOutput)
    {
        name += "ExclInput";
    }
    name.append("BW" + std::to_string(beamWidth));
    return name;
}

std::string generateTestNameCancelReq(testing::TestParamInfo<ParamCancelReqType> const& info)
{
    auto const streaming = std::get<0>(info.param);
    auto const& useOrchestratorMode = std::get<1>(info.param);
    int const beamWidth = std::get<2>(info.param);
    auto const modelName = std::get<3>(info.param);
    std::string name = "ExecutorTest";
    if (streaming)
    {
        name += "Streaming";
    }

    name.append("BW" + std::to_string(beamWidth));
    name.append("_" + modelName + "_");

    if (useOrchestratorMode)
    {
        name.append("OrchMode");
    }
    else
    {
        name.append("LeaderMode");
    }
    return name;
}

std::string generateTestNameStats(testing::TestParamInfo<ParamStatsType> const& info)
{
    int iterStatsMaxIterations = std::get<0>(info.param);
    auto const& useOrchestratorMode = std::get<1>(info.param);
    std::string name = "ExecutorTest_";
    name.append(std::to_string(iterStatsMaxIterations) + "_");
    if (useOrchestratorMode)
    {
        name.append("OrchMode");
    }
    else
    {
        name.append("LeaderMode");
    }
    return name;
}

std::string generateTestNameAllParams(testing::TestParamInfo<AllParamsType> const& info)
{
    auto const batchingType = std::get<0>(info.param);
    auto const streaming = std::get<1>(info.param);
    auto const& beamWidth = std::get<2>(info.param);
    auto const& computeLogProbs = std::get<3>(info.param);
    auto const& excludeInputInOutput = std::get<4>(info.param);
    auto const& returnContextLogits = std::get<5>(info.param);
    auto const& returnGenerationLogits = std::get<6>(info.param);
    auto const modelName = std::get<7>(info.param);
    auto const& useOrchestratorMode = std::get<8>(info.param);

    std::string name = "ExecutorTest_";

    switch (batchingType)
    {
    case BatchingType::kSTATIC: name.append("Static"); break;
    case BatchingType::kINFLIGHT: name.append("Ifb"); break;
    default: name.append("DefaultModel"); break;
    }

    if (streaming)
    {
        name += "Streaming";
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
    name.append("_" + modelName + "_");
    if (useOrchestratorMode)
    {
        name.append("OrchMode");
    }
    else
    {
        name.append("LeaderMode");
    }
    return name;
}

std::string generateTestNameEncDec(testing::TestParamInfo<EncDecParamsType> const& info)
{
    auto modelName = std::get<0>(info.param);
    auto const beamWidth = std::get<1>(info.param);
    auto const maxNewTokens = std::get<2>(info.param);
    auto const tp = std::get<3>(info.param);
    auto const pp = std::get<4>(info.param);

    // GTEST does not allow '-' in its test name
    for (auto& c : modelName)
    {
        if (c == '-')
        {
            c = '_';
        }
    }

    std::string name = "EncDecTest";
    name.append("_" + modelName);
    name.append("_BeamWidth" + std::to_string(beamWidth));
    name.append("_MaxNewTokens" + std::to_string(maxNewTokens));
    name.append("_TP" + std::to_string(tp));
    name.append("_PP" + std::to_string(pp));
    return name;
}

class ParamTest : public GptExecutorTest, public ::testing::WithParamInterface<ParamType>
{
};

class ParamStatsTest : public GptExecutorTest, public ::testing::WithParamInterface<ParamStatsType>
{
};

class AllParamsTest : public GptExecutorTest, public ::testing::WithParamInterface<AllParamsType>
{
};

class ParamCancelReqTest : public GptExecutorTest, public ::testing::WithParamInterface<ParamCancelReqType>
{
};

class EncDecParamsTest : public GptExecutorTest, public ::testing::WithParamInterface<EncDecParamsType>
{
};

TEST_F(GptExecutorTest, GetLatestStats)
{
    bool streaming = false;
    bool excludeInputFromOutput = false;
    OutputConfig outConfig;
    outConfig.excludeInputFromOutput = excludeInputFromOutput;

    SizeType32 beamWidth = 1;
    auto executorConfig = ExecutorConfig(beamWidth);
    auto trtEnginePath = GPT_MODEL_PATH / FP16_GPT_ATTENTION_PACKED_PAGED_DIR / "tp1-pp1-gpu";
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    // Create the request
    SizeType32 maxNewTokens = 5;
    VecTokens inputTokens{1, 2, 3, 4};
    auto request
        = Request(inputTokens, maxNewTokens, streaming, tensorrt_llm::executor::SamplingConfig(beamWidth), outConfig);
    auto requestId = executor.enqueueRequest(std::move(request));

    bool done = false;
    int iter = 0;
    while (!done && iter < mMaxWaitMs)
    {
        std::chrono::milliseconds waitTime(1);
        auto responses = executor.awaitResponses(requestId, waitTime);
        for (auto& response : responses)
        {
            if (response.hasError())
            {
                FAIL();
            }
            else
            {
                done = response.getResult().isFinal;
            }
        }
        ++iter;
    }
    EXPECT_LT(iter, mMaxWaitMs);

    // Expect 5 non-empty iterations
    auto stats = executor.getLatestIterationStats();
    EXPECT_EQ(stats.size(), 5);
    uint64_t currentIter = 0;
    for (auto const& stat : stats)
    {
        EXPECT_EQ(stat.timestamp.size(), 26);
        EXPECT_EQ(stat.iter, currentIter);
        EXPECT_EQ(stat.numActiveRequests, 1);
        EXPECT_EQ(stat.maxNumActiveRequests, 64);
        // Very loose check to make sure the memory stats are valid
        EXPECT_GT(stat.gpuMemUsage, 16);
        EXPECT_GT(stat.cpuMemUsage, 16);
        EXPECT_GT(stat.pinnedMemUsage, 16);

        // Stats for KV cache
        EXPECT_TRUE(stat.kvCacheStats.has_value());
        KvCacheStats const& kvStats = stat.kvCacheStats.value();
        EXPECT_GT(kvStats.maxNumBlocks, 0);
        EXPECT_GT(kvStats.freeNumBlocks, 0);
        EXPECT_EQ(kvStats.usedNumBlocks, currentIter == maxNewTokens ? 0 : 1);
        EXPECT_GT(kvStats.tokensPerBlock, 0);
        EXPECT_GT(kvStats.allocTotalBlocks, 0);
        EXPECT_GT(kvStats.allocNewBlocks, 0);
        EXPECT_GE(kvStats.reusedBlocks, 0);

        // Stats for inflight batching
        EXPECT_TRUE(stat.inflightBatchingStats.has_value() && !stat.staticBatchingStats.has_value());
        InflightBatchingStats const& modelStats = stat.inflightBatchingStats.value();
        EXPECT_EQ(modelStats.numScheduledRequests, currentIter == maxNewTokens ? 0 : 1);
        EXPECT_EQ(modelStats.numContextRequests, currentIter == 0 ? 1 : 0);
        EXPECT_EQ(modelStats.numGenRequests, currentIter == 0 || currentIter == maxNewTokens ? 0 : 1);
        EXPECT_EQ(modelStats.numPausedRequests, 0);
        EXPECT_EQ(modelStats.numCtxTokens, currentIter == 0 ? inputTokens.size() : 0);
        EXPECT_EQ(modelStats.microBatchId, 0);
        EXPECT_NEAR(modelStats.avgNumDecodedTokensPerIter, currentIter == 0 ? 0.f : 1.f, 1e-9f);

        auto jsonStr = JsonSerialization::toJsonStr(stat);
        EXPECT_THAT(jsonStr, testing::HasSubstr("\"iter\":" + std::to_string(currentIter)));
        EXPECT_THAT(jsonStr, testing::HasSubstr("\"staticBatchingStats\":null"));
        EXPECT_THAT(jsonStr, testing::HasSubstr("\"numCtxTokens\":" + std::to_string(modelStats.numCtxTokens)));
        EXPECT_THAT(jsonStr, testing::HasSubstr("\"numGenRequests\":" + std::to_string(modelStats.numGenRequests)));

        ++currentIter;
    }
}

TEST_F(GptExecutorTest, GetLatestRequestStats)
{
    bool streaming = false;
    bool excludeInputFromOutput = false;
    OutputConfig outConfig;
    outConfig.excludeInputFromOutput = excludeInputFromOutput;

    SizeType32 beamWidth = 1;
    auto executorConfig = ExecutorConfig(beamWidth);
    executorConfig.setRequestStatsMaxIterations(1000);
    executorConfig.setEnableChunkedContext(true);
    auto trtEnginePath = GPT_MODEL_PATH / FP16_GPT_ATTENTION_PACKED_PAGED_DIR / "tp1-pp1-gpu";
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    // Create the requests
    std::vector<std::pair<SizeType32, VecTokens>> requestParams = {
        // {maxNewTokens, inputTokens}
        {5, {1, 2, 3, 4}}, {4, {1, 1, 2, 3, 5}}, {1, {1}},
        {8, VecTokens(383, 1)} // Long enough to be chunked into multiple iterations
    };
    std::vector<Request> requests;
    for (auto requestParam : requestParams)
    {
        requests.emplace_back(requestParam.second, requestParam.first, streaming,
            tensorrt_llm::executor::SamplingConfig(beamWidth), outConfig);
    }
    auto requestIdsVec = executor.enqueueRequests(std::move(requests));
    std::map<IdType, SizeType32> requestIdToIndex;
    std::set<IdType> activeRequests;
    for (SizeType32 i = 0; i < requestIdsVec.size(); ++i)
    {
        auto requestId = requestIdsVec[i];
        activeRequests.insert(requestId);
        requestIdToIndex[requestId] = i;
    }

    int iter = 0;
    while (!activeRequests.empty() && iter < mMaxWaitMs)
    {
        for (auto i = activeRequests.begin(); i != activeRequests.end();)
        {
            auto requestId = *i;
            bool thisDone = false;
            std::chrono::milliseconds waitTime(1);
            auto responses = executor.awaitResponses(requestId, waitTime);
            for (auto& response : responses)
            {
                if (response.hasError())
                {
                    // Allow response with error only if awaitResponse processed a terminated request id
                    std::string err = "ReqId " + std::to_string(response.getRequestId())
                        + " has already been processed and was terminated.";
                    EXPECT_EQ(response.getErrorMsg(), err);
                }
                else
                {
                    thisDone = response.getResult().isFinal;
                }
            }
            if (thisDone)
            {
                // Erase completed request and move to the next one
                i = activeRequests.erase(i);
            }
            else
            {
                ++i;
            }
        }
        ++iter;
    }
    EXPECT_LT(iter, mMaxWaitMs);

    // Expect 5 non-empty iterations
    // Note: The 6th iteration with the last finished request will not be reported
    //       because execution will stop at getNewReqWithIds due to numActiveRequests == 0.
    auto stats = executor.getLatestRequestStats();
    EXPECT_EQ(stats.size(), 5);
    SizeType32 currentIter = 0;
    auto invalidStart = std::numeric_limits<SizeType32>::max();
    std::vector<SizeType32> genStart(requestParams.size(), invalidStart); // The iteration index when generation started
    std::set<IdType> completedRequests;
    for (auto const& stat : stats)
    {
        auto jsonStrIter = JsonSerialization::toJsonStr(stat);
        EXPECT_EQ(stat.iter, currentIter);
        EXPECT_THAT(jsonStrIter, testing::HasSubstr("\"iter\":" + std::to_string(currentIter)));
        EXPECT_EQ(stat.requestStats.size() + completedRequests.size(), requestParams.size());
        for (auto rStat : stat.requestStats)
        {
            auto jsonStr = JsonSerialization::toJsonStr(rStat);
            // Only a few requests here so all of them should be scheduled. A separate test
            // GetLatestRequestStatsScheduling will target the scheduling stats.
            if (rStat.stage != RequestStage::kGENERATION_COMPLETE)
            {
                EXPECT_TRUE(rStat.scheduled);
                EXPECT_THAT(jsonStr, testing::HasSubstr("\"scheduled\":true"));
            }
            EXPECT_TRUE(!rStat.paused);
            EXPECT_THAT(jsonStr, testing::HasSubstr("\"paused\":false"));
            EXPECT_TRUE(requestIdToIndex.count(rStat.id));
            EXPECT_THAT(jsonStr, testing::HasSubstr("\"id\":" + std::to_string(rStat.id)));
            auto requestIndex = requestIdToIndex[rStat.id];
            auto contextSize = requestParams[requestIndex].second.size();
            if (rStat.contextPrefillPosition == contextSize) // Check generation phase
            {
                bool firstIteration{false};
                // Context phase is done
                EXPECT_TRUE(rStat.stage == RequestStage::kGENERATION_IN_PROGRESS
                    || rStat.stage == RequestStage::kGENERATION_COMPLETE);
                EXPECT_THAT(jsonStr, testing::HasSubstr("\"stage\":\"GENERATION"));
                if (genStart[requestIndex] == invalidStart)
                {
                    // Just started generation
                    genStart[requestIndex] = currentIter;
                    firstIteration = true;
                }

                // One token per iteration
                EXPECT_TRUE(currentIter - genStart[requestIndex] == rStat.numGeneratedTokens);
                EXPECT_NEAR(rStat.avgNumDecodedTokensPerIter, firstIteration ? 0.f : 1.0f, 1e-9);
                if (rStat.stage == RequestStage::kGENERATION_COMPLETE)
                {
                    EXPECT_TRUE(requestParams[requestIndex].first >= rStat.numGeneratedTokens);
                    completedRequests.insert(requestIndex);
                }
                else
                {
                    EXPECT_FALSE(completedRequests.count(requestIndex));
                }
            }
            else if (rStat.contextPrefillPosition < contextSize) // Check context phase
            {
                // Must be chunked
                SizeType32 const maxChunkSize = 128;
                EXPECT_TRUE(rStat.contextPrefillPosition % maxChunkSize == 0);
                // Context phase is on-going
                EXPECT_TRUE(rStat.stage == RequestStage::kCONTEXT_IN_PROGRESS);
                // No tokens are generated
                EXPECT_TRUE(0 == rStat.numGeneratedTokens);
            }
            else
            {
                FAIL() << "Out-of-boundary contextPrefillPosition in stats: " << rStat.contextPrefillPosition
                       << " out of " << contextSize;
            }
        }
        ++currentIter;
    }
    // We should have visited all requests.
    // Take into consideration the last request has not been reported
    EXPECT_EQ(completedRequests.size() + 1, requestParams.size());
}

TEST_F(GptExecutorTest, GetLatestRequestStatsScheduling)
{
    // Specifically test the case where there are too many requests to be scheduled for a iteration
    bool streaming = false;
    bool excludeInputFromOutput = false;
    OutputConfig outConfig;
    outConfig.excludeInputFromOutput = excludeInputFromOutput;

    SizeType32 beamWidth = 1;
    auto executorConfig = ExecutorConfig(beamWidth);
    executorConfig.setRequestStatsMaxIterations(1000);
    executorConfig.setEnableChunkedContext(true);
    auto trtEnginePath = GPT_MODEL_PATH / FP16_GPT_ATTENTION_PACKED_PAGED_DIR / "tp1-pp1-gpu";
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    // Create 100 requests. Note the max batch size for this model is 64 so some requests won't be scheduled right away.
    std::vector<std::pair<SizeType32, VecTokens>> requestParams(100, {5, {1, 2, 3, 4}});
    std::vector<Request> requests;
    requests.reserve(requestParams.size());
    for (auto requestParam : requestParams)
    {
        requests.emplace_back(requestParam.second, requestParam.first, streaming,
            tensorrt_llm::executor::SamplingConfig(beamWidth), outConfig);
    }
    auto requestIdsVec = executor.enqueueRequests(std::move(requests));
    std::map<IdType, SizeType32> requestIdToIndex;
    std::set<IdType> activeRequests;
    for (SizeType32 i = 0; i < requestIdsVec.size(); ++i)
    {
        auto requestId = requestIdsVec[i];
        activeRequests.insert(requestId);
        requestIdToIndex[requestId] = i;
    }

    int iter = 0;
    while (!activeRequests.empty() && iter < mMaxWaitMs)
    {
        for (auto i = activeRequests.begin(); i != activeRequests.end();)
        {
            auto requestId = *i;
            bool thisDone = false;
            std::chrono::milliseconds waitTime(1);
            auto responses = executor.awaitResponses(requestId, waitTime);
            for (auto& response : responses)
            {
                if (response.hasError())
                {
                    // Allow response with error only if awaitResponse processed a terminated request id
                    std::string err = "ReqId " + std::to_string(response.getRequestId())
                        + " has already been processed and was terminated.";
                    EXPECT_EQ(response.getErrorMsg(), err);
                }
                else
                {
                    thisDone = response.getResult().isFinal;
                }
            }
            if (thisDone)
            {
                // Erase completed request and move to the next one
                i = activeRequests.erase(i);
            }
            else
            {
                ++i;
            }
        }
        ++iter;
    }
    EXPECT_LT(iter, mMaxWaitMs);

    auto stats = executor.getLatestRequestStats();
    SizeType32 numFinished = 0;
    SizeType32 const maxActiveSize = 64; // Decided by the model
    for (auto const& stat : stats)
    {
        SizeType32 numReqs = 0;
        SizeType32 numReqsActive = 0;
        SizeType32 numReqsQueued = 0;
        SizeType32 numReqsJustDone = 0;
        for (auto rStat : stat.requestStats)
        {
            ++numReqs;
            numReqsActive += rStat.scheduled ? 1 : 0;
            numReqsQueued += rStat.stage == RequestStage::kQUEUED ? 1 : 0;
            numReqsJustDone += rStat.stage == RequestStage::kGENERATION_COMPLETE ? 1 : 0;
        }
        EXPECT_EQ(numReqs, numReqsActive + numReqsQueued + numReqsJustDone);
        EXPECT_EQ(numReqs + numFinished, requestParams.size()); // Should report all unfinished requests
        EXPECT_TRUE(numReqsActive <= maxActiveSize); // Not all requests are active due to max active size limit.
        numFinished += numReqsJustDone;
    }
}

TEST_P(ParamTest, SingleRequestDemo)
{
    bool const streaming = std::get<0>(GetParam());
    bool const excludeInputFromOutput = std::get<1>(GetParam());
    auto const beamWidth = std::get<2>(GetParam());
    OutputConfig outConfig;
    outConfig.excludeInputFromOutput = excludeInputFromOutput;

    auto executorConfig = ExecutorConfig(beamWidth);
    auto trtEnginePath = GPT_MODEL_PATH / FP16_GPT_ATTENTION_PACKED_PAGED_DIR / "tp1-pp1-gpu";
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    // Create the request
    SizeType32 maxNewTokens = 5;
    VecTokens inputTokens{1, 2, 3, 4};
    auto request
        = Request(inputTokens, maxNewTokens, streaming, tensorrt_llm::executor::SamplingConfig(beamWidth), outConfig);

    // Enqueue the request
    auto requestId = executor.enqueueRequest(std::move(request));

    // Get the new tokens
    VecTokens tokens;
    SizeType32 numResponses{0};
    bool done = false;
    int iter = 0;
    std::chrono::milliseconds waitTime(1);
    while (!done && iter < mMaxWaitMs)
    {
        auto responses = executor.awaitResponses(requestId, waitTime);
        for (auto& response : responses)
        {
            ++numResponses;
            if (response.hasError())
            {
                // This request failed for some reason, get error msg
                std::string errStr
                    = "Request id " + std::to_string(requestId) + " failed with err " + response.getErrorMsg();
                FAIL();
            }
            else
            {
                auto result = response.getResult();
                done = result.isFinal;
                auto& newTokens = result.outputTokenIds.at(beamWidth - 1);
                auto const expectedSize = streaming
                    ? (beamWidth > 1 ? numResponses : 1)
                    : (maxNewTokens + (excludeInputFromOutput ? 0 : inputTokens.size()));
                EXPECT_EQ(newTokens.size(), expectedSize);

                if (streaming && beamWidth > 1)
                {
                    // replace tokens
                    tokens = newTokens;
                }
                else
                {
                    // Append tokens
                    tokens.insert(tokens.end(), newTokens.begin(), newTokens.end());
                }
            }
        }
        ++iter;
    }
    EXPECT_LT(iter, mMaxWaitMs);
    EXPECT_EQ(numResponses, streaming ? maxNewTokens : 1);
    EXPECT_EQ(
        tokens.size(), streaming ? maxNewTokens : (excludeInputFromOutput ? 0 : inputTokens.size()) + maxNewTokens);

    // Expect awaitResponse to return error message because the request is already terminated (isFinal = True)
    auto response = executor.awaitResponses(requestId, waitTime).at(0);
    EXPECT_TRUE(response.hasError());
    std::string err
        = "ReqId " + std::to_string(response.getRequestId()) + " has already been processed and was terminated.";
    EXPECT_EQ(response.getErrorMsg(), err);
}

TEST_P(ParamTest, MultipleRequestDemo)
{
    bool const streaming = std::get<0>(GetParam());
    bool const excludeInputFromOutput = std::get<1>(GetParam());
    auto const beamWidth = std::get<2>(GetParam());
    OutputConfig outConfig;
    outConfig.excludeInputFromOutput = excludeInputFromOutput;
    SizeType32 numRequests = 20;

    auto executorConfig = ExecutorConfig(beamWidth);
    auto trtEnginePath = GPT_MODEL_PATH / FP16_GPT_ATTENTION_PACKED_PAGED_DIR / "tp1-pp1-gpu";
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    SizeType32 maxPromptLen = 20;
    SizeType32 maxMaxNewTokens = 20;

    SizeType32 endId = -1;
    // Enqueue the requests
    std::unordered_map<IdType, VecTokens> tokens;
    std::unordered_map<IdType, SizeType32> expectedNumTokens;
    std::unordered_map<IdType, SizeType32> expectedNumResponses;
    for (SizeType32 req = 0; req < numRequests; ++req)
    {
        SizeType32 promptLen = rand() % maxPromptLen + 1;
        SizeType32 maxNewTokens = rand() % maxMaxNewTokens + 1;

        auto request = Request(VecTokens(promptLen, 1), maxNewTokens, streaming,
            tensorrt_llm::executor::SamplingConfig(beamWidth), outConfig, endId);
        auto reqId = executor.enqueueRequest(std::move(request));
        tokens[reqId] = {};
        expectedNumTokens[reqId] = ((streaming || excludeInputFromOutput) ? 0 : promptLen) + maxNewTokens;
        expectedNumResponses[reqId] = streaming ? maxNewTokens : 1;
    }

    // Get the new tokens for each requests
    int32_t numFinished = 0;
    int iter = 0;
    std::unordered_map<IdType, SizeType32> numResponses;
    while (numFinished < numRequests && iter < mMaxWaitMs)
    {
        std::chrono::milliseconds waitTime(1);
        auto responses = executor.awaitResponses(waitTime);
        for (auto& response : responses)
        {
            auto reqId = response.getRequestId();
            ++numResponses[reqId];
            if (!response.hasError())
            {
                auto result = response.getResult();
                numFinished += result.isFinal;
                auto& newTokens = result.outputTokenIds.at(beamWidth - 1);
                auto const expectedSize
                    = streaming ? (beamWidth > 1 ? numResponses[reqId] : 1) : expectedNumTokens[reqId];
                EXPECT_EQ(newTokens.size(), expectedSize);

                auto& reqTokens = tokens.at(response.getRequestId());
                if (streaming && beamWidth > 1)
                {
                    reqTokens = newTokens;
                }
                else
                {
                    reqTokens.insert(reqTokens.end(), newTokens.begin(), newTokens.end());
                }
            }
            else
            {
                // Allow response with error only if awaitResponse processed a terminated request id
                std::string err = "ReqId " + std::to_string(response.getRequestId())
                    + " has already been processed and was terminated.";
                EXPECT_EQ(response.getErrorMsg(), err);
            }
        }
        ++iter;
    }
    EXPECT_LT(iter, mMaxWaitMs);

    // Check that number of tokens matches expectations
    for (auto const& [reqId, numTokens] : expectedNumTokens)
    {
        EXPECT_EQ(expectedNumResponses[reqId], numResponses[reqId]) << "reqId " << reqId;
        EXPECT_EQ(expectedNumTokens[reqId], tokens[reqId].size()) << "reqId " << reqId;
    }
}

TEST_P(ParamStatsTest, MultipleRequestStats)
{
    bool streaming = false;
    bool excludeInputFromOutput = false;
    OutputConfig outConfig;
    outConfig.excludeInputFromOutput = excludeInputFromOutput;
    SizeType32 numRequests = 100;
    auto iterStatsMaxIterations = std::get<0>(GetParam());
    bool useOrchestratorMode = std::get<1>(GetParam());

    SizeType32 beamWidth = 1;
    auto executorConfig = ExecutorConfig(beamWidth);
    executorConfig.setIterStatsMaxIterations(iterStatsMaxIterations);
    auto trtEnginePath = GPT_MODEL_PATH / FP16_GPT_ATTENTION_PACKED_PAGED_DIR / "tp1-pp1-gpu";

    std::optional<OrchestratorConfig> orchestratorConfig = std::nullopt;
    if (useOrchestratorMode)
    {
        orchestratorConfig = OrchestratorConfig(true, EXECUTOR_WORKER_PATH.string());
    }
    auto parallelConfig = ParallelConfig(CommunicationType::kMPI,
        useOrchestratorMode ? CommunicationMode::kORCHESTRATOR : CommunicationMode::kLEADER, std::nullopt, std::nullopt,
        orchestratorConfig);
    executorConfig.setParallelConfig(parallelConfig);

    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    SizeType32 maxPromptLen = 20;
    SizeType32 maxMaxNewTokens = 20;

    SizeType32 endId = -1;
    // Enqueue the requests
    std::unordered_map<IdType, VecTokens> tokens;
    std::unordered_map<IdType, SizeType32> expectedNumTokens;
    for (SizeType32 req = 0; req < numRequests; ++req)
    {
        SizeType32 promptLen = rand() % maxPromptLen + 1;
        SizeType32 maxNewTokens = rand() % maxMaxNewTokens + 1;

        auto request = Request(VecTokens(promptLen, 1), maxNewTokens, streaming,
            tensorrt_llm::executor::SamplingConfig(beamWidth), outConfig, endId);
        auto reqId = executor.enqueueRequest(std::move(request));
        tokens[reqId] = {};
        expectedNumTokens[reqId] = (streaming ? 0 : (excludeInputFromOutput ? 0 : promptLen)) + maxNewTokens;
    }

    std::atomic<bool> statsThreadDone = false;
    std::atomic<int32_t> numFinished = 0;
    std::deque<IterationStats> iterStatsReceived;
    // Spawn a thread that continuously get stats
    auto statsThread = std::thread(
        [&executor, &numFinished, numRequests, &iterStatsReceived, &statsThreadDone]()
        {
            while (numFinished < numRequests)
            {
                auto reqStats = executor.getLatestIterationStats();
                iterStatsReceived.insert(iterStatsReceived.end(), std::make_move_iterator(reqStats.begin()),
                    std::make_move_iterator(reqStats.end()));
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            statsThreadDone = true;
        });

    // Get the new tokens for each requests
    int iter = 0;
    SizeType32 numResponses = 0;
    while (numFinished < numRequests && iter < mMaxWaitMs)
    {
        std::chrono::milliseconds waitTime(1);
        auto responses = executor.awaitResponses(waitTime);
        for (auto& response : responses)
        {
            numResponses++;
            if (!response.hasError())
            {
                auto result = response.getResult();
                numFinished += result.isFinal;
                auto& newTokens = result.outputTokenIds.at(beamWidth - 1);
                auto& reqTokens = tokens.at(response.getRequestId());
                reqTokens.insert(reqTokens.end(), std::make_move_iterator(newTokens.begin()),
                    std::make_move_iterator(newTokens.end()));
            }
            else
            {
                // Allow response with error only if awaitResponse processed a terminated request id
                std::string err = "ReqId " + std::to_string(response.getRequestId())
                    + " has already been processed and was terminated.";
                EXPECT_EQ(response.getErrorMsg(), err);
            }
        }
        ++iter;
    }
    EXPECT_LT(iter, mMaxWaitMs);

    // Check that number of tokens matches expectations
    for (auto const& [reqId, numTokens] : expectedNumTokens)
    {
        EXPECT_EQ(expectedNumTokens[reqId], tokens[reqId].size()) << "reqId " << reqId;
    }

    // Wait for stats thread to be done, fail otherwise
    iter = 0;
    while (!statsThreadDone && iter < mMaxWaitMs)
    {
        std::chrono::milliseconds waitTime(1);
        std::this_thread::sleep_for(std::chrono::milliseconds(waitTime));
        iter++;
    }
    ASSERT_TRUE(statsThreadDone);
    if (iterStatsMaxIterations > 0)
    {
        ASSERT_GT(iterStatsReceived.size(), 1);

        for (auto stats : iterStatsReceived)
        {
            EXPECT_GT(stats.numActiveRequests, 0);
            TLLM_LOG_INFO("%d %d", stats.iter, stats.numActiveRequests);

            EXPECT_TRUE(stats.inflightBatchingStats.has_value());
            if (stats.inflightBatchingStats.has_value())
            {
                EXPECT_GT(stats.inflightBatchingStats.value().numScheduledRequests, 0);
            }
        }
    }

    statsThread.join();
}

TEST_P(ParamTest, MultipleRequestBatchResponses)
{
    bool const streaming = std::get<0>(GetParam());
    bool const excludeInputFromOutput = std::get<1>(GetParam());
    auto const beamWidth = std::get<2>(GetParam());
    OutputConfig outConfig;
    outConfig.excludeInputFromOutput = excludeInputFromOutput;
    SizeType32 constexpr numRequests{20};

    auto executorConfig = ExecutorConfig(beamWidth);
    auto trtEnginePath = GPT_MODEL_PATH / FP16_GPT_ATTENTION_PACKED_PAGED_DIR / "tp1-pp1-gpu";
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    SizeType32 constexpr maxPromptLen{20};
    SizeType32 constexpr maxMaxNewTokens{20};

    SizeType32 endId = -1;
    // Enqueue the requests
    std::unordered_map<IdType, VecTokens> tokens;
    std::unordered_map<IdType, SizeType32> expectedNumTokens;
    std::vector<IdType> requestIds;
    for (SizeType32 req = 0; req < numRequests; ++req)
    {
        SizeType32 promptLen = rand() % maxPromptLen + 1;
        SizeType32 maxNewTokens = rand() % maxMaxNewTokens + 1;

        auto request = Request(VecTokens(promptLen, 1), maxNewTokens, streaming,
            tensorrt_llm::executor::SamplingConfig(beamWidth), outConfig, endId);
        auto reqId = executor.enqueueRequest(std::move(request));
        requestIds.push_back(reqId);
        tokens[reqId] = {};
        expectedNumTokens[reqId] = (streaming ? 0 : (excludeInputFromOutput ? 0 : promptLen)) + maxNewTokens;
    }

    // Get the new tokens for each requests
    int32_t numFinished = 0;
    int iter = 0;
    SizeType32 numResponses = 0;
    std::chrono::milliseconds waitTime(1);
    while (numFinished < numRequests && iter < mMaxWaitMs)
    {
        auto idResponses = executor.awaitResponses(requestIds, waitTime);
        for (unsigned i = 0; i < requestIds.size(); ++i)
        {
            auto& responses = idResponses[i];
            for (auto& response : responses)
            {
                numResponses++;
                if (!response.hasError())
                {
                    auto result = response.getResult();
                    numFinished += result.isFinal;
                    auto& newTokens = result.outputTokenIds.at(beamWidth - 1);
                    auto& reqTokens = tokens.at(response.getRequestId());
                    if (streaming && beamWidth > 1)
                    {
                        reqTokens = newTokens;
                    }
                    else
                    {
                        reqTokens.insert(reqTokens.end(), newTokens.begin(), newTokens.end());
                    }
                }
                else
                {
                    // Allow response with error only if awaitResponse processed a terminated request id
                    std::string err = "ReqId " + std::to_string(response.getRequestId())
                        + " has already been processed and was terminated.";
                    EXPECT_EQ(response.getErrorMsg(), err);
                }
            }
        }
        ++iter;
    }
    EXPECT_LT(iter, mMaxWaitMs);

    // Rerun awaitResponses again and we expect to only see terminated request id error.
    auto idResponses = executor.awaitResponses(requestIds, waitTime);
    for (auto const& responses : idResponses)
    {
        for (auto& response : responses)
        {
            EXPECT_TRUE(response.hasError());
            std::string err = "ReqId " + std::to_string(response.getRequestId())
                + " has already been processed and was terminated.";
            EXPECT_EQ(response.getErrorMsg(), err);
        }
    }

    // Check that number of tokens matches expectations
    for (auto const& [reqId, numTokens] : expectedNumTokens)
    {
        EXPECT_EQ(expectedNumTokens[reqId], tokens[reqId].size()) << "reqId " << reqId;
    }
}

TEST_P(ParamTest, GetNumResponsesReadyTest)
{
    bool const streaming = std::get<0>(GetParam());
    bool const excludeInputFromOutput = std::get<1>(GetParam());
    auto const beamWidth = std::get<2>(GetParam());
    OutputConfig outConfig;
    outConfig.excludeInputFromOutput = excludeInputFromOutput;

    auto executorConfig = ExecutorConfig(beamWidth);
    auto trtEnginePath = GPT_MODEL_PATH / FP16_GPT_ATTENTION_PACKED_PAGED_DIR / "tp1-pp1-gpu";
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    SizeType32 maxNumRequests = 50;
    SizeType32 maxPromptLen = 20;
    SizeType32 maxMaxNewTokens = 20;

    SizeType32 numRequests = rand() % maxNumRequests + 1;
    SizeType32 numExpectedResponses = 0;
    std::map<IdType, SizeType32> reqNumExpectedResponses;
    std::vector<IdType> ids;
    for (SizeType32 req = 0; req < numRequests; ++req)
    {
        SizeType32 promptLen = rand() % maxPromptLen + 1;
        SizeType32 maxNewTokens = rand() % maxMaxNewTokens + 1;

        auto request = Request(VecTokens(promptLen, 1), maxNewTokens, streaming,
            tensorrt_llm::executor::SamplingConfig(beamWidth), outConfig);
        auto id = executor.enqueueRequest(std::move(request));
        ids.emplace_back(id);
        reqNumExpectedResponses[id] = streaming ? maxNewTokens : 1;
        numExpectedResponses += reqNumExpectedResponses.at(id);
    }

    SizeType32 iter = 0;
    SizeType32 numReady = 0;
    while (numReady < numExpectedResponses && iter < mMaxWaitMs)
    {
        numReady = 0;
        for (auto id : ids)
        {
            numReady += executor.getNumResponsesReady(id);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        ++iter;
    }
    EXPECT_LT(iter, mMaxWaitMs);
    // Expect one response per request
    for (auto id : ids)
    {
        SizeType32 numReady = executor.getNumResponsesReady(id);
        EXPECT_EQ(numReady, reqNumExpectedResponses.at(id));
    }
    auto numResponsesReady = executor.getNumResponsesReady();
    EXPECT_EQ(numResponsesReady, numExpectedResponses);
}

namespace
{

struct FlakyTestInfo
{
    // Pair of batch ID + beam which are flaky
    std::set<std::pair<SizeType32, SizeType32>> batchIdBeams;
};

void verifyOutput(std::unordered_map<SizeType32, BeamTokens> const& beamTokens, TestData const& testData,
    std::vector<SizeType32> const& givenInputLengths, SizeType32 nbGivenInputs, bool streaming,
    bool excludeInputFromOutput, FlakyTestInfo flakyTestInfo, bool isSpeculativeDecoding)
{
    for (auto const& [batchId, tokens] : beamTokens)
    {
        auto const inputLength = givenInputLengths.at(batchId);
        SizeType32 const reqBeamWidth = tokens.size();
        auto const* const expectedOutputData = tr::bufferCast<TokenIdType const>(*testData.expectedOutputIds);
        auto const expectedOutputLengths = testData.expectedOutputLengths;
        auto const endId = testData.endIds[batchId];
        auto const maxSeqLen = testData.maxSeqLen;

        for (SizeType32 beam = 0; beam < reqBeamWidth; ++beam)
        {
            bool isFlaky = flakyTestInfo.batchIdBeams.count(std::make_pair(batchId, beam));
            if (isFlaky)
            {
                TLLM_LOG_WARNING("Disabling token comparison for batchId %d beam %d, test if flaky", batchId, beam);
            }

            auto expectedOutputLength = expectedOutputLengths[batchId * reqBeamWidth + beam];
            expectedOutputLength -= inputLength;

            bool inputNotIncluded = (streaming || excludeInputFromOutput);
            bool anyMismatch = false;
            auto predictedTokens = tokens.at(beam);
            // Remove the prompt
            if (!inputNotIncluded)
            {
                predictedTokens.erase(predictedTokens.begin(), predictedTokens.begin() + inputLength);
            }
            auto const numTokensToRemove = 0;
            auto numPredTokens = static_cast<SizeType32>(predictedTokens.size());
            EXPECT_EQ(predictedTokens.size(), expectedOutputLength) << "b: " << batchId << " beam: " << beam;

            if (isSpeculativeDecoding)
            {
                // WAR to ensure bulk execution of spec decoding.
                // We hope that no request in batch can finish 2x faster than any other request.
                // For the cases when BS < 8, some predicted tokens are mismatched to reference data.
                numPredTokens /= 2;
            }

            for (auto i = 0; i < numPredTokens; ++i)
            {
                // Use the expected data for that beamWidth
                auto const expectIndex = tc::flat_index3(batchId, beam, inputLength + i, reqBeamWidth, maxSeqLen);
                auto const expectedToken = expectedOutputData[expectIndex];
                if (expectedToken == endId)
                {
                    // TODO: can not find the error when (expectedToken == endId) && (predictedToken != endId)
                    break;
                }
                auto const predictedToken = predictedTokens.at(i);
                if (!isFlaky)
                {
                    EXPECT_EQ(predictedToken, expectedToken) << "b: " << batchId << " beam: " << beam << " i: " << i;
                    anyMismatch |= (predictedToken != expectedToken);
                }
            }
            EXPECT_FALSE(anyMismatch) << "b: " << batchId << " beam: " << beam;
        }
    }
}

void verifyLogProbs(bool computeLogProbs, TestData const& testData, bool streaming, bool excludeInputFromOutput,
    SizeType32 inputLength, SizeType32 beamWidth, BeamTokens const& beamTokens,
    std::optional<VecLogProbs> const& cumLogProbs, std::optional<std::vector<VecLogProbs>> const& logProbs,
    SizeType32 batchId, FlakyTestInfo flakyTestInfo)
{
    auto expectedCumLogProbs = testData.expectedCumLogProbs[batchId];
    auto expectedLogProbs = testData.expectedLogProbs[batchId];
    auto const expectedOutputLengths = testData.expectedOutputLengths;

    if (computeLogProbs)
    {
        EXPECT_TRUE(cumLogProbs.has_value()) << "bid: " << batchId;
        EXPECT_TRUE(logProbs.has_value()) << "bid: " << batchId;
        EXPECT_EQ(cumLogProbs.value().size(), beamWidth) << "bid: " << batchId;
        EXPECT_EQ(logProbs.value().size(), beamWidth) << "bid: " << batchId;

        bool removeInput = !excludeInputFromOutput && !streaming;

        for (SizeType32 beam = 0; beam < beamWidth; ++beam)
        {
            bool isFlaky = flakyTestInfo.batchIdBeams.count(std::make_pair(batchId, beam));
            if (isFlaky)
            {
                TLLM_LOG_WARNING("Disabling token comparison for batchId %d beam %d, test if flaky", batchId, beam);
            }

            auto expectedOutputLength = expectedOutputLengths[batchId * beamWidth + beam];
            expectedOutputLength -= inputLength;

            auto numPredTokens = logProbs.value().at(beam).size();
            // Check shape
            EXPECT_EQ(numPredTokens, beamTokens.at(beam).size() - (removeInput ? inputLength : 0))
                << "bid: " << batchId << " beam: " << beam;

            // If beamWidth == 1, compare log probs against python runtime
            if (beamWidth == 1)
            {
                auto* const reqExpectedCumLogProbs = tr::bufferCast<float>(*expectedCumLogProbs);
                // Only check cumLogProbs for the last generated token
                if (numPredTokens == expectedOutputLength && !isFlaky)
                {
                    EXPECT_TRUE(almostEqual(reqExpectedCumLogProbs[beam], cumLogProbs.value().at(beam), 2e-1, 5e-2))
                        << "expectedCumLogProbs : " << reqExpectedCumLogProbs[beam]
                        << " cumlogProbs : " << cumLogProbs.value().at(beam);
                }

                auto expectedLogProbsBeam = std::shared_ptr(tr::ITensor::slice(expectedLogProbs, beam, 1));
                expectedLogProbsBeam->squeeze(0);
                auto* const reqExpectedLogProbs = tr::bufferCast<float>(*expectedLogProbsBeam);
                for (auto i = 0; i < numPredTokens; ++i)
                {
                    if (!isFlaky)
                    {
                        EXPECT_TRUE(
                            almostEqual(reqExpectedLogProbs[inputLength + i], logProbs.value()[beam][i], 5e-2, 5e-2))
                            << "expectedLogProbs : " << reqExpectedLogProbs[inputLength + i]
                            << " logProbs : " << logProbs.value()[beam][i];
                    }
                }
            }
        }
    }
    else
    {
        EXPECT_FALSE(cumLogProbs.has_value()) << "bid: " << batchId;
        EXPECT_FALSE(logProbs.has_value()) << "bid: " << batchId;
    }
}

void validateContextLogitsShape(bool getContextLogits, SizeType32 inputLength,
    std::optional<Tensor> const& contextLogits, SizeType32 vocabSizePadded, SizeType32 batchId)
{
    if (getContextLogits)
    {
        EXPECT_TRUE(contextLogits.has_value()) << "bid: " << batchId;
        EXPECT_EQ(contextLogits.value().getShape().size(), 2);
        EXPECT_EQ(contextLogits.value().getShape()[0], inputLength);
        EXPECT_EQ(contextLogits.value().getShape()[1], vocabSizePadded);
    }
    else
    {
        EXPECT_FALSE(contextLogits.has_value()) << "bid: " << batchId;
    }
}

void validateGenerationLogitsShape(bool getGenLogits, bool streaming, bool excludeInputFromOutput,
    SizeType32 inputLength, SizeType32 maxOutputLen, SizeType32 beamWidth, BeamTokens const& beamTokens,
    std::optional<Tensor> const& genLogits, SizeType32 vocabSizePadded, SizeType32 batchId)
{
    if (getGenLogits)
    {
        EXPECT_TRUE(genLogits.has_value()) << "bid: " << batchId;
        EXPECT_EQ(genLogits.value().getShape().size(), 3);
        EXPECT_EQ(genLogits.value().getShape()[0], beamWidth);
        EXPECT_EQ(genLogits.value().getShape()[1], maxOutputLen);
        EXPECT_EQ(genLogits.value().getShape()[2], vocabSizePadded);
    }
    else
    {
        EXPECT_FALSE(genLogits.has_value()) << "bid: " << batchId;
    }
}

void runTest(Executor& executor, fs::path const& inputPath, ModelIds const& modelIds,
    FlakyTestInfo const& flakyTestInfo, bool streaming, SizeType32 const vocabSizePadded, BeamResult const& beamResult,
    OutputConfig const& outConfig, bool isSpeculativeDecoding, int maxWaitMs)
{
    auto const beamWidth = beamResult.beamWidth;

    std::unordered_map<IdType, SizeType32> reqIdToBatchId;
    std::unordered_map<SizeType32, BeamTokens> tokens;

    auto manager = tr::BufferManager(std::make_shared<tr::CudaStream>());
    auto const& givenInput = tr::utils::loadNpy(manager, inputPath.string(), tr::MemoryType::kCPU);
    auto [givenInputLengths, nbGivenInputs, maxInputLength] = getGivenInputLengths(*givenInput, modelIds.padId);
    auto const* const givenInputData = tr::bufferCast<TokenIdType const>(*givenInput);

    auto const& inputShape = givenInput->getShape();
    ASSERT_EQ(inputShape.nbDims, 2);
    ASSERT_GT(inputShape.d[0], 0);

    // Load expected outputs for each beam width value
    auto testData = loadTestData(beamResult, *givenInput, beamWidth, manager, outConfig, modelIds);
    auto const maxSeqLen = testData.maxSeqLen;

    // Load expected outputs and inputs
    SizeType32 numRequests = static_cast<SizeType32>(givenInputLengths.size());
    SizeType32 maxRequests = numRequests;
    std::vector<Request> requests;
    std::vector<SizeType32> reqMaxNewTokens;
    for (SizeType32 req = 0; req < maxRequests; ++req)
    {
        SizeType32 inputLen = givenInputLengths.at(req);
        auto maxNewTokens = maxSeqLen - maxInputLength;
        reqMaxNewTokens.push_back(maxNewTokens);
        SizeType32 endId = -1;
        auto const* const seqBegin = givenInputData + req * maxInputLength;
        VecTokens tokens(seqBegin, seqBegin + inputLen);
        requests.emplace_back(VecTokens(seqBegin, seqBegin + inputLen), maxNewTokens, streaming,
            tensorrt_llm::executor::SamplingConfig(beamWidth), outConfig, endId);
    }

    auto& comm = tensorrt_llm::mpi::MpiComm::world();
    auto const worldRank = comm.getRank();
    auto const worldSize = comm.getSize();

    std::vector<IdType> reqIds;
    if (worldRank == 0)
    {
        reqIds = executor.enqueueRequests(std::move(requests));

        for (SizeType32 req = 0; req < reqIds.size(); ++req)
        {
            tokens[req] = BeamTokens(beamWidth);
            reqIdToBatchId[reqIds.at(req)] = req;
        }

        // Get the new tokens for each requests
        int32_t numFinished = 0;
        int iter = 0;
        SizeType32 numResponses = 0;
        while (numFinished < maxRequests && iter < maxWaitMs)
        {
            std::chrono::milliseconds waitTime(1);
            auto responses = executor.awaitResponses(waitTime);
            for (auto& response : responses)
            {
                numResponses++;
                if (!response.hasError())
                {
                    auto result = response.getResult();
                    numFinished += result.isFinal;
                    auto batchId = reqIdToBatchId.at(response.getRequestId());

                    auto& contextLogits = result.contextLogits;
                    auto& genLogits = result.generationLogits;
                    auto& outputTokenIds = result.outputTokenIds;

                    for (SizeType32 beam = 0; beam < beamWidth; ++beam)
                    {
                        auto& newTokens = outputTokenIds.at(beam);
                        auto& reqTokens = tokens.at(batchId).at(beam);

                        reqTokens.insert(reqTokens.end(), newTokens.begin(), newTokens.end());
                    }

                    auto& cumLogProbs = result.cumLogProbs;
                    auto& logProbs = result.logProbs;
                    verifyLogProbs(outConfig.returnLogProbs, testData, streaming, outConfig.excludeInputFromOutput,
                        givenInputLengths.at(batchId), beamWidth, tokens.at(batchId), cumLogProbs, logProbs, batchId,
                        flakyTestInfo);
                    validateContextLogitsShape(outConfig.returnContextLogits, givenInputLengths.at(batchId),
                        contextLogits, vocabSizePadded, batchId);
                    validateGenerationLogitsShape(outConfig.returnGenerationLogits, streaming,
                        outConfig.excludeInputFromOutput, givenInputLengths.at(batchId), reqMaxNewTokens.at(batchId),
                        beamWidth, tokens.at(batchId), genLogits, vocabSizePadded, batchId);
                }
                else
                {
                    // Allow response with error only if awaitResponse processed a terminated request id
                    std::string err = "ReqId " + std::to_string(response.getRequestId())
                        + " has already been processed and was terminated.";
                    EXPECT_EQ(response.getErrorMsg(), err);
                }
            }
            ++iter;
        }
        EXPECT_LT(iter, maxWaitMs);
        verifyOutput(tokens, testData, givenInputLengths, nbGivenInputs, streaming, outConfig.excludeInputFromOutput,
            flakyTestInfo, isSpeculativeDecoding);
    }
}

void runTest(fs::path const& modelPath, ExecutorConfig const& executorConfig, fs::path const& inputPath,
    ModelIds const& modelIds, FlakyTestInfo const& flakyTestInfo, bool streaming, SizeType32 const vocabSizePadded,
    BeamResult const& beamResult, OutputConfig const& outConfig, bool isSpeculativeDecoding, int maxWaitMs)
{
    auto executor = Executor{modelPath, ModelType::kDECODER_ONLY, executorConfig};

    runTest(executor, inputPath, modelIds, flakyTestInfo, streaming, vocabSizePadded, beamResult, outConfig,
        isSpeculativeDecoding, maxWaitMs);
}

ExecutorConfig createExecutorConfig(BatchingType batchingType, SizeType32 maxBeamWidth, bool useOrchestratorMode)
{
    // Note: we reduce memory fraction for cases that return context/generation logits which require more free
    // memory
    FloatType constexpr freeGpuMemoryFraction{0.5F};
    KvCacheConfig kvCacheConfig(false, std::nullopt, std::nullopt, std::nullopt, freeGpuMemoryFraction);
    auto executorConfig = ExecutorConfig(maxBeamWidth);
    executorConfig.setBatchingType(batchingType);
    executorConfig.setKvCacheConfig(kvCacheConfig);
    executorConfig.setNormalizeLogProbs(false);

    std::optional<OrchestratorConfig> orchestratorConfig = std::nullopt;
    if (useOrchestratorMode)
    {
        orchestratorConfig = OrchestratorConfig(true, EXECUTOR_WORKER_PATH.string());
    }
    auto parallelConfig = ParallelConfig(CommunicationType::kMPI,
        useOrchestratorMode ? CommunicationMode::kORCHESTRATOR : CommunicationMode::kLEADER, std::nullopt, std::nullopt,
        orchestratorConfig);
    executorConfig.setParallelConfig(parallelConfig);

    return executorConfig;
}

} // namespace

TEST_P(AllParamsTest, TokenComparison)
{
    auto const batchingType = std::get<0>(GetParam());
    auto const streaming = std::get<1>(GetParam());
    auto const& beamWidth = std::get<2>(GetParam());
    OutputConfig outConfig;
    outConfig.returnLogProbs = std::get<3>(GetParam());
    outConfig.excludeInputFromOutput = std::get<4>(GetParam());
    outConfig.returnContextLogits = std::get<5>(GetParam());
    outConfig.returnGenerationLogits = std::get<6>(GetParam());
    auto const modelName = std::get<7>(GetParam());
    auto const useOrchestratorMode = std::get<8>(GetParam());

    auto const executorConfig = createExecutorConfig(batchingType, beamWidth, useOrchestratorMode);

    BeamResult beamResult{beamWidth};

    ASSERT_TRUE(fs::exists(DATA_PATH));

    fs::path modelPath;
    // set defaults and adjust if needed by different models
    fs::path inputPath = DATA_PATH / "input_tokens.npy";
    ModelIds modelIds{50256, 50256};
    bool isSpeculativeDecoding{false};

    // NOTE: This can be used to disable checks for certain prompt batch entries
    FlakyTestInfo flakyTestInfo;

    if (modelName == "gpt")
    {
        auto const resultsPath
            = GPT_DATA_PATH / ((beamWidth == 1) ? "sampling" : "beam_search_" + std::to_string(beamWidth));
        if (outConfig.returnContextLogits || outConfig.returnGenerationLogits)
        {
            modelPath = GPT_MODEL_PATH / FP16_GPT_ATTENTION_PACKED_PAGED_GATHER_DIR / "tp1-pp1-gpu";
            beamResult.resultsFile = resultsPath / FP16_PLUGIN_PACKED_PAGED_GATHER_RESULT_FILE;
            beamResult.contextLogitsFile = resultsPath / FP16_PLUGIN_PACKED_PAGED_CONTEXT_LOGITS_FILE;
            beamResult.genLogitsFile = resultsPath / FP16_PLUGIN_PACKED_PAGED_GENERATION_LOGITS_FILE;
            if (outConfig.returnLogProbs)
            {
                beamResult.cumLogProbsFile = resultsPath / FP16_PLUGIN_PACKED_PAGED_GATHER_CUM_LOG_PROBS_FILE;
                beamResult.logProbsFile = resultsPath / FP16_PLUGIN_PACKED_PAGED_GATHER_LOG_PROBS_FILE;
            }
        }
        else
        {
            modelPath = GPT_MODEL_PATH / FP16_GPT_ATTENTION_PACKED_PAGED_DIR / "tp1-pp1-gpu";
            beamResult.resultsFile = resultsPath / FP16_PLUGIN_PACKED_PAGED_RESULT_FILE;
            if (outConfig.returnLogProbs)
            {
                beamResult.cumLogProbsFile = resultsPath / FP16_PLUGIN_PACKED_PAGED_CUM_LOG_PROBS_FILE;
                beamResult.logProbsFile = resultsPath / FP16_PLUGIN_PACKED_PAGED_LOG_PROBS_FILE;
            }
        }
    }
    else if (modelName == "llama_tp4_pp1" || modelName == "llama_tp1_pp4" || modelName == "llama_tp2_pp2")
    {
        auto const resultsPath
            = LLAMA_DATA_PATH / ((beamWidth == 1) ? "sampling" : "beam_search_" + std::to_string(beamWidth));
        if (modelName == "llama_tp4_pp1")
        {
            beamResult.resultsFile = resultsPath / FP16_PLUGIN_PACKED_PAGED_RESULT_TP4_PP1_FILE;
            modelPath = LLAMA_MODEL_PATH / FP16_GPT_ATTENTION_PACKED_PAGED_DIR / "tp4-pp1-gpu";
        }
        else if (modelName == "llama_tp1_pp4")
        {
            beamResult.resultsFile = resultsPath / FP16_PLUGIN_PACKED_PAGED_RESULT_TP1_PP4_FILE;
            modelPath = LLAMA_MODEL_PATH / FP16_GPT_ATTENTION_PACKED_PAGED_DIR / "tp1-pp4-gpu";
        }
        else if (modelName == "llama_tp2_pp2")
        {
            beamResult.resultsFile = resultsPath / FP16_PLUGIN_PACKED_PAGED_RESULT_TP2_PP2_FILE;
            modelPath = LLAMA_MODEL_PATH / FP16_GPT_ATTENTION_PACKED_PAGED_DIR / "tp2-pp2-gpu";
        }
    }
    else if (modelName == "medusa")
    {
        TLLM_CHECK_WITH_INFO(beamWidth == 1, "Medusa does not support beam search.");
        auto const resultsPath = MEDUSA_DATA_PATH / "sampling";
        ModelSpec modelSpec
            = getDefaultModelSpec().useMedusa().setInputFile("input_tokens_long.npy").setMaxOutputLength(128);
        beamResult.resultsFile = resultsPath / modelSpec.getResultsFile();
        modelPath = MEDUSA_MODEL_PATH / modelSpec.getModelPath() / "tp1-pp1-gpu";

        inputPath = DATA_PATH / "input_vicuna.npy";
        modelIds.padId = 2;
        modelIds.endId = 2;
        isSpeculativeDecoding = true;
    }
    else if (modelName == "chatglm" || modelName == "chatglm2" || modelName == "chatglm3" || modelName == "glm")
    {
        fs::path resultsPath;
        if (modelName == "chatglm")
        {
            resultsPath = CHATGLM_DATA_PATH;
            modelPath = CHATGLM_MODEL_PATH;
        }
        else if (modelName == "chatglm2")
        {
            resultsPath = CHATGLM2_DATA_PATH;
            modelPath = CHATGLM2_MODEL_PATH;
        }
        else if (modelName == "chatglm3")
        {
            resultsPath = CHATGLM3_DATA_PATH;
            modelPath = CHATGLM3_MODEL_PATH;
        }
        else if (modelName == "glm")
        {
            resultsPath = GLM_DATA_PATH;
            modelPath = GLM_MODEL_PATH;
        }
        resultsPath /= (beamWidth == 1) ? "sampling" : "beam_search_" + std::to_string(beamWidth);
        if (batchingType == BatchingType::kSTATIC)
        {
            ModelSpec modelSpec{"input_tokens.npy", nvinfer1::DataType::kHALF};
            modelSpec.useGptAttentionPlugin();
            beamResult.resultsFile = resultsPath / modelSpec.getResultsFile();
            modelPath /= modelSpec.getModelPath() + "/tp1-pp1-gpu";
        }
        else
        {
            beamResult.resultsFile = resultsPath / FP16_PLUGIN_PACKED_PAGED_RESULT_FILE;
            modelPath = modelPath / FP16_GPT_ATTENTION_PACKED_PAGED_DIR / "tp1-pp1-gpu";
        }

        char versionChatglm{0};
        if (size_t index = modelPath.string().find("chatglm"); index != std::string::npos)
        {
            versionChatglm = modelPath.string()[index + 7];
            std::string const vChatglmString
                = (versionChatglm == '-') ? std::string("") : std::string(1, versionChatglm);
            inputPath = DATA_PATH / ("input_tokens_chatglm" + vChatglmString + "-6b.npy");
            modelIds.padId = (versionChatglm == '-') ? 3 : 0;
            modelIds.endId = (versionChatglm == '-') ? 130005 : 2;
        }
        else if (size_t index = modelPath.string().find("glm-10b"); index != std::string::npos)
        {
            inputPath = DATA_PATH / "input_tokens_glm-10b.npy";
            modelIds.padId = 50256;
            modelIds.endId = 50258;
        }

        if (versionChatglm != 0)
        {
            flakyTestInfo.batchIdBeams.insert(std::make_pair(1, 0));
        }
    }
    else
    {
        TLLM_THROW("Unrecognized modelName");
    }

    if (streaming && beamWidth > 1)
    {
        GTEST_SKIP() << "Test does not support streaming with beam search";
    }

    // Warning: This should be the last check before running the test.
    // It will initialize MPI which can take significant time.
    if (modelName == "llama_tp4_pp1" || modelName == "llama_tp1_pp4" || modelName == "llama_tp2_pp2")
    {
        // For llama model, only run for multiple GPUs
        // This is detected by setting an env variable when running the test
        char const* val = getenv("RUN_LLAMA_MULTI_GPU");
        if (val == NULL)
        {
            GTEST_SKIP() << "Skipping Llama test";
        }
        else
        {
            if (outConfig.returnLogProbs || outConfig.returnContextLogits || outConfig.returnGenerationLogits)
            {
                GTEST_SKIP() << "Skipping logits and log probs tests for mpi runs";
            }

            // Check that it was launched with right number of MPI ranks
            if (!useOrchestratorMode && COMM_SESSION.getSize() != 4)
            {
                // No orchestrator, need worldSize to match TP*PP
                FAIL() << "Leader mode and world size is not equal to 4";
            }
            else if (useOrchestratorMode && COMM_SESSION.getSize() != 1)
            {
                // No orchestrator, need worldSize to match TP*PP
                FAIL() << "Orchestrator mode and World size is not equal to 1";
            }
        }
    }

    SizeType32 constexpr vocabSizePadded{50257}; // gpt vocabSizePadded
    runTest(modelPath, executorConfig, inputPath, modelIds, flakyTestInfo, streaming, vocabSizePadded, beamResult,
        outConfig, isSpeculativeDecoding, mMaxWaitMs);
}

TEST_F(GptExecutorTest, ChangeBwError)
{
    SizeType32 constexpr maxBeamWidth{2};
    auto executorConfig = ExecutorConfig(maxBeamWidth);

    auto trtEnginePath = (GPT_MODEL_PATH / FP16_GPT_ATTENTION_PACKED_PAGED_DIR / "tp1-pp1-gpu");
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    SizeType32 constexpr beamWidth1{1};
    SizeType32 constexpr beamWidth2{2};
    SizeType32 constexpr maxNewTokens{5};
    VecTokens inputTokens{1, 2, 3, 4};

    // Create two req with different beam width
    std::vector<Request> requests;
    requests.emplace_back(inputTokens, maxNewTokens, false, tensorrt_llm::executor::SamplingConfig(beamWidth1));
    requests.emplace_back(inputTokens, maxNewTokens, false, tensorrt_llm::executor::SamplingConfig(beamWidth2));

    auto requestIds = executor.enqueueRequests(requests);

    bool done = false;
    int iter = 0;
    while (!done && iter < mMaxWaitMs)
    {
        std::chrono::milliseconds waitTime(1);
        auto responses = executor.awaitResponses(waitTime);
        for (auto& response : responses)
        {
            if (response.hasError())
            {
                auto err = response.getErrorMsg();
                std::cout << "err:" << err << std::endl;
                EXPECT_THAT(err, testing::HasSubstr("All active requests must have same beam width"));
                done = true;
            }
            else
            {
                FAIL() << "Should get a response with error";
            }
        }
        ++iter;
    }
    EXPECT_LT(iter, mMaxWaitMs);
}

TEST_F(GptExecutorTest, TokenComparisonChangeBw)
{
    SizeType32 constexpr maxBeamWidth{2};
    SizeType32 constexpr vocabSizePadded{50257}; // gpt vocabSizePadded
    auto constexpr streaming = false;

    // Create executor config
    auto executorConfig = ExecutorConfig(maxBeamWidth);

    // Create executor
    auto trtEnginePath = (GPT_MODEL_PATH / FP16_GPT_ATTENTION_PACKED_PAGED_GATHER_DIR / "tp1-pp1-gpu");
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    auto const inputPath = DATA_PATH / "input_tokens.npy";
    ModelIds modelIds{50256, 50256};

    OutputConfig outConfig;
    FlakyTestInfo flakyTestInfo;
    bool constexpr isSpeculativeDecoding{false};

    for (SizeType32 beamWidth : {1, 2})
    {
        BeamResult beamResult{beamWidth};
        auto const resultsPath
            = GPT_DATA_PATH / ((beamWidth == 1) ? "sampling" : "beam_search_" + std::to_string(beamWidth));
        beamResult.resultsFile = resultsPath / FP16_PLUGIN_PACKED_PAGED_GATHER_RESULT_FILE;
        beamResult.contextLogitsFile = resultsPath / FP16_PLUGIN_PACKED_PAGED_CONTEXT_LOGITS_FILE;
        beamResult.genLogitsFile = resultsPath / FP16_PLUGIN_PACKED_PAGED_GENERATION_LOGITS_FILE;

        runTest(executor, inputPath, modelIds, flakyTestInfo, streaming, vocabSizePadded, beamResult, outConfig,
            isSpeculativeDecoding, mMaxWaitMs);
    }
}

TEST_F(GptExecutorTest, TimedOut)
{
    SizeType32 beamWidth = 1;
    auto executorConfig = ExecutorConfig(beamWidth);
    auto trtEnginePath = GPT_MODEL_PATH / FP16_GPT_ATTENTION_PACKED_PAGED_DIR / "tp1-pp1-gpu";
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    // No requests enqueued, expect no responses
    auto numResponsesReady = executor.getNumResponsesReady();
    EXPECT_EQ(numResponsesReady, 0);

    std::chrono::milliseconds waitTime(10);
    auto responses = executor.awaitResponses(waitTime);
    EXPECT_EQ(responses.size(), 0);
}

TEST_F(GptExecutorTest, LogitsPostProcessor)
{
    SizeType32 constexpr endId{50256};
    SizeType32 constexpr vocabSizePadded{50257}; // gpt vocabSizePadded
    // We just use tokenIdCalculator to generate a token_id based on request index, output position and max new tokens.
    // Then LogitsPostProcessor set all other logits except the generated token_id to large negative value.
    // So the output token should be the generated token by tokenIdCalculator.
    auto tokenIdCalculator = [endId, vocabSizePadded](IdType req, SizeType32 pos)
    {
        SizeType32 tokenId = (req * 1000 + pos) % vocabSizePadded;
        if (tokenId == endId)
        {
            tokenId = 0;
        }
        return tokenId;
    };

    // Configuration options common to batched and non-batched logits processor test
    bool const streaming = false;
    bool excludeInputFromOutput = false;
    OutputConfig outConfig;
    outConfig.excludeInputFromOutput = excludeInputFromOutput;
    SizeType32 numRequests = 20;
    IdType const kClientId = 1234;

    SizeType32 beamWidth = 1;
    SizeType32 maxPromptLen = 20;
    SizeType32 maxMaxNewTokens = 20;

    std::unordered_map<IdType, VecTokens> tokens;
    std::unordered_map<IdType, SizeType32> expectedNumTokens;
    std::unordered_map<IdType, VecTokens> expectedOutputTokens;

    // Define helper lambdas used by batched and non-batched test

    // Enqueue the requests
    auto enqueueRequests = [&](Executor& executor, std::string const logitsProcessorName)
    {
        tokens.clear();
        expectedNumTokens.clear();
        expectedOutputTokens.clear();

        for (SizeType32 req = 0; req < numRequests; ++req)
        {
            SizeType32 promptLen = rand() % maxPromptLen + 1;
            SizeType32 maxNewTokens = rand() % maxMaxNewTokens + 1;

            auto request = Request(VecTokens(promptLen, 1), maxNewTokens, streaming,
                tensorrt_llm::executor::SamplingConfig(beamWidth), outConfig, endId);
            request.setClientId(kClientId);
            request.setLogitsPostProcessorName(logitsProcessorName);
            auto reqId = executor.enqueueRequest(std::move(request));
            tokens[reqId] = {};
            expectedNumTokens[reqId] = (streaming ? 0 : (excludeInputFromOutput ? 0 : promptLen)) + maxNewTokens;
            expectedOutputTokens[reqId] = {};
            if (!streaming && !excludeInputFromOutput)
            {
                expectedOutputTokens[reqId].resize(promptLen, 1);
            }
            for (SizeType32 outputPos = 0; outputPos < maxNewTokens; ++outputPos)
            {
                SizeType32 outputTokenId = tokenIdCalculator(reqId, outputPos + promptLen);
                expectedOutputTokens[reqId].push_back(outputTokenId);
            }
        }
    };

    // Get the new tokens for each requests
    auto collectResponses = [&](Executor& executor)
    {
        int32_t numFinished = 0;
        int iter = 0;
        SizeType32 numResponses = 0;
        while (numFinished < numRequests && iter < mMaxWaitMs)
        {
            std::chrono::milliseconds waitTime(1);
            auto responses = executor.awaitResponses(waitTime);
            for (auto& response : responses)
            {
                numResponses++;
                if (!response.hasError())
                {
                    auto result = response.getResult();
                    numFinished += result.isFinal;
                    auto& newTokens = result.outputTokenIds.at(beamWidth - 1);
                    auto& reqTokens = tokens.at(response.getRequestId());
                    reqTokens.insert(reqTokens.end(), std::make_move_iterator(newTokens.begin()),
                        std::make_move_iterator(newTokens.end()));
                }
                else
                {
                    // Allow response with error only if awaitResponse processed a terminated request id
                    std::string err = "ReqId " + std::to_string(response.getRequestId())
                        + " has already been processed and was terminated.";
                    EXPECT_EQ(response.getErrorMsg(), err);
                }
            }
            ++iter;
        }
        EXPECT_LT(iter, mMaxWaitMs);
    };

    // Check that tokens matches expectations
    auto checkOutput = [&]()
    {
        for (auto const& [reqId, numTokens] : expectedNumTokens)
        {
            EXPECT_EQ(expectedNumTokens[reqId], tokens[reqId].size()) << "reqId " << reqId;
            for (SizeType32 tokenPos = 0;
                 tokenPos < std::min<SizeType32>(expectedNumTokens[reqId], tokens[reqId].size()); ++tokenPos)
            {
                EXPECT_EQ(expectedOutputTokens[reqId][tokenPos], tokens[reqId][tokenPos])
                    << "reqId=" << reqId << ", tokenPos=" << tokenPos;
            }
        }
    };

    // Test non-batched logits processor
    std::string const logitsProcessorName = "SelectToken";

    auto logitsPostProcessorFn = [kClientId, tokenIdCalculator](IdType reqId, Tensor& logits, BeamTokens const& tokens,
                                     StreamPtr const& streamPtr, std::optional<IdType> clientId)
    {
        EXPECT_TRUE(clientId.value() == kClientId);
        SizeType32 numTokens = tokens.at(0).size();
        SizeType32 pos = numTokens;
        SizeType32 outputTokenId = tokenIdCalculator(reqId, pos);
        auto logitsDataType = logits.getDataType();
        EXPECT_TRUE(logitsDataType == DataType::kFP16 || logitsDataType == DataType::kBF16
            || logitsDataType == DataType::kFP32);
        // logits has shape [draftLength + 1, reqBeamWidth, vocabSize]
        auto logitsCpu = tensorrt_llm::executor::Tensor::cpu(logitsDataType, logits.getShape());
        auto* dataPtr = logitsCpu.getData();
        auto eltSize = logitsCpu.getSizeInBytes() / logitsCpu.getSize();
        EXPECT_TRUE(eltSize == 2 || eltSize == 4);
        if (eltSize == 2)
        {
            auto* dataPtrU16 = static_cast<uint16_t*>(dataPtr);
            uint16_t hugeNegValue = logitsDataType == DataType::kFP16 ? 0xFBFF : 0xFF7F; // a huge negative value
            for (size_t i = 0; i < logitsCpu.getSize(); ++i)
            {
                dataPtrU16[i] = hugeNegValue;
            }
            dataPtrU16[outputTokenId] = 0;
        }
        else
        {
            auto* dataPtrFloat = static_cast<float*>(dataPtr);
            for (size_t i = 0; i < logitsCpu.getSize(); ++i)
            {
                dataPtrFloat[i] = -HUGE_VALF;
            }
            dataPtrFloat[outputTokenId] = 0.0f;
        }

        logits.setFrom(logitsCpu, streamPtr);
    };

    auto executorConfig = ExecutorConfig(beamWidth);
    executorConfig.setLogitsPostProcessorMap(
        std::unordered_map<std::string, tensorrt_llm::executor::LogitsPostProcessor>{
            {logitsProcessorName, logitsPostProcessorFn}});
    auto trtEnginePath = GPT_MODEL_PATH / FP16_GPT_ATTENTION_PACKED_PAGED_DIR / "tp1-pp1-gpu";
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    enqueueRequests(executor, logitsProcessorName);
    collectResponses(executor);
    checkOutput();

    // Test batched logits processor
    auto logitsPostProcessorBatchedFn
        = [logitsPostProcessorFn](std::vector<IdType> const& reqIdBatch, std::vector<Tensor>& logitsBatch,
              std::vector<std::reference_wrapper<BeamTokens const>> const& tokensBatch, StreamPtr const& streamPtr,
              std::vector<std::optional<IdType>> const& clientIdBatch)
    {
        for (int sample = 0; sample < reqIdBatch.size(); sample++)
        {
            logitsPostProcessorFn(
                reqIdBatch[sample], logitsBatch[sample], tokensBatch[sample], streamPtr, clientIdBatch[sample]);
        }
    };

    auto batchedExecutorConfig = ExecutorConfig(beamWidth);
    batchedExecutorConfig.setLogitsPostProcessorBatched(logitsPostProcessorBatchedFn);
    auto batchedExecutor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, batchedExecutorConfig);

    enqueueRequests(batchedExecutor, Request::kBatchedPostProcessorName);
    collectResponses(batchedExecutor);
    checkOutput();
}

TEST_F(GptExecutorTest, LogitsPostProcessorThrow)
{
    SizeType32 beamWidth = 1;
    auto executorConfig = ExecutorConfig(beamWidth);
    auto trtEnginePath = GPT_MODEL_PATH / FP16_GPT_ATTENTION_PACKED_PAGED_DIR / "tp1-pp1-gpu";
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    std::string const logitsProcessorName = "UnExistProcessor";

    auto request
        = Request(VecTokens(10, 1), 10, false, tensorrt_llm::executor::SamplingConfig(beamWidth), OutputConfig());
    request.setLogitsPostProcessorName(logitsProcessorName);
    EXPECT_THROW({ auto reqId = executor.enqueueRequest(std::move(request)); }, tensorrt_llm::common::TllmException);
}

class MockedModel : public Model
{
    using LlmRequestPtr = std::shared_ptr<tb::LlmRequest>;
    using RequestList = std::list<LlmRequestPtr>;

public:
    MOCK_METHOD(void, forwardSync, (), ());
    MOCK_METHOD(void, forwardAsync, (RequestList const&), ());
    MOCK_METHOD(void, terminateRequest, (std::shared_ptr<tb::LlmRequest> const& llmRequest, bool pause), ());
    MOCK_METHOD(SizeType32, getMaxNumSequences, (), (const));
    MOCK_METHOD(SizeType32, getMaxInputLen, (), (const));
    MOCK_METHOD(SizeType32, getHiddenSize, (), (const));
    MOCK_METHOD(SizeType32, getMaxSequenceLen, (), (const));
    MOCK_METHOD(SizeType32, getVocabSizePadded, (), (const));
    MOCK_METHOD(SizeType32, getMaxDraftLen, (), (const));
    MOCK_METHOD(nvinfer1::DataType, getLogitDataType, (), (const));
    MOCK_METHOD(void, getCurrentIterationStats, (IterationStats&), (const));
    MOCK_METHOD(void, getCurrentRequestStats, (RequestStatsPerIteration&), (const));
    MOCK_METHOD(tr::WorldConfig const&, getWorldConfig, (), (const));
    MOCK_METHOD(tr::ModelConfig const&, getModelConfig, (), (const));
    MOCK_METHOD(tr::BufferManager const&, getBufferManager, (), (const));
    MOCK_METHOD(tr::BufferManager::CudaStreamPtr, getRuntimeStreamPtr, (), (const));
    MOCK_METHOD(void, updatePeftCache, (LlmRequestPtr const& llmReqeust), ());
    MOCK_METHOD(void, setLogitsPostProcessorBatched, (std::optional<LogitsPostProcessorBatched>), ());
};

TEST_P(ParamTest, MockedModel)
{
    using LlmRequestPtr = std::shared_ptr<tb::LlmRequest>;
    using RequestList = std::list<LlmRequestPtr>;

    bool const streaming = std::get<0>(GetParam());
    bool const excludeInputFromOutput = std::get<1>(GetParam());
    auto const beamWidth = std::get<2>(GetParam());
    OutputConfig outConfig;
    outConfig.excludeInputFromOutput = excludeInputFromOutput;
    auto model = std::make_shared<MockedModel>();

    EXPECT_CALL(*model, terminateRequest(_, _)).Times(0);
    EXPECT_CALL(*model, getVocabSizePadded()).Times(0);
    EXPECT_CALL(*model, getLogitDataType()).Times(0);

    SizeType32 callCount = 0;
    EXPECT_CALL(*model, forwardAsync(_))
        .WillRepeatedly(Invoke(
            [&](RequestList const& requestList)
            {
                for (auto const& llmReq : requestList)
                {
                    // Don't add any tokens to simulate no output tokens
                    llmReq->addNewTokens({1});
                    llmReq->mState = tb::REQUEST_STATE_GENERATION_IN_PROGRESS;
                    if (llmReq->getMaxNumGeneratedTokens() >= llmReq->mMaxNewTokens)
                    {
                        llmReq->mState = tb::REQUEST_STATE_GENERATION_COMPLETE;
                    }
                }
                callCount++;
            }));

    EXPECT_CALL(*model, getMaxNumSequences()).WillRepeatedly(Invoke([&]() { return 10; }));
    EXPECT_CALL(*model, getMaxInputLen()).WillRepeatedly(Invoke([&]() { return 10; }));
    EXPECT_CALL(*model, getMaxSequenceLen()).WillRepeatedly(Invoke([&]() { return 10; }));
    EXPECT_CALL(*model, getMaxDraftLen()).WillRepeatedly(Invoke([&]() { return 0; }));
    tr::WorldConfig const dummyWorldConfig;
    EXPECT_CALL(*model, getWorldConfig())
        .WillRepeatedly(Invoke([&]() -> tr::WorldConfig const& { return dummyWorldConfig; }));
    EXPECT_CALL(*model, getCurrentIterationStats(_)).WillRepeatedly(Invoke([&](IterationStats& stats) { return; }));
    EXPECT_CALL(*model, getCurrentRequestStats(_))
        .WillRepeatedly(Invoke([&](RequestStatsPerIteration& stats) { return; }));

    ExecutorConfig executorConfig(beamWidth);
    auto executor = Executor(model, executorConfig);

    // Create the request
    SizeType32 maxNewTokens = 5;
    VecTokens inputTokens{1, 2, 3, 4};
    auto request
        = Request(inputTokens, maxNewTokens, streaming, tensorrt_llm::executor::SamplingConfig(beamWidth), outConfig);

    // Enqueue the request
    auto requestId = executor.enqueueRequest(std::move(request));

    bool done = false;
    int iter = 0;
    while (!done && iter < mMaxWaitMs)
    {
        std::chrono::milliseconds waitTime(1);
        auto responses = executor.awaitResponses(requestId, waitTime);
        for (auto& response : responses)
        {
            auto result = response.getResult();
            done = result.isFinal;
        }
        ++iter;
    }

    EXPECT_LT(iter, mMaxWaitMs);
    EXPECT_EQ(callCount, maxNewTokens);
}

TEST_F(GptExecutorTest, MockedModelMaxQueueSize)
{
    using LlmRequestPtr = std::shared_ptr<tb::LlmRequest>;
    using RequestList = std::list<LlmRequestPtr>;

    auto model = std::make_shared<MockedModel>();

    EXPECT_CALL(*model, terminateRequest(_, _)).Times(0);
    EXPECT_CALL(*model, getVocabSizePadded()).Times(0);
    EXPECT_CALL(*model, getLogitDataType()).Times(0);

    SizeType32 callCount = 0;
    EXPECT_CALL(*model, forwardAsync(_))
        .WillRepeatedly(Invoke(
            [&](RequestList const& requestList)
            {
                for (auto const& llmReq : requestList)
                {
                    // Sleep to allow queue to fill up
                    std::this_thread::sleep_for(std::chrono::milliseconds(200));
                    // Don't add any tokens to simulate no output tokens
                    llmReq->addNewTokens({1});
                    llmReq->mState = tb::REQUEST_STATE_GENERATION_IN_PROGRESS;
                    if (llmReq->getMaxNumGeneratedTokens() >= llmReq->mMaxNewTokens)
                    {
                        llmReq->mState = tb::REQUEST_STATE_GENERATION_COMPLETE;
                    }
                }
                callCount++;
            }));

    EXPECT_CALL(*model, getMaxNumSequences()).WillRepeatedly(Invoke([&]() { return 10; }));
    EXPECT_CALL(*model, getMaxInputLen()).WillRepeatedly(Invoke([&]() { return 10; }));
    EXPECT_CALL(*model, getMaxSequenceLen()).WillRepeatedly(Invoke([&]() { return 10; }));
    EXPECT_CALL(*model, getMaxDraftLen()).WillRepeatedly(Invoke([&]() { return 0; }));
    tr::WorldConfig const dummyWorldConfig;
    EXPECT_CALL(*model, getWorldConfig())
        .WillRepeatedly(Invoke([&]() -> tr::WorldConfig const& { return dummyWorldConfig; }));
    EXPECT_CALL(*model, getCurrentIterationStats(_)).WillRepeatedly(Invoke([&](IterationStats& stats) { return; }));
    EXPECT_CALL(*model, getCurrentRequestStats(_))
        .WillRepeatedly(Invoke([&](RequestStatsPerIteration& stats) { return; }));

    SizeType32 maxQueueSize = 6;
    ExecutorConfig executorConfig;
    executorConfig.setMaxQueueSize(maxQueueSize);

    auto executor = Executor(model, executorConfig);

    // Create the request
    SizeType32 maxNewTokens = 5;
    VecTokens inputTokens{1, 2, 3, 4};
    auto request = Request(inputTokens, maxNewTokens);

    // Enqueue as many requests as the queue can manage
    for (int i = 0; i < maxQueueSize; i++)
    {
        auto requestId = executor.enqueueRequest(std::move(request));
    }
    try
    {
        auto requestId = executor.enqueueRequest(std::move(request));

        FAIL() << "Expected TllmException";
    }
    catch (std::exception const& e)
    {
        EXPECT_THAT(e.what(), testing::HasSubstr("Maximum queue size of 6 has been reached, please try again later"));
    }

    // Wait for requests to get scheduled to free up space in queue
    std::this_thread::sleep_for(std::chrono::milliseconds(maxQueueSize * 200));
    auto requestId = executor.enqueueRequest(std::move(request));
}

TEST_F(GptExecutorTest, OrchestratorMaxQueueSize)
{
    auto trtEnginePath = GPT_MODEL_PATH / FP16_GPT_ATTENTION_PACKED_PAGED_DIR / "tp1-pp1-gpu";
    SizeType32 maxQueueSize = 6;
    ExecutorConfig executorConfig;
    executorConfig.setMaxQueueSize(maxQueueSize);
    auto orchestratorConfig = OrchestratorConfig(true, EXECUTOR_WORKER_PATH.string());
    auto parallelConfig = ParallelConfig(
        CommunicationType::kMPI, CommunicationMode::kORCHESTRATOR, std::nullopt, std::nullopt, orchestratorConfig);
    executorConfig.setParallelConfig(parallelConfig);

    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    // Create the request
    SizeType32 maxNewTokens = 100;
    VecTokens inputTokens{1, 2, 3, 4};
    auto request = Request(inputTokens, maxNewTokens);
    std::vector<IdType> requestIds;
    auto numberOfRequests = maxQueueSize * 5;
    requestIds.reserve(numberOfRequests);

    // Enqueue more requests than the queue can manage
    for (int i = 0; i < numberOfRequests; i++)
    {
        auto requestId = executor.enqueueRequest(request);
        requestIds.emplace_back(requestId);
    }

    auto responseVectors = executor.awaitResponses(std::move(requestIds));
    bool failedWithFullQueue = false;
    for (auto& responseVector : responseVectors)
    {
        for (auto& response : responseVector)
        {
            if (response.hasError())
            {
                EXPECT_THAT(response.getErrorMsg(),
                    testing::HasSubstr("Maximum queue size of 6 has been reached, please try again later"));
                failedWithFullQueue = true;
            }
        }
    }
    EXPECT_TRUE(failedWithFullQueue) << "Expected requests to fail due to maximum queue size reached";

    // Wait for requests to get scheduled to free up space in queue
    std::this_thread::sleep_for(std::chrono::milliseconds(maxQueueSize * 200));
    auto requestId = executor.enqueueRequest(std::move(request));
    auto responses = executor.awaitResponses(requestId);
    for (auto& response : responses)
    {
        EXPECT_FALSE(response.hasError());
    }
}

TEST_F(GptExecutorTest, MockedModelReqStatsBug)
{
    using LlmRequestPtr = std::shared_ptr<tb::LlmRequest>;
    using RequestList = std::list<LlmRequestPtr>;

    bool streaming = false;
    bool excludeInputFromOutput = false;
    OutputConfig outConfig;
    outConfig.excludeInputFromOutput = excludeInputFromOutput;
    auto model = std::make_shared<MockedModel>();

    EXPECT_CALL(*model, terminateRequest(_, _)).Times(0);
    EXPECT_CALL(*model, getVocabSizePadded()).Times(0);
    EXPECT_CALL(*model, getLogitDataType()).Times(0);
    EXPECT_CALL(*model, updatePeftCache(_)).WillRepeatedly(Invoke([&]() { return; }));

    SizeType32 callCount = 0;
    RequestList currentReq;
    EXPECT_CALL(*model, forwardAsync(_))
        .WillRepeatedly(Invoke(
            [&](RequestList const& requestList)
            {
                currentReq = requestList;
                for (auto const& llmReq : requestList)
                {
                    // Don't add any tokens to simulate no output tokens
                    llmReq->addNewTokens({1});
                    llmReq->mState = tb::REQUEST_STATE_GENERATION_IN_PROGRESS;
                }
                callCount++;
            }));

    EXPECT_CALL(*model, forwardSync())
        .WillRepeatedly(Invoke(
            [&]()
            {
                for (auto const& llmReq : currentReq)
                {
                    if (llmReq->getMaxNumGeneratedTokens() >= llmReq->mMaxNewTokens)
                    {
                        llmReq->mState = tb::REQUEST_STATE_GENERATION_COMPLETE;
                    }
                }
                return;
            }));

    EXPECT_CALL(*model, getMaxNumSequences()).WillRepeatedly(Invoke([&]() { return 10; }));
    EXPECT_CALL(*model, getMaxInputLen()).WillRepeatedly(Invoke([&]() { return 10; }));
    EXPECT_CALL(*model, getMaxSequenceLen()).WillRepeatedly(Invoke([&]() { return 10; }));
    EXPECT_CALL(*model, getMaxDraftLen()).WillRepeatedly(Invoke([&]() { return 0; }));
    tr::WorldConfig const dummyWorldConfig;
    EXPECT_CALL(*model, getWorldConfig())
        .WillRepeatedly(Invoke([&]() -> tr::WorldConfig const& { return dummyWorldConfig; }));
    EXPECT_CALL(*model, getCurrentIterationStats(_)).WillRepeatedly(Invoke([&](IterationStats& stats) { return; }));
    EXPECT_CALL(*model, getCurrentRequestStats(_))
        .WillRepeatedly(Invoke([&](RequestStatsPerIteration& stats) { return; }));

    SizeType32 beamWidth = 1;
    ExecutorConfig executorConfig(beamWidth);
    executorConfig.setRequestStatsMaxIterations(1000);
    auto executor = Executor(model, executorConfig);

    // Create the request
    SizeType32 maxNewTokens = 5;
    VecTokens inputTokens{1, 2, 3, 4};
    int numRequests = 10000;
    auto request
        = Request(inputTokens, maxNewTokens, streaming, tensorrt_llm::executor::SamplingConfig(beamWidth), outConfig);

    auto done = std::atomic<bool>{false};
    auto statsThreadDone = false;
    // Spawn a thread that continuously get stats
    auto statsThread = std::thread(
        [&executor, &done, &statsThreadDone]()
        {
            while (!done)
            {
                auto reqStats = executor.getLatestRequestStats();
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
            statsThreadDone = true;
        });

    // Spawn a thread that enqueues the requests
    std::vector<IdType> requestIds;
    auto enqueueThread = std::thread(
        [&executor, &requestIds, &request, &done, numRequests]()
        {
            for (int i = 0; i < numRequests; ++i)
            {
                requestIds.push_back(executor.enqueueRequest(request));
            }
            done = true;
        });
    enqueueThread.join();
    ASSERT_EQ(requestIds.size(), numRequests);

    // Wait for stats thread to be done, fail otherwise
    int iter = 0;
    while (!statsThreadDone && iter < mMaxWaitMs)
    {
        std::chrono::milliseconds waitTime(1);
        std::this_thread::sleep_for(std::chrono::milliseconds(waitTime));
        iter++;
    }
    ASSERT_TRUE(statsThreadDone);
    statsThread.join();
}

TEST_F(GptExecutorTest, MockedModelEvictRestartValidityTest)
{
    using LlmRequestPtr = std::shared_ptr<tb::LlmRequest>;
    using RequestList = std::list<LlmRequestPtr>;

    bool excludeInputFromOutput = false;
    OutputConfig outConfig;
    outConfig.excludeInputFromOutput = excludeInputFromOutput;
    auto model = std::make_shared<MockedModel>();

    EXPECT_CALL(*model, terminateRequest(_, _)).Times(0);
    EXPECT_CALL(*model, getVocabSizePadded()).Times(0);
    EXPECT_CALL(*model, getLogitDataType()).Times(0);
    EXPECT_CALL(*model, updatePeftCache(_)).WillRepeatedly(Invoke([&]() { return; }));

    SizeType32 callCount = 0;
    RequestList currentReq;
    EXPECT_CALL(*model, forwardAsync(_))
        .WillRepeatedly(Invoke(
            [&](RequestList const& requestList)
            {
                currentReq = requestList;
                for (auto const& llmReq : requestList)
                {
                    // Don't add any tokens to simulate no output tokens
                    llmReq->addNewTokens({1});
                    llmReq->mState = tb::REQUEST_STATE_GENERATION_IN_PROGRESS;
                }
                callCount++;
            }));

    EXPECT_CALL(*model, forwardSync())
        .WillRepeatedly(Invoke(
            [&]()
            {
                for (auto const& llmReq : currentReq)
                {
                    if (llmReq->getMaxNumGeneratedTokens() >= llmReq->mMaxNewTokens)
                    {
                        llmReq->mState = tb::REQUEST_STATE_GENERATION_COMPLETE;
                    }
                }
                return;
            }));

    EXPECT_CALL(*model, getMaxNumSequences()).WillRepeatedly(Invoke([&]() { return 10; }));
    EXPECT_CALL(*model, getMaxInputLen()).WillRepeatedly(Invoke([&]() { return 6; }));
    EXPECT_CALL(*model, getMaxSequenceLen()).WillRepeatedly(Invoke([&]() { return 10; }));
    EXPECT_CALL(*model, getMaxDraftLen()).WillRepeatedly(Invoke([&]() { return 0; }));
    tr::WorldConfig const dummyWorldConfig;
    EXPECT_CALL(*model, getWorldConfig())
        .WillRepeatedly(Invoke([&]() -> tr::WorldConfig const& { return dummyWorldConfig; }));
    EXPECT_CALL(*model, getCurrentIterationStats(_)).WillRepeatedly(Invoke([&](IterationStats& stats) { return; }));
    EXPECT_CALL(*model, getCurrentRequestStats(_))
        .WillRepeatedly(Invoke([&](RequestStatsPerIteration& stats) { return; }));

    SizeType32 beamWidth = 1;
    ExecutorConfig executorConfig(beamWidth,
        SchedulerConfig(CapacitySchedulerPolicy::kMAX_UTILIZATION)); // Condition 1 : MAX_UTILIZATION scheduling policy
    executorConfig.setEnableChunkedContext(false);                   // Condition 2 : Chunked context disabled
    executorConfig.setRequestStatsMaxIterations(1000);
    auto executor = Executor(model, executorConfig);

    // Create the request
    bool streaming = true;                       // Condition 3 : Streaming enabled
    SizeType32 maxNewTokens = 5;
    VecTokens tooLongInputTokens{1, 2, 3, 4, 5}; // Condition 4 : prompt input len + maxNewTokens > MaxInputLen
    auto tooLongRequest = Request(
        tooLongInputTokens, maxNewTokens, streaming, tensorrt_llm::executor::SamplingConfig(beamWidth), outConfig);

    // Enqueue the request
    auto longRequestId = executor.enqueueRequest(std::move(tooLongRequest));
    bool done = false;
    int iter = 0;
    while (!done && iter < mMaxWaitMs)
    {
        std::chrono::milliseconds waitTime(1);
        auto responses = executor.awaitResponses(longRequestId, waitTime);
        for (auto& response : responses)
        {
            EXPECT_EQ(response.hasError(), true);
            EXPECT_THAT(response.getErrorMsg(),
                testing::HasSubstr("sequence length is potentially greater than max input length"));
            done = true;
        }
        ++iter;
    }
}

#if ENABLE_MULTI_DEVICE
// This test can be run manually to test multiGPU execution
// mpirun --allow-run-as-root -n 5 ./executorTest --gtest_filter="*MockedModelMultiGpu/ExecutorTest"
// Number of MPI ranks can be greater than tp

TEST_P(ParamTest, MockedModelMultiGpu)
{
    auto& world = tensorrt_llm::mpi::MpiComm::world();
    auto const worldRank = world.getRank();
    auto const worldSize = world.getSize();

    // In this test, allow worldSize to be greater than tp = 4
    // If so, set participant ids to be the last 4 ranks
    SizeType32 tp = std::min(4, worldSize);

    using LlmRequestPtr = std::shared_ptr<tb::LlmRequest>;
    using RequestList = std::list<LlmRequestPtr>;

    bool const streaming = std::get<0>(GetParam());
    bool const excludeInputFromOutput = std::get<1>(GetParam());
    auto const beamWidth = std::get<2>(GetParam());
    OutputConfig outConfig;
    outConfig.excludeInputFromOutput = excludeInputFromOutput;
    auto model = std::make_shared<MockedModel>();

    // Create the request
    SizeType32 maxNewTokens = 5;
    VecTokens inputTokens{1, 2, 3, 4};
    auto request
        = Request(inputTokens, maxNewTokens, streaming, tensorrt_llm::executor::SamplingConfig(beamWidth), outConfig);

    EXPECT_CALL(*model, terminateRequest(_, _)).Times(0);
    EXPECT_CALL(*model, getVocabSizePadded()).Times(0);
    EXPECT_CALL(*model, getLogitDataType()).Times(0);

    SizeType32 callCount = 0;
    SizeType32 reqCallCount = 0;
    EXPECT_CALL(*model, forwardAsync(_))
        .WillRepeatedly(Invoke(
            [&](RequestList const& requestList)
            {
                for (auto const& llmReq : requestList)
                {
                    EXPECT_EQ(llmReq->getTokens().size(), beamWidth);
                    // Verify that all MPI ranks get the expected request, even though only rank 0 actually gets the
                    // request
                    if (reqCallCount == 0)
                    {
                        EXPECT_EQ(llmReq->getOrigPromptLen(), request.getInputTokenIds().size());
                        for (int i = 0; i < llmReq->getOrigPromptLen(); ++i)
                        {
                            EXPECT_EQ(llmReq->getTokens(beamWidth - 1).at(i), request.getInputTokenIds().at(i));
                        }
                    }
                    EXPECT_EQ(llmReq->isStreaming(), request.getStreaming());
                    EXPECT_EQ(llmReq->mMaxNewTokens, request.getMaxNewTokens());
                    EXPECT_EQ(
                        llmReq->getTokens(beamWidth - 1).size(), request.getInputTokenIds().size() + reqCallCount);

                    SizeType32 tokenId = 1;
                    COMM_SESSION.bcastValue(tokenId, 0);
                    // Don't add any tokens to simulate no output tokens
                    // Simulate leader rank communicating with comm session
                    VecTokens newTokens(beamWidth, tokenId);
                    llmReq->addNewTokens(newTokens);
                    llmReq->mState = tb::REQUEST_STATE_GENERATION_IN_PROGRESS;
                    if (llmReq->getMaxNumGeneratedTokens() >= llmReq->mMaxNewTokens)
                    {
                        llmReq->mState = tb::REQUEST_STATE_GENERATION_COMPLETE;
                    }
                    reqCallCount++;
                }
                callCount++;
            }));

    EXPECT_CALL(*model, getMaxNumSequences()).WillRepeatedly(Invoke([&]() { return 10; }));
    EXPECT_CALL(*model, getMaxInputLen()).WillRepeatedly(Invoke([&]() { return 10; }));
    EXPECT_CALL(*model, getMaxSequenceLen()).WillRepeatedly(Invoke([&]() { return 10; }));
    EXPECT_CALL(*model, getCurrentIterationStats(_)).WillRepeatedly(Invoke([&](IterationStats& stats) { return; }));
    EXPECT_CALL(*model, getCurrentRequestStats(_))
        .WillRepeatedly(Invoke([&](RequestStatsPerIteration& stats) { return; }));

    tr::WorldConfig dummyWorldConfig = tr::WorldConfig(tp, 1, worldRank, tp);
    EXPECT_CALL(*model, getWorldConfig())
        .WillRepeatedly(Invoke([&]() -> tr::WorldConfig const& { return dummyWorldConfig; }));

    ParallelConfig parallelConfig;

    // Set participant ids to be of size tp, starting at worldSize - 1
    std::vector<SizeType32> participantIds;
    for (int i = 0; i < tp; ++i)
    {
        participantIds.push_back(worldSize - tp + i);
    }
    bool isLeader = (worldRank == participantIds.front());
    parallelConfig.setParticipantIds(participantIds);

    // Set device ids
    std::vector<SizeType32> deviceIds(tp);
    std::iota(deviceIds.begin(), deviceIds.end(), 0);
    parallelConfig.setDeviceIds(deviceIds);

    ExecutorConfig executorConfig(beamWidth);
    executorConfig.setParallelConfig(parallelConfig);
    auto executor = Executor(model, executorConfig);

    // Enqueue the request
    IdType requestId = 0;
    if (isLeader)
    {
        requestId = executor.enqueueRequest(request);

        SizeType32 numResponses{0};
        bool done = false;
        int iter = 0;
        while (!done && iter < mMaxWaitMs)
        {
            std::chrono::milliseconds waitTime(1);
            auto responses = executor.awaitResponses(waitTime);
            for (auto& response : responses)
            {
                ++numResponses;
                auto result = response.getResult();
                EXPECT_EQ(result.outputTokenIds.size(), beamWidth);
                auto expectedSize = streaming ? (beamWidth > 1 ? numResponses : 1)
                                              : (maxNewTokens + (excludeInputFromOutput ? 0 : inputTokens.size()));
                EXPECT_EQ(result.outputTokenIds.at(beamWidth - 1).size(), expectedSize);
                done = result.isFinal;
            }
            ++iter;
        }

        EXPECT_LT(iter, mMaxWaitMs);
        EXPECT_EQ(numResponses, streaming ? maxNewTokens : 1);
        EXPECT_EQ(callCount, maxNewTokens);
    }
}
#endif // ENABLE_MULTI_DEVICE

TEST_F(GptExecutorTest, MockedModelWithError)
{
    using LlmRequestPtr = std::shared_ptr<tb::LlmRequest>;
    using RequestList = std::list<LlmRequestPtr>;

    struct MockedModelParams
    {
        SizeType32 maxInputLen;
        SizeType32 maxSeqLen;
        SizeType32 expectedTerminateCnt;
        SizeType32 expectedForwardCnt;
        bool computeGenLogits;
        bool computeContextLogits;
        std::string expectedError;
    };

    std::vector<MockedModelParams> mockedModelParams;
    // Mocked error in forward call
    mockedModelParams.emplace_back(MockedModelParams{10, 20, 1, 1, true, true, "mocked error"});
    // prompt longer than maxInputLen
    mockedModelParams.emplace_back(MockedModelParams{1, 20, 0, 0, true, true, "exceeds maximum input length"});
    // Model doesn't support context logits output
    mockedModelParams.emplace_back(
        MockedModelParams{10, 20, 0, 0, false, true, "need to build engine with gather_generation"});
    // Model doesn't support gen logits output
    mockedModelParams.emplace_back(
        MockedModelParams{10, 20, 0, 0, true, false, "need to build engine with gather_context"});

    for (auto const& mockedModelParam : mockedModelParams)
    {
        auto model = std::make_shared<MockedModel>();
        SizeType32 beamWidth = 1;

        // One request should be terminated
        EXPECT_CALL(*model, terminateRequest(_, _)).Times(mockedModelParam.expectedTerminateCnt);
        EXPECT_CALL(*model, getVocabSizePadded()).WillRepeatedly(Invoke([&]() { return 1024; }));
        EXPECT_CALL(*model, getLogitDataType()).WillRepeatedly(Invoke([&]() { return nvinfer1::DataType::kFLOAT; }));
        EXPECT_CALL(*model, getCurrentIterationStats(_)).WillRepeatedly(Invoke([&](IterationStats& stats) { return; }));
        EXPECT_CALL(*model, getCurrentRequestStats(_))
            .WillRepeatedly(Invoke([&](RequestStatsPerIteration& stats) { return; }));

        SizeType32 callCount = 0;
        EXPECT_CALL(*model, forwardAsync(_))
            .WillRepeatedly(Invoke(
                [&](RequestList const&)
                {
                    callCount++;
                    // There was a bug where we were missing a notify call when errors were encountered
                    // and this test was not catching it, probably because the error was reported
                    // before the first call to awaitResponses. So we add a sleep here to make sure
                    // the awaitResponses is called before the error is thrown
                    std::this_thread::sleep_for(std::chrono::milliseconds(200));
                    throw std::runtime_error("mocked error");
                }));

        EXPECT_CALL(*model, getMaxNumSequences()).WillRepeatedly(Invoke([&]() { return 10; }));
        EXPECT_CALL(*model, getMaxInputLen()).WillRepeatedly(Invoke([&]() { return mockedModelParam.maxInputLen; }));
        EXPECT_CALL(*model, getMaxSequenceLen()).WillRepeatedly(Invoke([&]() { return mockedModelParam.maxSeqLen; }));
        tr::WorldConfig const dummyWorldConfig;
        EXPECT_CALL(*model, getWorldConfig())
            .WillRepeatedly(Invoke([&]() -> tr::WorldConfig const& { return dummyWorldConfig; }));
        tr::ModelConfig dummyModelConfig(0, 0, 0, 0, 0, nvinfer1::DataType::kHALF);
        dummyModelConfig.computeContextLogits(mockedModelParam.computeContextLogits);
        dummyModelConfig.computeGenerationLogits(mockedModelParam.computeGenLogits);
        EXPECT_CALL(*model, getModelConfig())
            .WillRepeatedly(Invoke([&]() -> tr::ModelConfig const& { return dummyModelConfig; }));
        EXPECT_CALL(*model, getCurrentIterationStats(_)).WillRepeatedly(Invoke([&](IterationStats& stats) { return; }));
        EXPECT_CALL(*model, getCurrentRequestStats(_))
            .WillRepeatedly(Invoke([&](RequestStatsPerIteration& stats) { return; }));

        ExecutorConfig executorConfig(beamWidth);
        auto executor = Executor(model, executorConfig);

        // Create the request
        SizeType32 maxNewTokens = 5;
        VecTokens inputTokens{1, 2, 3, 4};

        OutputConfig outConfig;
        outConfig.returnContextLogits = true;
        outConfig.returnGenerationLogits = true;

        auto streaming = false;
        auto request = Request(
            inputTokens, maxNewTokens, streaming, tensorrt_llm::executor::SamplingConfig(beamWidth), outConfig);

        // Enqueue the request
        auto requestId = executor.enqueueRequest(std::move(request));

        bool done = false;
        auto responses = executor.awaitResponses(requestId);
        for (auto& response : responses)
        {
            if (!response.hasError())
            {
                FAIL() << "Expecting an error to be received";
            }
            else
            {
                auto err = response.getErrorMsg();
                EXPECT_THAT(err, testing::HasSubstr(mockedModelParam.expectedError));
                done = true;
            }
        }

        EXPECT_TRUE(done);
        EXPECT_EQ(callCount, mockedModelParam.expectedForwardCnt);
    }
}

TEST_F(GptExecutorTest, MockedModelCancelRequest)
{
    using LlmRequestPtr = std::shared_ptr<tb::LlmRequest>;
    using RequestList = std::list<LlmRequestPtr>;

    bool streaming = true;
    auto model = std::make_shared<MockedModel>();

    // One request should be terminated
    EXPECT_CALL(*model, terminateRequest(_, _)).Times(1);
    EXPECT_CALL(*model, getVocabSizePadded()).Times(0);
    EXPECT_CALL(*model, getLogitDataType()).Times(0);
    tr::WorldConfig const dummyWorldConfig;
    EXPECT_CALL(*model, getWorldConfig())
        .WillRepeatedly(Invoke([&]() -> tr::WorldConfig const& { return dummyWorldConfig; }));
    EXPECT_CALL(*model, getCurrentIterationStats(_)).WillRepeatedly(Invoke([&](IterationStats& stats) { return; }));
    EXPECT_CALL(*model, getCurrentRequestStats(_))
        .WillRepeatedly(Invoke([&](RequestStatsPerIteration& stats) { return; }));

    SizeType32 callCount = 0;
    EXPECT_CALL(*model, forwardAsync(_))
        .WillRepeatedly(Invoke(
            [&](RequestList const& requestList)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));

                for (auto const& llmReq : requestList)
                {
                    // Don't add any tokens to simulate no output tokens
                    llmReq->addNewTokens({1});
                    llmReq->mState = tb::REQUEST_STATE_GENERATION_IN_PROGRESS;
                    if (llmReq->getMaxNumGeneratedTokens() >= llmReq->mMaxNewTokens)
                    {
                        llmReq->mState = tb::REQUEST_STATE_GENERATION_COMPLETE;
                    }
                }
                callCount++;
            }));

    EXPECT_CALL(*model, getMaxNumSequences()).WillRepeatedly(Invoke([&]() { return 10; }));
    EXPECT_CALL(*model, getMaxInputLen()).WillRepeatedly(Invoke([&]() { return 100; }));
    EXPECT_CALL(*model, getMaxSequenceLen()).WillRepeatedly(Invoke([&]() { return 200; }));

    SizeType32 beamWidth = 1;
    ExecutorConfig executorConfig(beamWidth);
    auto executor = Executor(model, executorConfig);

    // Create the request
    SizeType32 maxNewTokens = 150;
    VecTokens inputTokens{1, 2, 3, 4};
    auto request = Request(inputTokens, maxNewTokens, streaming);

    // Enqueue the request
    auto requestId = executor.enqueueRequest(std::move(request));

    // Cancel the request
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    executor.cancelRequest(requestId);

    bool done = false;
    int iter = 0;
    while (!done && iter < mMaxWaitMs)
    {
        std::chrono::milliseconds waitTime(1);
        auto responses = executor.awaitResponses(requestId, waitTime);
        for (auto& response : responses)
        {

            if (response.hasError())
            {
                FAIL() << "Not expecting an error to be received";
            }
            else
            {
                auto result = response.getResult();
                done = result.isFinal;
            }
        }
        ++iter;
    }

    EXPECT_LT(iter, mMaxWaitMs);
    // Expecting to receiving fewer tokens than maxNewTokens
    EXPECT_LT(callCount, maxNewTokens);
}

TEST_F(GptExecutorTest, SingleRequestInvalidInputs)
{
    bool streaming = true;

    SizeType32 beamWidth = 1;
    auto executorConfig = ExecutorConfig(beamWidth);
    auto trtEnginePath = GPT_MODEL_PATH / FP16_GPT_ATTENTION_PACKED_PAGED_DIR / "tp1-pp1-gpu";
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    // Create the request
    SizeType32 maxNewTokens = 5;
    VecTokens inputTokens{1, 2, 3, 4};

    std::vector<std::string> expectedErrMsgs;
    std::vector<Request> requests;

    // Invalid embedding bias shape
    {
        requests.emplace_back(inputTokens, maxNewTokens, streaming);
        auto embeddingBias = Tensor::cpu(DataType::kFP32, {1});
        requests.back().setEmbeddingBias(embeddingBias);
        expectedErrMsgs.emplace_back("embedding bias shape is not as expected");
    }

    for (auto req = 0; req < requests.size(); ++req)
    {
        auto& request = requests.at(req);
        auto const& expectedErrMsg = expectedErrMsgs.at(req);

        auto requestId = executor.enqueueRequest(std::move(request));

        // Try to get the new tokens
        bool done = false;
        int iter = 0;
        while (!done && iter < mMaxWaitMs)
        {
            std::chrono::milliseconds waitTime(1);
            auto responses = executor.awaitResponses(requestId, waitTime);
            for (auto& response : responses)
            {
                if (response.hasError())
                {

                    auto err = response.getErrorMsg();
                    EXPECT_THAT(err, testing::HasSubstr(expectedErrMsg));
                    done = true;
                }
                else
                {
                    FAIL() << "Expected an err: " << expectedErrMsg;
                }
            }
            ++iter;
        }
        EXPECT_EQ(done, true);
    }
}

TEST_F(GptExecutorTest, SingleRequestLora)
{
    bool streaming = true;

    SizeType32 beamWidth = 1;
    auto executorConfig = ExecutorConfig(beamWidth);
    auto trtEnginePath = GPT_MODEL_PATH / FP16_GPT_ATTENTION_PACKED_PAGED_DIR / "tp1-pp1-gpu";
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    // Load lora weights, config
    auto manager = tr::BufferManager(std::make_shared<tr::CudaStream>());
    auto loraWeightsTensor
        = std::shared_ptr(tr::utils::loadNpy(manager, LORA_WEIGHTS_FILE.string(), tr::MemoryType::kCPU));
    auto loraConfigTensor
        = std::shared_ptr(tr::utils::loadNpy(manager, LORA_CONFIG_FILE.string(), tr::MemoryType::kCPU));

    // Create the request
    SizeType32 maxNewTokens = 5;
    VecTokens inputTokens{1, 2, 3, 4};
    auto request = Request(inputTokens, maxNewTokens, streaming, tensorrt_llm::executor::SamplingConfig());
    auto loraConfig = LoraConfig(0, detail::ofITensor(loraWeightsTensor), detail::ofITensor(loraConfigTensor));
    request.setLoraConfig(loraConfig);

    // Enqueue the request
    auto requestId = executor.enqueueRequest(std::move(request));

    // Get the new tokens
    VecTokens tokens;
    bool done = false;
    int iter = 0;
    std::chrono::milliseconds waitTime(1);
    while (!done && iter < mMaxWaitMs)
    {
        auto responses = executor.awaitResponses(requestId, waitTime);
        for (auto& response : responses)
        {
            if (response.hasError())
            {
                // This request failed for some reason, get error msg
                std::string errStr
                    = "Request id " + std::to_string(requestId) + " failed with err " + response.getErrorMsg();
                FAIL();
            }
            else
            {
                auto result = response.getResult();
                done = result.isFinal;
                // Append tokens
                auto& newTokens = result.outputTokenIds.at(beamWidth - 1);
                tokens.insert(
                    tokens.end(), std::make_move_iterator(newTokens.begin()), std::make_move_iterator(newTokens.end()));
            }
        }
        ++iter;
    }
    EXPECT_LT(iter, mMaxWaitMs);
    EXPECT_EQ(tokens.size(), maxNewTokens);
}

TEST_P(ParamTest, SingleRequestCancelRequest)
{
    bool const streaming = std::get<0>(GetParam());
    bool const excludeInputFromOutput = std::get<1>(GetParam());
    auto const beamWidth = std::get<2>(GetParam());
    OutputConfig outConfig;
    outConfig.excludeInputFromOutput = excludeInputFromOutput;

    auto executorConfig = ExecutorConfig(beamWidth);
    auto trtEnginePath = GPT_MODEL_PATH / FP16_GPT_ATTENTION_PACKED_PAGED_DIR / "tp1-pp1-gpu";
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    // Create the request
    SizeType32 maxNewTokens = 300;
    VecTokens inputTokens{1, 2, 3, 4};
    auto request
        = Request(inputTokens, maxNewTokens, streaming, tensorrt_llm::executor::SamplingConfig(beamWidth), outConfig);

    auto requestId = executor.enqueueRequest(std::move(request));

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    executor.cancelRequest(requestId);

    // Try to get the new tokens
    bool done = false;
    int iter = 0;
    VecTokens tokens;
    while (!done && iter < mMaxWaitMs)
    {
        std::chrono::milliseconds waitTime(1);
        auto responses = executor.awaitResponses(requestId, waitTime);
        for (auto& response : responses)
        {
            if (response.hasError())
            {
                FAIL() << "Did not expect errors";
            }
            else
            {
                auto result = response.getResult();
                done = result.isFinal;
                // Append tokens
                auto& newTokens = result.outputTokenIds.at(beamWidth - 1);
                if (streaming && beamWidth > 1)
                {
                    tokens = newTokens;
                }
                else
                {
                    tokens.insert(tokens.end(), newTokens.begin(), newTokens.end());
                }
            }
        }
        ++iter;
    }
    EXPECT_EQ(done, true);
    EXPECT_LT(iter, mMaxWaitMs);
    auto expectedNumTokens
        = streaming ? maxNewTokens : (excludeInputFromOutput ? 0 : inputTokens.size()) + maxNewTokens;
    TLLM_LOG_INFO("num tokens: %d, expected %d", tokens.size(), expectedNumTokens);
    EXPECT_LT(tokens.size(), expectedNumTokens);
}

TEST_F(GptExecutorTest, orchModeFetchNewReqErr)
{
    SizeType32 beamWidth = 1;
    auto executorConfig = ExecutorConfig(beamWidth);

    auto orchestratorConfig = OrchestratorConfig(true, EXECUTOR_WORKER_PATH.string());
    auto parallelConfig = ParallelConfig(
        CommunicationType::kMPI, CommunicationMode::kORCHESTRATOR, std::nullopt, std::nullopt, orchestratorConfig);
    executorConfig.setParallelConfig(parallelConfig);

    auto trtEnginePath = (GPT_MODEL_PATH / FP16_GPT_ATTENTION_PACKED_PAGED_DIR / "tp1-pp1-gpu");
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    // Create a req with invalid parameters
    SizeType32 maxNewTokens = 5;
    // Create very long prompt which should result in error during request validate
    VecTokens inputTokens(10000000);

    auto request = Request(inputTokens, maxNewTokens, false, tensorrt_llm::executor::SamplingConfig(beamWidth));
    auto requestId = executor.enqueueRequest(request);
    auto requestId2 = executor.enqueueRequest(request);

    bool done = false;
    int iter = 0;
    while (!done && iter < mMaxWaitMs)
    {
        std::chrono::milliseconds waitTime(1);
        auto responses = executor.awaitResponses(waitTime);
        for (auto& response : responses)
        {
            if (response.hasError())
            {
                auto err = response.getErrorMsg();
                EXPECT_THAT(err, testing::HasSubstr("exceeds maximum input length"));
                EXPECT_THAT(err, testing::HasSubstr("Encountered an error when fetching new request:"));
                done = true;
            }
            else
            {
                FAIL() << "Should get a response with error";
            }
        }
        ++iter;
    }
    EXPECT_LT(iter, mMaxWaitMs);
}

TEST_F(GptExecutorTest, orchModeForwardError)
{
    SizeType32 constexpr maxBeamWidth{1};
    auto executorConfig = ExecutorConfig(maxBeamWidth);

    auto orchestratorConfig = OrchestratorConfig(true, EXECUTOR_WORKER_PATH.string());
    auto parallelConfig = ParallelConfig(
        CommunicationType::kMPI, CommunicationMode::kORCHESTRATOR, std::nullopt, std::nullopt, orchestratorConfig);
    executorConfig.setParallelConfig(parallelConfig);

    auto trtEnginePath = (GPT_MODEL_PATH / FP16_GPT_ATTENTION_PACKED_PAGED_DIR / "tp1-pp1-gpu");
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    // Setting request beam width to 2 which should cause failure
    SizeType32 constexpr beamWidth{2};
    SizeType32 constexpr maxNewTokens{5};
    VecTokens inputTokens{1, 2, 3, 4};

    auto request = Request(inputTokens, maxNewTokens, false, tensorrt_llm::executor::SamplingConfig(beamWidth));
    auto requestId = executor.enqueueRequest(request);
    auto requestId2 = executor.enqueueRequest(request);

    bool done = false;
    int iter = 0;
    while (!done && iter < mMaxWaitMs)
    {
        std::chrono::milliseconds waitTime(1);
        auto responses = executor.awaitResponses(waitTime);
        for (auto& response : responses)
        {
            if (response.hasError())
            {
                auto err = response.getErrorMsg();
                std::cout << "err:" << err << std::endl;
                EXPECT_THAT(
                    err, testing::HasSubstr("Requested beam width 2 is larger than configured max beam width 1"));
                done = true;
            }
            else
            {
                FAIL() << "Should get a response with error";
            }
        }
        ++iter;
    }
    EXPECT_LT(iter, mMaxWaitMs);
}

TEST_P(ParamCancelReqTest, MultipleRequestsMultiGpuCancelRequest)
{
    bool const streaming = std::get<0>(GetParam());
    bool const useOrchestratorMode = std::get<1>(GetParam());
    auto const beamWidth = std::get<2>(GetParam());
    auto const modelName = std::get<3>(GetParam());

    OutputConfig outConfig;

    auto executorConfig = ExecutorConfig(beamWidth);
    std::filesystem::path modelPath;
    if (modelName == "llama_tp4_pp1" || modelName == "llama_tp1_pp4" || modelName == "llama_tp2_pp2")
    {
        if (modelName == "llama_tp4_pp1")
        {
            modelPath = LLAMA_MODEL_PATH / FP16_GPT_ATTENTION_PACKED_PAGED_DIR / "tp4-pp1-gpu";
        }
        else if (modelName == "llama_tp1_pp4")
        {
            modelPath = LLAMA_MODEL_PATH / FP16_GPT_ATTENTION_PACKED_PAGED_DIR / "tp1-pp4-gpu";
        }
        else if (modelName == "llama_tp2_pp2")
        {
            modelPath = LLAMA_MODEL_PATH / FP16_GPT_ATTENTION_PACKED_PAGED_DIR / "tp2-pp2-gpu";
        }
    }

    // For llama model, only run for multiple GPUs
    // This is detected by setting an env variable when running the test
    char const* val = getenv("RUN_LLAMA_MULTI_GPU");
    if (val == NULL)
    {
        GTEST_SKIP() << "Skipping Llama test";
    }
    else
    {
        // Check that it was launched with right number of MPI ranks
        if (!useOrchestratorMode && COMM_SESSION.getSize() != 4)
        {
            // No orchestrator, need worldSize to match TP*PP
            FAIL() << "Leader mode and world size is not equal to 4";
        }
        else if (useOrchestratorMode && COMM_SESSION.getSize() != 1)
        {
            // No orchestrator, need worldSize to match TP*PP
            FAIL() << "Orchestrator mode and World size is not equal to 1";
        }
    }

    if (useOrchestratorMode)
    {
        auto orchestratorConfig = OrchestratorConfig(true, EXECUTOR_WORKER_PATH.string());
        auto parallelConfig = ParallelConfig(CommunicationType::kMPI,
            useOrchestratorMode ? CommunicationMode::kORCHESTRATOR : CommunicationMode::kLEADER, std::nullopt,
            std::nullopt, orchestratorConfig);
        executorConfig.setParallelConfig(parallelConfig);
    }

    auto executor = Executor(modelPath, ModelType::kDECODER_ONLY, executorConfig);

    // Create the request
    SizeType32 maxNewTokens = 50;
    VecTokens inputTokens{1, 2, 3, 4};
    auto request
        = Request(inputTokens, maxNewTokens, streaming, tensorrt_llm::executor::SamplingConfig(beamWidth), outConfig);

    if (executor.canEnqueueRequests())
    {
        auto requestId = executor.enqueueRequest(request);
        // Enqueue another request
        auto requestId2 = executor.enqueueRequest(request);

        // Cancel the first request
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        executor.cancelRequest(requestId);

        SizeType32 expectedNumToken = (streaming ? 0 : inputTokens.size()) + maxNewTokens;

        std::unordered_map<IdType, VecTokens> tokens;
        tokens[requestId] = {};
        tokens[requestId2] = {};

        std::unordered_map<IdType, SizeType32> expectedNumTokens;
        expectedNumTokens[requestId] = expectedNumToken;
        expectedNumTokens[requestId2] = expectedNumToken;

        // Get the new tokens for each requests
        int32_t numFinished = 0;
        int iter = 0;
        SizeType32 numResponses = 0;
        while (numFinished < 2 && iter < mMaxWaitMs)
        {
            std::chrono::milliseconds waitTime(1);
            auto responses = executor.awaitResponses(waitTime);
            for (auto& response : responses)
            {
                numResponses++;
                if (!response.hasError())
                {
                    auto result = response.getResult();
                    numFinished += result.isFinal;
                    auto& newTokens = result.outputTokenIds.at(beamWidth - 1);
                    auto& reqTokens = tokens.at(response.getRequestId());
                    if (streaming && beamWidth > 1)
                    {
                        reqTokens = newTokens;
                    }
                    else
                    {
                        reqTokens.insert(reqTokens.end(), newTokens.begin(), newTokens.end());
                    }
                }
                else
                {
                    FAIL() << "Did not expect errors";
                }
            }
            ++iter;
        }

        EXPECT_LE(numResponses, streaming ? 2 * expectedNumToken : 2);
        EXPECT_EQ(numFinished, 2);
        EXPECT_LT(iter, mMaxWaitMs);

        std::cout << "req1: " << tokens[requestId].size() << " expected:" << expectedNumTokens[requestId] << std::endl;
        EXPECT_LT(tokens[requestId].size(), expectedNumTokens[requestId]);
        EXPECT_EQ(tokens[requestId2].size(), expectedNumTokens[requestId2]);
    }
}

TEST_F(GptExecutorTest, validateParallelConfig)
{

    auto trtEnginePath = (GPT_MODEL_PATH / FP16_GPT_ATTENTION_PACKED_PAGED_DIR / "tp1-pp1-gpu");
    {
        auto executorConfig = ExecutorConfig();
        auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);
    }

    {
        std::string expectedErrMsg = "OrchestratorConfig must be set";
        try
        {
            auto executorConfig = ExecutorConfig();
            auto parallelConfig = ParallelConfig(CommunicationType::kMPI, CommunicationMode::kORCHESTRATOR);
            executorConfig.setParallelConfig(parallelConfig);
            auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);
            FAIL() << "Expected TllmException";
        }
        catch (tc::TllmException& e)
        {
            EXPECT_THAT(e.what(), testing::HasSubstr(expectedErrMsg));
        }
        catch (std::exception const& e)
        {
            FAIL() << "Expected TllmException";
        }
    }
}

TEST_P(EncDecParamsTest, validEncDecCtor)
{
    auto const modelName = std::get<0>(GetParam());
    SizeType32 const beamWidth = std::get<1>(GetParam());
    SizeType32 const maxNewTokens = std::get<2>(GetParam());
    SizeType32 const tp = std::get<3>(GetParam());
    SizeType32 const pp = std::get<4>(GetParam());

    auto const enginePathName = getEncDecEnginePath(modelName, tp, pp);
    std::filesystem::path encEnginePath = ENC_DEC_ENGINE_BASE / enginePathName / "encoder";
    std::filesystem::path decEnginePath = ENC_DEC_ENGINE_BASE / enginePathName / "decoder";
    ExecutorConfig executorConfig{};
    FloatType freeGpuMemoryFraction = 0.45f;
    KvCacheConfig kvCacheConfig{false, std::nullopt, std::nullopt, std::nullopt, freeGpuMemoryFraction};
    executorConfig.setKvCacheConfig(kvCacheConfig);
    auto executor = Executor(encEnginePath, decEnginePath, ModelType::kENCODER_DECODER, executorConfig);
}

TEST_P(EncDecParamsTest, Forward)
{
    bool constexpr VERBOSE = false;
    auto const modelName = std::get<0>(GetParam());
    SizeType32 const beamWidth = std::get<1>(GetParam());
    SizeType32 const maxNewTokens = std::get<2>(GetParam());
    SizeType32 const tp = std::get<3>(GetParam());
    SizeType32 const pp = std::get<4>(GetParam());
    bool const streaming = false;

    auto const enginePathName = getEncDecEnginePath(modelName, tp, pp);
    std::filesystem::path encEnginePath = ENC_DEC_ENGINE_BASE / enginePathName / "encoder";
    std::filesystem::path decEnginePath = ENC_DEC_ENGINE_BASE / enginePathName / "decoder";

    // load ground truth input & output data
    auto manager = tr::BufferManager(std::make_shared<tr::CudaStream>());
    auto inputsIdsHost
        = tr::utils::loadNpy(manager, (ENC_DEC_DATA_BASE / "input_ids.npy").string(), tr::MemoryType::kCPU);
    auto inputsIdsPtr = tr::bufferCast<TokenIdType>(*inputsIdsHost);
    auto inputLengthsHost
        = tr::utils::loadNpy(manager, (ENC_DEC_DATA_BASE / "input_lengths.npy").string(), tr::MemoryType::kCPU);
    auto inputLengthsPtr = tr::bufferCast<SizeType32>(*inputLengthsHost);
    auto encoderOutputHost
        = tr::utils::loadNpy(manager, (ENC_DEC_DATA_BASE / "encoder_output.npy").string(), tr::MemoryType::kCPU);
    auto encoderOutputPtr = tr::bufferCast<half>(*encoderOutputHost);
    auto decoderOutputHost
        = tr::utils::loadNpy(manager, (ENC_DEC_DATA_BASE / "output_ids.npy").string(), tr::MemoryType::kCPU);
    auto decoderOutputPtr = tr::bufferCast<TokenIdType>(*decoderOutputHost);

    // Rank and size info
    auto& comm = tensorrt_llm::mpi::MpiComm::world();
    auto const worldRank = comm.getRank();
    auto const worldSize = comm.getSize();

    // create executor
    BatchingType const batchingType = BatchingType::kINFLIGHT;
    FloatType freeGpuMemoryFraction = 0.45f;
    KvCacheConfig kvCacheConfig{false, std::nullopt, std::nullopt, std::nullopt, freeGpuMemoryFraction};

    ExecutorConfig executorConfig{beamWidth};
    executorConfig.setBatchingType(batchingType);
    executorConfig.setKvCacheConfig(kvCacheConfig);
    executorConfig.setNormalizeLogProbs(false);

    // TODO: OrchestratorMode test does not pass
    bool const useOrchestratorMode = (tp * pp) > worldSize;
    std::optional<OrchestratorConfig> orchestratorConfig = std::nullopt;
    if (useOrchestratorMode)
    {
        orchestratorConfig = OrchestratorConfig(true, EXECUTOR_WORKER_PATH.string());
    }
    auto parallelConfig = ParallelConfig(CommunicationType::kMPI,
        useOrchestratorMode ? CommunicationMode::kORCHESTRATOR : CommunicationMode::kLEADER, std::nullopt, std::nullopt,
        orchestratorConfig);
    executorConfig.setParallelConfig(parallelConfig);

    auto executor = Executor(encEnginePath, decEnginePath, ModelType::kENCODER_DECODER, executorConfig);

    OutputConfig outConfig;
    outConfig.excludeInputFromOutput = false;
    outConfig.returnLogProbs = false;
    outConfig.returnGenerationLogits = false;
    outConfig.returnContextLogits = false;
    outConfig.returnEncoderOutput = false;

    TokenIdType bosId = getDecTokenFromJsonConfig(decEnginePath, "bos_token_id");
    TokenIdType padId = getDecTokenFromJsonConfig(decEnginePath, "pad_token_id");
    TokenIdType eosId = getDecTokenFromJsonConfig(decEnginePath, "eos_token_id");
    TokenIdType decoderStartTokenId = getDecTokenFromJsonConfig(decEnginePath, "decoder_start_token_id");

    // create requests
    SizeType32 const nbRequests = inputLengthsHost->getShape().d[0];
    std::vector<Request> requests;
    for (int i = 0, cumInputLen = 0; i < nbRequests; i++)
    {
        auto encoderInput = VecTokens(&inputsIdsPtr[cumInputLen],
            &inputsIdsPtr[cumInputLen] + inputLengthsPtr[i]); // assume inputIds is flattened / no-padding
        cumInputLen += inputLengthsPtr[i];
        auto decoderInput = VecTokens{decoderStartTokenId};
        Request req(decoderInput, maxNewTokens, streaming, tensorrt_llm::executor::SamplingConfig(beamWidth), outConfig,
            eosId, padId);
        req.setEncoderInputTokenIds(encoderInput);
        requests.emplace_back(req);
    }

    using namespace std::chrono;

    // enqueue requests
    if (worldRank == 0)
    {
        auto tik = high_resolution_clock::now();
        std::vector<IdType> reqIds = executor.enqueueRequests(std::move(requests));

        // get responses
        milliseconds waitTime(5000);
        auto responsesAll = executor.awaitResponses(reqIds, waitTime);
        auto tok = high_resolution_clock::now();
        TLLM_LOG_DEBUG("TRT-LLM C++ E2E time %d ms", duration_cast<milliseconds>(tok - tik).count());
        TLLM_LOG_DEBUG("Number of responses: %d", responsesAll.size());

        int32_t numFinished = 0;
        int iter = 0;
        SizeType32 numResponses = 0;
        std::unordered_map<IdType, std::vector<VecTokens>> outputTokens;
        for_each(reqIds.begin(), reqIds.end(),
            [&outputTokens, &beamWidth](auto const& id)
            {
                TLLM_LOG_DEBUG("Request IDs: %d", id);
                outputTokens[id] = {};
                for (int i = 0; i < beamWidth; i++)
                {
                    outputTokens[id].emplace_back(VecTokens{});
                }
            });
        for (int i = 0; i < reqIds.size(); i++)
        {
            auto& responses = responsesAll[i];
            for (auto& response : responses)
            {
                numResponses++;
                if (!response.hasError())
                {
                    auto result = response.getResult();
                    numFinished += result.isFinal;
                    for (int beam = 0; beam < beamWidth; beam++)
                    {
                        auto& resTokens = result.outputTokenIds.at(beam);
                        auto& outTokens = outputTokens.at(response.getRequestId()).at(beam);
                        outTokens.insert(outTokens.end(), std::make_move_iterator(resTokens.begin()),
                            std::make_move_iterator(resTokens.end()));
                    }
                }
                else
                {
                    // Allow response with error only if awaitResponse processed a terminated request id
                    std::string err = "ReqId " + std::to_string(response.getRequestId())
                        + " has already been processed and was terminated.";
                    EXPECT_EQ(response.getErrorMsg(), err);
                }
            }
        }

        // print output & check correctness with ground truth
        for (auto const& [reqId, tokens] : outputTokens)
        {
            SizeType32 gtMaxLength = decoderOutputHost->getShape().d[1];
            auto gtOutput = decoderOutputPtr + (reqId - 1) * gtMaxLength;

            if constexpr (VERBOSE)
            {
                std::cout << ">>> Request ID: " << reqId << std::endl;
                for (int beam = 0; beam < beamWidth; beam++)
                {
                    std::cout << "output tokens, beam " << beam << ", output length " << tokens[beam].size() << ": "
                              << std::endl;
                    for_each(tokens[beam].begin(), tokens[beam].end(),
                        [](auto const& token) { std::cout << token << ", "; });
                    std::cout << std::endl;
                }
                std::cout << "ground truth tokens: " << std::endl;

                SizeType32 gtLength = 0;
                for (int i = 0; i < gtMaxLength; i++)
                {
                    if (gtOutput[i] != eosId)
                    {
                        std::cout << gtOutput[i] << ", ";
                        gtLength++;
                    }
                }
                std::cout << std::endl;
                std::cout << "ground truth length: " << gtLength << std::endl;
            }

            // check token-by-token match between beam 0 & ground truth
            ASSERT_TRUE(tokens.size() <= gtMaxLength)
                << "Request ID " << reqId << "'s generated length is longer than ground truth length" << gtMaxLength;
            for (int i = 0; i < gtMaxLength; i++)
            {
                if (outConfig.excludeInputFromOutput)
                {
                    // if results exclude decoder start token, skip it in ground truth too
                    continue;
                }
                if (i < tokens[0].size())
                {
                    ASSERT_EQ(tokens[0][i], gtOutput[i])
                        << "Generated token id: " << tokens[0][i] << " v.s. ground truth: " << gtOutput[i];
                }
                else
                {
                    ASSERT_EQ(gtOutput[i], eosId)
                        << "Request ID " << reqId << "'s generated length is shorter than ground truth length"
                        << gtMaxLength;
                }
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(GptExecutorTest, ParamTest,
    testing::Combine(testing::Values(false, true), // streaming
        testing::Values(false, true),              // excludeInputFromOutput
        testing::Values(1, 2)                      // beamWidth
        ),
    generateTestName);

INSTANTIATE_TEST_SUITE_P(GptExecutorTest, ParamStatsTest,
    testing::Combine(testing::Values(0, 1000), testing::Values(false, true)), generateTestNameStats);

INSTANTIATE_TEST_SUITE_P(LlamaExecutorTest, ParamCancelReqTest,
    testing::Combine(testing::Values(false, true), testing::Values(false, true), testing::Values(1, 2),
        testing::Values("llama_tp1_pp4", "llama_tp4_pp1", "llama_tp2_pp2")),
    generateTestNameCancelReq);

INSTANTIATE_TEST_SUITE_P(GptExecutorTest, AllParamsTest,
    testing::Combine(testing::Values(BatchingType::kSTATIC, BatchingType::kINFLIGHT), testing::Values(false, true),
        testing::Values(1, 2), testing::Values(false, true), testing::Values(false, true), testing::Values(false, true),
        testing::Values(false, true), testing::Values("gpt"), testing::Values(false, true)),
    generateTestNameAllParams);

INSTANTIATE_TEST_SUITE_P(LlamaExecutorTest, AllParamsTest,
    testing::Combine(testing::Values(BatchingType::kINFLIGHT), testing::Values(false, true), testing::Values(1, 2),
        testing::Values(false, true), testing::Values(false, true), testing::Values(false, true),
        testing::Values(false, true), testing::Values("llama_tp1_pp4", "llama_tp4_pp1", "llama_tp2_pp2"),
        testing::Values(false, true)),
    generateTestNameAllParams);

INSTANTIATE_TEST_SUITE_P(MedusaExecutorTest, AllParamsTest,
    testing::Combine(testing::Values(BatchingType::kINFLIGHT), // batchingType
        testing::Values(false, true),                          // streaming
        testing::Values(1),                                    // beamWidth
        testing::Values(false),                                // computeLogProbs
        testing::Values(false, true),                          // excludeInputInOutput
        testing::Values(false),                                // returnContextLogits
        testing::Values(false),                                // returnGenerationLogits
        testing::Values("medusa"),                             // modelName
        testing::Values(false, true)                           // useOrchestratorMode
        ),
    generateTestNameAllParams);

// Disable some of ChatGLM's tests since they are the same as gpt's.
INSTANTIATE_TEST_SUITE_P(ChatGlmExecutorTest, AllParamsTest,
    testing::Combine(testing::Values(BatchingType::kSTATIC, BatchingType::kINFLIGHT), testing::Values(false),
        testing::Values(1, 2), testing::Values(false), testing::Values(false), testing::Values(false),
        testing::Values(false), testing::Values("chatglm"), testing::Values(false)),
    generateTestNameAllParams);

// ChatGlm0 Test is for glm-10b.
INSTANTIATE_TEST_SUITE_P(ChatGlm0ExecutorTest, AllParamsTest,
    testing::Combine(testing::Values(BatchingType::kINFLIGHT), testing::Values(false), testing::Values(1),
        testing::Values(false), testing::Values(false), testing::Values(false), testing::Values(false),
        testing::Values("glm"), testing::Values(false)),
    generateTestNameAllParams);

INSTANTIATE_TEST_SUITE_P(ChatGlm2ExecutorTest, AllParamsTest,
    testing::Combine(testing::Values(BatchingType::kSTATIC), testing::Values(false), testing::Values(1),
        testing::Values(false), testing::Values(false), testing::Values(false), testing::Values(false),
        testing::Values("chatglm2"), testing::Values(false)),
    generateTestNameAllParams);

INSTANTIATE_TEST_SUITE_P(ChatGlm3ExecutorTest, AllParamsTest,
    testing::Combine(testing::Values(BatchingType::kSTATIC), testing::Values(false), testing::Values(1),
        testing::Values(false), testing::Values(false), testing::Values(false), testing::Values(false),
        testing::Values("chatglm3"), testing::Values(false)),
    generateTestNameAllParams);

INSTANTIATE_TEST_SUITE_P(T5BasicTest, EncDecParamsTest,
    testing::Combine(
        testing::Values(T5_NAME), testing::Values(1), testing::Values(64), testing::Values(1), testing::Values(1)),
    generateTestNameEncDec);

INSTANTIATE_TEST_SUITE_P(T5MultiGPUTest, EncDecParamsTest,
    testing::Combine(
        testing::Values(T5_NAME), testing::Values(1), testing::Values(64), testing::Values(4), testing::Values(1)),
    generateTestNameEncDec);

INSTANTIATE_TEST_SUITE_P(BartBasicTest, EncDecParamsTest,
    testing::Combine(
        testing::Values(BART_NAME), testing::Values(1), testing::Values(64), testing::Values(1), testing::Values(1)),
    generateTestNameEncDec);

INSTANTIATE_TEST_SUITE_P(BartMultiGPUTest, EncDecParamsTest,
    testing::Combine(
        testing::Values(BART_NAME), testing::Values(1), testing::Values(64), testing::Values(4), testing::Values(1)),
    generateTestNameEncDec);
