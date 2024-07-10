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

#include "tensorrt_llm/batch_manager/trtGptModel.h"
#include "tensorrt_llm/batch_manager/trtGptModelFactory.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/mpiUtils.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"
#include "tensorrt_llm/runtime/speculativeDecodingMode.h"
#include "tensorrt_llm/runtime/tllmLogger.h"
#include "tensorrt_llm/runtime/utils/numpyUtils.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm::batch_manager;
namespace fs = std::filesystem;
namespace tc = tensorrt_llm::common;
namespace texec = tensorrt_llm::executor;

namespace
{
using TensorPtr = tensorrt_llm::runtime::ITensor::SharedPtr;

auto const TEST_RESOURCE_PATH = fs::path{TOP_LEVEL_DIR} / "cpp/tests/resources";
auto const ENGINE_PATH = TEST_RESOURCE_PATH / "models/rt_engine";
auto const DATA_PATH = TEST_RESOURCE_PATH / "data";

auto constexpr GPT_MODEL_DIR = "gpt2";
auto constexpr GPTJ_MODEL_DIR = "gpt-j-6b";
auto constexpr LLAMA_MODEL_DIR = "llama-7b-hf";
auto constexpr MEDUSA_MODEL_DIR = "vicuna-7b-v1.3";
auto constexpr MAMBA_MODEL_DIR = "mamba-2.8b-hf";
auto constexpr RECURRENTGEMMA_MODEL_DIR = "recurrentgemma-2b";
auto constexpr EXPLICIT_DRAFT_MODEL_DIR = "vicuna-explicit_draft";
auto constexpr CHATGLM_MODEL_DIR = "chatglm-6b";

auto constexpr FP16_GPT_ATTENTION_PACKED_DIR = "fp16-plugin-packed";
auto constexpr FP16_GPT_ATTENTION_PACKED_PAGED_DIR = "fp16-plugin-packed-paged";
auto constexpr FP16_GPT_ATTENTION_PACKED_PAGED_DRAFT_TOKENS_DIR = "fp16-plugin-packed-paged-draft-tokens";
auto constexpr FP16_GPT_ATTENTION_PACKED_PAGED_NPROFILES_DIR = "fp16-plugin-packed-paged-nprofiles";
auto constexpr FP16_GPT_ATTENTION_PACKED_PAGED_LA_DECODING_DIR = "fp16-plugin-packed-paged-la-decoding";
auto constexpr FP16_GPT_ATTENTION_PACKED_PAGED_SQ_DIR = "fp16-plugin-packed-paged-sq";
auto constexpr FP16_GPT_ATTENTION_PACKED_PAGED_IN128_DIR = "fp16-plugin-packed-paged-in128";
auto constexpr FP16_GPT_ATTENTION_PACKED_PAGED_GATHER_DIR = "fp16-plugin-packed-paged-gather";
auto constexpr FP16_GPT_ATTENTION_PACKED_PAGED_RETURN_ACCEPTED_TOKENS_LOGITS_DIR
    = "fp16-plugin-packed-paged-return-accepted-tokens-logits";

auto constexpr FP8_GPT_ATTENTION_PLUGIN_IFB_PACKED_PATH = "fp8-plugin";

auto constexpr INPUT_FILE = "input_tokens.npy";
auto constexpr LONG_INPUT_FILE = "input_tokens_long.npy";
auto constexpr CHATGLM_INPUT_FILE = "input_tokens_chatglm-6b.npy";

// Expected outputs need to be generated using cpp/tests/resources/scripts/generate_expected_gpt_output.py.
auto constexpr FP16_PLUGIN_PACKED_RESULT_FILE = "output_tokens_fp16_plugin_packed_tp1_pp1.npy";
auto constexpr FP16_PLUGIN_PACKED_PAGED_RESULT_FILE = "output_tokens_fp16_plugin_packed_paged_tp1_pp1.npy";
auto constexpr FP16_PLUGIN_PACKED_PAGED_SQ_RESULT_FILE = "output_tokens_fp16_plugin_packed_paged_sq_tp1_pp1.npy";
auto constexpr FP16_PLUGIN_PACKED_PAGED_LONG_RESULT_FILE = "output_tokens_long_fp16_plugin_packed_paged_tp1_pp1.npy";
auto constexpr FP16_PLUGIN_PACKED_PAGED_LONG_INPUT_RESULT_FILE
    = "output_tokens_long_input_fp16_plugin_packed_paged_tp1_pp1.npy";
auto constexpr FP16_PLUGIN_PACKED_PAGED_GATHER_RESULT_FILE
    = "output_tokens_fp16_plugin_packed_paged_gather_tp1_pp1.npy";
auto constexpr FP16_PLUGIN_PACKED_PAGED_GENERATION_LOGITS_FILE
    = "output_tokens_fp16_plugin_packed_paged_gather_tp1_pp1_logits_generation.npy";
auto constexpr FP16_PLUGIN_PACKED_PAGED_CONTEXT_LOGITS_FILE
    = "output_tokens_fp16_plugin_packed_paged_gather_tp1_pp1_logits_context.npy";
auto constexpr FP16_PLUGIN_PACKED_PAGED_CUM_LOG_PROBS_FILE
    = "output_tokens_fp16_plugin_packed_paged_tp1_pp1_cum_log_probs.npy";
auto constexpr FP16_PLUGIN_PACKED_PAGED_LOG_PROBS_FILE = "output_tokens_fp16_plugin_packed_paged_tp1_pp1_log_probs.npy";
auto constexpr FP16_PLUGIN_PACKED_PAGED_RESULT_TP1_PP4_FILE = "output_tokens_fp16_plugin_packed_paged_tp1_pp4.npy";
auto constexpr FP16_PLUGIN_PACKED_PAGED_RESULT_TP4_PP1_FILE = "output_tokens_fp16_plugin_packed_paged_tp4_pp1.npy";
auto constexpr FP16_PLUGIN_PACKED_PAGED_RESULT_TP2_PP2_FILE = "output_tokens_fp16_plugin_packed_paged_tp2_pp2.npy";

auto constexpr FP8_PLUGIN_RESULT_FILE = "output_tokens_fp8_plugin_tp1_pp1.npy";

inline bool almostEqual(float a, float b, float atol = 1e-2, float rtol = 1e-3)
{
    // Params: a = value to compare and b = reference
    // This function follows implementation of numpy.isclose(), which checks
    //   abs(a - b) <= (atol + rtol * abs(b)).
    // Note that the inequality above is asymmetric where b is considered as
    // a reference value. To account into both absolute/relative errors, it
    // uses absolute tolerance and relative tolerance at the same time. The
    // default values of atol and rtol borrowed from numpy.isclose(). For the
    // case of nan value, the result will be true.
    if (isnan(a) && isnan(b))
    {
        return true;
    }
    return fabs(a - b) <= (atol + rtol * fabs(b));
}

struct ModelIds
{
    int endId;
    int padId;
};

struct ModelParams
{
    char const* baseDir;
    ModelIds ids;
};

class ModelSpec
{
public:
    ModelSpec(fs::path modelPath, fs::path inputFile, fs::path resultsFile, nvinfer1::DataType dtype,
        fs::path generationLogitsFile = "", fs::path contextLogitsFile = "", fs::path cumLogProbsFile = "",
        fs::path logProbsFile = "")
        : mModelPath{std::move(modelPath)}
        , mInputFile{std::move(inputFile)}
        , mResultsFile{std::move(resultsFile)}
        , mDataType{dtype}
        , mGenerationLogitsFile{std::move(generationLogitsFile)}
        , mContextLogitsFile{std::move(contextLogitsFile)}
        , mCumLogProbsFile{std::move(cumLogProbsFile)}
        , mLogProbsFile{std::move(logProbsFile)}
        , mBatchSizes{1, 2, 8}
    {
    }

    ModelSpec& useGptAttentionPlugin()
    {
        mUseGptAttentionPlugin = true;
        return *this;
    }

    ModelSpec& usePackedInput()
    {
        mUsePackedInput = true;
        return *this;
    }

    ModelSpec& usePagedKvCache()
    {
        mUsePagedKvCache = true;
        return *this;
    }

    ModelSpec& useDecoderPerRequest()
    {
        mDecoderPerRequest = true;
        return *this;
    }

    ModelSpec& useTensorParallelism(int tensorParallelism)
    {
        mTPSize = tensorParallelism;
        return *this;
    }

    ModelSpec& usePipelineParallelism(int pipelineParallelism)
    {
        mPPSize = pipelineParallelism;
        return *this;
    }

    ModelSpec& useRandomEndId()
    {
        mRandomEndId = true;
        return *this;
    }

    ModelSpec& setDraftTokens(SizeType32 maxDraftTokens)
    {
        mSpecDecodingMode = SpeculativeDecodingMode::DraftTokensExternal();
        mMaxDraftTokens = maxDraftTokens;
        return *this;
    }

    ModelSpec& useAcceptByLogits()
    {
        mAcceptDraftByLogits = true;
        return *this;
    }

    ModelSpec& gatherLogits()
    {
        mGatherLogits = true;
        return *this;
    }

    ModelSpec& returnAcceptedTokensLogits()
    {
        mReturnAcceptedTokensLogits = true;
        return *this;
    }

    ModelSpec& replaceLogits()
    {
        mReplaceLogits = true;
        return *this;
    }

    ModelSpec& returnLogProbs()
    {
        mReturnLogProbs = true;
        return *this;
    }

    ModelSpec& smokeTest()
    {
        mSmokeTest = true;
        return *this;
    }

    ModelSpec& useMedusa()
    {
        mSpecDecodingMode = SpeculativeDecodingMode::Medusa();
        return *this;
    }

    ModelSpec& useLookaheadDecoding()
    {
        mSpecDecodingMode = SpeculativeDecodingMode::LookaheadDecoding();
        return *this;
    }

    ModelSpec& useExplicitDraftTokensDecoding()
    {
        mSpecDecodingMode = SpeculativeDecodingMode::ExplicitDraftTokens();
        return *this;
    }

    [[nodiscard]] bool useLogits() const
    {
        return mGatherLogits || mReplaceLogits;
    }

    ModelSpec& setBatchSizes(std::vector<SizeType32> batchSizes)
    {
        mBatchSizes = std::move(batchSizes);
        return *this;
    }

    fs::path mModelPath;
    fs::path mInputFile;
    fs::path mResultsFile;
    nvinfer1::DataType mDataType;
    fs::path mGenerationLogitsFile;
    fs::path mContextLogitsFile;
    fs::path mCumLogProbsFile;
    fs::path mLogProbsFile;
    bool mUseGptAttentionPlugin{false};
    bool mUsePackedInput{false};
    bool mUsePagedKvCache{false};
    bool mDecoderPerRequest{false};
    int mPPSize{1};
    int mTPSize{1};
    bool mRandomEndId{false};
    int mMaxDraftTokens{0};
    bool mAcceptDraftByLogits{false};
    bool mGatherLogits{false};
    bool mReplaceLogits{false};
    bool mReturnLogProbs{false};
    bool mSmokeTest{false};
    bool mReturnAcceptedTokensLogits{false};
    SpeculativeDecodingMode mSpecDecodingMode{SpeculativeDecodingMode::None()};
    std::vector<SizeType32> mBatchSizes;
};

using BeamResults
    = std::vector<std::tuple<SizeType32, fs::path, std::pair<fs::path, fs::path>, std::pair<fs::path, fs::path>>>;

} // namespace

class TrtModelRealDecoderTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
protected:
    TrtModelRealDecoderTest() {}

    void SetUp() override
    {
        mDeviceCount = tc::getDeviceCount();
        if (mDeviceCount == 0)
        {
            GTEST_SKIP() << "No GPUs found";
        }

        mLogger = std::make_shared<TllmLogger>();

        initTrtLlmPlugins(mLogger.get());
    }

    void TearDown() override {}

    int mDeviceCount{};
    std::shared_ptr<nvinfer1::ILogger> mLogger{};
};

enum class TrtGptModelIfbTestType
{
    BULK,
    WAVEFRONT,
    RANDOM
};

namespace
{

std::tuple<SizeType32, SizeType32> getRequestGivenInputIdxLength(
    std::uint64_t requestId, SizeType32 nbGivenInputs, std::vector<SizeType32> const& givenInputLengths)
{
    auto const givenInputIdx = requestId % nbGivenInputs;
    auto const inputLength = givenInputLengths.at(givenInputIdx);
    return {givenInputIdx, inputLength};
}

std::tuple<std::vector<SizeType32>, SizeType32, SizeType32> getGivenInputLengths(
    ITensor const& givenInput, SizeType32 padId)
{
    auto const& inputShape = givenInput.getShape();
    auto const nbGivenInputs = static_cast<SizeType32>(inputShape.d[0]);
    auto const maxInputLength = static_cast<SizeType32>(inputShape.d[1]);
    auto const* const givenInputData = bufferCast<TokenIdType const>(givenInput);

    std::vector<SizeType32> givenInputLengths(nbGivenInputs);
    for (SizeType32 i = 0; i < nbGivenInputs; ++i)
    {
        auto const* const seqBegin = givenInputData + i * maxInputLength;
        auto const* const it = std::find(seqBegin, seqBegin + maxInputLength, padId);
        givenInputLengths[i] = std::distance(seqBegin, it);
    }

    return {givenInputLengths, nbGivenInputs, maxInputLength};
}

struct TestData
{
    ITensor::SharedPtr expectedOutputIds;
    std::vector<SizeType32> expectedOutputLengths;
    SizeType32 maxSeqLen;
    std::vector<SizeType32> endIds;
    std::vector<LlmRequest::VecTokens> draftTokens;
    std::vector<ITensor::SharedPtr> draftLogits;
    std::vector<SizeType32> acceptedDraftTokensLengths;
    std::vector<ITensor::SharedPtr> expectedGenerationLogits;
    std::vector<ITensor::SharedPtr> expectedContextLogits;
    std::vector<ITensor::SharedPtr> expectedCumLogProbs;
    std::vector<ITensor::SharedPtr> expectedLogProbs;
};

template <typename T>
bool invokeCompareLogits(
    ITensor const& groundTruthLogits, ITensor const& outputLogits, float atol = 1e-2, float rtol = 1e-3)
{
    bool allMatch = true;
    T const* const gtLogitsPtr = bufferCast<T>(groundTruthLogits);
    T const* const outputLogitsPtr = bufferCast<T>(outputLogits);

    size_t outputSize = outputLogits.getSize();
    int errorNumber = 0;

    for (size_t i = 0; i < outputSize; i++)
    {
        if (!almostEqual(outputLogitsPtr[i], gtLogitsPtr[i], atol, rtol))
        {
            TLLM_LOG_DEBUG("Mismatch value. Position of logits: %d, expected value: %f, output value: %f", i,
                gtLogitsPtr[i], outputLogitsPtr[i]);
            allMatch = false;
            errorNumber++;
            if (errorNumber == 10)
            {
                break;
            }
        }
    }
    return allMatch;
}

bool compareLogits(ITensor const& groundTruthLogits, ITensor const& outputLogits, float atol = 1e-2, float rtol = 1e-3)
{
    EXPECT_EQ(groundTruthLogits.getDataType(), outputLogits.getDataType());
    switch (groundTruthLogits.getDataType())
    {
    case nvinfer1::DataType::kFLOAT: return invokeCompareLogits<float>(groundTruthLogits, outputLogits, atol, rtol);
    case nvinfer1::DataType::kHALF: return invokeCompareLogits<half>(groundTruthLogits, outputLogits, atol, rtol);
    default: TLLM_THROW("Unsupported data type");
    }
}

void verifyOutput(RequestList const& finishedRequestList,
    std::unordered_map<SizeType32, TestData> const& beamWidthTestData, std::vector<SizeType32> const& givenInputLengths,
    SizeType32 nbGivenInputs, ModelSpec const& modelSpec)
{
    auto const checkRawLogits = modelSpec.mGatherLogits;
    auto const smokeTest = modelSpec.mSmokeTest;
    auto const returnLogProbs = modelSpec.mReturnLogProbs;
    auto const checkAcceptedTokenLogits = modelSpec.mReturnAcceptedTokensLogits;

    if (smokeTest)
    {
        return;
    }

    for (auto const& llmReqPtr : finishedRequestList)
    {
        auto const& llmReq = *llmReqPtr;
        auto const requestId = llmReq.mRequestId;
        auto const [givenInputIdx, inputLength]
            = getRequestGivenInputIdxLength(requestId, nbGivenInputs, givenInputLengths);
        auto const reqBeamWidth = llmReq.mSamplingConfig.beamWidth;
        auto const& testData = beamWidthTestData.at(reqBeamWidth);
        auto const* const expectedOutputData = bufferCast<TokenIdType const>(*testData.expectedOutputIds);
        auto const expectedOutputLengths = testData.expectedOutputLengths;
        auto const acceptedDraftTokensLengths = testData.acceptedDraftTokensLengths;
        auto const endId = testData.endIds[givenInputIdx];
        auto const maxSeqLen = testData.maxSeqLen;
        auto const draftLogits = testData.draftLogits;
        auto const expectedGenerationLogits = testData.expectedGenerationLogits;
        auto const expectedContextLogits = testData.expectedContextLogits;
        auto const expectedCumLogProbs = testData.expectedCumLogProbs;
        auto const expectedLogProbs = testData.expectedLogProbs;
        auto const draftTokens = llmReq.getDraftTokens();
        auto const hasDraftTokens = llmReq.hasDraftTokens() && modelSpec.mSpecDecodingMode.isDraftTokensExternal();

        for (auto beam = 0; beam < reqBeamWidth; ++beam)
        {
            auto const expectedOutputLength = expectedOutputLengths[givenInputIdx * reqBeamWidth + beam];
            bool anyMismatch = false;
            auto const predictedTokens = llmReq.getTokens(beam);
            auto numPredTokens = static_cast<SizeType32>(predictedTokens.size() - inputLength);
            if (hasDraftTokens)
            {
                numPredTokens
                    = std::min(numPredTokens, acceptedDraftTokensLengths[givenInputIdx * reqBeamWidth + beam] + 1);
            }
            if (modelSpec.mSpecDecodingMode.isMedusa() || modelSpec.mSpecDecodingMode.isLookaheadDecoding()
                || modelSpec.mSpecDecodingMode.isExplicitDraftTokens())
            {
                // WAR to ensure bulk execution of spec decoding.
                // We hope that no request in batch can finish 2x faster than any other request.
                // For the cases when BS < 8, some predicted tokens are mismatched to reference data.
                numPredTokens /= 2;
            }
            EXPECT_EQ(predictedTokens.size(), expectedOutputLength) << "b: " << requestId << " beam: " << beam;
            for (auto i = 0; i < numPredTokens; ++i)
            {
                // Use the expected data for that beamWidth
                auto const expectIndex = tc::flat_index3(givenInputIdx, beam,
                    inputLength + i + static_cast<SizeType32>(hasDraftTokens), reqBeamWidth, maxSeqLen);

                auto const expectedToken = expectedOutputData[expectIndex];
                if (expectedToken == endId)
                {
                    break;
                }
                auto const predictIndex = hasDraftTokens ? llmReq.mPromptLen + i : inputLength + i;
                auto const predictedToken = predictedTokens.at(predictIndex);
                EXPECT_EQ(predictedToken, expectedToken) << "b: " << requestId << " beam: " << beam << " i: " << i;
                anyMismatch |= (predictedToken != expectedToken);
            }
            EXPECT_FALSE(anyMismatch) << "b: " << requestId << " beam: " << beam;

            if (returnLogProbs)
            {
                auto cumLogProbs = llmReq.getCumLogProbs();
                auto* const reqExpectedCumLogProbs = bufferCast<float>(*expectedCumLogProbs[requestId]);
                EXPECT_TRUE(almostEqual(reqExpectedCumLogProbs[beam], cumLogProbs[beam]));

                auto logProbs = llmReq.getLogProbs(beam);
                auto expectedLogProbsBeam = std::shared_ptr(ITensor::slice(expectedLogProbs[requestId], beam, 1));
                expectedLogProbsBeam->squeeze(0);
                auto* const reqExpectedLogProbs = bufferCast<float>(*expectedLogProbsBeam);

                for (auto i = 0; i < numPredTokens; ++i)
                {
                    EXPECT_TRUE(almostEqual(reqExpectedLogProbs[inputLength + i], logProbs[i], 5e-2, 5e-2))
                        << "expectedLogProbs : " << reqExpectedLogProbs[inputLength + i]
                        << " logProbs : " << logProbs[i];
                }
            }

            if (checkAcceptedTokenLogits)
            {
                TLLM_CHECK_WITH_INFO(reqBeamWidth == 1, "speculative decoding only works for beam width == 1");

                TensorPtr const& acceptedTokensLogits = llmReq.getGenerationLogitsHost();
                auto const acceptedTokensLogitsShape = acceptedTokensLogits->getShape();

                EXPECT_EQ(acceptedTokensLogitsShape.nbDims, 2);
                EXPECT_EQ(numPredTokens, acceptedTokensLogitsShape.d[0]);

                TensorPtr const& expectedLogits = ITensor::slice(expectedGenerationLogits[requestId], 1, numPredTokens);

                // For hyperparameters
                // Greater tolerance for the accepted logits of the target model.
                float atol = 0.f;
                float rtol = 0.01f;
                EXPECT_TRUE(compareLogits(*expectedLogits, *acceptedTokensLogits, atol, rtol));
            }

            if (checkRawLogits)
            {
                // Check generation logits
                TensorPtr const& expectedGenerationLogitsSliced
                    = ITensor::slice(expectedGenerationLogits[requestId], 0, numPredTokens);

                TensorPtr const& llmReqGeneration = llmReq.getGenerationLogitsHost();
                auto generationLogitsBeam = std::shared_ptr(ITensor::slice(llmReqGeneration, beam, 1));
                generationLogitsBeam->squeeze(0);
                TensorPtr const& generationLogitsSliced = ITensor::slice(generationLogitsBeam, 0, numPredTokens);
                EXPECT_TRUE(compareLogits(*expectedGenerationLogitsSliced, *generationLogitsSliced));
            }
        }

        if (checkRawLogits)
        {
            // Check context logits
            TensorPtr const& llmReqContext = llmReq.getContextLogitsHost();
            EXPECT_TRUE(compareLogits(*expectedContextLogits[requestId], *llmReqContext));
        }
    }
}

std::tuple<std::vector<SizeType32>, std::unordered_map<SizeType32, TestData>> loadTestData(ModelSpec const& modelSpec,
    TrtGptModelType const& modelType, ModelIds const modelIds, BeamResults const& resultsFilesBeamWidths,
    ITensor const& givenInput, SizeType32 const maxBeamWidth, BufferManager& manager)
{
    // Map between beam width, and expected results for that beam width
    std::unordered_map<SizeType32, TestData> beamWidthTestData;
    std::vector<SizeType32> beamWidths;

    for (auto const& [beamWidth, resultsFile, logitsFiles, logProbsFiles] : resultsFilesBeamWidths)
    {
        auto expectedOutputs = utils::loadNpy(manager, resultsFile.string(), MemoryType::kCPU);

        auto* const expectedOutputData = bufferCast<TokenIdType>(*expectedOutputs);
        auto const& outputShape = expectedOutputs->getShape();
        EXPECT_EQ(outputShape.nbDims, 2);
        EXPECT_EQ(givenInput.getShape().d[0] * beamWidth, outputShape.d[0]);
        auto const& inputShape = givenInput.getShape();
        auto const maxSeqLen = static_cast<SizeType32>(outputShape.d[1]);
        EXPECT_LE(beamWidth, maxBeamWidth);
        EXPECT_EQ(std::find(beamWidths.begin(), beamWidths.end(), beamWidth), beamWidths.end());

        auto const padId = modelIds.padId;
        auto const [givenInputLengths, nbGivenInputs, maxInputLength] = getGivenInputLengths(givenInput, padId);
        auto const maxNewTokens = maxSeqLen - maxInputLength;

        std::srand(42);
        std::vector<TokenIdType> endIds;
        std::vector<SizeType32> expectedLengths(nbGivenInputs * beamWidth);

        if (modelSpec.mRandomEndId)
        {
            // Pick a different endId at random from one of the expected tokens

            // For V1, all entries in batch must have same endId
            if (modelType == TrtGptModelType::V1)
            {
                SizeType32 skippedTokenIndex = 0;
                TokenIdType endId = 0;
                do
                {
                    auto const endIdRow = std::rand() % nbGivenInputs;
                    auto const endIdBeam = std::rand() % beamWidth;
                    // We skip 1st token because minLength is 1
                    auto const endIdCol = givenInputLengths[endIdRow] + 1 + std::rand() % (maxNewTokens - 1);
                    auto const endIdIndex = tc::flat_index2((endIdRow * beamWidth + endIdBeam), endIdCol, maxSeqLen);
                    skippedTokenIndex
                        = tc::flat_index2((endIdRow * beamWidth + endIdBeam), givenInputLengths[endIdRow], maxSeqLen);
                    endId = expectedOutputData[endIdIndex];
                } while (endId == expectedOutputData[skippedTokenIndex]);
                endIds.insert(endIds.end(), nbGivenInputs, endId);
            }
            else
            {
                // For IFB, pick one of the output tokens as endId
                for (SizeType32 bi = 0; bi < nbGivenInputs; ++bi)
                {
                    SizeType32 skippedTokenIndex1 = 0;
                    SizeType32 skippedTokenIndex2 = 0;
                    SizeType32 endIdIndex = 0;
                    TokenIdType endId = 0;
                    do
                    {
                        auto const endIdRow = bi;
                        auto const endIdBeam = std::rand() % beamWidth;
                        // We do not use the 1st token for EndId because of Speculative Decoding test design
                        // We skip 1st token because minLength is 1
                        auto const endIdCol = givenInputLengths[endIdRow] + (2 + std::rand() % (maxNewTokens - 2));
                        endIdIndex = tc::flat_index2((endIdRow * beamWidth + endIdBeam), endIdCol, maxSeqLen);

                        skippedTokenIndex1 = tc::flat_index2(
                            (endIdRow * beamWidth + endIdBeam), givenInputLengths[endIdRow], maxSeqLen);
                        skippedTokenIndex2 = tc::flat_index2(
                            (endIdRow * beamWidth + endIdBeam), givenInputLengths[endIdRow] + 1, maxSeqLen);
                        endId = expectedOutputData[endIdIndex];
                    } while (endId == expectedOutputData[skippedTokenIndex1]
                        || endId == expectedOutputData[skippedTokenIndex2]);
                    // Workaround: The first example has endIdIndex 14, where the generation logits are almost same at
                    // token ids 257 and 373, which causes unstable generation results. Hence, we use the one previous
                    // token as endId.
                    if (bi == 0)
                    {
                        endId = expectedOutputData[endIdIndex - 1];
                    }
                    endIds.push_back(endId);
                }
            }
        }
        else
        {
            endIds.insert(endIds.end(), nbGivenInputs, modelIds.endId);
        }

        std::vector<LlmRequest::VecTokens> draftTokens(nbGivenInputs);
        std::vector<ITensor::SharedPtr> draftLogits(nbGivenInputs);
        std::vector<SizeType32> acceptedDraftTokensLengths(nbGivenInputs);

        std::vector<ITensor::SharedPtr> expectedGenerationLogits(nbGivenInputs);
        std::vector<ITensor::SharedPtr> expectedContextLogits(nbGivenInputs);
        ITensor::SharedPtr expectedGenerationLogitsPtr = nullptr;
        ITensor::SharedPtr expectedContextLogitsPtr = nullptr;

        std::vector<ITensor::SharedPtr> expectedCumLogProbs(nbGivenInputs);
        std::vector<ITensor::SharedPtr> expectedLogProbs(nbGivenInputs);
        ITensor::SharedPtr expectedCumLogProbsPtr = nullptr;
        ITensor::SharedPtr expectedLogProbsPtr = nullptr;

        if (modelSpec.mAcceptDraftByLogits)
        {
            TLLM_CHECK_WITH_INFO(logitsFiles.first != "",
                "Testing Draft token, but missing the expected generation logits config in the modelSpec.");
            expectedGenerationLogitsPtr
                = std::shared_ptr(utils::loadNpy(manager, logitsFiles.first.string(), MemoryType::kCPU));
        }
        if (modelSpec.useLogits() || modelSpec.mReturnAcceptedTokensLogits)
        {
            TLLM_CHECK_WITH_INFO(logitsFiles.first != "",
                "Testing with gather or replace logits, but missing the expected generation logits config in the "
                "modelSpec.");
            expectedGenerationLogitsPtr
                = std::shared_ptr(utils::loadNpy(manager, logitsFiles.first.string(), MemoryType::kCPU));

            TLLM_CHECK_WITH_INFO(logitsFiles.second != "",
                "Testing with gather or replace logits, but missing the expected context logits config in the "
                "modelSpec.");
            expectedContextLogitsPtr
                = std::shared_ptr(utils::loadNpy(manager, logitsFiles.second.string(), MemoryType::kCPU));
        }

        if (modelSpec.mReturnLogProbs)
        {
            TLLM_CHECK_WITH_INFO(logProbsFiles.first != "",
                "Testing return log probs, but missing the expected cum log probs config in the modelSpec.");
            expectedCumLogProbsPtr
                = std::shared_ptr(utils::loadNpy(manager, logProbsFiles.first.string(), MemoryType::kCPU));

            TLLM_CHECK_WITH_INFO(logProbsFiles.second != "",
                "Testing return log probs, but missing the expected log probs config in the modelSpec.");
            expectedLogProbsPtr
                = std::shared_ptr(utils::loadNpy(manager, logProbsFiles.second.string(), MemoryType::kCPU));
        }

        int promptOffset = 0;
        for (SizeType32 bi = 0; bi < nbGivenInputs; ++bi)
        {
            auto const endId = endIds[bi];
            for (SizeType32 beam = 0; beam < beamWidth; ++beam)
            {
                SizeType32 expectedLen = givenInputLengths[bi] + maxNewTokens;
                for (SizeType32 si = givenInputLengths[bi]; si < maxSeqLen; ++si)
                {
                    auto const expectIndex = tc::flat_index2((bi * beamWidth + beam), si, maxSeqLen);
                    if (expectedOutputData[expectIndex] == endId)
                    {
                        expectedLen = si;
                        break;
                    }
                }
                // Fill new EOS token to the expected data
                for (SizeType32 si = expectedLen; si < maxSeqLen; ++si)
                {
                    auto const expectIndex = tc::flat_index2((bi * beamWidth + beam), si, maxSeqLen);
                    expectedOutputData[expectIndex] = endId;
                }

                if (modelSpec.useLogits() || modelSpec.mReturnAcceptedTokensLogits)
                {
                    auto expectedGenerationLogitBatchSlice
                        = std::shared_ptr(ITensor::slice(expectedGenerationLogitsPtr, bi, 1));
                    expectedGenerationLogitBatchSlice->squeeze(0); // bs
                    expectedGenerationLogitBatchSlice->squeeze(0); // beam
                    expectedGenerationLogits[bi]
                        = expectedGenerationLogitBatchSlice;       // shape: [max_output_len, vocab_size]

                    auto expectedContextLogitBatchSlice = std::shared_ptr(
                        ITensor::slice(expectedContextLogitsPtr, promptOffset, givenInputLengths[bi]));
                    expectedContextLogits[bi] = expectedContextLogitBatchSlice; // shape: [prompt_length, vocab_size]
                }

                if (modelSpec.mReturnLogProbs)
                {
                    auto expectedCumLogProbsBatchSlice = std::shared_ptr(ITensor::slice(expectedCumLogProbsPtr, bi, 1));
                    expectedCumLogProbsBatchSlice->squeeze(0);               // bs
                    expectedCumLogProbs[bi] = expectedCumLogProbsBatchSlice; // shape: [beamWidth]

                    auto expectedLogProbsBatchSlice = std::shared_ptr(ITensor::slice(expectedLogProbsPtr, bi, 1));
                    expectedLogProbsBatchSlice->squeeze(0);            // bs
                    expectedLogProbs[bi] = expectedLogProbsBatchSlice; // shape: [beamWidth, numOutputTokens]
                }

                if (modelSpec.mMaxDraftTokens != 0)
                {
                    auto const draftLen
                        = std::rand() % std::min((maxSeqLen - (givenInputLengths[bi] + 1)), modelSpec.mMaxDraftTokens)
                        + 1;
                    auto acceptedLen = std::rand() % draftLen;

                    if (modelSpec.mAcceptDraftByLogits)
                    {
                        auto expectedLogitBatchSlice
                            = std::shared_ptr(ITensor::slice(expectedGenerationLogitsPtr, bi, 1));
                        expectedLogitBatchSlice->squeeze(0); // bs
                        expectedLogitBatchSlice->squeeze(0); // beam
                        auto expectedLogitBatchStepSlice
                            = std::shared_ptr(ITensor::slice(expectedLogitBatchSlice, 1, draftLen));
                        auto expectedLogitBatchStepView = ITensor::view(expectedLogitBatchStepSlice,
                            ITensor::makeShape({draftLen, 1, 1, expectedLogitBatchStepSlice->getShape().d[1]}));
                        draftLogits[bi] = manager.copyFrom(*expectedLogitBatchStepView, MemoryType::kCPU);
                    }

                    for (SizeType32 si = 0; si < draftLen; ++si)
                    {
                        auto const draftIndex
                            = tc::flat_index3(bi, beam, givenInputLengths[bi] + si + 1, beamWidth, maxSeqLen);
                        auto draftToken = expectedOutputData[draftIndex];
                        if (draftToken == endId)
                        {
                            acceptedLen = std::min(acceptedLen, si);
                        }
                        if (si >= acceptedLen)
                        {
                            draftToken = -1;
                            if (modelSpec.mAcceptDraftByLogits)
                            {
                                auto vocabSizePadded = expectedGenerationLogitsPtr->getShape().d[3];
                                auto* draftLogitsPtr = bufferCast<float>(*draftLogits[bi]);
                                for (SizeType32 vi = 0; vi < vocabSizePadded; ++vi)
                                {
                                    draftLogitsPtr[si * vocabSizePadded + vi] = 0.f;
                                }
                            }
                        }
                        draftTokens[bi].push_back(draftToken);
                    }
                    acceptedDraftTokensLengths[bi] = acceptedLen;
                    expectedLen = std::min(expectedLen, (givenInputLengths[bi] + 1) + acceptedLen + 1);
                }
                expectedLengths[bi * beamWidth + beam] = expectedLen;
            }
            promptOffset += givenInputLengths[bi];
        }

        TestData testData{std::move(expectedOutputs), expectedLengths, maxSeqLen, endIds, draftTokens, draftLogits,
            acceptedDraftTokensLengths, expectedGenerationLogits, expectedContextLogits, expectedCumLogProbs,
            expectedLogProbs};
        beamWidthTestData.emplace(beamWidth, std::move(testData));
        beamWidths.push_back(beamWidth);
    }

    return {beamWidths, beamWidthTestData};
}

RequestList runGptModelInference(std::shared_ptr<TrtGptModel>& trtGptModel, std::vector<SizeType32> const& beamWidths,
    std::unordered_map<SizeType32, TestData> const& beamWidthTestData, SizeType32 batchSize, SizeType32 nbGivenInputs,
    SizeType32 maxInputLength, SizeType32 padId, std::vector<SizeType32> const& givenInputLengths,
    TokenIdType const* givenInputData, ModelSpec const& modelSpec, TrtGptModelIfbTestType testType,
    TrtGptModelType modelType, int maxReqPerStep = 0, bool prepopulateKVCache = false)
{
    // Fill the requests using givenInput
    // requestList will have batchSize requests
    RequestList requestList;

    SizeType32 requestId = 0;
    RequestList finishedRequestList;
    std::vector<SizeType32> reqVec;
    // Advance the requests until they are all finished
    if (COMM_SESSION.getRank() == 0)
    {
        SizeType32 numReq = 0;
        while (numReq < batchSize)
        {
            // Add appropriate number of requests in each iteration. For WAVEFRONT, this is always 1.
            // For RANDOM, it could be any integer <= maxReqPerStep including 0.
            SizeType32 reqThisStep{0};
            switch (testType)
            {
            case TrtGptModelIfbTestType::WAVEFRONT: reqThisStep = 1; break;
            case TrtGptModelIfbTestType::RANDOM: reqThisStep = rand() % (maxReqPerStep + 1); break;
            case TrtGptModelIfbTestType::BULK: [[fallthrough]];
            default: reqThisStep = batchSize; break;
            }
            reqThisStep = std::min(reqThisStep, (batchSize - numReq));
            reqVec.push_back(reqThisStep);
            numReq += reqThisStep;
        }
    }
    COMM_SESSION.bcast(reqVec, 0);

    SizeType32 reqVecIdx = 0;
    while (requestId < batchSize || !requestList.empty())
    {
        SizeType32 reqThisStep = reqVecIdx < reqVec.size() ? reqVec[reqVecIdx++] : 0;
        for (SizeType32 req = 0; req < reqThisStep; req++)
        {
            // Alternate between beamWidths
            SizeType32 beamWidth = beamWidths.at(requestId % beamWidths.size());
            auto const& testData = beamWidthTestData.at(beamWidth);
            auto const* const expectedOutputData = bufferCast<TokenIdType const>(*testData.expectedOutputIds);
            auto const maxSeqLen = testData.maxSeqLen;

            SamplingConfig samplingConfig{beamWidth};
            samplingConfig.temperature = std::vector{1.0f};
            samplingConfig.minLength = std::vector{1};
            samplingConfig.randomSeed = std::vector{static_cast<uint64_t>(42ull)};
            samplingConfig.topK = std::vector{1};
            samplingConfig.topP = std::vector{0.0f};
            samplingConfig.draftAcceptanceThreshold = std::vector{0.3f};
            samplingConfig.noRepeatNgramSize = std::vector{1 << 30};

            if (modelType != TrtGptModelType::V1)
            {
                // Check that sampling config can work with diverging specialization of sampling params.
                if (req % 2 == 0)
                {
                    // minLength is defaulted as 1.
                    samplingConfig.minLength = std::nullopt;
                }
            }

            auto const [givenInputIdx, inputLength]
                = getRequestGivenInputIdxLength(requestId, nbGivenInputs, givenInputLengths);
            SizeType32 endId = testData.endIds[givenInputIdx];
            auto draftTokens = std::make_shared<std::vector<int32_t>>(testData.draftTokens[givenInputIdx]);

            auto maxNewTokens = maxSeqLen - maxInputLength;
            if (modelSpec.mMaxDraftTokens > 0)
            {
                maxNewTokens = modelSpec.mMaxDraftTokens + 1;
            }
            // Run model only to produce a single token and prepopulate KV cache
            if (prepopulateKVCache)
            {
                maxNewTokens = 1;
            }
            auto const* const seqBegin = givenInputData + givenInputIdx * maxInputLength;
            auto tokens = std::make_shared<std::vector<int32_t>>(seqBegin, seqBegin + inputLength);
            if (!prepopulateKVCache && modelSpec.mMaxDraftTokens > 0)
            {
                // Append the 1st predicted token to the prompt to get the match with prepopulated KV cache
                auto const expectIndex = tc::flat_index3(givenInputIdx, 0, inputLength, 1, maxSeqLen);
                auto expectedToken = expectedOutputData[expectIndex];
                tokens->push_back(expectedToken);
            }
            auto r = std::make_shared<LlmRequest>(requestId, maxNewTokens, tokens, samplingConfig, false, endId, padId);
            SizeType32 maxDraftTokens{0};
            if (trtGptModel->getModelConfig().hasSpeculativeDecodingModule())
            {
                maxDraftTokens
                    = trtGptModel->getModelConfig().getSpeculativeDecodingModulePtr()->getMaxDecodingDraftTokens();
            }
            r->validate(trtGptModel->getMaxInputLen(), trtGptModel->getMaxSequenceLen(), maxDraftTokens);
            auto logits = testData.draftLogits[givenInputIdx];
            std::optional<ITensor::SharedPtr> draftLogits
                = modelSpec.mAcceptDraftByLogits ? std::make_optional<ITensor::SharedPtr>(logits) : std::nullopt;
            if (!prepopulateKVCache)
            {
                r->setDraftTokens(draftTokens);
                r->setDraftLogits(draftLogits);
            }

            if (modelSpec.mGatherLogits)
            {
                auto const vocabSizePadded
                    = trtGptModel->getModelConfig().getVocabSizePadded(trtGptModel->getWorldConfig().getSize());
                TensorPtr contextLogitsHost = BufferManager::cpu(
                    ITensor::makeShape({r->mPromptLen, vocabSizePadded}), trtGptModel->getLogitDataType());
                TensorPtr generationLogitsHost = BufferManager::cpu(
                    ITensor::makeShape({r->mSamplingConfig.beamWidth, r->mMaxNewTokens, vocabSizePadded}),
                    trtGptModel->getLogitDataType());

                r->setContextLogitsHost(contextLogitsHost);
                r->setGenerationLogitsHost(generationLogitsHost);
                r->setReturnContextLogits(true);
                r->setReturnGenerationLogits(true);
            }

            if (!prepopulateKVCache && modelSpec.mReturnAcceptedTokensLogits)
            {
                auto const vocabSizePadded
                    = trtGptModel->getModelConfig().getVocabSizePadded(trtGptModel->getWorldConfig().getSize());
                TensorPtr generationLogitsHost = BufferManager::cpu(
                    ITensor::makeShape({r->getNumDraftTokens() + 1, vocabSizePadded}), trtGptModel->getLogitDataType());
                r->setGenerationLogitsHost(generationLogitsHost);
                r->setReturnGenerationLogits(true);
            }

            if (modelSpec.mReplaceLogits)
            {
                LlmRequest::LogitsPostProcessor logitsCb
                    = [&testData](uint64_t rId, tensorrt_llm::runtime::ITensor::SharedPtr& logits,
                          LlmRequest::BeamTokens const& tokens,
                          tensorrt_llm::runtime::BufferManager::CudaStreamPtr streamPtr)
                {
                    auto const expectedGenerationLogits = testData.expectedGenerationLogits[rId];
                    auto const expectedContextLogits = testData.expectedContextLogits[rId];
                    SizeType32 acceptedDraftTokensLengths = testData.acceptedDraftTokensLengths[rId];

                    auto manager = BufferManager(streamPtr);

                    auto const beamWidth = tokens.size();
                    TLLM_CHECK_WITH_INFO(beamWidth == 1, "Logits substitution is not supported for beam search");

                    auto const genLogitsOffset = tokens[0].size() - expectedContextLogits->getShape().d[0];
                    // TODO(xiweny): Avoid static cast in TRT 10.0
                    auto const numLogits = static_cast<SizeType32>(logits->getShape().d[0]);
                    auto const numVerifyLogits = std::min(numLogits, acceptedDraftTokensLengths + 1);
                    TensorPtr logitsHost = manager.copyFrom(*logits, MemoryType::kCPU);
                    auto logitsSliceHost = ITensor::slice(logitsHost, 0, numVerifyLogits);

                    TensorPtr refLogitsHost
                        = ITensor::slice(expectedGenerationLogits, genLogitsOffset, numVerifyLogits);

                    streamPtr->synchronize();
                    EXPECT_TRUE(compareLogits(*refLogitsHost, *logitsSliceHost, 0.f, 1e-2)) << "reqId: " << rId;

                    TensorPtr correctedLogitsSlice = ITensor::slice(logits, 0, numVerifyLogits);

                    manager.copy(*refLogitsHost, *correctedLogitsSlice);
                };

                r->mLogitsPostProcessor = logitsCb;
            }

            if (modelSpec.mReturnLogProbs)
            {
                r->setReturnLogProbs(true);
            }
            requestList.push_back(r);
            ++requestId;
        }

        //  Advance all active requests by one step
        trtGptModel->forwardAsync(requestList);
        trtGptModel->forwardSync();

        // Check which requests are done, move them out
        for (auto it = requestList.cbegin(); it != requestList.cend();)
        {
            if ((*it)->mState == REQUEST_STATE_GENERATION_COMPLETE)
            {
                finishedRequestList.push_back(*it);
                requestList.erase(it++);
            }
            else
            {
                ++it;
            }
        }
    }
    return finishedRequestList;
}

void runIfbTest(fs::path const& modelPath, ModelSpec const& modelSpec, ModelIds const modelIds,
    TrtGptModelType modelType, std::vector<int32_t> const& batchSizes, BeamResults const& resultsFilesBeamWidths,
    TrtGptModelIfbTestType testType, int maxReqPerStep, TrtGptModelOptionalParams const& optionalParams)
{
    auto manager = BufferManager(std::make_shared<CudaStream>());
    auto const padId = modelIds.padId;

    // Load input data
    ASSERT_TRUE(fs::exists(DATA_PATH));
    auto const inputPath = DATA_PATH / modelSpec.mInputFile;
    auto const& givenInput = utils::loadNpy(manager, inputPath.string(), MemoryType::kCPU);
    auto [givenInputLengths, nbGivenInputs, maxInputLength] = getGivenInputLengths(*givenInput, padId);
    auto const* const givenInputData = bufferCast<TokenIdType const>(*givenInput);

    auto const& inputShape = givenInput->getShape();
    ASSERT_EQ(inputShape.nbDims, 2);
    ASSERT_GT(inputShape.d[0], 0);

    TLLM_CHECK(optionalParams.maxBeamWidth.has_value());
    auto const maxBeamWidth = optionalParams.maxBeamWidth.value();
    // Load expected outputs for each beam width value
    auto [beamWidths, beamWidthTestData]
        = loadTestData(modelSpec, modelType, modelIds, resultsFilesBeamWidths, *givenInput, maxBeamWidth, manager);

    int const worldSize = modelSpec.mTPSize * modelSpec.mPPSize;
    auto const worldConfig = WorldConfig::mpi(worldSize, modelSpec.mTPSize, modelSpec.mPPSize);

    ASSERT_TRUE(fs::exists(modelPath));

    auto trtGptModel = TrtGptModelFactory::create(modelPath, modelType, optionalParams);

    for (auto batchSize : batchSizes)
    {
        std::cout << "=== batchSize:" << batchSize << " ===\n";

        // Prepopulate KV cache for speculative decoding test
        bool const prepopulateKVCache = modelSpec.mMaxDraftTokens > 0;
        auto finishedRequestList = runGptModelInference(trtGptModel, beamWidths, beamWidthTestData, batchSize,
            nbGivenInputs, maxInputLength, padId, givenInputLengths, givenInputData, modelSpec, testType, modelType,
            maxReqPerStep, prepopulateKVCache);

        if (prepopulateKVCache)
        {
            // Call the 2nd time with prefilled KV cache
            finishedRequestList = runGptModelInference(trtGptModel, beamWidths, beamWidthTestData, batchSize,
                nbGivenInputs, maxInputLength, padId, givenInputLengths, givenInputData, modelSpec, testType, modelType,
                maxReqPerStep, false);
        }

        // WAR: disabled verification because of switched beams for different batch composition
        if (worldConfig.isFirstPipelineParallelRank()
            && (testType == TrtGptModelIfbTestType::BULK || maxBeamWidth == 1))
        {
            verifyOutput(finishedRequestList, beamWidthTestData, givenInputLengths, nbGivenInputs, modelSpec);
        }
    }
}

struct BeamConfig
{
    SizeType32 maxBeamWidth;
    std::vector<SizeType32> beamWidths;
};

} // namespace

using ParamType = std::tuple<ModelParams, ModelSpec, TrtGptModelType, TrtGptModelIfbTestType, BeamConfig,
    std::optional<int32_t>, std::optional<float>, bool, bool>;

std::string generateTestName(testing::TestParamInfo<ParamType> const& info)
{
    auto const modelSpec = std::get<1>(info.param);
    std::string name{modelSpec.mDataType == nvinfer1::DataType::kFLOAT ? "Float" : "Half"};
    auto const modelType = std::get<2>(info.param);
    switch (modelType)
    {
    case TrtGptModelType::V1: name.append("V1Model"); break;
    case TrtGptModelType::InflightBatching: name.append("IbModel"); break;
    case TrtGptModelType::InflightFusedBatching: name.append("FusedIbModel"); break;
    default: name.append("DefaultModel"); break;
    }
    if (modelSpec.mUsePagedKvCache)
    {
        name.append("PagedKvCache");
    }
    auto const testType = std::get<3>(info.param);
    switch (testType)
    {
    case TrtGptModelIfbTestType::BULK: name.append("Bulk"); break;
    case TrtGptModelIfbTestType::WAVEFRONT: name.append("Wavefront"); break;
    case TrtGptModelIfbTestType::RANDOM: name.append("Random"); break;
    default: name.append("DefaultTest"); break;
    }
    BeamConfig const beamConfig = std::get<4>(info.param);
    name.append("MaxBeamWidth" + std::to_string(beamConfig.maxBeamWidth));
    for (auto const beamWdith : beamConfig.beamWidths)
    {
        name.append("Bw" + std::to_string(beamWdith));
    }

    auto const maxTokensInPagedKvCache = std::get<5>(info.param);
    if (maxTokensInPagedKvCache.has_value())
    {
        name.append("KvCacheSize" + std::to_string(maxTokensInPagedKvCache.value()));
    }

    auto const freeGpuMemoryFraction = std::get<6>(info.param);
    if (freeGpuMemoryFraction.has_value())
    {
        name.append("GpuFrac");
    }

    auto const enableTrtOverlap = std::get<7>(info.param);
    if (enableTrtOverlap)
    {
        name.append("TrtOverlap");
    }

    auto const enableChunkedContext = std::get<8>(info.param);
    if (enableChunkedContext)
    {
        name.append("Chunked");
    }

    if (modelSpec.mTPSize > 1)
    {
        name.append("TP" + std::to_string(modelSpec.mTPSize));
    }

    if (modelSpec.mPPSize > 1)
    {
        name.append("PP" + std::to_string(modelSpec.mPPSize));
    }

    if (modelSpec.mRandomEndId)
    {
        name.append("EndId");
    }

    if (modelSpec.mMaxDraftTokens > 0)
    {
        name.append("DraftTokens" + std::to_string(modelSpec.mMaxDraftTokens));
    }

    if (modelSpec.mAcceptDraftByLogits)
    {
        name.append("AcceptByLogits");
    }

    if (modelSpec.mReturnAcceptedTokensLogits)
    {
        name.append("ReturnAcceptedTokenLogits");
    }

    return name;
}

class ParamTest : public TrtModelRealDecoderTest, public ::testing::WithParamInterface<ParamType>
{
};

TEST_P(ParamTest, Test)
{

    auto const& beamConfig = std::get<4>(GetParam());
    auto const& beamWidths = beamConfig.beamWidths;

    auto const modelParams = std::get<0>(GetParam());
    auto const modelIds = modelParams.ids;
    auto const* const modelDir = modelParams.baseDir;
    auto const modelSpec = std::get<1>(GetParam());

    std::vector<int32_t> batchSizes = modelSpec.mBatchSizes;

    std::ostringstream gpuSizePath;
    gpuSizePath << "tp" << modelSpec.mTPSize << "-pp" << modelSpec.mPPSize << "-gpu";

    auto const modelPath{ENGINE_PATH / modelDir / modelSpec.mModelPath / gpuSizePath.str()};

    auto const inputPath = DATA_PATH / modelSpec.mInputFile;

    BeamResults beamResults;
    beamResults.reserve(beamWidths.size());
    for (auto beamWidth : beamWidths)
    {
        fs::path resultsPath
            = DATA_PATH / modelDir / ((beamWidth == 1) ? "sampling" : "beam_search_" + std::to_string(beamWidth));
        fs::path generationLogitsPath
            = modelSpec.mGenerationLogitsFile.empty() ? "" : (resultsPath / modelSpec.mGenerationLogitsFile).string();
        fs::path contextLogitsPath
            = modelSpec.mContextLogitsFile.empty() ? "" : (resultsPath / modelSpec.mContextLogitsFile).string();
        fs::path cumLogProbsPath
            = modelSpec.mCumLogProbsFile.empty() ? "" : (resultsPath / modelSpec.mCumLogProbsFile).string();
        fs::path logProbsPath = modelSpec.mLogProbsFile.empty() ? "" : (resultsPath / modelSpec.mLogProbsFile).string();

        beamResults.emplace_back(std::make_tuple(beamWidth, (resultsPath / modelSpec.mResultsFile).string(),
            std::make_pair(generationLogitsPath, contextLogitsPath), std::make_pair(cumLogProbsPath, logProbsPath)));

        if (modelSpec.mRandomEndId && beamWidth > 1)
        {
            GTEST_SKIP() << "Test does not support endId test with beam search";
        }
    }

    auto const modelType = std::get<2>(GetParam());
    auto const testType = std::get<3>(GetParam());

    if (modelType != TrtGptModelType::V1 && !(modelSpec.mUsePackedInput && modelSpec.mUsePagedKvCache))
    {
        GTEST_SKIP() << "Inflight batching requires packed input and paged KV cache.";
    }

    if (!modelSpec.mUsePackedInput && modelSpec.mRandomEndId)
    {
        GTEST_SKIP() << "Test does not support endId test with padded inputs";
    }

    for (auto beamWidth : beamWidths)
    {
        if (modelSpec.mMaxDraftTokens > 0 && (beamWidth > 1 || modelType == TrtGptModelType::V1))
        {
            GTEST_SKIP() << "Target model in speculative decoding does not support beam search and V1";
        }
    }

    TrtGptModelOptionalParams modelOptionalParams;
    modelOptionalParams.kvCacheConfig.maxTokens = std::get<5>(GetParam());
    modelOptionalParams.kvCacheConfig.enableBlockReuse = modelSpec.mMaxDraftTokens > 0;
    modelOptionalParams.kvCacheConfig.freeGpuMemoryFraction = std::get<6>(GetParam());
    modelOptionalParams.enableTrtOverlap = std::get<7>(GetParam());
    modelOptionalParams.enableChunkedContext = std::get<8>(GetParam());
    modelOptionalParams.normalizeLogProbs = false;
    modelOptionalParams.maxBeamWidth = beamConfig.maxBeamWidth;
    texec::CapacitySchedulerPolicy capacitySchedulerPolicy = (modelType == TrtGptModelType::V1)
        ? texec::CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT
        : texec::CapacitySchedulerPolicy::kMAX_UTILIZATION;
    modelOptionalParams.schedulerConfig = texec::SchedulerConfig{capacitySchedulerPolicy};

    if (modelType == TrtGptModelType::V1
        && (modelOptionalParams.kvCacheConfig.maxTokens.has_value() || modelOptionalParams.enableTrtOverlap))
    {
        GTEST_SKIP() << "Not running V1 with Inflight batching optional params";
    }

    // Warning: This should be the last check before running the test.
    // It will initialize MPI which can take significant time.
    if (modelSpec.mTPSize * modelSpec.mPPSize != COMM_SESSION.getSize())
    {
        GTEST_SKIP() << "Model's world size " << modelSpec.mPPSize * modelSpec.mTPSize
                     << " is not equal to the system world size";
    }

    runIfbTest(modelPath, modelSpec, modelIds, modelType, batchSizes, beamResults, testType, 2, modelOptionalParams);
}

auto constexpr gptModelParams = ModelParams{GPT_MODEL_DIR, {50256, 50256}};

INSTANTIATE_TEST_SUITE_P(GptV1Tests, ParamTest,
    testing::Combine(testing::Values(gptModelParams),
        testing::Values(
            //
            ModelSpec{
                FP16_GPT_ATTENTION_PACKED_DIR, INPUT_FILE, FP16_PLUGIN_PACKED_RESULT_FILE, nvinfer1::DataType::kHALF}
                .usePackedInput(),
            ModelSpec{FP16_GPT_ATTENTION_PACKED_PAGED_DIR, INPUT_FILE, FP16_PLUGIN_PACKED_PAGED_RESULT_FILE,
                nvinfer1::DataType::kHALF}
                .usePackedInput()
                .usePagedKvCache(),
            ModelSpec{FP16_GPT_ATTENTION_PACKED_PAGED_DIR, INPUT_FILE, FP16_PLUGIN_PACKED_PAGED_RESULT_FILE,
                nvinfer1::DataType::kHALF}
                .usePackedInput()
                .usePagedKvCache()
                .useRandomEndId()

                ),
        testing::Values(TrtGptModelType::V1),
        testing::Values(
            TrtGptModelIfbTestType::BULK, TrtGptModelIfbTestType::WAVEFRONT, TrtGptModelIfbTestType::RANDOM),
        testing::Values(
            // enable more tests when mixed beam width is supported
            BeamConfig{1, {1}}, BeamConfig{2, {2}} // , BeamConfig{2, {1, 2}}
            ),
        testing::Values(std::nullopt),             // maxTokensInPagedKvCache
        testing::Values(std::nullopt, 0.8),        // freeGpuMemoryFraction
        testing::Values(false),                    // enableTrtOverlap
        testing::Values(false)                     // enableChunkedContext
        ),
    generateTestName);

INSTANTIATE_TEST_SUITE_P(GptTests, ParamTest,
    testing::Combine(testing::Values(gptModelParams),
        testing::Values(
            //
            ModelSpec{FP16_GPT_ATTENTION_PACKED_PAGED_DIR, INPUT_FILE, FP16_PLUGIN_PACKED_PAGED_RESULT_FILE,
                nvinfer1::DataType::kHALF}
                .usePackedInput()
                .usePagedKvCache(),
            ModelSpec{FP16_GPT_ATTENTION_PACKED_PAGED_DIR, INPUT_FILE, FP16_PLUGIN_PACKED_PAGED_RESULT_FILE,
                nvinfer1::DataType::kHALF}
                .usePackedInput()
                .usePagedKvCache()
                .useRandomEndId()

                ),
        testing::Values(TrtGptModelType::InflightBatching, TrtGptModelType::InflightFusedBatching),
        testing::Values(
            TrtGptModelIfbTestType::BULK, TrtGptModelIfbTestType::WAVEFRONT, TrtGptModelIfbTestType::RANDOM),
        testing::Values(
            // enable more tests when mixed beam width is supported
            BeamConfig{1, {1}}, BeamConfig{2, {2}} // , BeamConfig{2, {1, 2}}
            ),
        testing::Values(std::nullopt, 1280),       // maxTokensInPagedKvCache
        testing::Values(std::nullopt, 0.8),        // freeGpuMemoryFraction
        testing::Values(false, true),              // enableTrtOverlap
        testing::Values(false)                     // enableChunkedContext
        ),
    generateTestName);

INSTANTIATE_TEST_SUITE_P(GptNProfilesTests, ParamTest,
    testing::Combine(testing::Values(gptModelParams),
        testing::Values(ModelSpec{FP16_GPT_ATTENTION_PACKED_PAGED_NPROFILES_DIR, INPUT_FILE,
            FP16_PLUGIN_PACKED_PAGED_RESULT_FILE, nvinfer1::DataType::kHALF}
                            .usePackedInput()
                            .usePagedKvCache()
                            .useRandomEndId()),
        testing::Values(TrtGptModelType::InflightBatching, TrtGptModelType::InflightFusedBatching),
        testing::Values(TrtGptModelIfbTestType::BULK),
        testing::Values(
            // enable more tests when mixed beam width is supported
            BeamConfig{1, {1}}, BeamConfig{2, {2}} // , BeamConfig{2, {1, 2}}
            ),
        testing::Values(std::nullopt, 1280),       // maxTokensInPagedKvCache
        testing::Values(std::nullopt, 0.8),        // freeGpuMemoryFraction
        testing::Values(false, true),              // enableTrtOverlap
        testing::Values(false)                     // enableChunkedContext
        ),
    generateTestName);

INSTANTIATE_TEST_SUITE_P(GptSqTests, ParamTest,
    testing::Combine(testing::Values(gptModelParams),
        testing::Values(
            //
            ModelSpec{FP16_GPT_ATTENTION_PACKED_PAGED_SQ_DIR, INPUT_FILE, FP16_PLUGIN_PACKED_PAGED_SQ_RESULT_FILE,
                nvinfer1::DataType::kHALF}
                .usePackedInput()
                .usePagedKvCache()),
        testing::Values(TrtGptModelType::InflightBatching),
        testing::Values(
            TrtGptModelIfbTestType::BULK, TrtGptModelIfbTestType::WAVEFRONT, TrtGptModelIfbTestType::RANDOM),
        testing::Values(
            // enable more tests when mixed beam width is supported
            // disabled flaky beam search tests (https://nvbugspro.nvidia.com/bug/4646234)
            BeamConfig{1, {1}}         //, BeamConfig{2, {2}}
            ),
        testing::Values(std::nullopt), // maxTokensInPagedKvCache
        testing::Values(std::nullopt), // freeGpuMemoryFraction
        testing::Values(false),        // enableTrtOverlap
        testing::Values(false)         // enableChunkedContext
        ),
    generateTestName);

// disabled because paused requests generate different tokens after resuming
INSTANTIATE_TEST_SUITE_P(DISABLED_GptChunkedContextTests, ParamTest,
    testing::Combine(testing::Values(gptModelParams),
        testing::Values(
            //
            ModelSpec{FP16_GPT_ATTENTION_PACKED_PAGED_IN128_DIR, INPUT_FILE, FP16_PLUGIN_PACKED_PAGED_LONG_RESULT_FILE,
                nvinfer1::DataType::kHALF}
                .usePackedInput()
                .usePagedKvCache()

                ),
        testing::Values(TrtGptModelType::InflightBatching, TrtGptModelType::InflightFusedBatching),
        testing::Values(TrtGptModelIfbTestType::BULK),
        testing::Values(
            // enable more tests when mixed beam width is supported
            BeamConfig{1, {1}}),
        testing::Values(257),          // maxTokensInPagedKvCache
        testing::Values(std::nullopt), // freeGpuMemoryFraction
        testing::Values(false),        // enableTrtOverlap
        testing::Values(true)          // enableChunkedContext
        ),
    generateTestName);

INSTANTIATE_TEST_SUITE_P(GptChunkedLongContextTests, ParamTest,
    testing::Combine(testing::Values(gptModelParams),
        testing::Values(
            //
            ModelSpec{FP16_GPT_ATTENTION_PACKED_PAGED_IN128_DIR, LONG_INPUT_FILE,
                FP16_PLUGIN_PACKED_PAGED_LONG_INPUT_RESULT_FILE, nvinfer1::DataType::kHALF}
                .usePackedInput()
                .usePagedKvCache(),
            ModelSpec{FP16_GPT_ATTENTION_PACKED_PAGED_DRAFT_TOKENS_DIR, LONG_INPUT_FILE,
                FP16_PLUGIN_PACKED_PAGED_LONG_INPUT_RESULT_FILE, nvinfer1::DataType::kHALF}
                .usePackedInput()
                .usePagedKvCache()
                .setDraftTokens(5)),
        testing::Values(TrtGptModelType::InflightBatching, TrtGptModelType::InflightFusedBatching),
        testing::Values(
            TrtGptModelIfbTestType::BULK, TrtGptModelIfbTestType::WAVEFRONT, TrtGptModelIfbTestType::RANDOM),
        testing::Values(
            // enable more tests when mixed beam width is supported
            BeamConfig{1, {1}}),
        testing::Values(std::nullopt, 1024), // maxTokensInPagedKvCache
        testing::Values(std::nullopt),       // freeGpuMemoryFraction
        testing::Values(false),              // enableTrtOverlap
        testing::Values(true)                // enableChunkedContext
        ),
    generateTestName);

INSTANTIATE_TEST_SUITE_P(GptDraftTests, ParamTest,
    testing::Combine(testing::Values(gptModelParams),
        testing::Values(
            //
            ModelSpec{FP16_GPT_ATTENTION_PACKED_PAGED_DRAFT_TOKENS_DIR, INPUT_FILE,
                FP16_PLUGIN_PACKED_PAGED_GATHER_RESULT_FILE, nvinfer1::DataType::kHALF,
                FP16_PLUGIN_PACKED_PAGED_GENERATION_LOGITS_FILE, FP16_PLUGIN_PACKED_PAGED_CONTEXT_LOGITS_FILE}
                .usePackedInput()
                .usePagedKvCache()
                .useRandomEndId()
                .setDraftTokens(5)
                .replaceLogits(),
            ModelSpec{FP16_GPT_ATTENTION_PACKED_PAGED_DRAFT_TOKENS_DIR, INPUT_FILE,
                FP16_PLUGIN_PACKED_PAGED_GATHER_RESULT_FILE, nvinfer1::DataType::kHALF,
                FP16_PLUGIN_PACKED_PAGED_GENERATION_LOGITS_FILE, FP16_PLUGIN_PACKED_PAGED_CONTEXT_LOGITS_FILE}
                .usePackedInput()
                .usePagedKvCache()
                .useRandomEndId()
                .setDraftTokens(5)
                .useAcceptByLogits()
                .replaceLogits()),
        testing::Values(TrtGptModelType::InflightBatching, TrtGptModelType::InflightFusedBatching),
        testing::Values(
            TrtGptModelIfbTestType::BULK, TrtGptModelIfbTestType::WAVEFRONT, TrtGptModelIfbTestType::RANDOM),
        testing::Values(BeamConfig{1, {1}}), // beamConfig
        testing::Values(std::nullopt),       // maxTokensInPagedKvCache
        testing::Values(std::nullopt),       // freeGpuMemoryFraction
        testing::Values(false),              // enableTrtOverlap
        testing::Values(true, false)         // enableChunkedContext
        ),
    generateTestName);

INSTANTIATE_TEST_SUITE_P(GptReturnAcceptedTokenLogitsTests, ParamTest,
    testing::Combine(testing::Values(gptModelParams),
        testing::Values(ModelSpec{FP16_GPT_ATTENTION_PACKED_PAGED_RETURN_ACCEPTED_TOKENS_LOGITS_DIR, INPUT_FILE,
            FP16_PLUGIN_PACKED_PAGED_GATHER_RESULT_FILE, nvinfer1::DataType::kHALF,
            FP16_PLUGIN_PACKED_PAGED_GENERATION_LOGITS_FILE, FP16_PLUGIN_PACKED_PAGED_CONTEXT_LOGITS_FILE}
                            .usePackedInput()
                            .usePagedKvCache()
                            .setDraftTokens(5)
                            .useAcceptByLogits()
                            .returnAcceptedTokensLogits()),
        testing::Values(TrtGptModelType::InflightBatching, TrtGptModelType::InflightFusedBatching),
        testing::Values(
            TrtGptModelIfbTestType::BULK, TrtGptModelIfbTestType::WAVEFRONT, TrtGptModelIfbTestType::RANDOM),
        testing::Values(BeamConfig{1, {1}}), // beamConfig
        testing::Values(std::nullopt),       // maxTokensInPagedKvCache
        testing::Values(std::nullopt),       // freeGpuMemoryFraction
        testing::Values(false),              // enableTrtOverlap
        testing::Values(false)               // enableChunkedContext
        ),

    generateTestName);

INSTANTIATE_TEST_SUITE_P(GptLogitsTests, ParamTest,
    testing::Combine(testing::Values(gptModelParams),
        testing::Values(
            // modelSpec
            ModelSpec{FP16_GPT_ATTENTION_PACKED_PAGED_GATHER_DIR, INPUT_FILE,
                FP16_PLUGIN_PACKED_PAGED_GATHER_RESULT_FILE, nvinfer1::DataType::kHALF,
                FP16_PLUGIN_PACKED_PAGED_GENERATION_LOGITS_FILE, FP16_PLUGIN_PACKED_PAGED_CONTEXT_LOGITS_FILE}
                .usePackedInput()
                .usePagedKvCache()
                .gatherLogits()
                .useRandomEndId()),
        testing::Values(TrtGptModelType::InflightBatching, TrtGptModelType::InflightFusedBatching), // modelType
        testing::Values(TrtGptModelIfbTestType::BULK, TrtGptModelIfbTestType::WAVEFRONT,
            TrtGptModelIfbTestType::RANDOM),                                                        // testType
        testing::Values(BeamConfig{1, {1}}),                                                        // beamConfig
        testing::Values(std::nullopt), // maxTokensInPagedKvCache
        testing::Values(std::nullopt), // freeGpuMemoryFraction
        testing::Values(false),        // enableTrtOverlap
        testing::Values(false)         // enableChunkedContext
        ),
    generateTestName);

INSTANTIATE_TEST_SUITE_P(GptLogProbsTests, ParamTest,
    testing::Combine(testing::Values(gptModelParams),
        testing::Values(
            // modelSpec
            ModelSpec{FP16_GPT_ATTENTION_PACKED_PAGED_DIR, INPUT_FILE, FP16_PLUGIN_PACKED_PAGED_RESULT_FILE,
                nvinfer1::DataType::kHALF, "", "", FP16_PLUGIN_PACKED_PAGED_CUM_LOG_PROBS_FILE,
                FP16_PLUGIN_PACKED_PAGED_LOG_PROBS_FILE}
                .usePackedInput()
                .usePagedKvCache()
                .returnLogProbs()),
        testing::Values(TrtGptModelType::InflightFusedBatching), // modelType
        testing::Values(TrtGptModelIfbTestType::BULK, TrtGptModelIfbTestType::WAVEFRONT,
            TrtGptModelIfbTestType::RANDOM),                     // testType
        testing::Values(BeamConfig{1, {1}}),                     // beamConfig
        testing::Values(std::nullopt),                           // maxTokensInPagedKvCache
        testing::Values(std::nullopt),                           // freeGpuMemoryFraction
        testing::Values(false),                                  // enableTrtOverlap
        testing::Values(false)                                   // enableChunkedContext
        ),
    generateTestName);

INSTANTIATE_TEST_SUITE_P(GptjTests, ParamTest,
    testing::Combine(testing::Values(ModelParams{GPTJ_MODEL_DIR, {50256, 50256}}),
        testing::Values(
            //
            ModelSpec{
                FP16_GPT_ATTENTION_PACKED_DIR, INPUT_FILE, FP16_PLUGIN_PACKED_RESULT_FILE, nvinfer1::DataType::kHALF}
                .usePackedInput(),
            ModelSpec{FP16_GPT_ATTENTION_PACKED_PAGED_DIR, INPUT_FILE, FP16_PLUGIN_PACKED_PAGED_RESULT_FILE,
                nvinfer1::DataType::kHALF}
                .usePackedInput()
                .usePagedKvCache()

                ),
        testing::Values(TrtGptModelType::V1, TrtGptModelType::InflightBatching, TrtGptModelType::InflightFusedBatching),
        // WAR: disable wavefront and random tests on because of switched beams
        testing::Values(TrtGptModelIfbTestType::BULK
            /* , TrtGptModelIfbTestType::WAVEFRONT, TrtGptModelIfbTestType::RANDOM */),
        testing::Values(
            // enable more tests when mixed beam width is supported
            BeamConfig{1, {1}}, BeamConfig{2, {2}} // , BeamConfig{2, {1, 2}}
            ),
        testing::Values(std::nullopt),             // maxTokensInPagedKvCache
        testing::Values(std::nullopt),             // freeGpuMemoryFraction
        testing::Values(false),                    // enableTrtOverlap
        testing::Values(false)                     // enableChunkedContext
        ),
    generateTestName);

INSTANTIATE_TEST_SUITE_P(MambaTests, ParamTest,
    testing::Combine(testing::Values(ModelParams{MAMBA_MODEL_DIR, {0, 1}}),
        testing::Values(ModelSpec{FP16_GPT_ATTENTION_PACKED_DIR, INPUT_FILE, FP16_PLUGIN_PACKED_RESULT_FILE,
                            nvinfer1::DataType::kHALF}
                            .usePackedInput(),
            ModelSpec{FP16_GPT_ATTENTION_PACKED_PAGED_DIR, INPUT_FILE, FP16_PLUGIN_PACKED_PAGED_RESULT_FILE,
                nvinfer1::DataType::kHALF}
                .usePackedInput()
                .usePagedKvCache()),
        testing::Values(TrtGptModelType::V1, TrtGptModelType::InflightBatching),
        testing::Values(
            TrtGptModelIfbTestType::BULK, TrtGptModelIfbTestType::WAVEFRONT, TrtGptModelIfbTestType::RANDOM),
        testing::Values(BeamConfig{1, {1}}),
        testing::Values(std::nullopt), // maxTokensInPagedKvCache
        testing::Values(std::nullopt), // freeGpuMemoryFraction
        testing::Values(false),        // enableTrtOverlap
        testing::Values(false)         // enableChunkedContext
        ),
    generateTestName);

INSTANTIATE_TEST_SUITE_P(RecurrentGemmaTests, ParamTest,
    testing::Combine(testing::Values(ModelParams{RECURRENTGEMMA_MODEL_DIR, {0, 1}}),
        testing::Values(ModelSpec{FP16_GPT_ATTENTION_PACKED_PAGED_DIR, INPUT_FILE, FP16_PLUGIN_PACKED_PAGED_RESULT_FILE,
            nvinfer1::DataType::kHALF}
                            .usePackedInput()
                            .usePagedKvCache()),
        testing::Values(TrtGptModelType::InflightBatching),
        testing::Values(
            TrtGptModelIfbTestType::BULK, TrtGptModelIfbTestType::WAVEFRONT, TrtGptModelIfbTestType::RANDOM),
        testing::Values(BeamConfig{1, {1}}),
        testing::Values(std::nullopt), // maxTokensInPagedKvCache
        testing::Values(std::nullopt), // freeGpuMemoryFraction
        testing::Values(false),        // enableTrtOverlap
        testing::Values(false)         // enableChunkedContext
        ),
    generateTestName);

INSTANTIATE_TEST_SUITE_P(LlamaTests, ParamTest,
    testing::Combine(testing::Values(ModelParams{LLAMA_MODEL_DIR, {2, 2}}),
        testing::Values(
            //
            ModelSpec{FP16_GPT_ATTENTION_PACKED_PAGED_DIR, INPUT_FILE, FP16_PLUGIN_PACKED_PAGED_RESULT_FILE,
                nvinfer1::DataType::kHALF}
                .usePackedInput()
                .usePagedKvCache(),
            ModelSpec{FP16_GPT_ATTENTION_PACKED_PAGED_DIR, INPUT_FILE, FP16_PLUGIN_PACKED_PAGED_RESULT_TP1_PP4_FILE,
                nvinfer1::DataType::kHALF}
                .usePackedInput()
                .usePagedKvCache()
                .usePipelineParallelism(4),
            ModelSpec{FP16_GPT_ATTENTION_PACKED_PAGED_DIR, INPUT_FILE, FP16_PLUGIN_PACKED_PAGED_RESULT_TP4_PP1_FILE,
                nvinfer1::DataType::kHALF}
                .usePackedInput()
                .usePagedKvCache()
                .useTensorParallelism(4),
            ModelSpec{FP16_GPT_ATTENTION_PACKED_PAGED_DIR, INPUT_FILE, FP16_PLUGIN_PACKED_PAGED_RESULT_TP2_PP2_FILE,
                nvinfer1::DataType::kHALF}
                .usePackedInput()
                .usePagedKvCache()
                .usePipelineParallelism(2)
                .useTensorParallelism(2)

                ),
        testing::Values(TrtGptModelType::V1, TrtGptModelType::InflightBatching, TrtGptModelType::InflightFusedBatching),
        testing::Values(
            TrtGptModelIfbTestType::BULK, TrtGptModelIfbTestType::WAVEFRONT, TrtGptModelIfbTestType::RANDOM),
        testing::Values(
            // enable more tests when mixed beam width is supported
            BeamConfig{1, {1}}, BeamConfig{2, {2}} // , BeamConfig{2, {1, 2}}
            ),
        testing::Values(std::nullopt),             // maxTokensInPagedKvCache
        testing::Values(std::nullopt),             // freeGpuMemoryFraction
        testing::Values(false),                    // enableTrtOverlap
        testing::Values(false)                     // enableChunkedContext
        ),
    generateTestName);

INSTANTIATE_TEST_SUITE_P(ChatGlmTests, ParamTest,
    testing::Combine(testing::Values(ModelParams{CHATGLM_MODEL_DIR, {130005, 3}}),
        testing::Values(
            //
            ModelSpec{FP16_GPT_ATTENTION_PACKED_PAGED_DIR, CHATGLM_INPUT_FILE, FP16_PLUGIN_PACKED_PAGED_RESULT_FILE,
                nvinfer1::DataType::kHALF}
                .usePackedInput()
                .usePagedKvCache()),
        testing::Values(TrtGptModelType::InflightFusedBatching),
        testing::Values(
            TrtGptModelIfbTestType::BULK, TrtGptModelIfbTestType::WAVEFRONT, TrtGptModelIfbTestType::RANDOM),
        testing::Values(BeamConfig{1, {1}}),
        testing::Values(std::nullopt), // maxTokensInPagedKvCache
        testing::Values(std::nullopt), // freeGpuMemoryFraction
        testing::Values(false),        // enableTrtOverlap
        testing::Values(false)         // enableChunkedContext
        ),
    generateTestName);

// https://nvbugspro.nvidia.com/bug/4640177
// WAVEFRONT and RANDOM are disabled because of the accuracy mismatch
INSTANTIATE_TEST_SUITE_P(MedusaTests, ParamTest,
    testing::Combine(testing::Values(ModelParams{MEDUSA_MODEL_DIR, {2, 2}}),
        testing::Values(
            //
            ModelSpec{FP16_GPT_ATTENTION_PACKED_PAGED_DIR, INPUT_FILE, FP16_PLUGIN_PACKED_PAGED_LONG_RESULT_FILE,
                nvinfer1::DataType::kHALF}
                .usePackedInput()
                .usePagedKvCache()
                .useMedusa()
                .setBatchSizes({8})),
        testing::Values(TrtGptModelType::InflightFusedBatching), testing::Values(TrtGptModelIfbTestType::BULK),
        testing::Values(
            // enable more tests when mixed beam width is supported
            BeamConfig{1, {1}}),
        testing::Values(std::nullopt), // maxTokensInPagedKvCache
        testing::Values(std::nullopt), // freeGpuMemoryFraction
        testing::Values(false),        // enableTrtOverlap
        testing::Values(false)         // enableChunkedContext
        ),
    generateTestName);

INSTANTIATE_TEST_SUITE_P(DISABLED_GptLookaheadDecodingTests, ParamTest,
    testing::Combine(testing::Values(gptModelParams),
        testing::Values(
            //
            ModelSpec{FP16_GPT_ATTENTION_PACKED_PAGED_LA_DECODING_DIR, INPUT_FILE, FP16_PLUGIN_PACKED_PAGED_RESULT_FILE,
                nvinfer1::DataType::kHALF}
                .usePackedInput()
                .usePagedKvCache()
                .useRandomEndId()
                .useLookaheadDecoding()
                .smokeTest()),
        testing::Values(TrtGptModelType::InflightFusedBatching),
        testing::Values(
            TrtGptModelIfbTestType::BULK, TrtGptModelIfbTestType::WAVEFRONT, TrtGptModelIfbTestType::RANDOM),
        testing::Values(BeamConfig{1, {1}}), // beamConfig
        testing::Values(std::nullopt),       // maxTokensInPagedKvCache
        testing::Values(std::nullopt),       // freeGpuMemoryFraction
        testing::Values(false),              // enableTrtOverlap
        testing::Values(false)               // enableChunkedContext
        ),

    generateTestName);

// TODO(rkobus): enable tests
INSTANTIATE_TEST_SUITE_P(DISABLED_GptExplicitDraftTokensDecodingTests, ParamTest,
    testing::Combine(testing::Values(ModelParams{EXPLICIT_DRAFT_MODEL_DIR, {2, 2}}),
        testing::Values(
            //
            ModelSpec{FP16_GPT_ATTENTION_PACKED_PAGED_DIR, INPUT_FILE, FP16_PLUGIN_PACKED_PAGED_RESULT_FILE,
                nvinfer1::DataType::kHALF}
                .usePackedInput()
                .usePagedKvCache()
                .useExplicitDraftTokensDecoding()
                .smokeTest()
                .setBatchSizes({1})),
        testing::Values(TrtGptModelType::InflightFusedBatching), testing::Values(TrtGptModelIfbTestType::BULK),
        testing::Values(BeamConfig{1, {1}}), // beamConfig
        testing::Values(std::nullopt),       // maxTokensInPagedKvCache
        testing::Values(std::nullopt),       // freeGpuMemoryFraction
        testing::Values(false),              // enableTrtOverlap
        testing::Values(false)               // enableChunkedContext
        ),

    generateTestName);

#ifdef ENABLE_FP8
// Using IFB-enabled engine
INSTANTIATE_TEST_SUITE_P(GptjFP8Tests, ParamTest,
    testing::Combine(testing::Values(ModelParams{GPTJ_MODEL_DIR, {50256, 50256}}),
        testing::Values(
            //
            ModelSpec{
                FP8_GPT_ATTENTION_PLUGIN_IFB_PACKED_PATH, INPUT_FILE, FP8_PLUGIN_RESULT_FILE, nvinfer1::DataType::kHALF}

            ),
        testing::Values(TrtGptModelType::V1, TrtGptModelType::InflightBatching, TrtGptModelType::InflightFusedBatching),
        testing::Values(
            TrtGptModelIfbTestType::BULK, TrtGptModelIfbTestType::WAVEFRONT, TrtGptModelIfbTestType::RANDOM),
        testing::Values(
            // enable more tests when supported
            BeamConfig{1, {1}}         // , BeamConfig{2, {2}}, BeamConfig{2, {1, 2}}
            ),
        testing::Values(std::nullopt), // maxTokensInPagedKvCache
        testing::Values(std::nullopt), // freeGpuMemoryFraction
        testing::Values(false),        // enableTrtOverlap
        testing::Values(false)         // enableChunkedContext

        ),
    generateTestName);

#endif
