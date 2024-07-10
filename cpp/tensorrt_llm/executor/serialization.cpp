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

#include "tensorrt_llm/executor/serialization.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/requestImpl.h"
#include "tensorrt_llm/executor/responseImpl.h"
#include "tensorrt_llm/executor/serializeUtils.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include <iostream>
#include <memory>
#include <type_traits>

namespace su = tensorrt_llm::executor::serialize_utils;

namespace tensorrt_llm::executor
{

// SamplingConfig
SamplingConfig Serialization::deserializeSamplingConfig(std::istream& is)
{
    auto beamWidth = su::deserialize<SizeType32>(is);
    auto topK = su::deserialize<std::optional<SizeType32>>(is);
    auto topP = su::deserialize<std::optional<FloatType>>(is);
    auto topPMin = su::deserialize<std::optional<FloatType>>(is);
    auto topPResetIds = su::deserialize<std::optional<TokenIdType>>(is);
    auto topPDecay = su::deserialize<std::optional<FloatType>>(is);
    auto randomSeed = su::deserialize<std::optional<RandomSeedType>>(is);
    auto temperature = su::deserialize<std::optional<FloatType>>(is);
    auto minLength = su::deserialize<std::optional<SizeType32>>(is);
    auto beamSearchDiversityRate = su::deserialize<std::optional<FloatType>>(is);
    auto repetitionPenalty = su::deserialize<std::optional<FloatType>>(is);
    auto presencePenalty = su::deserialize<std::optional<FloatType>>(is);
    auto frequencyPenalty = su::deserialize<std::optional<FloatType>>(is);
    auto lengthPenalty = su::deserialize<std::optional<FloatType>>(is);
    auto earlyStopping = su::deserialize<std::optional<SizeType32>>(is);
    auto noRepeatNgramSize = su::deserialize<std::optional<SizeType32>>(is);

    return SamplingConfig{beamWidth, topK, topP, topPMin, topPResetIds, topPDecay, randomSeed, temperature, minLength,
        beamSearchDiversityRate, repetitionPenalty, presencePenalty, frequencyPenalty, lengthPenalty, earlyStopping,
        noRepeatNgramSize};
}

void Serialization::serialize(SamplingConfig const& config, std::ostream& os)
{
    su::serialize(config.mBeamWidth, os);
    su::serialize(config.mTopK, os);
    su::serialize(config.mTopP, os);
    su::serialize(config.mTopPMin, os);
    su::serialize(config.mTopPResetIds, os);
    su::serialize(config.mTopPDecay, os);
    su::serialize(config.mRandomSeed, os);
    su::serialize(config.mTemperature, os);
    su::serialize(config.mMinLength, os);
    su::serialize(config.mBeamSearchDiversityRate, os);
    su::serialize(config.mRepetitionPenalty, os);
    su::serialize(config.mPresencePenalty, os);
    su::serialize(config.mFrequencyPenalty, os);
    su::serialize(config.mLengthPenalty, os);
    su::serialize(config.mEarlyStopping, os);
    su::serialize(config.mNoRepeatNgramSize, os);
}

size_t Serialization::serializedSize(SamplingConfig const& config)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(config.mBeamWidth);
    totalSize += su::serializedSize(config.mTopK);
    totalSize += su::serializedSize(config.mTopP);
    totalSize += su::serializedSize(config.mTopPMin);
    totalSize += su::serializedSize(config.mTopPResetIds);
    totalSize += su::serializedSize(config.mTopPDecay);
    totalSize += su::serializedSize(config.mRandomSeed);
    totalSize += su::serializedSize(config.mTemperature);
    totalSize += su::serializedSize(config.mMinLength);
    totalSize += su::serializedSize(config.mBeamSearchDiversityRate);
    totalSize += su::serializedSize(config.mRepetitionPenalty);
    totalSize += su::serializedSize(config.mPresencePenalty);
    totalSize += su::serializedSize(config.mFrequencyPenalty);
    totalSize += su::serializedSize(config.mLengthPenalty);
    totalSize += su::serializedSize(config.mEarlyStopping);
    totalSize += su::serializedSize(config.mNoRepeatNgramSize);
    return totalSize;
}

// OutputConfig
OutputConfig Serialization::deserializeOutputConfig(std::istream& is)
{
    auto returnLogProbs = su::deserialize<bool>(is);
    auto returnContextLogits = su::deserialize<bool>(is);
    auto returnGenerationLogits = su::deserialize<bool>(is);
    auto excludeInputFromOutput = su::deserialize<bool>(is);
    auto returnEncoderOutput = su::deserialize<bool>(is);
    return OutputConfig{
        returnLogProbs, returnContextLogits, returnGenerationLogits, excludeInputFromOutput, returnEncoderOutput};
}

void Serialization::serialize(OutputConfig const& config, std::ostream& os)
{
    su::serialize(config.returnLogProbs, os);
    su::serialize(config.returnContextLogits, os);
    su::serialize(config.returnGenerationLogits, os);
    su::serialize(config.excludeInputFromOutput, os);
    su::serialize(config.returnEncoderOutput, os);
}

size_t Serialization::serializedSize(OutputConfig const& config)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(config.returnLogProbs);
    totalSize += su::serializedSize(config.returnContextLogits);
    totalSize += su::serializedSize(config.returnGenerationLogits);
    totalSize += su::serializedSize(config.excludeInputFromOutput);
    totalSize += su::serializedSize(config.returnEncoderOutput);
    return totalSize;
}

// ExternalDraftTokensConfig
ExternalDraftTokensConfig Serialization::deserializeExternalDraftTokensConfig(std::istream& is)
{
    auto tokens = su::deserialize<VecTokens>(is);
    auto logits = su::deserialize<std::optional<Tensor>>(is);
    auto acceptanceThreshold = su::deserialize<std::optional<FloatType>>(is);

    return ExternalDraftTokensConfig{std::move(tokens), std::move(logits), acceptanceThreshold};
}

void Serialization::serialize(ExternalDraftTokensConfig const& config, std::ostream& os)
{
    su::serialize(config.mTokens, os);
    su::serialize(config.mLogits, os);
    su::serialize(config.mAcceptanceThreshold, os);
}

size_t Serialization::serializedSize(ExternalDraftTokensConfig const& config)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(config.mTokens);
    totalSize += su::serializedSize(config.mLogits);
    totalSize += su::serializedSize(config.mAcceptanceThreshold);
    return totalSize;
}

// PromptTuningConfig
PromptTuningConfig Serialization::deserializePromptTuningConfig(std::istream& is)
{
    auto tensor = su::deserialize<Tensor>(is);
    return PromptTuningConfig{std::move(tensor)};
}

void Serialization::serialize(PromptTuningConfig const& config, std::ostream& os)
{
    su::serialize(config.mEmbeddingTable, os);
}

size_t Serialization::serializedSize(PromptTuningConfig const& config)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(config.mEmbeddingTable);
    return totalSize;
}

// LoraConfig
LoraConfig Serialization::deserializeLoraConfig(std::istream& is)
{
    auto taskId = su::deserialize<IdType>(is);
    auto weights = su::deserialize<std::optional<Tensor>>(is);
    auto config = su::deserialize<std::optional<Tensor>>(is);
    return LoraConfig{taskId, std::move(weights), std::move(config)};
}

void Serialization::serialize(LoraConfig const& config, std::ostream& os)
{
    su::serialize(config.mTaskId, os);
    su::serialize(config.mWeights, os);
    su::serialize(config.mConfig, os);
}

size_t Serialization::serializedSize(LoraConfig const& config)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(config.mTaskId);
    totalSize += su::serializedSize(config.mWeights);
    totalSize += su::serializedSize(config.mConfig);
    return totalSize;
}

// Request
Request Serialization::deserializeRequest(std::istream& is)
{
    auto inputTokenIds = su::deserialize<VecTokens>(is);
    auto maxNewTokens = su::deserialize<SizeType32>(is);
    auto streaming = su::deserialize<bool>(is);
    auto samplingConfig = su::deserialize<SamplingConfig>(is);
    auto outputConfig = su::deserialize<OutputConfig>(is);
    auto endId = su::deserialize<std::optional<SizeType32>>(is);
    auto padId = su::deserialize<std::optional<SizeType32>>(is);
    auto badWords = su::deserialize<std::optional<std::list<VecTokens>>>(is);
    auto stopWords = su::deserialize<std::optional<std::list<VecTokens>>>(is);
    auto embeddingBias = su::deserialize<std::optional<Tensor>>(is);
    auto externalDraftTokensConfig = su::deserialize<std::optional<ExternalDraftTokensConfig>>(is);
    auto pTuningConfig = su::deserialize<std::optional<PromptTuningConfig>>(is);
    auto loraConfig = su::deserialize<std::optional<LoraConfig>>(is);
    auto logitsPostProcessorName = su::deserialize<std::optional<std::string>>(is);
    auto encoderInputTokenIds = su::deserialize<std::optional<VecTokens>>(is);

    return Request(std::move(inputTokenIds), maxNewTokens, streaming, samplingConfig, outputConfig, endId, padId,
        std::move(badWords), std::move(stopWords), std::move(embeddingBias), std::move(externalDraftTokensConfig),
        std::move(pTuningConfig), std::move(loraConfig), std::move(logitsPostProcessorName),
        std::move(encoderInputTokenIds));
}

void Serialization::serialize(Request const& request, std::ostream& os)
{
    request.mImpl->serialize(os);
}

size_t Serialization::serializedSize(Request const& request)
{
    return request.mImpl->serializedSize();
}

// Tensor
Tensor Serialization::deserializeTensor(std::istream& is)
{
    // DataType
    DataType dataType;
    is.read(reinterpret_cast<char*>(&dataType), sizeof(dataType));

    // Shape
    size_t shapeSize;
    is.read(reinterpret_cast<char*>(&shapeSize), sizeof(size_t));
    static constexpr int32_t MAX_DIMS{8};
    TLLM_CHECK(shapeSize < MAX_DIMS);

    Shape::DimType64 dims[MAX_DIMS];
    is.read(reinterpret_cast<char*>(&dims[0]), shapeSize * sizeof(Shape::DimType64));
    Shape shape(&dims[0], shapeSize);

    // Memory Type
    MemoryType memoryType;
    is.read(reinterpret_cast<char*>(&memoryType), sizeof(memoryType));

    // Size in bytes
    size_t sizeInBytes;
    is.read(reinterpret_cast<char*>(&sizeInBytes), sizeof(size_t));

    Tensor tensor;
    switch (memoryType)
    {
    case MemoryType::kCPU:
    {
        tensor = Tensor::cpu(dataType, shape);
        is.read(reinterpret_cast<char*>(tensor.getData()), static_cast<std::streamsize>(sizeInBytes));
        break;
    }
    case MemoryType::kCPU_PINNED:
    {
        tensor = Tensor::pinned(dataType, shape);
        is.read(reinterpret_cast<char*>(tensor.getData()), static_cast<std::streamsize>(sizeInBytes));
        break;
    }
    case MemoryType::kUVM:
    {
        tensor = Tensor::managed(dataType, shape);
        is.read(reinterpret_cast<char*>(tensor.getData()), static_cast<std::streamsize>(sizeInBytes));
        break;
    }
    case MemoryType::kGPU:
    {
        // TODO: Eventually we might want to support serialization/deserialization in GPU memory
        //       Until then created Pinned tensor and move to GPU
        auto pinnedTensor = Tensor::pinned(dataType, shape);
        is.read(reinterpret_cast<char*>(pinnedTensor.getData()), static_cast<std::streamsize>(sizeInBytes));
        auto stream = std::make_shared<tensorrt_llm::runtime::CudaStream>();
        tensor = pinnedTensor.copyToGpu(stream);
        stream->synchronize();
        break;
    }
    case MemoryType::kUNKNOWN:
    {
        TLLM_THROW("Cannot deserialize tensor with UNKNOWN type.");
        break;
    }
    default:
    {
        TLLM_THROW("Memory type not supported in deserializeTensor.");
        break;
    }
    }

    return tensor;
}

void Serialization::serialize(Tensor const& tensor, std::ostream& os)
{
    auto dataType = tensor.getDataType();
    os.write(reinterpret_cast<char const*>(&dataType), sizeof(dataType));
    auto shape = tensor.getShape();
    auto shapeSize = shape.size();
    os.write(reinterpret_cast<char const*>(&shapeSize), sizeof(shapeSize));
    os.write(reinterpret_cast<char const*>(&shape[0]), shapeSize * sizeof(Shape::DimType64));

    // Memory Type
    auto memoryType = tensor.getMemoryType();
    os.write(reinterpret_cast<char const*>(&memoryType), sizeof(memoryType));

    std::size_t sizeInBytes = tensor.getSizeInBytes();
    os.write(reinterpret_cast<char const*>(&sizeInBytes), sizeof(sizeInBytes));

    if (memoryType == MemoryType::kCPU || memoryType == MemoryType::kCPU_PINNED || memoryType == MemoryType::kUVM)
    {
        void const* data = tensor.getData();
        os.write(reinterpret_cast<char const*>(data), std::streamsize(sizeInBytes));
    }
    // Need special treatment for GPU type
    else if (memoryType == MemoryType::kGPU)
    {
        auto stream = std::make_shared<tensorrt_llm::runtime::CudaStream>();
        auto pinnedTensor = tensor.copyToPinned(stream);
        stream->synchronize();
        void const* data = pinnedTensor.getData();
        os.write(reinterpret_cast<char const*>(data), std::streamsize(sizeInBytes));
    }
    else if (memoryType == MemoryType::kUNKNOWN)
    {
        TLLM_THROW("Memory type unknown when serializing tensor");
    }
}

size_t Serialization::serializedSize(Tensor const& tensor)
{
    size_t totalSize = 0;
    totalSize += sizeof(tensor.getDataType()); // datatype
    auto const shape = tensor.getShape();
    auto const shapeSize = shape.size();
    totalSize += sizeof(decltype(shapeSize)); // number of dims
    TLLM_CHECK(shapeSize > 0);
    totalSize += shapeSize * sizeof(decltype(shape[0]));

    auto memoryType = tensor.getMemoryType();
    totalSize += sizeof(memoryType); // memory type

    totalSize += sizeof(size_t);     // Size in bytes
    totalSize += tensor.getSizeInBytes();
    return totalSize;
}

// Result
Result Serialization::deserializeResult(std::istream& is)
{

    auto isFinal = su::deserialize<bool>(is);
    auto outputTokenIds = su::deserialize<BeamTokens>(is);
    auto cumLogProbs = su::deserialize<std::optional<VecLogProbs>>(is);
    auto logProbs = su::deserialize<std::optional<std::vector<VecLogProbs>>>(is);
    auto contextLogits = su::deserialize<std::optional<Tensor>>(is);
    auto generationLogits = su::deserialize<std::optional<Tensor>>(is);
    auto encoderOutput = su::deserialize<std::optional<Tensor>>(is);

    return Result{isFinal, outputTokenIds, cumLogProbs, logProbs, contextLogits, generationLogits, encoderOutput};
}

void Serialization::serialize(Result const& result, std::ostream& os)
{
    su::serialize(result.isFinal, os);
    su::serialize(result.outputTokenIds, os);
    su::serialize(result.cumLogProbs, os);
    su::serialize(result.logProbs, os);
    su::serialize(result.contextLogits, os);
    su::serialize(result.generationLogits, os);
    su::serialize(result.encoderOutput, os);
}

size_t Serialization::serializedSize(Result const& result)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(result.isFinal);
    totalSize += su::serializedSize(result.outputTokenIds);
    totalSize += su::serializedSize(result.cumLogProbs);
    totalSize += su::serializedSize(result.logProbs);
    totalSize += su::serializedSize(result.contextLogits);
    totalSize += su::serializedSize(result.generationLogits);
    totalSize += su::serializedSize(result.encoderOutput);
    return totalSize;
}

// Response
Response Serialization::deserializeResponse(std::istream& is)
{
    auto requestId = su::deserialize<IdType>(is);
    auto errOrResult = su::deserialize<std::variant<std::string, Result>>(is);

    return std::holds_alternative<std::string>(errOrResult) ? Response{requestId, std::get<std::string>(errOrResult)}
                                                            : Response{requestId, std::get<Result>(errOrResult)};
}

void Serialization::serialize(Response const& response, std::ostream& os)
{
    su::serialize(response.mImpl->mRequestId, os);
    su::serialize(response.mImpl->mErrOrResult, os);
}

size_t Serialization::serializedSize(Response const& response)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(response.mImpl->mRequestId);
    totalSize += su::serializedSize(response.mImpl->mErrOrResult);
    return totalSize;
}

// Vector of responses
std::vector<Response> Serialization::deserializeResponses(std::vector<char>& buffer)
{
    std::vector<Response> responses;
    su::VectorWrapBuf<char> strbuf(buffer);
    std::istream is(&strbuf);
    std::size_t numResponses = su::deserialize<std::size_t>(is);
    for (std::size_t resp = 0; resp < numResponses; ++resp)
    {
        responses.emplace_back(std::move(Serialization::deserializeResponse(is)));
    }
    return responses;
}

std::vector<char> Serialization::serialize(std::vector<Response> const& responses)
{
    // Compute the size of serialized buffer
    size_t totalSize = 0;
    totalSize += sizeof(size_t);
    for (auto const& response : responses)
    {
        totalSize += su::serializedSize(response);
    }

    std::vector<char> buffer(totalSize);
    std::stringbuf strbuf(std::ios_base::out | std::ios_base::in);
    strbuf.pubsetbuf(&buffer[0], buffer.size());
    std::ostream os(&strbuf);

    su::serialize(responses.size(), os);
    for (auto const& response : responses)
    {
        su::serialize(response, os);
    }
    return buffer;
}

// ExecutorConfig
ExecutorConfig Serialization::deserializeExecutorConfig(std::istream& is)
{
    auto maxBeamWidth
        = su::deserialize<std::invoke_result_t<decltype(&ExecutorConfig::getMaxBeamWidth), ExecutorConfig>>(is);
    auto maxBatchSize
        = su::deserialize<std::invoke_result_t<decltype(&ExecutorConfig::getMaxBatchSize), ExecutorConfig>>(is);
    auto maxNumTokens
        = su::deserialize<std::invoke_result_t<decltype(&ExecutorConfig::getMaxNumTokens), ExecutorConfig>>(is);
    auto schedulerConfig
        = su::deserialize<std::invoke_result_t<decltype(&ExecutorConfig::getSchedulerConfig), ExecutorConfig>>(is);
    auto kvCacheConfig
        = su::deserialize<std::invoke_result_t<decltype(&ExecutorConfig::getKvCacheConfig), ExecutorConfig>>(is);
    auto enableChunkedContext
        = su::deserialize<std::invoke_result_t<decltype(&ExecutorConfig::getEnableChunkedContext), ExecutorConfig>>(is);
    auto normalizeLogProbs
        = su::deserialize<std::invoke_result_t<decltype(&ExecutorConfig::getNormalizeLogProbs), ExecutorConfig>>(is);
    auto iterStatsMaxIterations
        = su::deserialize<std::invoke_result_t<decltype(&ExecutorConfig::getIterStatsMaxIterations), ExecutorConfig>>(
            is);
    auto requestStatsMaxIterations = su::deserialize<
        std::invoke_result_t<decltype(&ExecutorConfig::getRequestStatsMaxIterations), ExecutorConfig>>(is);
    auto batchingType
        = su::deserialize<std::invoke_result_t<decltype(&ExecutorConfig::getBatchingType), ExecutorConfig>>(is);
    auto parallelConfig
        = su::deserialize<std::invoke_result_t<decltype(&ExecutorConfig::getParallelConfig), ExecutorConfig>>(is);
    auto peftCacheConfig
        = su::deserialize<std::invoke_result_t<decltype(&ExecutorConfig::getPeftCacheConfig), ExecutorConfig>>(is);
    auto decodingConfig
        = su::deserialize<std::invoke_result_t<decltype(&ExecutorConfig::getDecodingConfig), ExecutorConfig>>(is);

    return ExecutorConfig{maxBeamWidth, schedulerConfig, kvCacheConfig, enableChunkedContext, normalizeLogProbs,
        iterStatsMaxIterations, requestStatsMaxIterations, batchingType, maxBatchSize, maxNumTokens, parallelConfig,
        peftCacheConfig, std::nullopt, std::nullopt, decodingConfig};
}

size_t Serialization::serializedSize(ExecutorConfig const& executorConfig)
{
    TLLM_CHECK_WITH_INFO(!executorConfig.getLogitsPostProcessorMap().has_value(),
        "Serialization of executorConfig with logitsPostProcessor is currently not supported.");

    // Compute the size of serialized buffer
    size_t totalSize = 0;
    totalSize += su::serializedSize(executorConfig.getMaxBeamWidth());
    totalSize += su::serializedSize(executorConfig.getMaxBatchSize());
    totalSize += su::serializedSize(executorConfig.getMaxNumTokens());
    totalSize += su::serializedSize(executorConfig.getSchedulerConfig());
    totalSize += su::serializedSize(executorConfig.getKvCacheConfig());
    totalSize += su::serializedSize(executorConfig.getEnableChunkedContext());
    totalSize += su::serializedSize(executorConfig.getNormalizeLogProbs());
    totalSize += su::serializedSize(executorConfig.getIterStatsMaxIterations());
    totalSize += su::serializedSize(executorConfig.getRequestStatsMaxIterations());
    totalSize += su::serializedSize(executorConfig.getBatchingType());
    totalSize += su::serializedSize(executorConfig.getParallelConfig());
    totalSize += su::serializedSize(executorConfig.getPeftCacheConfig());
    totalSize += su::serializedSize(executorConfig.getDecodingConfig());

    return totalSize;
}

void Serialization::serialize(ExecutorConfig const& executorConfig, std::ostream& os)
{
    TLLM_CHECK_WITH_INFO(!executorConfig.getLogitsPostProcessorMap().has_value(),
        "Serialization of executorConfig with logitsPostProcessor is currently not supported.");

    su::serialize(executorConfig.getMaxBeamWidth(), os);
    su::serialize(executorConfig.getMaxBatchSize(), os);
    su::serialize(executorConfig.getMaxNumTokens(), os);
    su::serialize(executorConfig.getSchedulerConfig(), os);
    su::serialize(executorConfig.getKvCacheConfig(), os);
    su::serialize(executorConfig.getEnableChunkedContext(), os);
    su::serialize(executorConfig.getNormalizeLogProbs(), os);
    su::serialize(executorConfig.getIterStatsMaxIterations(), os);
    su::serialize(executorConfig.getRequestStatsMaxIterations(), os);
    su::serialize(executorConfig.getBatchingType(), os);
    su::serialize(executorConfig.getParallelConfig(), os);
    su::serialize(executorConfig.getPeftCacheConfig(), os);
    su::serialize(executorConfig.getDecodingConfig(), os);
}

// KvCacheConfig
KvCacheConfig Serialization::deserializeKvCacheConfig(std::istream& is)
{
    auto enableBlockReuse = su::deserialize<bool>(is);
    auto maxTokens = su::deserialize<std::optional<SizeType32>>(is);
    auto maxAttentionWindow = su::deserialize<std::optional<SizeType32>>(is);
    auto sinkTokenLength = su::deserialize<std::optional<SizeType32>>(is);
    auto freeGpuMemoryFraction = su::deserialize<std::optional<FloatType>>(is);
    auto hostCacheSize = su::deserialize<std::optional<size_t>>(is);
    auto onboardBlocks = su::deserialize<bool>(is);

    return KvCacheConfig{enableBlockReuse, maxTokens, maxAttentionWindow, sinkTokenLength, freeGpuMemoryFraction,
        hostCacheSize, onboardBlocks};
}

void Serialization::serialize(KvCacheConfig const& kvCacheConfig, std::ostream& os)
{
    su::serialize(kvCacheConfig.getEnableBlockReuse(), os);
    su::serialize(kvCacheConfig.getMaxTokens(), os);
    su::serialize(kvCacheConfig.getMaxAttentionWindow(), os);
    su::serialize(kvCacheConfig.getSinkTokenLength(), os);
    su::serialize(kvCacheConfig.getFreeGpuMemoryFraction(), os);
    su::serialize(kvCacheConfig.getHostCacheSize(), os);
    su::serialize(kvCacheConfig.getOnboardBlocks(), os);
}

size_t Serialization::serializedSize(KvCacheConfig const& kvCacheConfig)
{

    size_t totalSize = 0;
    totalSize += su::serializedSize(kvCacheConfig.getEnableBlockReuse());
    totalSize += su::serializedSize(kvCacheConfig.getMaxTokens());
    totalSize += su::serializedSize(kvCacheConfig.getMaxAttentionWindow());
    totalSize += su::serializedSize(kvCacheConfig.getSinkTokenLength());
    totalSize += su::serializedSize(kvCacheConfig.getFreeGpuMemoryFraction());
    totalSize += su::serializedSize(kvCacheConfig.getHostCacheSize());
    totalSize += su::serializedSize(kvCacheConfig.getOnboardBlocks());
    return totalSize;
}

// SchedulerConfig
SchedulerConfig Serialization::deserializeSchedulerConfig(std::istream& is)
{
    auto capacitySchedulerPolicy = su::deserialize<CapacitySchedulerPolicy>(is);
    auto contextChunkingPolicy = su::deserialize<std::optional<ContextChunkingPolicy>>(is);
    return SchedulerConfig{capacitySchedulerPolicy, contextChunkingPolicy};
}

void Serialization::serialize(SchedulerConfig const& schedulerConfig, std::ostream& os)
{
    su::serialize(schedulerConfig.getCapacitySchedulerPolicy(), os);
    su::serialize(schedulerConfig.getContextChunkingPolicy(), os);
}

size_t Serialization::serializedSize(SchedulerConfig const& schedulerConfig)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(schedulerConfig.getCapacitySchedulerPolicy());
    totalSize += su::serializedSize(schedulerConfig.getContextChunkingPolicy());
    return totalSize;
}

// ParallelConfig
ParallelConfig Serialization::deserializeParallelConfig(std::istream& is)
{
    auto commType = su::deserialize<CommunicationType>(is);
    auto commMode = su::deserialize<CommunicationMode>(is);
    auto deviceIds = su::deserialize<std::optional<std::vector<SizeType32>>>(is);
    auto participantids = su::deserialize<std::optional<std::vector<SizeType32>>>(is);
    auto orchestratorConfig = su::deserialize<std::optional<OrchestratorConfig>>(is);

    return ParallelConfig{commType, commMode, deviceIds, participantids, orchestratorConfig};
}

void Serialization::serialize(ParallelConfig const& parallelConfig, std::ostream& os)
{
    su::serialize(parallelConfig.getCommunicationType(), os);
    su::serialize(parallelConfig.getCommunicationMode(), os);
    su::serialize(parallelConfig.getDeviceIds(), os);
    su::serialize(parallelConfig.getParticipantIds(), os);
    su::serialize(parallelConfig.getOrchestratorConfig(), os);
}

size_t Serialization::serializedSize(ParallelConfig const& parallelConfig)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(parallelConfig.getCommunicationType());
    totalSize += su::serializedSize(parallelConfig.getCommunicationMode());
    totalSize += su::serializedSize(parallelConfig.getDeviceIds());
    totalSize += su::serializedSize(parallelConfig.getParticipantIds());
    totalSize += su::serializedSize(parallelConfig.getOrchestratorConfig());
    return totalSize;
}

// PeftCacheConfig
PeftCacheConfig Serialization::deserializePeftCacheConfig(std::istream& is)
{
    auto numHostModuleLayer = su::deserialize<SizeType32>(is);
    auto numDeviceModuleLayer = su::deserialize<SizeType32>(is);
    auto optimalAdapterSize = su::deserialize<SizeType32>(is);
    auto maxAdapterSize = su::deserialize<SizeType32>(is);
    auto numPutWorkers = su::deserialize<SizeType32>(is);
    auto numEnsureWorkers = su::deserialize<SizeType32>(is);
    auto numCopyStreams = su::deserialize<SizeType32>(is);
    auto maxPagesPerBlockHost = su::deserialize<SizeType32>(is);
    auto maxPagesPerBlockDevice = su::deserialize<SizeType32>(is);
    auto deviceCachePercent = su::deserialize<std::optional<FloatType>>(is);
    auto hostCacheSize = su::deserialize<std::optional<size_t>>(is);

    return PeftCacheConfig{numHostModuleLayer, numDeviceModuleLayer, optimalAdapterSize, maxAdapterSize, numPutWorkers,
        numEnsureWorkers, numCopyStreams, maxPagesPerBlockHost, maxPagesPerBlockDevice, deviceCachePercent,
        hostCacheSize};
}

void Serialization::serialize(PeftCacheConfig const& peftCacheConfig, std::ostream& os)
{
    su::serialize(peftCacheConfig.getNumHostModuleLayer(), os);
    su::serialize(peftCacheConfig.getNumDeviceModuleLayer(), os);
    su::serialize(peftCacheConfig.getOptimalAdapterSize(), os);
    su::serialize(peftCacheConfig.getMaxAdapterSize(), os);
    su::serialize(peftCacheConfig.getNumPutWorkers(), os);
    su::serialize(peftCacheConfig.getNumEnsureWorkers(), os);
    su::serialize(peftCacheConfig.getNumCopyStreams(), os);
    su::serialize(peftCacheConfig.getMaxPagesPerBlockHost(), os);
    su::serialize(peftCacheConfig.getMaxPagesPerBlockDevice(), os);
    su::serialize(peftCacheConfig.getDeviceCachePercent(), os);
    su::serialize(peftCacheConfig.getHostCacheSize(), os);
}

size_t Serialization::serializedSize(PeftCacheConfig const& peftCacheConfig)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(peftCacheConfig.getNumHostModuleLayer());
    totalSize += su::serializedSize(peftCacheConfig.getNumDeviceModuleLayer());
    totalSize += su::serializedSize(peftCacheConfig.getOptimalAdapterSize());
    totalSize += su::serializedSize(peftCacheConfig.getMaxAdapterSize());
    totalSize += su::serializedSize(peftCacheConfig.getNumPutWorkers());
    totalSize += su::serializedSize(peftCacheConfig.getNumEnsureWorkers());
    totalSize += su::serializedSize(peftCacheConfig.getNumCopyStreams());
    totalSize += su::serializedSize(peftCacheConfig.getMaxPagesPerBlockHost());
    totalSize += su::serializedSize(peftCacheConfig.getMaxPagesPerBlockDevice());
    totalSize += su::serializedSize(peftCacheConfig.getDeviceCachePercent());
    totalSize += su::serializedSize(peftCacheConfig.getHostCacheSize());
    return totalSize;
}

// OrchestratorConfig
OrchestratorConfig Serialization::deserializeOrchestratorConfig(std::istream& is)
{
    auto isOrchestrator = su::deserialize<bool>(is);
    auto path = su::deserialize<std::string>(is);
    // Note we ignore mpiComm since we don't need to exchange it
    return OrchestratorConfig{isOrchestrator, path};
}

void Serialization::serialize(OrchestratorConfig const& orchestratorConfig, std::ostream& os)
{
    su::serialize(orchestratorConfig.getIsOrchestrator(), os);
    su::serialize(orchestratorConfig.getWorkerExecutablePath(), os);
}

size_t Serialization::serializedSize(OrchestratorConfig const& orchestratorConfig)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(orchestratorConfig.getIsOrchestrator());
    totalSize += su::serializedSize(orchestratorConfig.getWorkerExecutablePath());
    return totalSize;
}

// DecodingMode
DecodingMode Serialization::deserializeDecodingMode(std::istream& is)
{
    auto mode = su::deserialize<DecodingMode::UnderlyingType>(is);

    return DecodingMode{mode};
}

void Serialization::serialize(DecodingMode const& decodingMode, std::ostream& os)
{
    su::serialize(decodingMode.getState(), os);
}

size_t Serialization::serializedSize(DecodingMode const& decodingMode)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(decodingMode.getState());
    return totalSize;
}

// LookaheadDecodingConfig
LookaheadDecodingConfig Serialization::deserializeLookaheadDecodingConfig(std::istream& is)
{
    auto ngramSize = su::deserialize<SizeType32>(is);
    auto windowSize = su::deserialize<SizeType32>(is);
    auto verificationSetSize = su::deserialize<SizeType32>(is);

    return LookaheadDecodingConfig{windowSize, ngramSize, verificationSetSize};
}

void Serialization::serialize(LookaheadDecodingConfig const& lookaheadDecodingConfig, std::ostream& os)
{
    su::serialize(lookaheadDecodingConfig.getNgramSize(), os);
    su::serialize(lookaheadDecodingConfig.getWindowSize(), os);
    su::serialize(lookaheadDecodingConfig.getVerificationSetSize(), os);
}

size_t Serialization::serializedSize(LookaheadDecodingConfig const& lookaheadDecodingConfig)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(lookaheadDecodingConfig.getNgramSize());
    totalSize += su::serializedSize(lookaheadDecodingConfig.getWindowSize());
    totalSize += su::serializedSize(lookaheadDecodingConfig.getVerificationSetSize());
    return totalSize;
}

// DecodingConfig
DecodingConfig Serialization::deserializeDecodingConfig(std::istream& is)
{
    auto decodingMode = su::deserialize<std::optional<DecodingMode>>(is);
    auto lookaheadDecodingConfig = su::deserialize<std::optional<LookaheadDecodingConfig>>(is);
    auto medusaChoices = su::deserialize<std::optional<MedusaChoices>>(is);

    return DecodingConfig{decodingMode, lookaheadDecodingConfig, medusaChoices};
}

void Serialization::serialize(DecodingConfig const& decodingConfig, std::ostream& os)
{
    su::serialize(decodingConfig.getDecodingMode(), os);
    su::serialize(decodingConfig.getLookaheadDecodingConfig(), os);
    su::serialize(decodingConfig.getMedusaChoices(), os);
}

size_t Serialization::serializedSize(DecodingConfig const& decodingConfig)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(decodingConfig.getDecodingMode());
    totalSize += su::serializedSize(decodingConfig.getLookaheadDecodingConfig());
    totalSize += su::serializedSize(decodingConfig.getMedusaChoices());
    return totalSize;
}

// KvCacheStats
KvCacheStats Serialization::deserializeKvCacheStats(std::istream& is)
{
    auto maxNumBlocks = su::deserialize<SizeType32>(is);
    auto freeNumBlocks = su::deserialize<SizeType32>(is);
    auto usedNumBlocks = su::deserialize<SizeType32>(is);
    auto tokensPerBlock = su::deserialize<SizeType32>(is);
    auto allocTotalBlocks = su::deserialize<SizeType32>(is);
    auto allocNewBlocks = su::deserialize<SizeType32>(is);
    auto reusedBlocks = su::deserialize<SizeType32>(is);

    return KvCacheStats{
        maxNumBlocks, freeNumBlocks, usedNumBlocks, tokensPerBlock, allocTotalBlocks, allocNewBlocks, reusedBlocks};
}

void Serialization::serialize(KvCacheStats const& kvCacheStats, std::ostream& os)
{
    su::serialize(kvCacheStats.maxNumBlocks, os);
    su::serialize(kvCacheStats.freeNumBlocks, os);
    su::serialize(kvCacheStats.usedNumBlocks, os);
    su::serialize(kvCacheStats.tokensPerBlock, os);
    su::serialize(kvCacheStats.allocTotalBlocks, os);
    su::serialize(kvCacheStats.allocNewBlocks, os);
    su::serialize(kvCacheStats.reusedBlocks, os);
}

size_t Serialization::serializedSize(KvCacheStats const& kvCacheStats)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(kvCacheStats.maxNumBlocks);
    totalSize += su::serializedSize(kvCacheStats.freeNumBlocks);
    totalSize += su::serializedSize(kvCacheStats.usedNumBlocks);
    totalSize += su::serializedSize(kvCacheStats.tokensPerBlock);
    totalSize += su::serializedSize(kvCacheStats.allocTotalBlocks);
    totalSize += su::serializedSize(kvCacheStats.allocNewBlocks);
    totalSize += su::serializedSize(kvCacheStats.reusedBlocks);
    return totalSize;
}

// StaticBatchingStats
StaticBatchingStats Serialization::deserializeStaticBatchingStats(std::istream& is)
{
    auto numScheduledRequests = su::deserialize<SizeType32>(is);
    auto numContextRequests = su::deserialize<SizeType32>(is);
    auto numCtxTokens = su::deserialize<SizeType32>(is);
    auto numGenTokens = su::deserialize<SizeType32>(is);
    auto emptyGenSlots = su::deserialize<SizeType32>(is);
    return StaticBatchingStats{numScheduledRequests, numContextRequests, numCtxTokens, numGenTokens, emptyGenSlots};
}

void Serialization::serialize(StaticBatchingStats const& staticBatchingStats, std::ostream& os)
{
    su::serialize(staticBatchingStats.numScheduledRequests, os);
    su::serialize(staticBatchingStats.numContextRequests, os);
    su::serialize(staticBatchingStats.numCtxTokens, os);
    su::serialize(staticBatchingStats.numGenTokens, os);
    su::serialize(staticBatchingStats.emptyGenSlots, os);
}

size_t Serialization::serializedSize(StaticBatchingStats const& staticBatchingStats)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(staticBatchingStats.numScheduledRequests);
    totalSize += su::serializedSize(staticBatchingStats.numContextRequests);
    totalSize += su::serializedSize(staticBatchingStats.numCtxTokens);
    totalSize += su::serializedSize(staticBatchingStats.numGenTokens);
    totalSize += su::serializedSize(staticBatchingStats.emptyGenSlots);
    return totalSize;
}

// InflightBatchingStats
InflightBatchingStats Serialization::deserializeInflightBatchingStats(std::istream& is)
{
    auto numScheduledRequests = su::deserialize<SizeType32>(is);
    auto numContextRequests = su::deserialize<SizeType32>(is);
    auto numGenRequests = su::deserialize<SizeType32>(is);
    auto numPausedRequests = su::deserialize<SizeType32>(is);
    auto numCtxTokens = su::deserialize<SizeType32>(is);
    auto microBatchId = su::deserialize<SizeType32>(is);
    return InflightBatchingStats{
        numScheduledRequests, numContextRequests, numGenRequests, numPausedRequests, numCtxTokens, microBatchId};
}

void Serialization::serialize(InflightBatchingStats const& inflightBatchingStats, std::ostream& os)
{
    su::serialize(inflightBatchingStats.numScheduledRequests, os);
    su::serialize(inflightBatchingStats.numContextRequests, os);
    su::serialize(inflightBatchingStats.numGenRequests, os);
    su::serialize(inflightBatchingStats.numPausedRequests, os);
    su::serialize(inflightBatchingStats.numCtxTokens, os);
    su::serialize(inflightBatchingStats.microBatchId, os);
}

size_t Serialization::serializedSize(InflightBatchingStats const& staticBatchingStats)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(staticBatchingStats.numScheduledRequests);
    totalSize += su::serializedSize(staticBatchingStats.numContextRequests);
    totalSize += su::serializedSize(staticBatchingStats.numGenRequests);
    totalSize += su::serializedSize(staticBatchingStats.numPausedRequests);
    totalSize += su::serializedSize(staticBatchingStats.numCtxTokens);
    totalSize += su::serializedSize(staticBatchingStats.microBatchId);
    return totalSize;
}

// IterationStats

IterationStats Serialization::deserializeIterationStats(std::istream& is)
{
    IterationStats iterStats;
    auto timestamp = su::deserialize<std::string>(is);
    auto iter = su::deserialize<IterationType>(is);
    auto iterLatencyMS = su::deserialize<double>(is);
    auto numActiveRequests = su::deserialize<SizeType32>(is);
    auto numQueuedRequests = su::deserialize<SizeType32>(is);
    auto maxNumActiveRequests = su::deserialize<SizeType32>(is);
    auto gpuMemUsage = su::deserialize<size_t>(is);
    auto cpuMemUsage = su::deserialize<size_t>(is);
    auto pinnedMemUsage = su::deserialize<size_t>(is);
    auto kvCacheStats = su::deserialize<std::optional<KvCacheStats>>(is);
    auto crossKvCacheStats = su::deserialize<std::optional<KvCacheStats>>(is);
    auto staticBatchingStats = su::deserialize<std::optional<StaticBatchingStats>>(is);
    auto inflightBatchingStats = su::deserialize<std::optional<InflightBatchingStats>>(is);

    return IterationStats{timestamp, iter, iterLatencyMS, numActiveRequests, numQueuedRequests, maxNumActiveRequests,
        gpuMemUsage, cpuMemUsage, pinnedMemUsage, kvCacheStats, crossKvCacheStats, staticBatchingStats,
        inflightBatchingStats};
}

IterationStats Serialization::deserializeIterationStats(std::vector<char>& buffer)
{
    su::VectorWrapBuf<char> strbuf(buffer);
    std::istream is(&strbuf);

    return Serialization::deserializeIterationStats(is);
}

size_t Serialization::serializedSize(IterationStats const& iterStats)
{
    // Compute the size of serialized buffer
    size_t totalSize = 0;

    totalSize += su::serializedSize(iterStats.timestamp);
    totalSize += su::serializedSize(iterStats.iter);
    totalSize += su::serializedSize(iterStats.iterLatencyMS);
    totalSize += su::serializedSize(iterStats.numActiveRequests);
    totalSize += su::serializedSize(iterStats.numQueuedRequests);
    totalSize += su::serializedSize(iterStats.maxNumActiveRequests);
    totalSize += su::serializedSize(iterStats.gpuMemUsage);
    totalSize += su::serializedSize(iterStats.cpuMemUsage);
    totalSize += su::serializedSize(iterStats.pinnedMemUsage);
    totalSize += su::serializedSize(iterStats.kvCacheStats);
    totalSize += su::serializedSize(iterStats.crossKvCacheStats);
    totalSize += su::serializedSize(iterStats.staticBatchingStats);
    totalSize += su::serializedSize(iterStats.inflightBatchingStats);

    return totalSize;
}

void Serialization::serialize(IterationStats const& iterStats, std::ostream& os)
{
    su::serialize(iterStats.timestamp, os);
    su::serialize(iterStats.iter, os);
    su::serialize(iterStats.iterLatencyMS, os);
    su::serialize(iterStats.numActiveRequests, os);
    su::serialize(iterStats.numQueuedRequests, os);
    su::serialize(iterStats.maxNumActiveRequests, os);
    su::serialize(iterStats.gpuMemUsage, os);
    su::serialize(iterStats.cpuMemUsage, os);
    su::serialize(iterStats.pinnedMemUsage, os);
    su::serialize(iterStats.kvCacheStats, os);
    su::serialize(iterStats.crossKvCacheStats, os);
    su::serialize(iterStats.staticBatchingStats, os);
    su::serialize(iterStats.inflightBatchingStats, os);
}

std::vector<char> Serialization::serialize(IterationStats const& iterStats)
{
    auto totalSize = Serialization::serializedSize(iterStats);
    std::vector<char> buffer(totalSize);

    std::stringbuf strbuf(std::ios_base::out | std::ios_base::in);
    strbuf.pubsetbuf(&buffer[0], buffer.size());
    std::ostream os(&strbuf);

    Serialization::serialize(iterStats, os);

    return buffer;
}

// String
std::string Serialization::deserializeString(std::istream& is)
{
    return su::deserialize<std::string>(is);
}

// Bool
bool Serialization::deserializeBool(std::istream& is)
{
    return su::deserialize<bool>(is);
}

// ModelType
ModelType Serialization::deserializeModelType(std::istream& is)
{
    return su::deserialize<ModelType>(is);
}

} // namespace tensorrt_llm::executor
