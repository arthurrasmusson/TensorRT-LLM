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

#include "tensorrt_llm/executor/serializeUtils.h"
#include "tensorrt_llm/executor/contextPhaseState.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/types.h"
#include <gtest/gtest.h>
#include <optional>
#include <variant>

namespace su = tensorrt_llm::executor::serialize_utils;
namespace texec = tensorrt_llm::executor;

void compareKvCacheStats(texec::KvCacheStats const& lh, texec::KvCacheStats const& rh)
{
    EXPECT_TRUE(std::make_tuple(lh.maxNumBlocks, lh.freeNumBlocks, lh.usedNumBlocks, lh.tokensPerBlock)
        == std::make_tuple(rh.maxNumBlocks, rh.freeNumBlocks, rh.usedNumBlocks, rh.tokensPerBlock));
}

void compareStaticBatchingStats(texec::StaticBatchingStats const& lh, texec::StaticBatchingStats const& rh)
{
    EXPECT_TRUE(std::make_tuple(
                    lh.numScheduledRequests, lh.numContextRequests, lh.numCtxTokens, lh.numGenTokens, lh.emptyGenSlots)
        == std::make_tuple(
            rh.numScheduledRequests, rh.numContextRequests, rh.numCtxTokens, rh.numGenTokens, rh.emptyGenSlots));
}

void compareInflightBatchingStats(texec::InflightBatchingStats const& lh, texec::InflightBatchingStats const& rh)
{
    EXPECT_TRUE(std::make_tuple(lh.numScheduledRequests, lh.numContextRequests, lh.numGenRequests, lh.numPausedRequests,
                    lh.numCtxTokens, lh.microBatchId)
        == std::make_tuple(rh.numScheduledRequests, rh.numContextRequests, rh.numGenRequests, rh.numPausedRequests,
            rh.numCtxTokens, rh.microBatchId));
}

void compareIterationStats(texec::IterationStats const& lh, texec::IterationStats const& rh)
{
    EXPECT_EQ(lh.timestamp, rh.timestamp);
    EXPECT_EQ(lh.iter, rh.iter);
    EXPECT_EQ(lh.numActiveRequests, rh.numActiveRequests);
    EXPECT_EQ(lh.maxNumActiveRequests, rh.maxNumActiveRequests);
    EXPECT_EQ(lh.gpuMemUsage, rh.gpuMemUsage);
    EXPECT_EQ(lh.cpuMemUsage, rh.cpuMemUsage);
    EXPECT_EQ(lh.pinnedMemUsage, rh.pinnedMemUsage);
    EXPECT_EQ(lh.kvCacheStats.has_value(), rh.kvCacheStats.has_value());
    if (lh.kvCacheStats.has_value())
    {
        compareKvCacheStats(lh.kvCacheStats.value(), rh.kvCacheStats.value());
    }
    EXPECT_EQ(lh.staticBatchingStats.has_value(), rh.staticBatchingStats.has_value());
    if (lh.staticBatchingStats.has_value())
    {
        compareStaticBatchingStats(lh.staticBatchingStats.value(), rh.staticBatchingStats.value());
    }
    EXPECT_EQ(lh.inflightBatchingStats.has_value(), rh.inflightBatchingStats.has_value());
    if (lh.inflightBatchingStats.has_value())
    {
        compareInflightBatchingStats(lh.inflightBatchingStats.value(), rh.inflightBatchingStats.value());
    }
}

void compareResult(texec::Result res, texec::Result res2)
{
    EXPECT_EQ(res.isFinal, res2.isFinal);
    EXPECT_EQ(res.outputTokenIds, res2.outputTokenIds);
    EXPECT_EQ(res.cumLogProbs, res2.cumLogProbs);
    EXPECT_EQ(res.logProbs, res2.logProbs);
    EXPECT_EQ(res.finishReasons, res2.finishReasons);
    EXPECT_EQ(res.decodingIter, res2.decodingIter);
    EXPECT_EQ(res.sequenceIndex, res2.sequenceIndex);
    EXPECT_EQ(res.isSequenceFinal, res2.isSequenceFinal);
}

void compareResponse(texec::Response res, texec::Response res2)
{
    EXPECT_EQ(res.hasError(), res2.hasError());
    EXPECT_EQ(res.getRequestId(), res2.getRequestId());
    if (res.hasError())
    {
        EXPECT_EQ(res.getErrorMsg(), res2.getErrorMsg());
    }
    else
    {
        compareResult(res.getResult(), res2.getResult());
    }
}

template <typename T>
T serializeDeserialize(T val)
{
    auto size = su::serializedSize(val);
    std::ostringstream oss;
    su::serialize(val, oss);
    EXPECT_EQ(oss.str().size(), size);

    std::istringstream iss(oss.str());
    return su::deserialize<T>(iss);
}

template <typename T>
void testSerializeDeserialize(T val)
{
    auto val2 = serializeDeserialize(val);
    if constexpr (std::is_same<T, texec::Result>::value)
    {
        compareResult(val, val2);
    }
    else if constexpr (std::is_same<T, texec::Response>::value)
    {
        compareResponse(val, val2);
    }
    else if constexpr (std::is_same<T, texec::KvCacheStats>::value)
    {
        compareKvCacheStats(val, val2);
    }
    else if constexpr (std::is_same<T, texec::StaticBatchingStats>::value)
    {
        compareStaticBatching(val, val2);
    }
    else if constexpr (std::is_same<T, texec::InflightBatchingStats>::value)
    {
        compareInflightBatchingStats(val, val2);
    }
    else if constexpr (std::is_same<T, texec::IterationStats>::value)
    {
        compareIterationStats(val, val2);
    }
    else
    {
        EXPECT_EQ(val2, val) << typeid(T).name();
    }
}

template <typename T, typename T2>
void testSerializeDeserializeVariant(T val)
{
    auto val2 = serializeDeserialize(val);
    EXPECT_TRUE(std::holds_alternative<T2>(val2));
    if constexpr (std::is_same<T2, texec::Result>::value)
    {
        compareResult(std::get<T2>(val), std::get<T2>(val2));
    }
    else
    {
        EXPECT_EQ(std::get<T2>(val), std::get<T2>(val2));
    }
}

TEST(SerializeUtilsTest, FundamentalTypes)
{
    testSerializeDeserialize(int32_t(99));
    testSerializeDeserialize(int64_t(99));
    testSerializeDeserialize(uint32_t(99));
    testSerializeDeserialize(uint64_t(99));
    testSerializeDeserialize(float(99.f));
    testSerializeDeserialize(double(99.));
    testSerializeDeserialize(char('c'));
}

TEST(SerializeUtilsTest, Vector)
{
    {
        std::vector<int32_t> vec{1, 2, 3, 4};
        testSerializeDeserialize(vec);
    }
    {
        std::vector<char> vec{'a', 'b', 'c', 'd'};
        testSerializeDeserialize(vec);
    }
    {
        std::vector<float> vec{1.f, 2.f, 3.f, 4.f};
        testSerializeDeserialize(vec);
    }
}

TEST(SerializeUtilsTest, List)
{
    {
        std::list<int32_t> list{1, 2, 3, 4};
        testSerializeDeserialize(list);
    }
    {
        std::list<char> list{'a', 'b', 'c', 'd'};
        testSerializeDeserialize(list);
    }
    {
        std::list<float> list{9.0f, 3.333f};
        testSerializeDeserialize(list);
    }
}

TEST(SerializeUtilsTest, String)
{
    {
        std::string str{"abcdefg"};
        testSerializeDeserialize(str);
    }
}

TEST(SerializeUtilsTest, Optional)
{
    {
        std::optional<int32_t> opt = std::nullopt;
        testSerializeDeserialize(opt);
    }
    {
        std::optional<int32_t> opt = 1;
        testSerializeDeserialize(opt);
    }
    {
        std::optional<char> opt = std::nullopt;
        testSerializeDeserialize(opt);
    }
    {
        std::optional<char> opt = 'c';
        testSerializeDeserialize(opt);
    }
    {
        std::optional<float> opt = std::nullopt;
        testSerializeDeserialize(opt);
    }
    {
        std::optional<float> opt = 1.f;
        testSerializeDeserialize(opt);
    }
}

TEST(SerializeUtilsTest, Variant)
{
    {
        std::variant<bool, int32_t> val = int32_t(10);
        testSerializeDeserializeVariant<std::variant<bool, int32_t>, int32_t>(val);
    }
    {
        std::variant<bool, texec::Result> val = texec::Result{false, {{1, 2, 3}}};
        testSerializeDeserializeVariant<std::variant<bool, texec::Result>, texec::Result>(val);
    }
    {
        std::variant<bool, texec::Result> val = true;
        testSerializeDeserializeVariant<std::variant<bool, texec::Result>, bool>(val);
    }
}

TEST(SerializeUtilsTest, SamplingConfig)
{
    {
        texec::SamplingConfig val(2);
        testSerializeDeserialize(val);
    }
}

TEST(SerializeUtilsTest, Nested)
{
    {
        std::optional<std::vector<int32_t>> val = std::nullopt;
        testSerializeDeserialize(val);
    }
    {
        std::optional<std::vector<int32_t>> val = std::vector<int32_t>{1, 2, 3, 5};
        testSerializeDeserialize(val);
    }
    {
        std::list<std::vector<int32_t>> val = {{2, 3}, {5, 6, 7}};
        testSerializeDeserialize(val);
    }
    {
        std::list<std::vector<std::optional<float>>> val = {{2.f, 3.f}, {5.f, 6.f, 7.f}, {std::nullopt, 3.f}};
        testSerializeDeserialize(val);
    }
    // Unsupported won't build
    //{
    // std::map<int, int> val;
    // val[1] = 1;
    // testSerializeDeserialize(val);
    //}
    {
        auto const val = std::make_optional(std::vector<texec::SamplingConfig>{
            texec::SamplingConfig{1, 1, 0.05, 0.2}, texec::SamplingConfig{2, std::nullopt}});
        testSerializeDeserialize(val);
    }
    {
        auto const val = std::make_optional(texec::ExternalDraftTokensConfig({1, 1}));
        auto const size = su::serializedSize(val);
        std::ostringstream oss;
        su::serialize(val, oss);
        EXPECT_EQ(oss.str().size(), size);

        std::istringstream iss(oss.str());
        auto const val2 = su::deserialize<std::optional<texec::ExternalDraftTokensConfig>>(iss);
        EXPECT_EQ(val2.value().getTokens(), val.value().getTokens());
    }
}

TEST(SerializeUtilsTest, ResultResponse)
{
    texec::Result res = texec::Result{false, {{1, 2, 3}}, texec::VecLogProbs{1.0, 2.0},
        std::vector<texec::VecLogProbs>{{1.1, 2.2}, {3.3, 4.4}}, std::nullopt, std::nullopt, std::nullopt,
        std::vector<texec::FinishReason>{texec::FinishReason::kLENGTH}, texec::ContextPhaseParams({9, 37}), 3, 2, true};
    {
        testSerializeDeserialize(res);
    }
    {
        auto val = texec::Response(1, res);
        testSerializeDeserialize(val);
    }
    {
        auto val = texec::Response(1, "my error msg");
        testSerializeDeserialize(val);
    }
}

TEST(SerializeUtilsTest, VectorResponses)
{
    int numResponses = 10;
    std::vector<texec::Response> responsesIn;
    for (int i = 0; i < numResponses; ++i)
    {
        if (i < 5)
        {
            texec::Result res = texec::Result{false, {{i + 1, i + 2, i + 3}}, texec::VecLogProbs{1.0, 2.0},
                std::vector<texec::VecLogProbs>{{1.1, 2.2}, {3.3, 4.4}}, std::nullopt, std::nullopt, std::nullopt,
                std::vector<texec::FinishReason>{texec::FinishReason::kEND_ID}};
            responsesIn.emplace_back(i, res);
        }
        else
        {
            std::string errMsg = "my_err_msg" + std::to_string(i);
            responsesIn.emplace_back(i, errMsg);
        }
    }

    auto buffer = texec::Serialization::serialize(responsesIn);
    auto responsesOut = texec::Serialization::deserializeResponses(buffer);

    EXPECT_EQ(responsesIn.size(), responsesOut.size());

    for (int i = 0; i < numResponses; ++i)
    {
        compareResponse(responsesIn.at(i), responsesOut.at(i));
    }
}

TEST(SerializeUtilsTest, KvCacheConfig)
{
    texec::KvCacheConfig kvCacheConfig(true, 10, std::vector(1, 100), 2, 0.1, 10000, false);
    auto kvCacheConfig2 = serializeDeserialize(kvCacheConfig);

    EXPECT_EQ(kvCacheConfig.getEnableBlockReuse(), kvCacheConfig2.getEnableBlockReuse());
    EXPECT_EQ(kvCacheConfig.getMaxTokens(), kvCacheConfig2.getMaxTokens());
    EXPECT_EQ(kvCacheConfig.getMaxAttentionWindowVec(), kvCacheConfig2.getMaxAttentionWindowVec());
    EXPECT_EQ(kvCacheConfig.getSinkTokenLength(), kvCacheConfig2.getSinkTokenLength());
    EXPECT_EQ(kvCacheConfig.getFreeGpuMemoryFraction(), kvCacheConfig2.getFreeGpuMemoryFraction());
    EXPECT_EQ(kvCacheConfig.getHostCacheSize(), kvCacheConfig2.getHostCacheSize());
    EXPECT_EQ(kvCacheConfig.getOnboardBlocks(), kvCacheConfig2.getOnboardBlocks());
}

TEST(SerializeUtilsTest, SchedulerConfig)
{
    texec::SchedulerConfig schedulerConfig(
        texec::CapacitySchedulerPolicy::kMAX_UTILIZATION, texec::ContextChunkingPolicy::kFIRST_COME_FIRST_SERVED);
    auto schedulerConfig2 = serializeDeserialize(schedulerConfig);
    EXPECT_EQ(schedulerConfig.getCapacitySchedulerPolicy(), schedulerConfig2.getCapacitySchedulerPolicy());
    EXPECT_EQ(schedulerConfig.getContextChunkingPolicy(), schedulerConfig2.getContextChunkingPolicy());
}

TEST(SerializeUtilsTest, ParallelConfig)
{
    texec::ParallelConfig parallelConfig(texec::CommunicationType::kMPI, texec::CommunicationMode::kLEADER,
        std::vector<texec::SizeType32>{1, 2, 7}, std::vector<texec::SizeType32>{0, 1, 4});

    auto parallelConfig2 = serializeDeserialize(parallelConfig);
    EXPECT_EQ(parallelConfig.getCommunicationType(), parallelConfig2.getCommunicationType());
    EXPECT_EQ(parallelConfig.getCommunicationMode(), parallelConfig2.getCommunicationMode());
    EXPECT_EQ(parallelConfig.getDeviceIds(), parallelConfig2.getDeviceIds());
    EXPECT_EQ(parallelConfig.getParticipantIds(), parallelConfig2.getParticipantIds());
}

TEST(SerializeUtilsTest, PeftCacheConfig)
{
    auto peftCacheConfig = texec::PeftCacheConfig(10, 9, 8, 7, 6, 5, 4, 3, 2, 0.9, 1000);
    testSerializeDeserialize(peftCacheConfig);
}

TEST(SerializeUtilsTest, LookaheadDecodingConfig)
{
    auto lookaheadDecodingConfig = texec::LookaheadDecodingConfig(3, 5, 7);
    auto lookaheadDecodingConfig2 = serializeDeserialize(lookaheadDecodingConfig);
    EXPECT_EQ(lookaheadDecodingConfig.getNgramSize(), lookaheadDecodingConfig2.getNgramSize());
    EXPECT_EQ(lookaheadDecodingConfig.getWindowSize(), lookaheadDecodingConfig2.getWindowSize());
    EXPECT_EQ(lookaheadDecodingConfig.getVerificationSetSize(), lookaheadDecodingConfig2.getVerificationSetSize());
}

TEST(SerializeUtilsTest, DecodingConfig)
{
    {
        texec::DecodingMode decodingMode{texec::DecodingMode::Lookahead()};
        texec::LookaheadDecodingConfig laConfig{3, 5, 7};
        auto specDecodingConfig = texec::DecodingConfig(decodingMode, laConfig);
        auto specDecodingConfig2 = serializeDeserialize(specDecodingConfig);
        EXPECT_EQ(specDecodingConfig.getDecodingMode(), specDecodingConfig2.getDecodingMode());
        EXPECT_EQ(specDecodingConfig.getLookaheadDecodingConfig(), specDecodingConfig2.getLookaheadDecodingConfig());
    }

    {
        texec::DecodingMode decodingMode{texec::DecodingMode::Medusa()};
        texec::MedusaChoices medusaChoices{{{0, 1, 2}}};
        auto specDecodingConfig = texec::DecodingConfig(decodingMode, std::nullopt, medusaChoices);
        auto specDecodingConfig2 = serializeDeserialize(specDecodingConfig);
        EXPECT_EQ(specDecodingConfig.getDecodingMode(), specDecodingConfig2.getDecodingMode());
        EXPECT_EQ(specDecodingConfig.getMedusaChoices(), specDecodingConfig2.getMedusaChoices());
    }
}

TEST(SerializeUtilsTest, DebugConfig)
{
    texec::DebugConfig debugConfig(true, true, {"test"}, 3);
    auto debugConfig2 = serializeDeserialize(debugConfig);
    EXPECT_EQ(debugConfig.getDebugInputTensors(), debugConfig2.getDebugInputTensors());
    EXPECT_EQ(debugConfig.getDebugOutputTensors(), debugConfig2.getDebugOutputTensors());
    EXPECT_EQ(debugConfig.getDebugTensorNames(), debugConfig2.getDebugTensorNames());
    EXPECT_EQ(debugConfig.getDebugTensorsMaxIterations(), debugConfig2.getDebugTensorsMaxIterations());
}

TEST(SerializeUtilsTest, OrchestratorConfig)
{
    auto orchConfig = texec::OrchestratorConfig(false, std::filesystem::current_path().string());
    auto orchConfig2 = serializeDeserialize(orchConfig);
    EXPECT_EQ(orchConfig.getIsOrchestrator(), orchConfig2.getIsOrchestrator());
    EXPECT_EQ(orchConfig.getWorkerExecutablePath(), orchConfig2.getWorkerExecutablePath());
}

TEST(SerializeUtilsTest, KvCacheStats)
{
    auto stats = texec::KvCacheStats{10, 20, 30, 40, 50, 60, 70};
    auto stats2 = serializeDeserialize(stats);
    compareKvCacheStats(stats, stats2);
}

TEST(SerializeUtilsTest, StaticBatchingStats)
{
    auto stats = texec::StaticBatchingStats{10, 20, 30, 40, 50};
    auto stats2 = serializeDeserialize(stats);
    compareStaticBatchingStats(stats, stats2);
}

TEST(SerializeUtilsTest, InflightBatchingStats)
{
    auto stats = texec::InflightBatchingStats{10, 20, 30, 40, 50, 60};
    auto stats2 = serializeDeserialize(stats);
    compareInflightBatchingStats(stats, stats2);
}

TEST(SerializeUtilsTest, IterationStats)
{
    auto timestamp = std::string{"05:01:00"};
    auto iter = texec::IterationType{10};
    auto iterLatencyMS = double{100};
    auto newActiveRequestsQueueLatencyMS = double{1000};
    auto numActiveRequests = texec::SizeType32{20};
    auto numQueuedRequests = texec::SizeType32{30};
    auto numCompletedRequests = texec::SizeType32{10};
    auto maxNumActiveRequests = texec::SizeType32{30};
    auto gpuMemUsage = size_t{1024};
    auto cpuMemUsage = size_t{2048};
    auto pinnedMemUsage = size_t{4096};
    auto kvCacheStats = texec::KvCacheStats{10, 20, 30, 40, 50, 60, 70};
    auto staticBatchingStats = texec::StaticBatchingStats{10, 20, 30, 40, 50};
    auto ifbBatchingStats = texec::InflightBatchingStats{10, 20, 30, 40, 50, 60};
    {
        {
            auto stats = texec::IterationStats{timestamp, iter, iterLatencyMS, newActiveRequestsQueueLatencyMS,
                numActiveRequests, numQueuedRequests, numCompletedRequests, maxNumActiveRequests, gpuMemUsage,
                cpuMemUsage, pinnedMemUsage, kvCacheStats, kvCacheStats, staticBatchingStats, ifbBatchingStats};

            // serialize and deserialize using std::vector<char>
            {
                auto buffer = texec::Serialization::serialize(stats);
                auto stats2 = texec::Serialization::deserializeIterationStats(buffer);
                compareIterationStats(stats, stats2);
            }
            // serialize deserialize using is, os
            {
                auto stats2 = serializeDeserialize(stats);
                compareIterationStats(stats, stats2);
            }
        }
    }

    for (auto kvStats : std::vector<std::optional<texec::KvCacheStats>>{std::nullopt, kvCacheStats})
    {
        for (auto staticBatchStats :
            std::vector<std::optional<texec::StaticBatchingStats>>{std::nullopt, staticBatchingStats})
        {
            for (auto ifbBatchStats :
                std::vector<std::optional<texec::InflightBatchingStats>>{std::nullopt, ifbBatchingStats})
            {
                auto stats = texec::IterationStats{timestamp, iter, iterLatencyMS, newActiveRequestsQueueLatencyMS,
                    numActiveRequests, numQueuedRequests, numCompletedRequests, maxNumActiveRequests, gpuMemUsage,
                    cpuMemUsage, pinnedMemUsage, kvStats, kvStats, staticBatchStats, ifbBatchStats};
                {
                    auto buffer = texec::Serialization::serialize(stats);
                    auto stats2 = texec::Serialization::deserializeIterationStats(buffer);
                    compareIterationStats(stats, stats2);
                }
                {
                    auto stats2 = serializeDeserialize(stats);
                    compareIterationStats(stats, stats2);
                }
            }
        }
    }
}

TEST(SerializeUtilsTest, ContextPhaseParams)
{
    {
        auto stats = texec::ContextPhaseParams({1});
        auto stats2 = serializeDeserialize(stats);
        EXPECT_EQ(stats, stats2);
    }

    {
        auto state = std::make_unique<texec::ContextPhaseState>(1);
        state->setCommState(texec::kv_cache::CommState{{10, 20}});
        auto stats = texec::ContextPhaseParams({10, 20, 30, 40, 50, 60}, state.release());
        auto stats2 = serializeDeserialize(stats);
        EXPECT_EQ(stats, stats2);
    }

    {
        auto state = std::make_unique<texec::ContextPhaseState>(1);
        state->setCommState(texec::kv_cache::CommState{12, "127.0.0.1"});
        state->setCacheState(texec::kv_cache::CacheState{10, 12, 128, 128, 8, 8, nvinfer1::DataType::kFLOAT});
        auto stats = texec::ContextPhaseParams({10, 20, 30, 40, 50, 60}, state.release());
        auto stats2 = serializeDeserialize(stats);
        EXPECT_EQ(stats, stats2);
    }
}

TEST(SerializeUtilsTest, ExecutorConfig)
{
    texec::ExecutorConfig executorConfig(2, texec::SchedulerConfig(texec::CapacitySchedulerPolicy::kMAX_UTILIZATION),
        texec::KvCacheConfig(true), true, false, 500, 200, texec::BatchingType::kSTATIC, 128, 64,
        texec::ParallelConfig(texec::CommunicationType::kMPI, texec::CommunicationMode::kORCHESTRATOR),
        texec::PeftCacheConfig(10), std::nullopt,
        texec::DecodingConfig(texec::DecodingMode::Lookahead(), texec::LookaheadDecodingConfig(3, 5, 7)), 0.5f, 8,
        texec::ExtendedRuntimePerfKnobConfig(true), texec::DebugConfig(true), 60000000);
    auto executorConfig2 = serializeDeserialize(executorConfig);

    EXPECT_EQ(executorConfig.getMaxBeamWidth(), executorConfig2.getMaxBeamWidth());
    EXPECT_EQ(executorConfig.getSchedulerConfig(), executorConfig2.getSchedulerConfig());
    EXPECT_EQ(executorConfig.getKvCacheConfig().getEnableBlockReuse(),
        executorConfig2.getKvCacheConfig().getEnableBlockReuse());
    EXPECT_EQ(executorConfig.getEnableChunkedContext(), executorConfig2.getEnableChunkedContext());
    EXPECT_EQ(executorConfig.getNormalizeLogProbs(), executorConfig2.getNormalizeLogProbs());
    EXPECT_EQ(executorConfig.getIterStatsMaxIterations(), executorConfig2.getIterStatsMaxIterations());
    EXPECT_EQ(executorConfig.getRequestStatsMaxIterations(), executorConfig2.getRequestStatsMaxIterations());
    EXPECT_EQ(executorConfig.getBatchingType(), executorConfig2.getBatchingType());
    EXPECT_EQ(executorConfig.getMaxBatchSize(), executorConfig2.getMaxBatchSize());
    EXPECT_EQ(executorConfig.getMaxNumTokens(), executorConfig2.getMaxNumTokens());
    EXPECT_EQ(executorConfig.getParallelConfig().value().getCommunicationMode(),
        executorConfig2.getParallelConfig().value().getCommunicationMode());
    EXPECT_EQ(executorConfig.getPeftCacheConfig(), executorConfig2.getPeftCacheConfig());
    EXPECT_EQ(executorConfig.getDecodingConfig(), executorConfig2.getDecodingConfig());
    EXPECT_EQ(executorConfig.getGpuWeightsPercent(), executorConfig2.getGpuWeightsPercent());
    EXPECT_EQ(executorConfig.getMaxQueueSize(), executorConfig2.getMaxQueueSize());
    EXPECT_EQ(executorConfig.getExtendedRuntimePerfKnobConfig(), executorConfig2.getExtendedRuntimePerfKnobConfig());
    EXPECT_EQ(executorConfig.getDebugConfig(), executorConfig2.getDebugConfig());
    EXPECT_EQ(executorConfig.getMaxSeqIdleMicroseconds(), executorConfig2.getMaxSeqIdleMicroseconds());
}
