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

#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/types.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace nlohmann
{

template <typename T>
struct adl_serializer<std::optional<T>>
{
    static void to_json(json& j, std::optional<T> const& opt)
    {
        if (opt == std::nullopt)
        {
            j = nullptr;
        }
        else
        {
            j = opt.value(); // this will call adl_serializer<T>::to_json which will
                             // find the free function to_json in T's namespace!
        }
    }

    static void from_json(json const& j, std::optional<T>& opt)
    {
        if (j.is_null())
        {
            opt = std::nullopt;
        }
        else
        {
            opt = j.template get<T>(); // same as above, but with
                                       // adl_serializer<T>::from_json
        }
    }
};
} // namespace nlohmann

namespace tensorrt_llm::executor
{

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(KvCacheStats, maxNumBlocks, freeNumBlocks, usedNumBlocks, tokensPerBlock,
    allocTotalBlocks, allocNewBlocks, reusedBlocks, missedBlocks, cacheHitRate);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
    StaticBatchingStats, numScheduledRequests, numContextRequests, numCtxTokens, numGenTokens, emptyGenSlots);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(InflightBatchingStats, numScheduledRequests, numContextRequests, numGenRequests,
    numPausedRequests, numCtxTokens, microBatchId, avgNumDecodedTokensPerIter);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(IterationStats, timestamp, iter, iterLatencyMS, newActiveRequestsQueueLatencyMS,
    numNewActiveRequests, numActiveRequests, numQueuedRequests, numCompletedRequests, maxNumActiveRequests,
    maxBatchSizeStatic, maxBatchSizeTunerRecommended, maxBatchSizeRuntime, gpuMemUsage, cpuMemUsage, pinnedMemUsage,
    kvCacheStats, staticBatchingStats, inflightBatchingStats);
NLOHMANN_JSON_SERIALIZE_ENUM(RequestStage,
    {{RequestStage::kQUEUED, "QUEUED"}, {RequestStage::kCONTEXT_IN_PROGRESS, "CONTEXT_IN_PROGRESS"},
        {RequestStage::kGENERATION_IN_PROGRESS, "GENERATION_IN_PROGRESS"},
        {RequestStage::kGENERATION_COMPLETE, "GENERATION_COMPLETE"}});
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(DisServingRequestStats, kvCacheTransferMS);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(RequestStats, id, stage, contextPrefillPosition, numGeneratedTokens,
    avgNumDecodedTokensPerIter, scheduled, paused, disServingStats, allocTotalBlocksPerRequest,
    allocNewBlocksPerRequest, reusedBlocksPerRequest, missedBlocksPerRequest, kvCacheHitRatePerRequest);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(RequestStatsPerIteration, iter, requestStats);

std::string JsonSerialization::toJsonStr(IterationStats const& iterationStats)
{
    json j = iterationStats;
    return j.dump();
}

std::string JsonSerialization::toJsonStr(RequestStatsPerIteration const& requestStatsPerIter)
{
    json j = requestStatsPerIter;
    return j.dump();
}

std::string JsonSerialization::toJsonStr(RequestStats const& requestStats)
{
    json j = requestStats;
    return j.dump();
}

} // namespace tensorrt_llm::executor
