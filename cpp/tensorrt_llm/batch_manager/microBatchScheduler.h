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

#pragma once

#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/runtime/common.h"

#include <optional>

namespace tensorrt_llm::batch_manager::batch_scheduler
{

struct ContextChunkingConfig
{
    tensorrt_llm::executor::ContextChunkingPolicy chunkingPolicy;
    /// The minimum size, also known as the chunk unit size. It generally
    /// needs to be equal to the size of the kv cache block or its integer
    /// multiples (except for the last context chunk) to avoid fragmentation.
    /// When set to null, it indicates that the context chunk is disabled.
    tensorrt_llm::runtime::SizeType32 chunkUnitSize;
};

/// @brief This scheduler takes into account the desired batch size and limits of the TRT engine to schedule requests.
class MicroBatchScheduler
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using ContextChunkingPolicy = tensorrt_llm::executor::ContextChunkingPolicy;

    explicit MicroBatchScheduler(SizeType32 maxBatchSize, std::optional<SizeType32> maxNumTokens = std::nullopt,
        std::optional<ContextChunkingConfig> ctxChunkConfig = std::nullopt,
        std::optional<SizeType32> maxContextLength = std::nullopt,
        LlmRequestState_t noScheduleUntilState = REQUEST_STATE_CONTEXT_INIT,
        LlmRequestState_t noScheduleAfterState = REQUEST_STATE_GENERATION_COMPLETE);

    ScheduledRequests scheduleRequests(RequestVector const& activeRequests, ReqIdsSet const& inflightReqIds);

    static void setCtxRequestsChunkSize(RequestVector const& contextsToBeChunked, ContextChunkingPolicy ctxChunkPolicy,
        std::optional<SizeType32> ctxTokensCapacity, SizeType32 chunkUnitSize,
        std::optional<SizeType32> const& maxContextLength);

private:
    template <ContextChunkingPolicy tPolicy>
    static void setCtxRequestsChunkSize(RequestVector const& contextsToBeChunked,
        std::optional<SizeType32> ctxTokensCapacity, SizeType32 chunkUnitSize,
        std::optional<SizeType32> const& maxContextLength);

    /// After the chunk sizes have been determined, this function will discard
    /// any draft tokens that don't fit.
    static void fitDraftTokens(RequestVector const& contextsToBeChunked, std::optional<SizeType32> ctxTokensCapacity,
        SizeType32 chunkUnitSize, std::optional<SizeType32> const& maxContextLength);

    /// The maximum number of requests returned by scheduleRequests
    SizeType32 mMaxBatchSize;

    /// The maximum number of tokens to include in a batch
    std::optional<SizeType32> mMaxNumTokens;

    /// The maximum length of the context. If the context exceeds this length,
    /// it must be chunked, otherwise it cannot be processed. Therefore, it
    /// needs to be set together with the chunk unit size to make sense.
    /// When set to null, it indicates that context length is unlimited.
    std::optional<SizeType32> mMaxContextLength;

    std::optional<ContextChunkingConfig> mCtxChunkConfig;

    /// The state until/after which the scheduler should not schedule requests
    LlmRequestState_t mNoScheduleUntilState;
    LlmRequestState_t mNoScheduleAfterState;
};

} // namespace tensorrt_llm::batch_manager::batch_scheduler
