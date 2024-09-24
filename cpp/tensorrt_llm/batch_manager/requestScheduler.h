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

#include "tensorrt_llm/batch_manager/capacityScheduler.h"
#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/microBatchScheduler.h"
#include "tensorrt_llm/runtime/common.h"

#include <memory>
#include <utility>

namespace tensorrt_llm::batch_manager
{
class BasePeftCacheManager;
}

namespace tensorrt_llm::batch_manager::batch_scheduler
{

/// Currently the usage is shared between both encoder and decoder models
/// TODO: abstract and separate into different classes
class RequestScheduler
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;

    //! @param noScheduleUntilState For multi-phase models such as encoder-decoder and mulitmodal models, some parts of
    //! the model may not start scheduling requests until a certain phase is reached. This allows request scheduler to
    //! skip scheduling requests
    RequestScheduler(SizeType32 maxBatchSize, SizeType32 numContexts,
        std::shared_ptr<kv_cache_manager::KVCacheManager> kvCacheManager,
        std::shared_ptr<kv_cache_manager::KVCacheManager> crossKvCacheManager,
        std::shared_ptr<BasePeftCacheManager> peftCacheManager, executor::SchedulerConfig const& schedulerConfig,
        std::optional<SizeType32> maxNumTokens = std::nullopt,
        std::optional<ContextChunkingConfig> ctxChunkConfig = std::nullopt,
        std::optional<SizeType32> maxContextLength = std::nullopt,
        LlmRequestState noScheduleUntilState = LlmRequestState::kCONTEXT_INIT,
        LlmRequestState noScheduleAfterState = LlmRequestState::kGENERATION_COMPLETE)
        : mMicroBatchScheduler{maxBatchSize, maxNumTokens, ctxChunkConfig, maxContextLength, noScheduleUntilState,
            noScheduleAfterState}
        , mCapacityScheduler{batch_scheduler::makeCapacityScheduler(numContexts * maxBatchSize, kvCacheManager,
              std::move(crossKvCacheManager), std::move(peftCacheManager), schedulerConfig.getCapacitySchedulerPolicy(),
              numContexts > 1, noScheduleUntilState, noScheduleAfterState)}
    {
        if (kvCacheManager && ctxChunkConfig)
        {
            TLLM_CHECK_WITH_INFO(ctxChunkConfig.value().chunkUnitSize % kvCacheManager->getTokensPerBlock() == 0,
                "To prevent cache fragmentation, the context chunk unit size (%d) should be divisible by the number of "
                "tokens per kv-cache block (%d).",
                ctxChunkConfig.value().chunkUnitSize, kvCacheManager->getTokensPerBlock());
        }
    }

    //! @brief Use CapacityScheduler and MicroBatchScheduler to schedule requests
    //! @return {contextRequests, generationRequests, pausedRequests}
    std::tuple<RequestVector, RequestVector, RequestVector> scheduleRequests(
        RequestList const& activeRequests, ReqIdsSet const& inflightReqIds)
    {
        auto [fittingRequests, pausedRequests] = mCapacityScheduler->scheduleRequests(activeRequests);
        auto [contextRequests, generationRequests]
            = mMicroBatchScheduler.scheduleRequests(fittingRequests, inflightReqIds);

        TLLM_LOG_DEBUG("[Summary] Capacity scheduler allows %d requests, pauses %d requests", fittingRequests.size(),
            pausedRequests.size());
        TLLM_LOG_DEBUG(
            "[Summary] Micro Batch scheduler schedules %d context/encoder requests, %d generation requests, %d "
            "requests inflight with the model already",
            contextRequests.size(), generationRequests.size(), inflightReqIds.size());

        return {std::move(contextRequests), std::move(generationRequests), std::move(pausedRequests)};
    }

private:
    MicroBatchScheduler mMicroBatchScheduler;
    std::unique_ptr<CapacityScheduler> mCapacityScheduler;
};

} // namespace tensorrt_llm::batch_manager::batch_scheduler
