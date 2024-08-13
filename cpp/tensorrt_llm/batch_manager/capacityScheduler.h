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
#include "tensorrt_llm/runtime/common.h"

#include <memory>
#include <unordered_set>

namespace tensorrt_llm::batch_manager
{
namespace kv_cache_manager
{
class KVCacheManager;
}
class BasePeftCacheManager;
} // namespace tensorrt_llm::batch_manager

namespace tensorrt_llm::batch_manager::batch_scheduler
{

/// @brief This scheduler takes into account the given request capacity and the KV cache capacity.
///        Depending on the CapacitySchedulerPolicy it will schedule already started and new requests,
///        or even pause previously started requests.
class CapacityScheduler
{
public:
    explicit CapacityScheduler(LlmRequestState_t noScheduleUntilState, LlmRequestState_t noScheduleAfterState)
        : mNoScheduleUntilState(noScheduleUntilState)
        , mNoScheduleAfterState(noScheduleAfterState)
    {
    }

    virtual ~CapacityScheduler() = default;

    using SizeType32 = tensorrt_llm::runtime::SizeType32;

    /// @brief Takes as input a sorted list of requests and outputs a sorted lists of requests
    ///        to update for this current iteration, and a map of requests to pause
    [[nodiscard]] virtual std::tuple<RequestVector, RequestVector> scheduleRequests(
        RequestList const& activeRequests) const
        = 0;

    [[nodiscard]] LlmRequestState_t constexpr getNoScheduleUntilState() const noexcept
    {
        return mNoScheduleUntilState;
    }

    [[nodiscard]] LlmRequestState_t constexpr getNoScheduleAfterState() const noexcept
    {
        return mNoScheduleAfterState;
    }

private:
    /// The state until/after which the scheduler should not schedule requests
    LlmRequestState_t mNoScheduleUntilState;
    LlmRequestState_t mNoScheduleAfterState;
};

/// @brief Schedule up to maxNumRequests requests
class MaxRequestsScheduler : public CapacityScheduler
{
public:
    explicit MaxRequestsScheduler(SizeType32 maxNumRequests,
        std::shared_ptr<kv_cache_manager::KVCacheManager> kvCacheManager,
        std::shared_ptr<kv_cache_manager::KVCacheManager> crossKvCacheManager,
        LlmRequestState_t noScheduleUntilState = REQUEST_STATE_CONTEXT_INIT,
        LlmRequestState_t noScheduleAfterState = REQUEST_STATE_GENERATION_COMPLETE);

    [[nodiscard]] std::tuple<RequestVector, RequestVector> scheduleRequests(
        RequestList const& activeRequests) const override;

private:
    SizeType32 mMaxNumRequests;
    std::shared_ptr<kv_cache_manager::KVCacheManager> mKvCacheManager{nullptr};
    std::shared_ptr<kv_cache_manager::KVCacheManager> mCrossKvCacheManager{nullptr};
};

/// @brief   Schedule requests using the MAX_UTILIZATION policy
/// @details Try reserving resources to advance requests by one step,
///          may pause previously started requests.
class MaxUtilizationScheduler : public CapacityScheduler
{
public:
    MaxUtilizationScheduler(SizeType32 maxNumRequests, std::shared_ptr<kv_cache_manager::KVCacheManager> kvCacheManager,
        std::shared_ptr<kv_cache_manager::KVCacheManager> crossKvCacheManager,
        std::shared_ptr<BasePeftCacheManager> peftCacheManager, bool manyMicroBatches,
        LlmRequestState_t noScheduleUntilState = REQUEST_STATE_CONTEXT_INIT,
        LlmRequestState_t noScheduleAfterState = REQUEST_STATE_GENERATION_COMPLETE);

    [[nodiscard]] std::tuple<RequestVector, RequestVector> scheduleRequests(
        RequestList const& activeRequests) const override;

private:
    /// @return {fitsKvCache, fitsPeft}
    std::pair<bool, bool> trySchedulingRequestMaxUtilization(std::shared_ptr<LlmRequest> const& req,
        RequestVector& scheduledRequests, SizeType32& numScheduledBlocks, SizeType32& numScheduledPeftPages,
        std::unordered_set<uint64_t>& seenTaskIds) const;

    SizeType32 mMaxNumRequests;
    std::shared_ptr<kv_cache_manager::KVCacheManager> mKvCacheManager{nullptr};
    std::shared_ptr<kv_cache_manager::KVCacheManager> mCrossKvCacheManager{nullptr};
    std::shared_ptr<BasePeftCacheManager> mPeftCacheManager{nullptr};
    /// @brief Boolean that indicates if multiple micro batches might be in flight
    bool mManyMicroBatches;
};

/// @brief Schedule requests using the GUARANTEED_NO_EVICT policy
class GuaranteedNoEvictScheduler : public CapacityScheduler
{
public:
    GuaranteedNoEvictScheduler(SizeType32 maxNumRequests,
        std::shared_ptr<kv_cache_manager::KVCacheManager> kvCacheManager,
        std::shared_ptr<kv_cache_manager::KVCacheManager> crossKvCacheManager,
        std::shared_ptr<BasePeftCacheManager> peftCacheManager,
        LlmRequestState_t noScheduleUntilState = REQUEST_STATE_CONTEXT_INIT,
        LlmRequestState_t noScheduleAfterState = REQUEST_STATE_GENERATION_COMPLETE);

    [[nodiscard]] std::tuple<RequestVector, RequestVector> scheduleRequests(
        RequestList const& activeRequests) const override;

private:
    SizeType32 mMaxNumRequests;
    std::shared_ptr<kv_cache_manager::KVCacheManager> mKvCacheManager{nullptr};
    std::shared_ptr<kv_cache_manager::KVCacheManager> mCrossKvCacheManager{nullptr};
    std::shared_ptr<BasePeftCacheManager> mPeftCacheManager{nullptr};
};

std::unique_ptr<CapacityScheduler> makeCapacityScheduler(tensorrt_llm::runtime::SizeType32 maxNumRequests,
    std::shared_ptr<kv_cache_manager::KVCacheManager> kvCacheManager,
    std::shared_ptr<kv_cache_manager::KVCacheManager> crossKvCacheManager,
    std::shared_ptr<BasePeftCacheManager> peftCacheManager, executor::CapacitySchedulerPolicy capacitySchedulerPolicy,
    bool manyMicroBatches = false, LlmRequestState_t noScheduleUntilState = REQUEST_STATE_CONTEXT_INIT,
    LlmRequestState_t noScheduleAfterState = REQUEST_STATE_GENERATION_COMPLETE);

} // namespace tensorrt_llm::batch_manager::batch_scheduler
