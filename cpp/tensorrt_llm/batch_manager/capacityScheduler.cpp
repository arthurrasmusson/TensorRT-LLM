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

#include "tensorrt_llm/batch_manager/capacityScheduler.h"

#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/batch_manager/peftCacheManager.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/nvtxUtils.h"

#include <deque>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace tensorrt_llm::batch_manager::batch_scheduler
{

std::unique_ptr<CapacityScheduler> makeCapacityScheduler(tensorrt_llm::runtime::SizeType32 maxNumRequests,
    std::shared_ptr<kv_cache_manager::KVCacheManager> kvCacheManager,
    std::shared_ptr<kv_cache_manager::KVCacheManager> crossKvCacheManager,
    std::shared_ptr<BasePeftCacheManager> peftCacheManager, executor::CapacitySchedulerPolicy capacitySchedulerPolicy,
    bool manyMicroBatches, LlmRequestState_t noScheduleUntilState, LlmRequestState_t noScheduleAfterState)
{
    if (!kvCacheManager && !peftCacheManager->enabled())
    {
        return std::make_unique<MaxRequestsScheduler>(maxNumRequests, noScheduleUntilState, noScheduleAfterState);
    }
    else if (capacitySchedulerPolicy == executor::CapacitySchedulerPolicy::kMAX_UTILIZATION)
    {
        return std::make_unique<MaxUtilizationScheduler>(maxNumRequests, std::move(kvCacheManager),
            std::move(crossKvCacheManager), std::move(peftCacheManager), manyMicroBatches, noScheduleUntilState,
            noScheduleAfterState);
    }
    else if (capacitySchedulerPolicy == executor::CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT)
    {
        return std::make_unique<GuaranteedNoEvictScheduler>(maxNumRequests, std::move(kvCacheManager),
            std::move(crossKvCacheManager), std::move(peftCacheManager), noScheduleUntilState, noScheduleAfterState);
    }
    else
    {
        throw std::runtime_error("Unsupported capacity scheduler policy");
    }
}

MaxRequestsScheduler::MaxRequestsScheduler(
    SizeType32 maxNumRequests, LlmRequestState_t noScheduleUntilState, LlmRequestState_t noScheduleAfterState)
    : CapacityScheduler(noScheduleUntilState, noScheduleAfterState)
    , mMaxNumRequests(maxNumRequests)
{
}

MaxUtilizationScheduler::MaxUtilizationScheduler(SizeType32 maxNumRequests,
    std::shared_ptr<kv_cache_manager::KVCacheManager> kvCacheManager,
    std::shared_ptr<kv_cache_manager::KVCacheManager> crossKvCacheManager,
    std::shared_ptr<BasePeftCacheManager> peftCacheManager, bool manyMicroBatches,
    LlmRequestState_t noScheduleUntilState, LlmRequestState_t noScheduleAfterState)
    : CapacityScheduler(noScheduleUntilState, noScheduleAfterState)
    , mMaxNumRequests(maxNumRequests)
    , mKvCacheManager(std::move(kvCacheManager))
    , mCrossKvCacheManager(std::move(crossKvCacheManager))
    , mPeftCacheManager(std::move(peftCacheManager))
    , mManyMicroBatches{manyMicroBatches}
{
}

GuaranteedNoEvictScheduler::GuaranteedNoEvictScheduler(SizeType32 maxNumRequests,
    std::shared_ptr<kv_cache_manager::KVCacheManager> kvCacheManager,
    std::shared_ptr<kv_cache_manager::KVCacheManager> crossKvCacheManager,
    std::shared_ptr<BasePeftCacheManager> peftCacheManager, LlmRequestState_t noScheduleUntilState,
    LlmRequestState_t noScheduleAfterState)
    : CapacityScheduler(noScheduleUntilState, noScheduleAfterState)
    , mMaxNumRequests(maxNumRequests)
    , mKvCacheManager(std::move(kvCacheManager))
    , mCrossKvCacheManager(std::move(crossKvCacheManager))
    , mPeftCacheManager(std::move(peftCacheManager))
{
}

std::tuple<RequestVector, RequestVector> MaxRequestsScheduler::scheduleRequests(RequestList const& activeRequests) const
{
    NVTX3_SCOPED_RANGE(capacitySchedulerScheduling);

    RequestVector scheduledRequests;
    for (auto const& req : activeRequests)
    {
        // if request cannot be scheduled yet or request should no longer be scheduled, skip
        if (!req->hasReachedState(getNoScheduleUntilState()) || req->hasReachedState(getNoScheduleAfterState()))
        {
            continue;
        }

        if (scheduledRequests.size() >= static_cast<std::size_t>(mMaxNumRequests))
        {
            break;
        }

        if (req->isEncoderInitState() || req->isContextInitState() || req->isGenerationInProgressState())
        {
            scheduledRequests.emplace_back(req);
        }
    }
    return {std::move(scheduledRequests), RequestVector{}};
}

std::tuple<RequestVector, RequestVector> GuaranteedNoEvictScheduler::scheduleRequests(
    RequestList const& activeRequests) const
{
    NVTX3_SCOPED_RANGE(capacitySchedulerScheduling);
    RequestVector scheduledRequests;

    // Now check if we can add pending requests
    auto const maxBlocks = mKvCacheManager->getMaxNumBlocks();
    auto const maxPeftCachePages = mPeftCacheManager->getMaxDevicePages();

    // If a request is already in progress, include it
    // If it's been allocated, it had resource to run to completion
    // Also keep track of blocks needed to drive all in-progress requests to completion
    SizeType32 reservedBlocks{0};
    SizeType32 claimedPeftPages{0};
    std::unordered_set<uint64_t> uniqTaskIds{};
    RequestVector pendingRequests;
    pendingRequests.reserve(activeRequests.size());
    for (auto const& req : activeRequests)
    {
        // if request cannot be scheduled yet or request should no longer be scheduled, skip
        if (!req->hasReachedState(getNoScheduleUntilState()) || req->hasReachedState(getNoScheduleAfterState()))
        {
            continue;
        }

        if (scheduledRequests.size() >= static_cast<std::size_t>(mMaxNumRequests) || reservedBlocks == maxBlocks)
        {
            break;
        }
        else if (req->isGenerationInProgressState())
        {
            scheduledRequests.emplace_back(req);
            reservedBlocks += mKvCacheManager->getNeededBlocksToCompletion(*req);

            bool const reqHasLora = req->getLoraTaskId().has_value();
            bool const isNewTask = reqHasLora && !uniqTaskIds.count(req->getLoraTaskId().value());
            if (isNewTask)
            {
                claimedPeftPages += mPeftCacheManager->determineNumPages(req);
                uniqTaskIds.insert(req->getLoraTaskId().value());
            }
        }
        else
        {
            pendingRequests.emplace_back(req);
        }
    }

    // Now check if we can add pending requests
    auto availableBlocks = maxBlocks - reservedBlocks;
    auto availablePeftPages = maxPeftCachePages - claimedPeftPages;
    for (auto const& req : pendingRequests)
    {
        if (scheduledRequests.size() >= static_cast<std::size_t>(mMaxNumRequests))
        {
            break;
        }
        else if (req->isContextInitState())
        {
            auto neededBlocks = mKvCacheManager->getNeededBlocksToCompletion(*req);
            bool reqHasLora = req->getLoraTaskId().has_value();
            bool isNewTask = reqHasLora && !uniqTaskIds.count(req->getLoraTaskId().value());
            auto neededPeftPages = isNewTask ? mPeftCacheManager->determineNumPages(req) : 0;

            if (neededBlocks <= availableBlocks && neededPeftPages <= availablePeftPages)
            {
                scheduledRequests.emplace_back(req);
                availableBlocks -= neededBlocks;
                availablePeftPages -= neededPeftPages;
                if (isNewTask)
                {
                    uniqTaskIds.insert(req->getLoraTaskId().value());
                }
            }
            else if (neededBlocks > availableBlocks)
            {
                // If one requests fails to be scheduled, break
                break;
            }
        }
    }
    return {std::move(scheduledRequests), RequestVector{}};
}

std::tuple<RequestVector, RequestVector> MaxUtilizationScheduler::scheduleRequests(
    RequestList const& activeRequests) const
{
    NVTX3_SCOPED_RANGE(capacitySchedulerScheduling);
    mKvCacheManager->startScheduling();

    // Keep track of number of requests and block needed for the scheduled requests
    SizeType32 numScheduledBlocks{0};
    SizeType32 numScheduledPeftPages{0};
    std::unordered_set<uint64_t> seenTaskIds;

    // Find last active in case we need to evict
    auto startedReqLambda = [this](std::shared_ptr<LlmRequest> const& req)
    {
        return (req->hasReachedState(getNoScheduleUntilState()) && !req->hasReachedState(getNoScheduleAfterState())
            && ((req->isContextInitState() && !req->isFirstContextChunk()) || req->isGenerationInProgressState()));
    };

    RequestVector scheduledRequests;
    RequestVector pausedRequests;
    auto reqItEnd = std::end(activeRequests);
    for (auto reqIt = std::begin(activeRequests); reqIt != reqItEnd;)
    {
        auto const& req = *reqIt;
        TLLM_LOG_DEBUG("MaxUtilizationScheduler: scheduling req %lu", req->mRequestId);

        // if request cannot be scheduled yet or request should no longer be scheduled, skip
        if (!req->hasReachedState(getNoScheduleUntilState()) || req->hasReachedState(getNoScheduleAfterState()))
        {
            TLLM_LOG_DEBUG("MaxUtilizationScheduler: req %lu cannot / should not be scheduled", req->mRequestId);
            reqIt++;
            continue;
        }

        auto const [fitsKvCache, fitsPeftCache] = trySchedulingRequestMaxUtilization(
            req, scheduledRequests, numScheduledBlocks, numScheduledPeftPages, seenTaskIds);
        if (fitsKvCache && fitsPeftCache)
        {
            TLLM_LOG_DEBUG("MaxUtilizationScheduler: req %lu -> start", req->mRequestId);
            reqIt++;
        }
        else
        {
            auto const rbegin = std::reverse_iterator(reqItEnd);
            auto const rend = std::reverse_iterator(reqIt);
            auto const lastStartedReqIt = std::find_if(rbegin, rend, startedReqLambda);
            if (lastStartedReqIt != rend)
            {
                // If we can't allocate a started request, we need to start freeing started requests
                // from the end of the vector and try again
                // Here we simulate freeing the kvCache blocks associated with that sequence
                mKvCacheManager->schedulingRemoveSequence((*lastStartedReqIt)->mSeqSlot.value());
                pausedRequests.emplace_back(*lastStartedReqIt);
                TLLM_LOG_DEBUG("MaxUtilizationScheduler: req %lu -> pause", (*lastStartedReqIt)->mRequestId);
                reqItEnd = std::next(lastStartedReqIt).base();
            }
            else
            {
                break;
            }
        }
    }

    return {std::move(scheduledRequests), std::move(pausedRequests)};
}

std::pair<bool, bool> MaxUtilizationScheduler::trySchedulingRequestMaxUtilization(
    std::shared_ptr<LlmRequest> const& req, RequestVector& scheduledRequests, SizeType32& numScheduledBlocks,
    SizeType32& numScheduledPeftPages, std::unordered_set<uint64_t>& seenTaskIds) const
{
    if (scheduledRequests.size() < static_cast<std::size_t>(mMaxNumRequests))
    {
        SizeType32 numRequiredBlocks = mKvCacheManager->getNeededBlocksOneStep(*req, mManyMicroBatches);
        TLLM_LOG_DEBUG("MaxUtilizationScheduler: req %lu required blocks: %i", req->mRequestId, numRequiredBlocks);

        bool reqHasLora = req->getLoraTaskId().has_value();
        bool isNewTask = reqHasLora && !seenTaskIds.count(req->getLoraTaskId().value());
        SizeType32 numRequiredPeftPages = isNewTask ? mPeftCacheManager->determineNumPages(req) : 0;
        TLLM_LOG_DEBUG(
            "MaxUtilizationScheduler: req %lu required peft pages: %i", req->mRequestId, numRequiredPeftPages);
        bool fitsKvCache
            = mKvCacheManager->getBlockManager().schedulingHasFreeBlocks(numScheduledBlocks + numRequiredBlocks);
        bool fitsPeft = numRequiredPeftPages + numScheduledPeftPages <= mPeftCacheManager->getMaxDevicePages();

        if (fitsKvCache && fitsPeft)
        {
            numScheduledBlocks += numRequiredBlocks;
            TLLM_LOG_DEBUG("MaxUtilizationScheduler: scheduled blocks: %i", numScheduledBlocks);
            numScheduledPeftPages += numRequiredPeftPages;
            TLLM_LOG_DEBUG("MaxUtilizationScheduler: scheduled peft pages: %i", numRequiredPeftPages);
            scheduledRequests.emplace_back(req);
            if (isNewTask)
            {
                seenTaskIds.insert(req->getLoraTaskId().value());
            }
            return std::make_pair(fitsKvCache, fitsPeft);
        }
        else
        {
            return std::make_pair(fitsKvCache, fitsPeft);
        }
    }
    else
    {
        return std::make_pair(false, false);
    }
}

} // namespace tensorrt_llm::batch_manager::batch_scheduler
