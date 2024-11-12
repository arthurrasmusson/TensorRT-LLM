/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "tensorrt_llm/batch_manager/pauseRequests.h"
#include "tensorrt_llm/batch_manager/utils/inflightBatchingUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/nvtxUtils.h"

void tensorrt_llm::batch_manager::PauseRequests::operator()(RequestVector& requestsToPause, ReqIdsSet& inflightReqIds,
    ReqIdsSet& reqIdsToPause, bool pauseFlagged, SequenceSlotManager& seqSlotManager,
    OptionalRef<KVCacheManager> kvCacheManager, OptionalRef<KVCacheManager> crossKvCacheManager,
    OptionalRef<BasePeftCacheManager> peftCacheManager) const
{
    NVTX3_SCOPED_RANGE(PauseRequests);
    if (!pauseFlagged)
    {
        // Loop over requests flagged to be paused, and if not in flight pause it right away
        for (auto& llmReq : requestsToPause)
        {
            auto const reqId = llmReq->mRequestId;
            if (inflightReqIds.find(reqId) == inflightReqIds.end())
            {
                // Not in flight, can terminate right away
                utils::terminateRequest(
                    seqSlotManager, *llmReq, mMaxInputLen, kvCacheManager, crossKvCacheManager, peftCacheManager, true);
            }
            else
            {
                // In flight, add to set for pausing later
                reqIdsToPause.insert(reqId);
            }
        }
    }
    else
    {
        for (auto& llmReq : requestsToPause)
        {
            auto const reqId = llmReq->mRequestId;
            inflightReqIds.erase(reqId);
            TLLM_LOG_DEBUG("request %lu removed from DECODER model inflight set", reqId);

            // If a request in this context had been flagged to be paused, pause it right away
            if (reqIdsToPause.find(reqId) != reqIdsToPause.end())
            {
                utils::terminateRequest(
                    seqSlotManager, *llmReq, mMaxInputLen, kvCacheManager, crossKvCacheManager, peftCacheManager, true);
                reqIdsToPause.erase(reqId);
            }
        }
    }
}
