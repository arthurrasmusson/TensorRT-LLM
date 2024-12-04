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

#include "tensorrt_llm/batch_manager/assignReqSeqSlots.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/nvtxUtils.h"

void tensorrt_llm::batch_manager::AssignReqSeqSlots::operator()(SequenceSlotManager& seqSlotManager,
    RequestVector const& contextRequests, RequestVector const& generationRequests) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(AssignReqSeqSlots);

    seqSlotManager.freeIdleSequenceSlots();
    for (auto const& requests : {contextRequests, generationRequests})
    {
        for (auto const& llmReq : requests)
        {
            auto const isReqNew = llmReq->isContextInitState() && llmReq->isFirstContextChunk();
            if (isReqNew && llmReq->getReturnPerfMetrics())
            {
                llmReq->setFirstScheduledTime(std::chrono::steady_clock::now());
            }
            auto const reqSeqSlot = seqSlotManager.getSequenceSlot(isReqNew, llmReq->mRequestId);
            TLLM_CHECK_WITH_INFO(reqSeqSlot, "Unable to get batch slot for reqId");
            llmReq->mSeqSlot = reqSeqSlot;
        }
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}
