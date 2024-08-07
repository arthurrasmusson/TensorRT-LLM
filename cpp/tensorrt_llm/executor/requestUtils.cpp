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

#include "tensorrt_llm/executor/requestUtils.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/executor/requestWithId.h"
#include <algorithm>

using tensorrt_llm::executor::RequestWithId;
using tensorrt_llm::batch_manager::RequestList;
using tensorrt_llm::batch_manager::LlmRequest;

void tensorrt_llm::executor::insertRequestInOrder(RequestList& reqList, std::shared_ptr<LlmRequest> const& req)
{
    auto const it = std::upper_bound(std::begin(reqList), std::end(reqList), req,
        [](std::shared_ptr<LlmRequest> const& a, std::shared_ptr<LlmRequest> const& b)
        { return a->priority() > b->priority(); });
    reqList.insert(it, req);
}

void tensorrt_llm::executor::insertRequestInOrder(std::deque<RequestWithId>& reqWithIdDeque, RequestWithId&& reqWithId)
{
    auto const it = std::upper_bound(std::begin(reqWithIdDeque), std::end(reqWithIdDeque), reqWithId,
        [](RequestWithId const& a, RequestWithId const& b) { return a.req.getPriority() > b.req.getPriority(); });
    reqWithIdDeque.insert(it, std::move(reqWithId));
}
