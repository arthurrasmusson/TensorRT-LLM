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
#include <deque>

namespace tensorrt_llm::executor
{

/// @brief Inserts a request into a request list sorted by priority / arrival time
void insertRequestInOrder(tensorrt_llm::batch_manager::RequestList& reqList,
    std::shared_ptr<tensorrt_llm::batch_manager::LlmRequest> const& req);

/// @brief Inserts a requestWithId into a request deque sorted by priority / arrival time
void insertRequestInOrder(std::deque<RequestWithId>& reqWithIdDeque, RequestWithId&& reqWithId);

} // namespace tensorrt_llm::executor
