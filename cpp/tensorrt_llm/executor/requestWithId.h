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

#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/serialization.h"
#include "tensorrt_llm/executor/serializeUtils.h"
#include "tensorrt_llm/executor/types.h"

#include <cstdint>
#include <vector>

namespace su = tensorrt_llm::executor::serialize_utils;

namespace tensorrt_llm::executor
{

class RequestWithId
{
public:
    Request req;
    IdType id;
    std::chrono::steady_clock::time_point queuedStart;

    static std::vector<char> serializeReqWithIds(std::vector<RequestWithId> const& reqWithIds);
    static std::vector<RequestWithId> deserializeReqWithIds(std::vector<char>& buffer);
};

} // namespace tensorrt_llm::executor
