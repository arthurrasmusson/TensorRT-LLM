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

#include <iostream>

#include "tensorrt_llm/executor/types.h"

namespace tensorrt_llm::executor
{
std::ostream& operator<<(std::ostream& os, CapacitySchedulerPolicy policy)
{
    switch (policy)
    {
    case CapacitySchedulerPolicy::kMAX_UTILIZATION: os << "MAX_UTILIZATION"; break;
    case CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT: os << "GUARANTEED_NO_EVICT"; break;
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, ContextChunkingPolicy policy)
{
    switch (policy)
    {
    case ContextChunkingPolicy::kEQUAL_PROGRESS: os << "EQUAL_PROGRESS"; break;
    case ContextChunkingPolicy::kFIRST_COME_FIRST_SERVED: os << "FIRST_COME_FIRST_SERVED"; break;
    }
    return os;
}
} // namespace tensorrt_llm::executor
