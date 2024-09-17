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

#include "mpiDataTransceiver.h"
#include "tensorrt_llm/batch_manager/cacheFormatter.h"
#include "tensorrt_llm/executor/contextPhaseState.h"

namespace tensorrt_llm::batch_manager
{

template class MpiDataSender<executor::kv_cache::CacheState>;

template class MpiDataReceiver<executor::kv_cache::CacheState>;

} // namespace tensorrt_llm::batch_manager
