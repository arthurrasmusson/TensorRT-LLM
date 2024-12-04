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

#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/executor/executor.h"

namespace tensorrt_llm::executor
{

KVCacheEventManager::KVCacheEventManager(
    std::shared_ptr<tensorrt_llm::batch_manager::kv_cache_manager::BaseKVCacheManager> kvCacheManager)
    : kvCacheManager{std::move(kvCacheManager)}
{
}

std::deque<KVCacheEvent> KVCacheEventManager::getLatestEvents(std::optional<std::chrono::milliseconds> timeout)
{
    return kvCacheManager->getLatestEvents(timeout);
}

} // namespace tensorrt_llm::executor
