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

#include "tensorrt_llm/batch_manager/kvCacheUtils.h"

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

BlockIterator getBlockBeginIt(
    KVCacheManager const& cacheManager, LlmRequest const& request, SizeType32 beam, SizeType32 poolIdx)
{
    auto const& req = cacheManager.getSequence(request.mSeqSlot.value());
    auto blockIds = req.getCacheBlockIds().at(beam);
    return BlockIterator{cacheManager.getBlockManager().getPrimaryPool(poolIdx), std::move(blockIds), 0};
}

BlockIterator getBlockEndIt(
    KVCacheManager const& cacheManager, LlmRequest const& request, SizeType32 beam, SizeType32 poolIdx)
{
    auto const& req = cacheManager.getSequence(request.mSeqSlot.value());
    auto blockIds = req.getCacheBlockIds().at(beam);
    auto const backIdsSize = blockIds.size();
    return BlockIterator{cacheManager.getBlockManager().getPrimaryPool(poolIdx), std::move(blockIds), backIdsSize};
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager
