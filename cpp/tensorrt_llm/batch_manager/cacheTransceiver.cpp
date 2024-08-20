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

#include "cacheTransceiver.h"
#include "tensorrt_llm/common/assert.h"

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

bool CacheBlockSender::inquireSupport(DataContext const* receiverContext)
{
    auto senderCacheContext = dynamic_cast<CacheContext const*>(receiverContext);
    if (!senderCacheContext)
    {
        return false;
    }
    return mSelfContext.getConfig() == senderCacheContext->getConfig();
}

void CacheBlockSender::send(LlmRequest const& request, DataContext const& destination)
{
    auto const& dst = dynamic_cast<CacheContext const&>(destination);
    auto dstRank = dst.getRanks().at(mSelfContext.getSelfIdx());
    TLLM_CHECK_WITH_INFO(request.mSamplingConfig.beamWidth == 1, "Currently only supports beam width 1.");
    constexpr SizeType32 beam{0};
    auto endIt = getBlockEndIt(*mCacheManager, request, beam);
    for (auto it = getBlockBeginIt(*mCacheManager, request, beam); it != endIt; ++it)
    {
        mComm->sendBuffer(*it, dstRank);
    }
}

bool CacheBlockReceiver::inquireSupport(DataContext const* senderContext)
{
    auto senderCacheContext = dynamic_cast<CacheContext const*>(senderContext);
    if (!senderCacheContext)
    {
        return false;
    }
    return mSelfContext.getConfig() == senderCacheContext->getConfig();
}

void CacheBlockReceiver::receive(LlmRequest const& request, DataContext const& source)
{
    auto const& src = dynamic_cast<CacheContext const&>(source);
    TLLM_CHECK(src.getRanks().size() > static_cast<size_t>(mSelfContext.getSelfIdx()));
    auto srcRank = src.getRanks().at(mSelfContext.getSelfIdx());
    TLLM_CHECK_WITH_INFO(request.mSamplingConfig.beamWidth == 1, "Currently only supports beam width 1.");
    constexpr SizeType32 beam{0};
    auto endIt = getBlockEndIt(*mCacheManager, request, beam);
    for (auto it = getBlockBeginIt(*mCacheManager, request, beam); it != endIt; ++it)
    {
        mComm->recvBuffer(*it, srcRank);
    }
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager
