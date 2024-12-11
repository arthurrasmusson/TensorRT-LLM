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

#include "tensorrt_llm/batch_manager/allocateKvCache.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/nvtxUtils.h"

void tensorrt_llm::batch_manager::AllocateKvCache::operator()(BaseKVCacheManager& kvCacheManager,
    RequestVector& contextRequests, RequestVector const& generationRequests, runtime::ModelConfig const& modelConfig,
    OptionalRef<BaseKVCacheManager> crossKvCacheManager) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(allocateKvCache);

    for (auto const& llmReq : contextRequests)
    {
        if (llmReq->isFirstContextChunk())
        {
            auto const requestId = llmReq->mRequestId;
            auto const promptLen = llmReq->mPromptLen;
            auto const reqBeamWidth = llmReq->mSamplingConfig.beamWidth;
            auto draftLength = llmReq->getNumDraftTokens();

            // Allocate/Reuse KV cache
            kvCacheManager.addSequence(requestId, promptLen, reqBeamWidth, llmReq);

            // EagleNet will increment kv cache up to maxPathLen to account for accepted tokens.
            // Then up to maxDecodingDraftTokens will be used to generate next draft tokens.
            if (modelConfig.getSpeculativeDecodingMode().isEagle())
            {
                draftLength = modelConfig.getSpeculativeDecodingModule().getMaxPathLen()
                    + modelConfig.getSpeculativeDecodingModule().getMaxDecodingTokens();
            }

            // Allocate more KV cache for speculative decoding
            if (draftLength > 0)
            {
                for (SizeType32 di = 0; di < draftLength; ++di)
                {
                    kvCacheManager.addToken(requestId);
                }
            }

            if (crossKvCacheManager)
            {
                crossKvCacheManager->addSequence(requestId, llmReq->getEncoderOutputLen(), reqBeamWidth, llmReq);
            }
        }
    }

    for (auto const& llmReq : generationRequests)
    {
        auto const requestId = llmReq->mRequestId;
        auto decodingTokens = llmReq->getNumDraftTokens() + 1;

        // EagleNet will increment kv cache up to maxPathLen to account for accepted tokens.
        // Then up to maxDecodingDraftTokens will be used to generate next draft tokens.
        if (modelConfig.getSpeculativeDecodingMode().isEagle())
        {
            decodingTokens = modelConfig.getSpeculativeDecodingModule().getMaxPathLen()
                + modelConfig.getSpeculativeDecodingModule().getMaxDecodingTokens();
        }

        for (SizeType32 di = 0; di < decodingTokens; ++di)
        {
            kvCacheManager.addToken(requestId);
        }
    }

    kvCacheManager.refreshBlocks();

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}
