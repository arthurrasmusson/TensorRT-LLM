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

#pragma once

#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/runtimeBuffers.h"
#include "tensorrt_llm/runtime/iTensor.h"

namespace tensorrt_llm::batch_manager::utils
{
using SizeType32 = runtime::SizeType32;
using TensorPtr = runtime::ITensor::SharedPtr;

TensorPtr collectRequestIds(RequestVector const& contextRequests, RequestVector const& generationRequests);

void setupMedusaLogits(std::vector<TensorPtr>& medusaLogitsHeads, TensorPtr& medusaLogitsDevice, SizeType32 medusaHeads,
    SizeType32 logitsIndex, SizeType32 numLogits);

//! @brief Copy logits from context phase to beginning of generation logits.
//! @details Usually, this concerns logits of 1 token. In speculative decoding this concerns draftLen + 1 tokens.
void copyLastContextLogits(
    TensorPtr const& contextLogits, LlmRequest& llmReq, runtime::BufferManager const& bufferManager);

//! @param beforeDecoder    Whether the function is called before the decoder. If it is true, correct the output offset.
//! @param numDroppedTokens The number of dropped tokens for each beam (e.g. when the requests finished early).
//!                         Generation logits for dropped tokens are ignored.
void copyGenerationLogits(RuntimeBuffers const& genRuntimeBuffers, runtime::BufferManager const& bufferManager,
    LlmRequest& llmReq, std::size_t batchIdx, bool beforeDecoder, std::vector<SizeType32> const& numDroppedTokens = {});

//! @brief Copy logits from generation phase under streaming mode.
void copyStreamingGenerationLogits(runtime::BufferManager const& bufferManager, LlmRequest& llmReq);

void allocateKvCache(ScheduledRequests const& scheduledRequests, kv_cache_manager::KVCacheManager* kvCacheManagerPtr,
    kv_cache_manager::KVCacheManager* crossKvCacheManagerPtr);

} // namespace tensorrt_llm::batch_manager::utils
