
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: NVIDIA TensorRT
 * Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

// we have blockNums Block, which is 3D  [PPs,TPs,(BlockIDs in one rank) tokens/tokens_per_block]

// input [PPs,TPs, BlockS] Block
// output [Blocks]Block. but each block has same tokens_per_block. so we can ignore tokens_per_block

#pragma once

#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iTensor.h"

#include <NvInferRuntimeBase.h>

namespace tensorrt_llm::executor::kv_cache
{
struct TargetRanksInfo
{
    int mDomainPPSize;
    int mDomainTPSize;
    std::vector<int> mIRanks;
};

TargetRanksInfo targetIRanks(
    kv_cache::CacheState const& iCacheState, kv_cache::CacheState const& oCacheState, int oRank);

void concatenateKVCacheDispatch(runtime::ITensor::SharedPtr* inputBlocks, int inputBlockNum,
    std::vector<int> const& inputRanks, kv_cache::CacheState const& iCacheState,
    runtime::ITensor::SharedPtr* outputBlocks, int outputBlockNum, int oRank, kv_cache::CacheState const& oCacheState,
    runtime::BufferManager const& bufferManager);
nvinfer1::Dims makeShapeFromCacheState(kv_cache::CacheState const& cacheState);

void splitKVCacheDispatch(std::vector<runtime::ITensor::SharedPtr> const& kVCacheBlocks,
    std::vector<runtime::ITensor::SharedPtr>& ouputSplitBlocks, kv_cache::CacheState const& iCacheState,
    kv_cache::CacheState const& oCacheState, int selfIdx, runtime::BufferManager const& bufferManager);

void concatenateKvCacheV2Dispatch(std::vector<runtime::ITensor::SharedPtr> const& inputSplitBlocks,
    std::vector<runtime::ITensor::SharedPtr>& outputKvCacheBlocks, kv_cache::CacheState const& iCacheState,
    kv_cache::CacheState const& oCacheState, int selfIdx, runtime::BufferManager const& bufferManager);
} // namespace tensorrt_llm::executor::kv_cache
