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
#include "tensorrt_llm/batch_manager/kvCacheConfig.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/worldConfig.h"

namespace tensorrt_llm::runtime
{
class TllmRuntime;
} // namespace tensorrt_llm::runtime

namespace tensorrt_llm::batch_manager
{

namespace kv_cache_manager
{
class KVCacheManager;
}

class TransformerBuffers
{
public:
    using SizeType32 = runtime::SizeType32;
    using TensorPtr = runtime::ITensor::SharedPtr;
    using TensorMap = runtime::StringPtrMap<runtime::ITensor>;
    using KvCacheType = batch_manager::kv_cache_manager::CacheType;

    TensorPtr pastKeyValueLengths; // Host tensor
    TensorPtr positionIds;

    // max kv cache lengths.
    TensorPtr maxAttentionWindows;
    // sink token lengths.
    TensorPtr sinkTokenLengths;
    TensorPtr cacheIndirection;
    TensorPtr kvCacheBlockPoolPointers;
    TensorPtr kvCacheBlockOffsetsHost;   // [numSequences, 2, maxBlocksPerSeq * 2]
    TensorPtr kvCacheBlockOffsetsDevice; // [numSequences, 2, maxBlocksPerSeq * 2]
    TensorPtr runtimePerfKnobsHost;

    TensorPtr crossKvCacheBlockPoolPointers = nullptr;
    TensorPtr crossKvCacheBlockOffsetsHost = nullptr;
    TensorPtr crossKvCacheBlockOffsetsDevice = nullptr;

    TensorPtr cacheIndirBatchedCopySrcOffsets;
    TensorPtr cacheIndirBatchedCopyDstOffsets;
    TensorPtr cacheIndirBatchedCopySizes;

    TensorPtr fillValuesAlt;
    TensorPtr fillValuesAltDevice;
    TensorPtr seqSlotsAlt;
    TensorPtr seqSlotsAltDevice;

    TransformerBuffers(SizeType32 maxBatchSize, SizeType32 maxBeamWidth, SizeType32 maxAttentionWindow,
        SizeType32 sinkTokenLen, executor::ExtendedRuntimePerfKnobConfig const& extendedRuntimePerfKnobConfig,
        runtime::TllmRuntime const& runtime, runtime::ModelConfig const& modelConfig,
        runtime::WorldConfig const& worldConfig);

    void reshape(SizeType32 numSequences);

    void reshapeKvTensors(SizeType32 maxBatchSize, SizeType32 maxBeamWidth, SizeType32 maxBlocksPerSeq,
        runtime::TllmRuntime const& runtime, KvCacheType kvCacheType = KvCacheType::kSELF);

    void setKvPoolPointers(kv_cache_manager::KVCacheManager& kvCacheManager);

    void getBuffers(TensorMap& inputBuffers) const;

    void copyPositionIds(
        runtime::TllmRuntime const& runtime, std::vector<SizeType32> const& positionIdsHost, bool isChatGlm);

    void resetCacheIndirection(RequestVector const& contextRequests, SizeType32 maxBeamWidth,
        SizeType32 maxAttentionWindow, TensorPtr const& decoderCacheIndirectionInput,
        TensorPtr const& decoderCacheIndirectionOutput, runtime::TllmRuntime const& runtime);

    void copyKvBlockOffsets(RequestVector const& contextRequests, RequestVector const& genRequests,
        kv_cache_manager::KVCacheManager const* kvCacheManager,
        kv_cache_manager::KVCacheManager const* crossKvCacheManager, runtime::TllmRuntime const& runtime);

    void copyCacheIndirection(RequestVector const& genRequests, TensorPtr const& decoderCacheIndirectionOutput,
        runtime::TllmRuntime const& runtime);
};

} // namespace tensorrt_llm::batch_manager
