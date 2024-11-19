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
#include "tensorrt_llm/runtime/bufferManager.h"
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
} // namespace kv_cache_manager

class TransformerBuffers
{
public:
    using SizeType32 = runtime::SizeType32;
    using TensorPtr = runtime::ITensor::SharedPtr;
    using TensorMap = runtime::StringPtrMap<runtime::ITensor>;
    using KvCacheType = batch_manager::kv_cache_manager::CacheType;

    static constexpr std::string_view kInputIdsTensorName = "input_ids";
    static constexpr std::string_view kAttentionMaskTensorName = "attention_mask";
    static constexpr std::string_view kCrossAttentionMaskTensorName = "cross_attention_mask";
    static constexpr std::string_view kCrossAttentionPackedMaskTensorName = "cross_attention_packed_mask";
    static constexpr std::string_view kPositionIdsTensorName = "position_ids";
    static constexpr std::string_view kContextLengthsTensorName = "context_lengths";
    static constexpr std::string_view kHostContextLengthsTensorName = "host_context_lengths";
    static constexpr std::string_view kSequenceLengthsTensorName = "sequence_length";
    static constexpr std::string_view kHiddenStatesInputTensorName = "hidden_states_input";
    static constexpr std::string_view kHiddenStatesOutputTensorName = "hidden_states_output";
    static constexpr std::string_view kLogitsTensorName = "logits";
    static constexpr std::string_view kLastTokenIdsTensorName = "last_token_ids";
    static constexpr std::string_view kCacheIndirectionsTensorName = "cache_indirection";
    static constexpr std::string_view kHostPastKeyValueLengthsTensorName = "host_past_key_value_lengths";
    static constexpr std::string_view kHostRequestTypesTensorName = "host_request_types";
    static constexpr std::string_view kHostSinkTokenLengthTensorName = "host_sink_token_length";
    static constexpr std::string_view kHostMaxAttentionWindowSizesTensorName = "host_max_attention_window_sizes";
    static constexpr std::string_view kHostRuntimePerfKnobsTensorName = "host_runtime_perf_knobs";
    static constexpr std::string_view kHostContextProgressTensorName = "host_context_progress";
    static constexpr std::string_view kKvCacheBlockOffsetsTensorName = "kv_cache_block_offsets";
    static constexpr std::string_view kHostKvCacheBlockOffsetsTensorName = "host_kv_cache_block_offsets";
    static constexpr std::string_view kHostKvCachePoolPointersTensorName = "host_kv_cache_pool_pointers";
    static constexpr std::string_view kHostKvCachePoolMappingTensorName = "host_kv_cache_pool_mapping";
    static constexpr std::string_view kCrossKvCacheBlockOffsetsTensorName = "cross_kv_cache_block_offsets";
    static constexpr std::string_view kHostCrossKvCacheBlockOffsetsTensorName = "host_cross_kv_cache_block_offsets";
    static constexpr std::string_view kHostCrossKvCachePoolPointersTensorName = "host_cross_kv_cache_pool_pointers";
    static constexpr std::string_view kHostCrossKvCachePoolMappingTensorName = "host_cross_kv_cache_pool_mapping";
    static constexpr std::string_view kSkipCrossAttentionBlocksTensorName = "skip_cross_attn_blocks";

    TensorPtr pastKeyValueLengths; // Host tensor
    TensorPtr positionIds;

    // max kv cache lengths.
    TensorPtr maxAttentionWindows;
    // sink token lengths.
    TensorPtr sinkTokenLengths;
    TensorPtr cacheIndirection;
    TensorPtr kvCacheBlockOffsetsHost;   // [numPools, maxBatch * maxBeamWidth, 2, maxBlocksPerSeq]
    TensorPtr kvCacheBlockOffsetsDevice; // [numPools, maxBatch * maxBeamWidth, 2, maxBlocksPerSeq]
    TensorPtr runtimePerfKnobsHost;
    TensorPtr contextProgressHost;

    // Cross attention buffers
    TensorPtr crossKvCacheBlockPoolPointers = nullptr;
    TensorPtr crossKvCacheBlockPoolMapping = nullptr;
    TensorPtr crossKvCacheBlockOffsetsHost = nullptr;
    TensorPtr crossKvCacheBlockOffsetsDevice = nullptr;
    TensorPtr crossAttentionMaskCopySrcOffsets = nullptr; // [maxNumRequest] pinned memory.
    TensorPtr crossAttentionMaskCopyDstOffsets = nullptr; // [maxNumRequest] pinned memory.
    TensorPtr crossAttentionMaskCopySizes = nullptr;      // [maxNumRequest] pinned memory.
    TensorPtr crossAttentionMaskDevice = nullptr;         // [maxNumTokens, maxEncoderOutputLen]
    // This is created to allow mixed memory types of crossAttentionMask (i.e. CPU and GPU).
    TensorPtr crossAttentionMaskPinnedHost = nullptr; // [maxNumTokens, maxEncoderOutputLen]
    // See more details in tensorrt_llm/kernels/contextFusedMultiHeadAttention/fmhaPackedMask.cu.
    // The attention packed mask for FMHA where each bit represents one mask.
    TensorPtr crossAttentionPackedMaskDevice
        = nullptr; // [maxBatchSize, maxInputLengthInBatch, roundUp(maxEncoderOutputLen, 32)]
    // The number of cumulative Q sequence lengths in the mask input, which is used to get mask offsets for different
    // requests.
    TensorPtr crossAttentionCuQSeqLensDevice = nullptr; // [maxBatchSize + 1]
    // The number of cumulative Q sequence lengths in the packed mask, which is used to get mask offsets for different
    // requests.
    TensorPtr crossAttentionPackedMaskCuMaskRowsDevice = nullptr; // [maxBatchSize + 1]

    TensorPtr cacheIndirBatchedCopySrcOffsets;
    TensorPtr cacheIndirBatchedCopyDstOffsets;
    TensorPtr cacheIndirBatchedCopySizes;

    TensorPtr fillValuesAlt;
    TensorPtr fillValuesAltDevice;
    TensorPtr seqSlotsAlt;
    TensorPtr seqSlotsAltDevice;
    TensorPtr skipCrossAttnBlocks;

    TransformerBuffers(SizeType32 maxBatchSize, SizeType32 maxBeamWidth, std::vector<SizeType32> maxAttentionWindowVec,
        SizeType32 maxAttentionWindow, SizeType32 sinkTokenLen,
        executor::ExtendedRuntimePerfKnobConfig const& extendedRuntimePerfKnobConfig,
        runtime::TllmRuntime const& runtime, runtime::ModelConfig const& modelConfig,
        runtime::WorldConfig const& worldConfig);

    void reshape(SizeType32 numSequences, SizeType32 numInputTokens);

    void reshapeKvTensors(SizeType32 maxBatchSize, SizeType32 maxBeamWidth, SizeType32 maxBlocksPerSeq,
        kv_cache_manager::CacheType kvCacheType, SizeType32 numPools, runtime::BufferManager const& manager);

    void getBuffers(TensorMap& inputBuffers) const;

    void reshapePositionIds(std::vector<SizeType32> const& positionIdsHost, bool isChatGlm);

    void copyPositionIds(runtime::TllmRuntime const& runtime, std::vector<SizeType32> const& positionIdsHost,
        bool isChatGlm, TensorPtr const& decoderPositionIds);

    void resetCacheIndirection(RequestVector const& contextRequests, SizeType32 maxBeamWidth,
        SizeType32 maxAttentionWindow, TensorPtr const& decoderCacheIndirectionInput,
        TensorPtr const& decoderCacheIndirectionOutput, runtime::BufferManager const& manager);

    void copyKvBlockOffsets(RequestVector const& contextRequests, RequestVector const& genRequests,
        kv_cache_manager::KVCacheManager const* kvCacheManager,
        kv_cache_manager::KVCacheManager const* crossKvCacheManager, runtime::BufferManager const& manager);

    void copyCacheIndirection(RequestVector const& genRequests, TensorPtr const& decoderCacheIndirectionOutput,
        runtime::CudaStream const& stream);

    void copyCrossAttentionMasks(RequestVector const& contextRequests, RequestVector const& genRequests,
        TensorPtr const& decoderContextLengthsDevice, TensorPtr const& encoderInputLengths,
        SizeType32 maxDecoderContextLength, SizeType32 maxEncoderInputLengthInBatch,
        runtime::TllmRuntime const& runtime);

    void copySkipCrossAttnBlocks(bool const& _skipCrossAttnBlocks, runtime::TllmRuntime const& runtime);

private:
    SizeType32 maxInputLen;
    SizeType32 maxEncoderOutputLen;
    SizeType32 maxNumTokens;
};

} // namespace tensorrt_llm::batch_manager
