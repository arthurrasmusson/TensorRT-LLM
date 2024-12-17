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

#include "encoderBuffers.h"
#include "loraBuffers.h"
#include "medusaBuffers.h"
#include "promptTuningBuffers.h"
#include "rnnStateBuffers.h"
#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/decoderBuffers.h"
#include "tensorrt_llm/runtime/eagleBuffers.h"
#include "tensorrt_llm/runtime/explicitDraftTokensBuffers.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/lookaheadBuffers.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/worldConfig.h"
#include "transformerBuffers.h"

#include <optional>
#include <vector>

namespace tensorrt_llm::runtime
{
class TllmRuntime;
} // namespace tensorrt_llm::runtime

namespace tensorrt_llm::batch_manager
{
enum CacheBufferSize
{
    GENERATION_LOGITS_BUFFER_LENGTH = 8
};

namespace kv_cache_manager
{
class BaseKVCacheManager;
} // namespace kv_cache_manager

class LlmRequest;

class RuntimeBuffers
{
public:
    static constexpr auto kLogitsTensorName = "logits";
    static constexpr auto kHiddenStatesOutputTensorName = "hidden_states_output";
    static constexpr auto kHiddenStatesInputTensorName = "hidden_states_input";
    static constexpr auto kInputIdsTensorName = "input_ids";
    static constexpr auto kLastTokenIdsTensorName = "last_token_ids";
    static constexpr auto kHostRequestTypesTensorName = "host_request_types";
    static constexpr auto kContextLengthsTensorName = "context_lengths";
    static constexpr auto kHostContextLengthsTensorName = "host_context_lengths";
    static constexpr auto kSequenceLengthsTensorName = "sequence_length";
    static constexpr auto kPromptEmbeddingTableTensorName = "prompt_embedding_table";
    static constexpr auto kTasksTensorName = "tasks";
    static constexpr auto kPromptVocabSizeTensorName = "prompt_vocab_size";
    static constexpr auto kMRopeRotaryCosSinTensorName = "mrope_rotary_cos_sin";
    static constexpr auto kMRopePositionDeltasTensorName = "mrope_position_deltas";

    using SizeType32 = runtime::SizeType32;
    using TensorPtr = runtime::ITensor::SharedPtr;
    using TensorMap = runtime::ITensor::TensorMap;
    using PeftTable = LoraBuffers::PeftTable;

    [[nodiscard]] SizeType32 constexpr getContextIndex() const noexcept
    {
        return contextIndex;
    };

    void constexpr setContextIndex(SizeType32 index) noexcept
    {
        contextIndex = index;
    };

    [[nodiscard]] SizeType32 constexpr getNumContextTokens() const noexcept
    {
        return numContextTokens;
    };

    [[nodiscard]] BatchState getBatchState() const noexcept
    {
        return {numContextRequests, numGenRequests, getNumTokens(), maxKvCacheLengthRounded};
    };

private:
    [[nodiscard]] SizeType32 constexpr getNumRequests() const noexcept
    {
        return numContextRequests + numGenRequests;
    };

    [[nodiscard]] SizeType32 constexpr getNumSequences() const noexcept
    {
        return numContextRequests + numGenSequences;
    };

    [[nodiscard]] SizeType32 constexpr getNumTokens() const noexcept
    {
        return numContextTokens + numGenTokens;
    };

    // sizes
    SizeType32 numContextRequests{};
    SizeType32 numGenRequests{};
    SizeType32 numGenSequences{};
    SizeType32 numContextTokens{};
    SizeType32 numGenTokens{};
    SizeType32 numLogits{};
    SizeType32 maxKvCacheLengthRounded{};

    // general
    TensorPtr inputsIds;

    TensorPtr contextLengthsHost;
    TensorPtr contextLengthsDevice;
    TensorPtr sequenceLengthsHost;

    /// @brief Index of selected runtime context.
    SizeType32 contextIndex{};
    SizeType32 maxContextLength{};

public:
    TensorPtr sequenceLengthsDevice;

private:
    // runtime
    TensorPtr requestTypes; // Host tensor, 0: context, 1: generation
    TensorPtr lastTokenIdsHost;
    TensorPtr lastTokenIdsDevice;
    TensorPtr logitsIdsHost;
    TensorPtr logitsIdsDevice;

    // pipeline parallelism
    TensorPtr hiddenStates;

    // Prompt tuning
    PromptTuningBuffers promptTuningBuffers;
    // Mrope
    TensorPtr mropeRotaryCosSin;
    TensorPtr mropePositionDeltas;

    // LoRA
    LoraBuffers loraBuffers;

    // Helper buffers
    TensorPtr fillValues;
    TensorPtr fillValuesDevice;

public:
    // additional buffers depending on model type
    std::optional<TransformerBuffers> transformerBuffers;
    std::optional<RnnStateBuffers> rnnStateBuffers;

    // Encoder-Decoder
    std::optional<EncoderBuffers> encoderBuffers;

    // Medusa
    std::optional<MedusaBuffers> medusaBuffers;

    // Lookahead decoding
    std::optional<runtime::LookaheadRuntimeBuffers> lookaheadBuffers;
    // Explicit draft tokens decoding
    std::optional<runtime::ExplicitDraftTokensBuffers> explicitDraftTokensBuffers;
    // Eagle decoding
    std::optional<runtime::EagleBuffers> eagleBuffers;

    // decoder
    TensorPtr decoderInputsIds;
    TensorPtr decoderInputLengthsHost;

    TensorPtr cacheIndirDecoderIOBatchedCopySrcOffsets;
    TensorPtr cacheIndirDecoderIOBatchedCopyDstOffsets;
    TensorPtr cacheIndirDecoderIOBatchedCopySizes;

    // logits
    std::vector<SizeType32> numContextLogits;
    TensorPtr logits;

    // Helper cache for store generation logits
    TensorPtr cacheTransposedGenerationLogits; // Temporarily store the transposed results of multiple fragment logits.
    TensorPtr cacheGenerationFragmentPointerDevice; // Temporarily store logits buffer address during the transposing.
    TensorPtr cacheGenerationFragmentPointerHost;   // Temporarily store logits buffer address during the transposing.

    // Helper for KV cache rewind
    TensorPtr seqSlots;
    TensorPtr seqSlotsDevice;
    TensorPtr sortedSeqSlots;
    // TODO(rkobus): move into decoderBuffers.DraftBuffers
    TensorPtr seqSlotRemappingHost;                                 // [numSequences]
    TensorPtr seqSlotRemappingDevice;                               // [numSequences]

    TensorPtr mCacheIndirDecoderIOBatchedCopySrcOffsetsSliceDevice; // [mMaxNumRequests], device: explicitly
                                                                    // device-copied src offsets to reduce warp stalls
                                                                    // in copy batch kernel invocation.
    TensorPtr mCacheIndirDecoderIOBatchedCopyDstOffsetsSliceDevice; // [mMaxNumRequests], device: explicitly
                                                                    // device-copied dst offsets to reduce warp stalls
                                                                    // in copy batch kernel invocation.
    TensorPtr
        mCacheIndirDecoderIOBatchedCopyCopySizesDevice; // [mMaxNumRequests], device: explicitly device-copied slice
                                                        // sizes to reduce warp stalls in copy batch kernel invocation.
private:
    // Re-capture cuda graph when max kv cache len of the batch has changed on kKV_CACHE_LEN_CUDA_GRAPH_ROUND_SIZE.
    static SizeType32 constexpr kKV_CACHE_LEN_CUDA_GRAPH_ROUND_SIZE{256};

    TensorPtr cacheGenerationLogits;           // Buffer for logits between steps to prevent from being overwritten.
    SizeType32 cacheGenerationLogitsOffset{0}; // Record the usage offset of the cacheGenerationLogits buffer.
    TensorMap mAdditionalOutputTensors;        // Tensors storing additional output tensors.

    // engine I/O
    TensorMap inputMap;
    TensorMap outputMap;

public:
    RuntimeBuffers(SizeType32 maxBatchSize, SizeType32 maxBeamWidth,
        std::vector<SizeType32> const& maxAttentionWindowVec, SizeType32 maxAttentionWindow, SizeType32 sinkTokenLen,
        runtime::TllmRuntime const& runtime, runtime::ModelConfig const& modelConfig,
        runtime::WorldConfig const& worldConfig, executor::DecodingConfig const& decodingConfig,
        std::optional<SizeType32> maxNumTokens = std::nullopt,
        std::optional<std::vector<std::string>> const& additionalOutputNames = std::nullopt);

    std::tuple<SizeType32, TensorMap const&, TensorMap&> prepareStep(RequestVector const& contextRequests,
        RequestVector const& genRequests, SizeType32 maxBeamWidth, SizeType32 maxAttentionWindow,
        DecoderBuffers& decoderBuffers, kv_cache_manager::BaseKVCacheManager* kvCacheManager,
        kv_cache_manager::BaseKVCacheManager* crossKvCacheManager, rnn_state_manager::RnnStateManager* rnnStateManager,
        PeftTable const& peftTable, runtime::TllmRuntime const& runtime, runtime::ModelConfig const& modelConfig,
        runtime::WorldConfig const& worldConfig);

    void prepareBuffersForCudaGraph(SizeType32 maxSequenceLength);

    void prepareExplicitDraftTokenBuffers(DecoderBuffers& decoderBuffers, runtime::TllmRuntime const& runtime,
        runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig);

    void prepareEagleBuffers(RequestVector const& contextRequests, RequestVector const& genRequests,
        DecoderBuffers& decoderBuffers, runtime::TllmRuntime const& runtime, runtime::ModelConfig const& modelConfig,
        runtime::WorldConfig const& worldConfig);

private:
    void create(SizeType32 maxBatchSize, SizeType32 maxBeamWidth, std::vector<SizeType32> maxAttentionWindowVec,
        SizeType32 maxAttentionWindow, SizeType32 sinkTokenLen, runtime::TllmRuntime const& runtime,
        runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig,
        executor::DecodingConfig const& decodingConfig,
        std::optional<std::vector<std::string>> const& additionalOutputNames = std::nullopt);

    void reshape(runtime::TllmRuntime const& runtime, runtime::ModelConfig const& modelConfig,
        runtime::WorldConfig const& worldConfig);

    //! @brief set max sizes for pre-allocation
    void setMaxBufferSizes(SizeType32 maxBatchSize, SizeType32 maxBeamWidth, runtime::ModelConfig const& modelConfig,
        std::optional<SizeType32> maxNumRuntimeTokens);

    //! @brief set sizes depending on scheduled requests
    void setBufferSizes(RequestVector const& contextRequests, RequestVector const& genRequests);

    void setFromInputs(RequestVector const& contextRequests, RequestVector const& genRequests, SizeType32 maxBeamWidth,
        SizeType32 maxAttentionWindow, DecoderBuffers& decoderBuffers,
        kv_cache_manager::BaseKVCacheManager* kvCacheManagerPtr,
        kv_cache_manager::BaseKVCacheManager* crossKvCacheManagerPtr,
        rnn_state_manager::RnnStateManager* rnnStateManagerPtr, PeftTable const& peftTable,
        runtime::TllmRuntime const& runtime, runtime::ModelConfig const& modelConfig,
        runtime::WorldConfig const& worldConfig);

    void fillIOMaps(runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig);
};

} // namespace tensorrt_llm::batch_manager
