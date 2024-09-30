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

#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/worldConfig.h"

namespace tensorrt_llm::batch_manager::rnn_state_manager
{

class RnnStateManager
{
public:
    using TensorPtr = runtime::ITensor::SharedPtr;
    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using TensorMap = runtime::StringPtrMap<runtime::ITensor>;

    RnnStateManager(SizeType32 maxNumSequences, tensorrt_llm::runtime::ModelConfig const& modelConfig,
        runtime::WorldConfig const& worldConfig, tensorrt_llm::runtime::BufferManager const& bufferManager);

    void getPtrBuffers(TensorMap& inputBuffers, runtime::ModelConfig const& modelConfig,
        runtime::WorldConfig const& worldConfig) const;

    void fillSlotMapping(
        runtime::ITensor& dstPointers, SizeType32 dstSlotOffset, SizeType32 seqSlotIdx, SizeType32 beamWidth) const;

private:
    // If we need support beam search, we may need mMaxBeamWidth + 1 slots and use separate input / output states.
    TensorPtr pagedRnnStates;  // [local_nb_layers, max_seq_num * max_beam_width, state_size, rnn_hidden_size] or
                               // [local_nb_layers, max_seq_num * max_beam_width, num_heads, state_size, rnn_head_size]
    TensorPtr pagedConvStates; // [local_nb_layers, max_seq_num * max_beam_width, conv_kernel - 1, rnn_hidden_size]

    TensorPtr rnnStatePtrs;    // [layer_count]
    TensorPtr convStatePtrs;   // [layer_count]

    std::vector<TensorPtr> rnnStatePtr;  // [1]
    std::vector<TensorPtr> convStatePtr; // [1]

    SizeType32 mMaxNumSequences = 0;
    SizeType32 mMaxBeamWidth = 0;
    SizeType32 mBeamSlotsPerSequence = 0;
};

} // namespace tensorrt_llm::batch_manager::rnn_state_manager
