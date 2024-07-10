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

#include "lookaheadDecodingBuffers.h"

namespace tensorrt_llm::batch_manager
{

LookaheadDecodingBuffers::LookaheadDecodingBuffers(SizeType32 maxBatchSize, SizeType32 maxBeamWidth,
    runtime::BufferManager const& manager, runtime::ModelConfig const& modelConfig,
    runtime::WorldConfig const& worldConfig, executor::DecodingConfig const& /* decodingConfig */,
    runtime::TllmRuntime const& runtime)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TLLM_CHECK_WITH_INFO(maxBeamWidth == 1, "Lookahead decoding does not support beam search");

    auto const tokensPerStep = modelConfig.getMaxDecodingDraftTokens();
    auto const numPackedMasks = static_cast<SizeType32>(tensorrt_llm::common::divUp(tokensPerStep, 32));

    // Copy buffers to device
    attentionPackedMaskDevice = manager.gpu(
        ITensor::makeShape({maxBatchSize * maxBeamWidth * tokensPerStep, numPackedMasks}), nvinfer1::DataType::kINT32);
    positionOffsetsDevice
        = manager.gpu(ITensor::makeShape({maxBatchSize * maxBeamWidth, tokensPerStep}), nvinfer1::DataType::kINT32);
    lookaheadGenerationLengthsDevice
        = manager.gpu(ITensor::makeShape({maxBatchSize * maxBeamWidth}), nvinfer1::DataType::kINT32);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void LookaheadDecodingBuffers::reshape(
    SizeType32 /* numCtxSequences */, SizeType32 numGenSequences, SizeType32 tokensPerStep)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto attentionPackedMaskShape = attentionPackedMaskDevice->getShape();
    attentionPackedMaskShape.d[0] = numGenSequences * tokensPerStep;
    attentionPackedMaskDevice->reshape(attentionPackedMaskShape);

    auto lookaheadGenerationLengthsShape = lookaheadGenerationLengthsDevice->getShape();
    lookaheadGenerationLengthsShape.d[0] = numGenSequences;
    lookaheadGenerationLengthsDevice->reshape(lookaheadGenerationLengthsShape);

    auto positionOffsetsShape = positionOffsetsDevice->getShape();
    positionOffsetsShape.d[0] = numGenSequences;
    positionOffsetsDevice->reshape(positionOffsetsShape);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void LookaheadDecodingBuffers::insertInputTensors(
    TensorMap& inputBuffers, TensorMap& /* outputBuffers */, runtime::WorldConfig const& /* worldConfig */) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    inputBuffers.insert_or_assign("spec_decoding_packed_mask", attentionPackedMaskDevice);
    inputBuffers.insert_or_assign("spec_decoding_generation_lengths", lookaheadGenerationLengthsDevice);
    inputBuffers.insert_or_assign("spec_decoding_position_offsets", positionOffsetsDevice);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

} // namespace tensorrt_llm::batch_manager
