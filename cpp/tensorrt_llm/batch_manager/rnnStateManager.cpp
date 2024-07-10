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

#include "rnnStateManager.h"

#include "tensorrt_llm/runtime/utils/sessionUtils.h"

using namespace tensorrt_llm::runtime;

namespace tensorrt_llm::batch_manager::rnn_state_manager
{

RnnStateManager::RnnStateManager(tensorrt_llm::runtime::ModelConfig const& modelConfig,
    runtime::WorldConfig const& worldConfig, tensorrt_llm::runtime::BufferManager::CudaStreamPtr stream,
    SizeType32 maxNumSequences)
    : mBufferManager{stream}
    , mMaxNumSequences(maxNumSequences)
{
    TLLM_CHECK_WITH_INFO(modelConfig.usePagedState(), "RnnStateManager should be used with Paged State enabled.");
    TLLM_CHECK_WITH_INFO(modelConfig.useMambaConv1dPlugin(), "RnnStateManager should be used with MambaConv1dPlugin.");
    mMaxBeamWidth = modelConfig.getMaxBeamWidth();
    TLLM_CHECK_WITH_INFO(mMaxBeamWidth == 1, "Beam search is not supported for Mamba now.");
    mBeamSlotsPerSequence = mMaxBeamWidth == 1 ? mMaxBeamWidth : mMaxBeamWidth + 1;
    // If we need support beam search, we may need mMaxBeamWidth + 1 slots and use separate input / output states.
    auto rnnConfig = modelConfig.getRnnConfig();
    TLLM_CHECK_WITH_INFO(rnnConfig.has_value(), "RnnStateManager should be used with rnnConfig");
    mConvKernel = rnnConfig->convKernel;
    mStateSize = rnnConfig->stateSize;
    mRnnHiddenSize = rnnConfig->rnnHiddenSize;
    mRnnHeadSize = rnnConfig->rnnHeadSize;
    mRnnConvDimSize = rnnConfig->rnnConvDimSize;
    auto const localNbLayers = modelConfig.getNbRnnLayers(worldConfig.getPipelineParallelism());
    mLocalNbLayers = localNbLayers;
    mDataType = modelConfig.getDataType();
    auto const rnnStateShape = [&]()
    {
        if (mRnnHeadSize > 0)
        {
            return tensorrt_llm::runtime::ITensor::makeShape({mLocalNbLayers, mMaxNumSequences * mBeamSlotsPerSequence,
                mRnnHiddenSize / mRnnHeadSize, mStateSize, mRnnHeadSize});
        }
        else
        {
            return tensorrt_llm::runtime::ITensor::makeShape(
                {mLocalNbLayers, mMaxNumSequences * mBeamSlotsPerSequence, mStateSize, mRnnHiddenSize});
        }
    }();
    auto const convStateShape = tensorrt_llm::runtime::ITensor::makeShape(
        {mLocalNbLayers, mMaxNumSequences * mBeamSlotsPerSequence, mConvKernel - 1, mRnnConvDimSize});
    pagedRnnStates = mBufferManager.gpu(rnnStateShape, nvinfer1::DataType::kFLOAT);
    pagedConvStates = mBufferManager.gpu(convStateShape, mDataType);

    auto statePtrsShape = tensorrt_llm::runtime::ITensor::makeShape({localNbLayers});
    rnnStatePtrs = tensorrt_llm::runtime::BufferManager::cpu(statePtrsShape, nvinfer1::DataType::kINT64);
    convStatePtrs = tensorrt_llm::runtime::BufferManager::cpu(statePtrsShape, nvinfer1::DataType::kINT64);
    void** rnnStatePtrArray = static_cast<void**>(rnnStatePtrs->data());
    void** convStatePtrArray = static_cast<void**>(convStatePtrs->data());

    rnnStatePtr.resize(mLocalNbLayers);
    convStatePtr.resize(mLocalNbLayers);
    for (int i = 0; i < mLocalNbLayers; i++)
    {
        auto layerRnnStates = tensorrt_llm::runtime::ITensor::slice(pagedRnnStates, i, 1);
        auto layerConvStates = tensorrt_llm::runtime::ITensor::slice(pagedConvStates, i, 1);
        rnnStatePtrArray[i] = layerRnnStates->data();
        convStatePtrArray[i] = layerConvStates->data();
        rnnStatePtr[i] = tensorrt_llm::runtime::ITensor::slice(rnnStatePtrs, i, 1);
        convStatePtr[i] = tensorrt_llm::runtime::ITensor::slice(convStatePtrs, i, 1);
    }
}

void RnnStateManager::getPtrBuffers(
    TensorMap& inputBuffers, runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig) const
{
    auto const localNbLayers = modelConfig.getNbRnnLayers(worldConfig.getPipelineParallelism());
    auto const firstLayerId = worldConfig.getPipelineParallelRank() * localNbLayers;
    auto const& layerTypes = modelConfig.getLayerTypes();

    utils::insertTensorVector(
        inputBuffers, "conv_state_ptr_", convStatePtr, firstLayerId, layerTypes, ModelConfig::LayerType::kRECURRENT);
    utils::insertTensorVector(
        inputBuffers, "rnn_state_ptr_", rnnStatePtr, firstLayerId, layerTypes, ModelConfig::LayerType::kRECURRENT);
}

void RnnStateManager::fillSlotMapping(
    runtime::ITensor& dstPointers, SizeType32 dstSlotOffset, SizeType32 seqSlotIdx, SizeType32 beamWidth) const
{
    auto* dstPtr = bufferCast<SizeType32>(dstPointers);
    if (beamWidth == 1)
    {
        dstPtr[dstSlotOffset] = seqSlotIdx * mBeamSlotsPerSequence;
    }
    else
    {
        // leave first for context.
        std::iota(dstPtr + dstSlotOffset, dstPtr + dstSlotOffset + beamWidth, seqSlotIdx * mBeamSlotsPerSequence + 1);
    }
}

} // namespace tensorrt_llm::batch_manager::rnn_state_manager
