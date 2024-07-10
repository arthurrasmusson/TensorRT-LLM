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

#include "rnnStateBuffers.h"
#include "rnnStateManager.h"

#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"
#include "tensorrt_llm/runtime/utils/sessionUtils.h"

#include "runtimeBuffers.h"

using namespace tensorrt_llm::runtime;

namespace tensorrt_llm::batch_manager
{

RnnStateBuffers::RnnStateBuffers(SizeType32 maxBatchSize, runtime::TllmRuntime const& runtime)
{
    auto const& manager = runtime.getBufferManager();

    slotMappingHost = BufferManager::cpu(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kINT32);
    slotMappingDevice = manager.gpu(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kINT32);
}

void RnnStateBuffers::reshape(SizeType32 numSequences)
{
    slotMappingHost->reshape(ITensor::makeShape({numSequences}));
    slotMappingDevice->reshape(ITensor::makeShape({numSequences}));
}

void RnnStateBuffers::getBuffers(rnn_state_manager::RnnStateManager* rnnStateManager, TensorMap& inputBuffers,
    ModelConfig const& modelConfig, WorldConfig const& worldConfig) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(rnnStateBuffersGetBuffers);

    inputBuffers.insert_or_assign("slot_mapping", slotMappingDevice);
    rnnStateManager->getPtrBuffers(inputBuffers, modelConfig, worldConfig);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void RnnStateBuffers::copySlotMappingH2D(runtime::TllmRuntime const& runtime)
{
    auto const& manager = runtime.getBufferManager();
    manager.copy(*slotMappingHost, *slotMappingDevice);
}

} // namespace tensorrt_llm::batch_manager
