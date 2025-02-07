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
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/loraManager.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"
#include "tensorrt_llm/runtime/worldConfig.h"

namespace tensorrt_llm::batch_manager
{

class LoraBuffers
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using PeftTable = runtime::LoraManager::PeftTable;
    using TensorPtr = runtime::ITensor::SharedPtr;
    using TensorMap = runtime::StringPtrMap<runtime::ITensor>;

    TensorPtr mLoraWeightsPointersHost;
    TensorPtr mLoraAdapterSizesHost;

    runtime::LoraManager mLoraManager;

    LoraBuffers(SizeType32 maxBatchSize, SizeType32 maxBeamWidth, runtime::TllmRuntime const& tllmRuntime,
        runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig);

    static void validate(std::optional<std::uint64_t> const& optTaskId,
        std::optional<TensorPtr> const& optReqLoraWeights, std::optional<TensorPtr> const& optReqLoraConfig,
        runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig);

    void fill(RequestVector const& contextRequests, RequestVector const& genRequests, PeftTable const& peftTable,
        runtime::BufferManager const& manager, runtime::ModelConfig const& modelConfig,
        runtime::WorldConfig const& worldConfig);

    void insertInputTensors(TensorMap& inputTensors, TensorPtr weightsPtrs, TensorPtr adapterSizes,
        runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig) const;

    void reshape(SizeType32 numSequences);
};
} // namespace tensorrt_llm::batch_manager
