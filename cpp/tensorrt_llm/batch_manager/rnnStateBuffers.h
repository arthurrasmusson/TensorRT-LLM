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
#include "tensorrt_llm/runtime/iTensor.h"

namespace tensorrt_llm::runtime
{
class TllmRuntime;
} // namespace tensorrt_llm::runtime

namespace tensorrt_llm::batch_manager
{

namespace rnn_state_manager
{
class RnnStateManager;
}

class RnnStateBuffers
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using TensorPtr = runtime::ITensor::SharedPtr;
    using TensorMap = runtime::StringPtrMap<runtime::ITensor>;

    // others should be in rnnStateManager, we only need slotMapping here.
    TensorPtr slotMappingHost;   // [batch_size]
    TensorPtr slotMappingDevice; // [batch_size]

    RnnStateBuffers(SizeType32 maxBatchSize, runtime::TllmRuntime const& runtime);

    void reshape(SizeType32 numSequences);

    void fillSlotMappings(RequestVector const& contextRequests, rnn_state_manager::RnnStateManager* rnnStateManager);

    void copySlotMappingH2D(runtime::TllmRuntime const& runtime);

    void getBuffers(TensorMap& inputBuffers) const;
};

} // namespace tensorrt_llm::batch_manager
