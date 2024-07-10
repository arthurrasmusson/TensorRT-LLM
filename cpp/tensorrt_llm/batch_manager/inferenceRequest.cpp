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

#include "tensorrt_llm/batch_manager/inferenceRequest.h"

using namespace tensorrt_llm::batch_manager;

std::vector<int64_t> InferenceRequest::serialize() const
{
    TLLM_CHECK_WITH_INFO(!mLogitsPostProcessor.has_value(),
        "Serializing InferenceRequest with logitsPostProcessor set is not supported."
        "Please set the callback after de-serialization");

    // request ID
    // num tensors
    // streaming
    size_t totalSize = 3;
    for (auto const& [name, tensor] : mInputTensors)
    {
        totalSize += NamedTensor(tensor, name).serializedSize();
        totalSize++;
    }

    std::vector<int64_t> vpacked(totalSize);
    int64_t* ptr = vpacked.data();
    *ptr++ = mRequestId;
    *ptr++ = static_cast<int64_t>(mInputTensors.size());
    for (auto const& [name, tensor] : mInputTensors)
    {
        auto namedTensor = NamedTensor(tensor, name);
        auto size = namedTensor.serializedSize();
        *ptr++ = size;
        namedTensor.serialize(ptr, size);
        ptr += size;
    }
    *ptr++ = mIsStreaming;

    return vpacked;
}

std::shared_ptr<InferenceRequest> InferenceRequest::deserialize(int64_t const* packed_ptr)
{
    auto const requestId = static_cast<uint64_t>(*packed_ptr++);
    auto ir = std::make_shared<InferenceRequest>(requestId);

    int64_t num_tensors = *packed_ptr++;
    for (int64_t i = 0; i < num_tensors; ++i)
    {
        int64_t n{*packed_ptr++};
        auto inputTensor = NamedTensor::deserialize(packed_ptr);
        packed_ptr += n;
        ir->emplaceInputTensor(inputTensor.name, inputTensor.tensor);
    }
    bool isStreaming = *packed_ptr != 0;
    ir->setIsStreaming(isStreaming);
    return ir;
}

std::shared_ptr<InferenceRequest> InferenceRequest::deserialize(std::vector<int64_t> const& packed)
{
    return InferenceRequest::deserialize(packed.data());
}
