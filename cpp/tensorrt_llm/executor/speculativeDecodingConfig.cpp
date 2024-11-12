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

#include "tensorrt_llm/executor/executor.h"

namespace tensorrt_llm::executor
{

SpeculativeDecodingConfig::SpeculativeDecodingConfig(bool inFastLogits)
    : fastLogits(inFastLogits)
{
}

bool SpeculativeDecodingConfig::operator==(SpeculativeDecodingConfig const& other) const
{
    return fastLogits == other.fastLogits;
}

Tensor SpeculativeDecodingFastLogitsInfo::toTensor() const
{
    size_t const numLogitsNeeded = (sizeof(*this) + 1) / sizeof(float);
    auto tensor = Tensor::cpu(DataType::kFP32, {1, numLogitsNeeded});
    std::memcpy(tensor.getData(), this, sizeof(*this));
    return tensor;
}

} // namespace tensorrt_llm::executor
