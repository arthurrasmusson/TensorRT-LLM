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
#include "tensorrt_llm/executor/types.h"

namespace tensorrt_llm::executor
{
PromptTuningConfig::PromptTuningConfig(Tensor embeddingTable, std::optional<VecTokenExtraIds> inputTokenExtraIds)
    : mEmbeddingTable(std::move(embeddingTable))
    , mInputTokenExtraIds(std::move(inputTokenExtraIds))
{
    TLLM_CHECK_WITH_INFO(mEmbeddingTable.getShape().size() == 2,
        "Expected prompt embedding table to have shape [vocabSize, hiddenSize]");
}

Tensor PromptTuningConfig::getEmbeddingTable() const
{
    return mEmbeddingTable;
}

std::optional<VecTokenExtraIds> PromptTuningConfig::getInputTokenExtraIds() const
{
    return mInputTokenExtraIds;
}

} // namespace tensorrt_llm::executor
