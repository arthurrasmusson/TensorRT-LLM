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

#include "tensorrt_llm/executor/types.h"

namespace tensorrt_llm::executor
{
class ContextPhaseState
{
public:
    using RequestIdType = std::uint64_t;

    ContextPhaseState() = default;

    ContextPhaseState(RequestIdType ReqId, std::vector<SizeType32> ranks)
        : mReqId{ReqId}
        , mRanks{std::move(ranks)}
    {
    }

    [[nodiscard]] std::vector<SizeType32> const& getRanks() const noexcept;

    [[nodiscard]] RequestIdType getReqId() const noexcept
    {
        return mReqId;
    }

    [[nodiscard]] bool operator==(ContextPhaseState const& other) const noexcept
    {
        return mReqId == other.mReqId && mRanks == other.mRanks;
    }

private:
    friend class Serialization;
    RequestIdType mReqId;
    std::vector<SizeType32> mRanks;
};

} // namespace tensorrt_llm::executor
