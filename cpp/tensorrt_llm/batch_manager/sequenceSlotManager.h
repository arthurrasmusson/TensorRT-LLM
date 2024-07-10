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

#include <chrono>
#include <cstdint>
#include <optional>
#include <queue>
#include <unordered_map>
#include <vector>

namespace tensorrt_llm::batch_manager
{

/// SequenceSlotManager
///
/// Helper class to manage sequence slots
/// This class is not thread-safe

class SequenceSlotManager
{
public:
    using SlotIdType = int32_t;
    using SequenceIdType = std::uint64_t;

    SequenceSlotManager(SlotIdType maxNumSlots, uint64_t maxSequenceIdleMicroseconds);

    /// Function that returns a slot for the provided sequenceId
    /// For a new sequence id, a new slot will be allocated
    /// In case no slot could be allocated or matched, optional will be empty
    std::optional<SlotIdType> getSequenceSlot(bool const& startFlag, SequenceIdType const& sequenceId);

    /// Function that frees the slot associated with the given sequence id
    void freeSequenceSlot(SequenceIdType sequenceId);

    /// Function that frees slots that have been idle for more than
    /// mMaxSequenceIdleMicroseconds
    void freeIdleSequenceSlots();

private:
    SlotIdType mMaxNumSlots;
    std::chrono::microseconds mMaxSequenceIdleMicroseconds;

    std::unordered_map<SequenceIdType, SlotIdType> mSequenceIdToSlot;
    std::queue<SlotIdType> mAvailableSlots;
    std::vector<std::chrono::steady_clock::time_point> mLastTimepoint;
};

} // namespace tensorrt_llm::batch_manager
