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

#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/executor/executor.h"

#include <optional>

namespace tensorrt_llm::executor
{

ContextPhaseParams::ContextPhaseParams(VecTokens firstGenTokens, RequestIdType reqId, void* state)
    : mReqId{reqId}
    , mFirstGenTokens{std::move(firstGenTokens)}
    , mState{StatePtr{state, deleter}}
{
}

ContextPhaseParams::ContextPhaseParams(VecTokens firstGenTokens, RequestIdType reqId)
    : mReqId{reqId}
    , mFirstGenTokens{std::move(firstGenTokens)}
{
}

ContextPhaseParams::ContextPhaseParams(ContextPhaseParams const& other)
{
    // Since the internal header files implement the destructor while using the declaration of this
    // type, a `unique_ptr` with a custom destructor member is used here.
    mReqId = other.mReqId;
    mFirstGenTokens = other.mFirstGenTokens;
    if (other.mState)
    {
        auto* otherState = static_cast<DataTransceiverState*>(other.mState.get());
        mState = StatePtr{std::make_unique<DataTransceiverState>(*otherState).release(), deleter};
    }
}

ContextPhaseParams::ContextPhaseParams(ContextPhaseParams&&) = default;

ContextPhaseParams& ContextPhaseParams::operator=(ContextPhaseParams const& other)
{
    *this = ContextPhaseParams{other};
    return *this;
}

ContextPhaseParams& ContextPhaseParams::operator=(ContextPhaseParams&&) = default;

VecTokens const& ContextPhaseParams::getFirstGenTokens() const& noexcept
{
    return mFirstGenTokens;
}

VecTokens ContextPhaseParams::popFirstGenTokens() && noexcept
{
    return std::move(mFirstGenTokens);
}

ContextPhaseParams::RequestIdType ContextPhaseParams::getReqId() const noexcept
{
    return mReqId;
}

void const* ContextPhaseParams::getState() const noexcept
{
    return mState.get();
}

void* ContextPhaseParams::getState() noexcept
{
    return mState.get();
}

void* ContextPhaseParams::releaseState() noexcept
{
    return mState.release();
}

void ContextPhaseParams::deleter(void const* data)
{
    using StateT = DataTransceiverState const;
    std::default_delete<StateT>()(static_cast<StateT*>(data));
}

bool ContextPhaseParams::operator==(ContextPhaseParams const& other) const noexcept
{
    if (mFirstGenTokens != other.mFirstGenTokens || mReqId != other.mReqId
        || static_cast<bool>(mState) != static_cast<bool>(other.mState))
    {
        return false;
    }
    return !mState
        || *static_cast<DataTransceiverState const*>(mState.get())
        == *static_cast<DataTransceiverState const*>(other.mState.get());
}

} // namespace tensorrt_llm::executor
