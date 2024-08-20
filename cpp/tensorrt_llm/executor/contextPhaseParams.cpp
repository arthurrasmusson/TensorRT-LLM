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

#include "tensorrt_llm/executor/contextPhaseState.h"
#include "tensorrt_llm/executor/executor.h"

#include <optional>

namespace tensorrt_llm::executor
{

ContextPhaseParams::ContextPhaseParams(VecTokens firstGenTokens, void* state)
    : mFirstGenTokens{std::move(firstGenTokens)}
    , mState{StatePtr{state, deleter}}
{
}

ContextPhaseParams::ContextPhaseParams(VecTokens firstGenTokens)
    : mFirstGenTokens{std::move(firstGenTokens)}
{
}

ContextPhaseParams::ContextPhaseParams(ContextPhaseParams const& other)
{
    // Since the internal header files implement the destructor while using the declaration of this
    // type, a `unique_ptr` with a custom destructor member is used here.
    mFirstGenTokens = other.mFirstGenTokens;
    if (other.mState)
    {
        auto* otherState = static_cast<ContextPhaseState*>(other.mState.get());
        mState = StatePtr{std::make_unique<ContextPhaseState>(*otherState).release(), deleter};
    }
}

ContextPhaseParams::ContextPhaseParams(ContextPhaseParams&&) = default;

ContextPhaseParams& ContextPhaseParams::operator=(ContextPhaseParams const& other)
{
    *this = other;
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

void const* ContextPhaseParams::getState() const noexcept
{
    return mState.get();
}

void* ContextPhaseParams::getState() noexcept
{
    return mState.get();
}

void ContextPhaseParams::deleter(void const* data)
{
    if (data)
    {
        delete static_cast<ContextPhaseState const*>(data);
    }
}

bool ContextPhaseParams::operator==(ContextPhaseParams const& other) const noexcept
{
    return mFirstGenTokens == other.mFirstGenTokens
        && *(static_cast<ContextPhaseState const*>(mState.get()))
        == *(static_cast<ContextPhaseState const*>(mState.get()));
}

} // namespace tensorrt_llm::executor
