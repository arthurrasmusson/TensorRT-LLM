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

#include "tensorrt_llm/batch_manager/contextProgress.h"

namespace tensorrt_llm::batch_manager
{

ContextProgress::ContextProgress(int numLayers)
{
    mCudaEventsRecorded = std::make_unique<std::atomic_bool[]>(numLayers);
    mCudaEvents.reserve(numLayers);
    for (int i = 0; i < numLayers; i++)
    {
        mCudaEventsRecorded[i] = false;
        mCudaEvents.emplace_back(cudaEventBlockingSync | cudaEventDisableTiming);
    }
    TLLM_LOG_DEBUG("ContextProgress created - expect %d layers", numLayers);
}

void ContextProgress::recordEvent(int layerIdx, cudaStream_t stream)
{
    TLLM_CHECK(layerIdx < getNumLayers());
    TLLM_CHECK_WITH_INFO(layerIdx == 0 || mCudaEventsRecorded[layerIdx - 1], "Layer %d is skipped", layerIdx - 1);
    TLLM_CHECK_WITH_INFO(!mCudaEventsRecorded[layerIdx], "Layer %d is recorded twice", layerIdx);
    TLLM_CUDA_CHECK(cudaEventRecord(mCudaEvents[layerIdx].get(), stream));
    mCudaEventsRecorded[layerIdx] = true;
    mConditionVariable.notify_all();
}

void ContextProgress::wait(int layerIdx)
{
    TLLM_CHECK(layerIdx < getNumLayers());
    while (!mCudaEventsRecorded[layerIdx])
    {
        std::unique_lock lock(mMutex);
        auto const timeout = std::chrono::milliseconds(10);
        mConditionVariable.wait_for(lock, timeout);
    }
    mCudaEvents[layerIdx].synchronize();
}

} // namespace tensorrt_llm::batch_manager
