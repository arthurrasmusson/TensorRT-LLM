/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "staticThreadPool.h"

namespace tensorrt_llm::batch_manager::utils
{

StaticThreadPool::StaticThreadPool(std::size_t numThreads)
{
    TLLM_CHECK_WITH_INFO(numThreads > 0, "The number of threads must be greater than 0.");
    try
    {
        for (std::size_t i = 0; i < numThreads; ++i)
        {
            mThreads.emplace_back(std::thread(&StaticThreadPool::workerThread, this));
        }
    }
    catch (...)
    {
        requestStop();
        join();
    }
}

void StaticThreadPool::join()
{
    for (auto& thread : mThreads)
    {
        thread.join();
    }
}

StaticThreadPool::~StaticThreadPool()
{
    requestStop();
    join();
}

void StaticThreadPool::requestStop()
{
    mTerminate = true;
}

void StaticThreadPool::workerThread()
{
    while (!mTerminate)
    {
        std::unique_lock lock(mQueueMutex);
        if (mQueue.size())
        {
            auto task = std::move(mQueue.front());
            mQueue.pop();
            lock.unlock();
            task();
        }
        else
        {
            std::this_thread::yield();
        }
    }
}

} // namespace tensorrt_llm::batch_manager::utils
