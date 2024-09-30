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

#pragma once

#include "tensorrt_llm/common/assert.h"
#include <atomic>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include <iostream>

namespace tensorrt_llm::batch_manager::utils
{

// A simple static thread pool to avoid the overhead caused by having too many threads.
class StaticThreadPool
{
public:
    explicit StaticThreadPool(std::size_t numThreads);

    StaticThreadPool(StaticThreadPool const&) = delete;
    StaticThreadPool& operator=(StaticThreadPool const&) = delete;

    ~StaticThreadPool();

    // TODO: Performance optimization.
    template <typename TFunction, typename... TArgs>
    [[nodiscard]] std::future<std::invoke_result_t<TFunction, TArgs...>> execute(TFunction&& f, TArgs&&... args)
    {
        TLLM_CHECK(!mTerminate);
        auto task = std::make_shared<std::packaged_task<std::invoke_result_t<TFunction, TArgs...>()>>(
            std::bind(std::forward<TFunction>(f), std::forward<TArgs>(args)...));
        auto res = task->get_future();
        {
            std::unique_lock lock(mQueueMutex);
            mQueue.push([taskCapture = std::move(task)] { (*taskCapture)(); });
        }
        return res;
    }

    void requestStop();

private:
    void workerThread();

    void join();

    std::atomic<bool> mTerminate{false};
    std::queue<std::function<void()>> mQueue;
    std::mutex mQueueMutex;
    std::vector<std::thread> mThreads;
};

} // namespace tensorrt_llm::batch_manager::utils
