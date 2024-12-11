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

#include "tensorrt_llm/runtime/iBuffer.h"
#include <future>
#include <variant>

namespace tensorrt_llm::batch_manager
{
class RequestInfo;
class UcxEndpoint;
} // namespace tensorrt_llm::batch_manager

namespace tensorrt_llm::executor::kv_cache
{

class ProcessInfo
{
public:
    explicit ProcessInfo(int32_t worldRank)
        : mInfo{worldRank}
    {
    }

    explicit ProcessInfo(batch_manager::UcxEndpoint const* ucxEndpoint)
        : mInfo{ucxEndpoint}
    {
    }

    int32_t getRank() const
    {
        return std::get<int32_t>(mInfo);
    }

    // TODO: remove the dependency of this interface on UCX.
    batch_manager::UcxEndpoint const* getEndpoint() const
    {
        auto* endpoint = std::get<batch_manager::UcxEndpoint const*>(mInfo);
        TLLM_CHECK(endpoint);
        return endpoint;
    }

private:
    std::variant<std::monostate, int32_t, batch_manager::UcxEndpoint const*> mInfo;
};

class DataContext
{
public:
    DataContext() = default;

    explicit DataContext(std::uint64_t requestId)
        : mRequestId{requestId}
    {
    }

    [[nodiscard]] std::uint64_t getRequestId() const
    {
        TLLM_CHECK(mRequestId);
        return mRequestId.value();
    }

private:
    std::optional<std::uint64_t> mRequestId;
};

// TODO:
// 1. Introduce a registration mechanism to avoid host overhead.
// 2. Adapt UCX to the `ProcessInfo` data structure.
// 3. Replace `Buffer` with basic types pointer and size.

class Communicator
{
public:
    [[nodiscard]] virtual bool isThreadSafe() const noexcept = 0;

    virtual void sendBuffer(runtime::IBuffer const& buf, DataContext const& context,
        executor::kv_cache::ProcessInfo const& processInfo) const
        = 0;

    virtual void recvBuffer(
        runtime::IBuffer& buf, DataContext const& context, executor::kv_cache::ProcessInfo const& processInfo) const
        = 0;

    [[nodiscard]] virtual std::unique_ptr<batch_manager::RequestInfo> recvRequestInfo(
        std::optional<executor::kv_cache::ProcessInfo> const& processInfo = std::nullopt) const
        = 0;

    virtual void sendRequestInfo(
        batch_manager::RequestInfo const& requestInfo, executor::kv_cache::ProcessInfo const& processInfo) const
        = 0;

    virtual ~Communicator() = default;
};

} // namespace tensorrt_llm::executor::kv_cache
