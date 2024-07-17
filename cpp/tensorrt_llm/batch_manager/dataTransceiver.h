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
#include <future>

#include "tensorrt_llm/batch_manager/llmRequest.h"

namespace tensorrt_llm::batch_manager
{

class DataContext
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;

    DataContext(std::vector<SizeType32> ranks, std::optional<SizeType32> selfIdx = std::nullopt)
        : mRanks{std::move(ranks)}
        , mSelfIdx{std::move(selfIdx)}
    {
    }

    [[nodiscard]] std::vector<SizeType32> const& getRanks() const noexcept
    {
        return mRanks;
    }

    [[nodiscard]] SizeType32 getSelfIdx() const
    {
        TLLM_CHECK_WITH_INFO(mSelfIdx, "The object is not used as self context.");
        return mSelfIdx.value();
    }

    virtual ~DataContext() = default;

private:
    std::vector<SizeType32> mRanks;
    std::optional<SizeType32> mSelfIdx;
};

// Used to support the data transmission with different layouts and different protocols.
class DataSender
{
public:
    /// @brief Synchronously send data.
    /// @param request The request object to which the data belongs.
    /// @param destination The destination for sending the data.
    virtual void send(LlmRequest const& request, DataContext const& destination) = 0;

    /// @brief Determine whether the sender is applicable to the source and target.
    /// @param receiverDataContext Receiver's data arrangement.
    /// @return Whether the sender is applicable to the source and target.
    [[nodiscard]] virtual bool inquireSupport(DataContext const* receiverContext) = 0;

    virtual ~DataSender() = default;
};

// Used to support the actual reception of data with different layouts and different protocols.
class DataReceiver
{
public:
    /// @brief Synchronously receive data.
    /// @param request The request object to which the data belongs.
    /// @param source The source for receiving the data.
    virtual void receive(LlmRequest const& request, DataContext const& source) = 0;

    /// @brief Determine whether the receiver is applicable to the source and target.
    /// @param senderDataContext Sender's data arrangement.
    /// @return Whether the receiver is applicable to the source and target.
    [[nodiscard]] virtual bool inquireSupport(DataContext const* senderContext) = 0;

    virtual ~DataReceiver() = default;
};

class DataResponder
{
public:
    /// @brief Asynchronously respond to the request and send data. For asynchronous reasons, the reference
    /// parameter must remain valid until the content of the future is fetched.
    /// @param llmRequest Request object. Its data should be ready when called, and the data for this request
    /// should remain valid until future synchronization.
    /// @return Once the data is fully sent, the future object will become valid.
    [[nodiscard]] virtual std::future<void> respondAndSendAsync(LlmRequest const& llmRequest) = 0;

    virtual ~DataResponder() = default;
};

class DataRequester
{
public:
    /// @brief Asynchronously send a request to receive data. For asynchronous reasons, the reference parameter
    /// must remain valid until the content of the future is fetched.
    /// @param llmRequest Request object. Its data should be in an allocated but unwritten state when called, and the
    /// data for this request should remain intact only after future synchronization.
    /// @param context The context which retains information about the resopnder, such as the ranks value.
    /// @return Once the data is fully received, the future object will become valid.
    [[nodiscard]] virtual std::future<void> requestAndReceiveAsync(
        LlmRequest const& llmRequest, DataContext const& context)
        = 0;

    virtual ~DataRequester() = default;
};

} // namespace tensorrt_llm::batch_manager
