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
#include "tensorrt_llm/executor/contextPhaseState.h"

namespace tensorrt_llm::batch_manager
{

// Used to support the data transmission with different layouts and different protocols.
template <typename TComm, typename TConfig>
class IOFormatter
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;

    /// @brief Perform data transmission with formatting actions.
    /// @param llmRequest The request associated with this data transmission.
    /// @param comm The communicator associated with this data transmission.
    virtual void operator()(LlmRequest const& llmRequest, typename TComm::TPtrContainer const& comm) = 0;

    /// @brief Determine whether the sender is applicable to the source and target.
    /// @param selfconfig Source data arrangement.
    /// @param destConfig Target data arrangement.
    /// @return Whether the sender is applicable to the source and target.
    [[nodiscard]] virtual bool inquireSupport(TConfig const& selfconfig, TConfig const& destConfig) const = 0;

    /// @brief Obtain the indies of the counterparts that need to be actually communicated with.
    /// @param selfconfig Source data arrangement.
    /// @param selfIdx The sequential index of the current executor process within the entire parallel group.
    /// @param destConfig Target data arrangement.
    /// @return The indies of the counterparts.
    [[nodiscard]] virtual std::vector<SizeType32> getCounterparts(
        TConfig const& selfconfig, SizeType32 selfIdx, TConfig const& destConfig) const
        = 0;

    /// @brief Destructor.
    virtual ~IOFormatter() = default;
};

// Operators required for data transmission in specific communication protocols.
class DataSender
{
public:
    /// @brief Receive the request id.
    /// @return The request id.
    [[nodiscard]] virtual LlmRequest::RequestIdType recvRequestId() = 0;

    /// @brief Synchronously send data.
    /// @param llmRequest The request object to which the data belongs.
    virtual void sendSync(LlmRequest const& llmRequest) = 0;

    /// @brief Return the internal communicator status.
    /// @return The communicator status.
    [[nodiscard]] virtual executor::kv_cache::CommState const& getCommState() const = 0;

    /// @brief Destructor.
    virtual ~DataSender() = default;
};

// Operators required for data transmission in specific communication protocols.
class DataReceiver
{
public:
    /// @brief Send the request id.
    /// @param llmRequest The request object to which the id belongs.
    virtual void sendRequestId(LlmRequest const& llmRequest) = 0;

    /// @brief Synchronously receive data.
    /// @param llmRequest The request object to which the data belongs.
    virtual void receiveSync(LlmRequest const& llmRequest) = 0;

    /// @brief Destructor.
    virtual ~DataReceiver() = default;
};

class DataResponder
{
public:
    /// @brief Constructor.
    /// @param sender The sender used at the underlying level.
    explicit DataResponder(std::unique_ptr<DataSender> sender);

    /// @brief Asynchronously respond to the request and send data.
    /// @param llmRequest Request object. Its data should be ready when called, and the data for this request
    /// should remain valid until future synchronization.
    /// @return Once the data is fully sent, the future object will become valid.
    [[nodiscard]] std::future<void> respondAndSendAsync(LlmRequest const& llmRequest) const;

    /// @brief Return the internal communicator status.
    /// @return The communicator status.
    [[nodiscard]] executor::kv_cache::CommState const& getCommState() const;

    /// @brief Destructor.
    ~DataResponder();

private:
    class Impl;
    std::unique_ptr<Impl> mImpl;
};

class DataRequester
{
public:
    /// @brief Constructor.
    /// @param receiver The receiver used at the underlying level.
    explicit DataRequester(std::unique_ptr<DataReceiver> receiver);

    /// @brief Asynchronously send a request to receive data.
    /// @param llmRequest Request object. Its data should be in an allocated but unwritten state when called, and the
    /// data for this request should remain intact only after future synchronization.
    /// @return Once the data is fully received, the future object will become valid.
    [[nodiscard]] std::future<void> requestAndReceiveAsync(LlmRequest const& llmRequest) const;

    /// @brief Destructor.
    ~DataRequester();

private:
    class Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace tensorrt_llm::batch_manager
