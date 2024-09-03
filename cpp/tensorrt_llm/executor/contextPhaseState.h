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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/executor/types.h"
#include <variant>

namespace tensorrt_llm::executor
{

class ContextPhaseState
{
public:
    using RequestIdType = std::uint64_t;

    struct MpiComm
    {
        [[nodiscard]] bool operator==(MpiComm const& other) const noexcept
        {
            return mRanks == other.mRanks;
        }

        std::vector<SizeType32> mRanks;
    };

    struct SocketComm
    {
        [[nodiscard]] bool operator==(SocketComm const& other) const noexcept
        {
            return mPort == other.mPort && mIp == other.mIp;
        }

        std::uint16_t mPort;
        std::string mIp;
    };

    ContextPhaseState() = default;

    ContextPhaseState(RequestIdType ReqId, std::vector<SizeType32> ranks)
        : mReqId{ReqId}
        , mComm{MpiComm{std::move(ranks)}}
        , mCommType{CommType::MPI}
    {
    }

    ContextPhaseState(RequestIdType ReqId, std::uint16_t port, std::string ip)
        : mReqId{ReqId}
        , mComm{SocketComm{port, std::move(ip)}}
        , mCommType{CommType::SOCKET}
    {
    }

    [[nodiscard]] RequestIdType getReqId() const noexcept
    {
        return mReqId;
    }

    [[nodiscard]] bool isMpiComm() const noexcept
    {
        return mCommType == CommType::MPI;
    }

    [[nodiscard]] bool isSocketComm() const noexcept
    {
        return mCommType == CommType::SOCKET;
    }

    [[nodiscard]] MpiComm const& getMpiComm() const
    {
        TLLM_CHECK(isMpiComm());
        return std::get<MpiComm>(mComm);
    }

    [[nodiscard]] SocketComm const& getSocketComm() const
    {
        TLLM_CHECK(isSocketComm());
        return std::get<SocketComm>(mComm);
    }

    [[nodiscard]] bool operator==(ContextPhaseState const& other) const noexcept
    {
        return mReqId == other.mReqId && mComm == other.mComm && mCommType == other.mCommType;
    }

private:
    enum class CommType : int8_t
    {
        UNK = 0,
        MPI,
        SOCKET
    };

    friend class Serialization;
    RequestIdType mReqId{0};
    std::variant<MpiComm, SocketComm> mComm;
    CommType mCommType{CommType::UNK};
};

} // namespace tensorrt_llm::executor
