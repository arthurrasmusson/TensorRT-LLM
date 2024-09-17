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
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/worldConfig.h"
#include <variant>

namespace tensorrt_llm::executor
{

class Serialization;

namespace kv_cache
{

// Describe the data structure for cache layout, which can be used to infer cache layouts and location
// associations between different processes, in order to determine suitable senders and receivers.
class CacheState final
{
public:
    CacheState(runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig)
        : mModelConfig{modelConfig.getNbAttentionLayers(1), modelConfig.getNbKvHeads(), modelConfig.getSizePerHead(),
            modelConfig.getTokensPerBlock()}
        , mParallelConfig{worldConfig.getTensorParallelism(), worldConfig.getPipelineParallelism()}
        , mDataType{modelConfig.getKvDataType()}
    {
    }

    CacheState(SizeType32 nbAttentionLayers, SizeType32 nbKvHeads, SizeType32 sizePerHead, SizeType32 tokensPerBlock,
        SizeType32 tensorParallelism, SizeType32 pipelineParallelism, nvinfer1::DataType dataType)
        : mModelConfig{nbAttentionLayers, nbKvHeads, sizePerHead, tokensPerBlock}
        , mParallelConfig{tensorParallelism, pipelineParallelism}
        , mDataType{dataType}
    {
    }

    [[nodiscard]] bool operator==(kv_cache::CacheState const& other) const noexcept
    {
        return mModelConfig == other.mModelConfig && mParallelConfig == other.mParallelConfig
            && mDataType == other.mDataType;
    }

private:
    friend class tensorrt_llm::executor::Serialization;

    struct ModelConfig
    {
        SizeType32 mNbAttentionLayers;
        SizeType32 mNbKvHeads;
        SizeType32 mSizePerHead;
        SizeType32 mTokensPerBlock;

        [[nodiscard]] bool operator==(ModelConfig const& other) const noexcept
        {
            return mNbAttentionLayers == other.mNbAttentionLayers && mNbKvHeads == other.mNbKvHeads
                && mSizePerHead == other.mSizePerHead && mTokensPerBlock == other.mTokensPerBlock;
        }
    };

    struct ParallelConfig
    {
        SizeType32 mTensorParallelism;
        SizeType32 mPipelineParallelism;

        [[nodiscard]] bool operator==(ParallelConfig const& other) const noexcept
        {
            return mTensorParallelism == other.mTensorParallelism && mPipelineParallelism == other.mPipelineParallelism;
        }
    };

    ModelConfig mModelConfig;
    ParallelConfig mParallelConfig;
    nvinfer1::DataType mDataType;
};

struct MpiState
{
    [[nodiscard]] bool operator==(MpiState const& other) const noexcept
    {
        return mRanks == other.mRanks;
    }

    std::vector<SizeType32> mRanks;
};

struct SocketState
{
    [[nodiscard]] bool operator==(SocketState const& other) const noexcept
    {
        return mPort == other.mPort && mIp == other.mIp;
    }

    std::uint16_t mPort;
    std::string mIp;
};

class CommState final
{
public:
    CommState() = default;

    explicit CommState(std::vector<SizeType32> ranks)
        : mState{MpiState{std::move(ranks)}}
    {
    }

    explicit CommState(std::vector<SocketState> socketState)
        : mState{std::move(socketState)}
    {
    }

    CommState(std::uint16_t port, std::string ip)
        : mState{std::vector<SocketState>{SocketState{port, std::move(ip)}}}
    {
    }

    [[nodiscard]] bool isMpiState() const noexcept
    {
        return std::holds_alternative<MpiState>(mState);
    }

    [[nodiscard]] bool isSocketState() const noexcept
    {
        return std::holds_alternative<std::vector<SocketState>>(mState);
    }

    [[nodiscard]] MpiState const& getMpiState() const
    {
        TLLM_CHECK(isMpiState());
        return std::get<MpiState>(mState);
    }

    [[nodiscard]] std::vector<SocketState> const& getSocketState() const
    {
        TLLM_CHECK(isSocketState());
        return std::get<std::vector<SocketState>>(mState);
    }

    [[nodiscard]] bool operator==(CommState const& other) const noexcept
    {
        return mState == other.mState;
    }

private:
    friend class tensorrt_llm::executor::Serialization;
    std::variant<std::monostate, MpiState, std::vector<SocketState>> mState;
};

} // namespace kv_cache

class ContextPhaseState final
{
public:
    using RequestIdType = std::uint64_t;

    ContextPhaseState() = default;

    explicit ContextPhaseState(RequestIdType ReqId)
        : mReqId{ReqId}
    {
    }

    [[nodiscard]] RequestIdType getReqId() const noexcept
    {
        return mReqId;
    }

    void setCacheState(kv_cache::CacheState state)
    {
        mCacheState = std::move(state);
    }

    [[nodiscard]] std::optional<kv_cache::CacheState> const& getCacheState() const noexcept
    {
        return mCacheState;
    }

    void setCommState(kv_cache::CommState state)
    {
        mCommState = std::move(state);
    }

    [[nodiscard]] std::optional<kv_cache::CommState> const& getCommState() const noexcept
    {
        return mCommState;
    }

    [[nodiscard]] bool operator==(ContextPhaseState const& other) const noexcept
    {
        return mReqId == other.mReqId && mCacheState == other.mCacheState && mCommState == other.mCommState;
    }

private:
    friend class Serialization;
    RequestIdType mReqId{0};
    std::optional<kv_cache::CacheState> mCacheState;
    std::optional<kv_cache::CommState> mCommState;
};

} // namespace tensorrt_llm::executor
