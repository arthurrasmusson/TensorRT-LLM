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

#include "mpiDataTransceiver.h"
#include "tensorrt_llm/batch_manager/kvCacheUtils.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include <iterator>

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

// Describe the data structure for cache layout, which can be used to infer cache layouts and location
// associations between different processes, in order to determine suitable senders and receivers.
class CacheConfig final
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;

    CacheConfig(
        runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig, nvinfer1::DataType dataType)
        : mModelConfig{modelConfig.getNbAttentionLayers(1), modelConfig.getNbKvHeads(), modelConfig.getSizePerHead(),
            modelConfig.getTokensPerBlock()}
        , mParallelConfig{worldConfig.getTensorParallelism(), worldConfig.getPipelineParallelism()}
        , mDataType{dataType}
    {
    }

    CacheConfig(SizeType32 nbAttentionLayers, SizeType32 nbKvHeads, SizeType32 sizePerHead, SizeType32 tokensPerBlock,
        SizeType32 tensorParallelism, SizeType32 pipelineParallelism, nvinfer1::DataType dataType)
        : mModelConfig{nbAttentionLayers, nbKvHeads, sizePerHead, tokensPerBlock}
        , mParallelConfig{tensorParallelism, pipelineParallelism}
        , mDataType{dataType}
    {
    }

    [[nodiscard]] bool operator==(CacheConfig const& other) const noexcept
    {
        return mModelConfig == other.mModelConfig && mParallelConfig == other.mParallelConfig
            && mDataType == other.mDataType;
    }

private:
    struct ParallelConfig
    {
        SizeType32 mTensorParallelism;
        SizeType32 mPipelineParallelism;

        [[nodiscard]] bool operator==(ParallelConfig const& other) const noexcept
        {
            return mTensorParallelism == other.mTensorParallelism && mPipelineParallelism == other.mPipelineParallelism;
        }
    };

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

    ModelConfig mModelConfig;
    ParallelConfig mParallelConfig;
    nvinfer1::DataType mDataType;
};

class CacheContext final : public DataContext
{
public:
    CacheContext(CacheConfig config, std::vector<SizeType32> ranks, std::optional<SizeType32> selfIdx = std::nullopt)
        : DataContext{std::move(ranks), std::move(selfIdx)}
        , mConfig{std::move(config)}
    {
    }

    [[nodiscard]] CacheConfig const& getConfig() const noexcept
    {
        return mConfig;
    }

private:
    bool isEqual(DataContext const& obj) const override
    {
        auto const& v = dynamic_cast<CacheContext const&>(obj);
        return DataContext::isEqual(obj) && mConfig == v.mConfig;
    }

    CacheConfig mConfig;
};

// Simple cache block copy. Because it does not involve data splitting or merging, it performs best when the
// parallel topology is completely identical, making it the preferred method.
class CacheBlockSender final : public DataSender
{
public:
    CacheBlockSender(KVCacheManager* cacheManager, MpiComm const& comm, CacheContext selfContext)
        : mComm{std::addressof(comm)}
        , mCacheManager{cacheManager}
        , mSelfContext{std::move(selfContext)}
    {
        TLLM_CHECK(cacheManager);
    }

    [[nodiscard]] bool inquireSupport(DataContext const* receiverContext) override;
    void send(LlmRequest const& request, DataContext const& destination) override;

private:
    MpiComm const* mComm{};
    KVCacheManager* mCacheManager{};
    CacheContext mSelfContext;
};

// Simple cache block copy. Because it does not involve data splitting or merging, it performs best when the
// parallel topology is completely identical, making it the preferred method.
class CacheBlockReceiver final : public DataReceiver
{
public:
    CacheBlockReceiver(KVCacheManager* cacheManager, MpiComm const& comm, CacheContext selfContext)
        : mComm{std::addressof(comm)}
        , mCacheManager{cacheManager}
        , mSelfContext{std::move(selfContext)}
    {
        TLLM_CHECK(cacheManager);
    }

    [[nodiscard]] bool inquireSupport(DataContext const* senderContext) override;
    void receive(LlmRequest const& request, DataContext const& source) override;

private:
    MpiComm const* mComm{};
    KVCacheManager* mCacheManager{};
    CacheContext mSelfContext;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager
