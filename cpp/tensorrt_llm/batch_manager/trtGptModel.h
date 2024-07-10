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

#include "tensorrt_llm/batch_manager/BatchManager.h"
#include "tensorrt_llm/batch_manager/peftCacheManager.h"
#include "tensorrt_llm/batch_manager/trtGptModelOptionalParams.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/stlUtils.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/model.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/worldConfig.h"

#include <memory>

namespace tensorrt_llm::batch_manager
{

class GptManager;
class LlmRequest;

namespace kv_cache_manager
{
class KVCacheManager;
}

class TrtGptModel : public executor::Model
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;

    TrtGptModel(runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig,
        TrtGptModelOptionalParams const& optionalParams)
        : mMaxBatchSize{optionalParams.maxBatchSize.value_or(modelConfig.getMaxBatchSize())}
        , mMaxBeamWidth{optionalParams.maxBeamWidth.value_or(modelConfig.getMaxBeamWidth())}
        , mMaxSequenceLen{modelConfig.getMaxSequenceLen()}
        , mMaxDraftLen{modelConfig.getMaxDecodingDraftTokens()}
        , mVocabSizePadded{modelConfig.getVocabSizePadded(worldConfig.getSize())}
        , mComputeContextLogits{modelConfig.computeContextLogits()}
        , mComputeGenerationLogits{modelConfig.computeGenerationLogits()}
        , mNormalizeLogProbs{optionalParams.normalizeLogProbs}
        , mEnableTrtOverlap{optionalParams.enableTrtOverlap}
    {
        TLLM_CHECK_WITH_INFO(mMaxBeamWidth <= modelConfig.getMaxBeamWidth(),
            "Runtime configured max beam width (%d) must not exceed engine max beam width (%d)", mMaxBeamWidth,
            modelConfig.getMaxBeamWidth());
        TLLM_CHECK_WITH_INFO(mMaxBatchSize <= modelConfig.getMaxBatchSize(),
            "Runtime configured max batch size (%d) must not exceed engine max batch size (%d)", mMaxBatchSize,
            modelConfig.getMaxBatchSize());

        if (optionalParams.kvCacheConfig.maxAttentionWindow.has_value()
            && optionalParams.kvCacheConfig.maxAttentionWindow.value() > mMaxSequenceLen)
        {
            TLLM_LOG_WARNING(
                "The value of maxAttentionWindow cannot exceed mMaxSequenceLen. "
                "Therefore, it has been adjusted to match the value of mMaxSequenceLen.");
        }
        mMaxAttentionWindow = optionalParams.kvCacheConfig.maxAttentionWindow.has_value()
            ? std::min(optionalParams.kvCacheConfig.maxAttentionWindow.value(), mMaxSequenceLen)
            : mMaxSequenceLen;
        TLLM_CHECK_WITH_INFO(mMaxAttentionWindow > 0, "Attention window size (mMaxAttentionWindow) must be > 0");
        mSinkTokenLen = optionalParams.kvCacheConfig.sinkTokenLength.has_value()
            ? optionalParams.kvCacheConfig.sinkTokenLength.value()
            : 0;

        auto const numBatches
            = worldConfig.isPipelineParallel() ? worldConfig.getPipelineParallelism() : (mEnableTrtOverlap ? 2 : 1);
        mMaxNumSequences = numBatches * mMaxBatchSize;

        TLLM_LOG_INFO("TRTGptModel maxNumSequences: %d", mMaxNumSequences);
        TLLM_LOG_INFO("TRTGptModel maxBatchSize: %d", mMaxBatchSize);
        TLLM_LOG_INFO("TRTGptModel maxBeamWidth: %d", mMaxBeamWidth);
        TLLM_LOG_INFO("TRTGptModel maxSequenceLen: %d", mMaxSequenceLen);
        TLLM_LOG_INFO("TRTGptModel maxDraftLen: %d", mMaxDraftLen);

        TLLM_LOG_INFO("TRTGptModel mMaxAttentionWindowSize: %d", mMaxAttentionWindow);

        TLLM_LOG_INFO("TRTGptModel computeContextLogits: %d", mComputeContextLogits);
        TLLM_LOG_INFO("TRTGptModel computeGenerationLogits: %d", mComputeGenerationLogits);
        TLLM_LOG_INFO("TRTGptModel enableTrtOverlap: %d", mEnableTrtOverlap);
        TLLM_LOG_INFO("TRTGptModel normalizeLogProbs: %d", mNormalizeLogProbs);

        mMaxNumTokens = modelConfig.getMaxNumTokens();
        if (optionalParams.maxNumTokens && mMaxNumTokens)
        {
            if (optionalParams.maxNumTokens.value() > mMaxNumTokens.value())
            {
                TLLM_LOG_WARNING(
                    "Runtime configured max num tokens (%d) is larger than model max num tokens (%d) and will be "
                    "ignored.",
                    optionalParams.maxNumTokens.value(), mMaxNumTokens.value());
            }
            else
            {
                mMaxNumTokens = optionalParams.maxNumTokens;
            }
        }
        if (mMaxNumTokens)
        {
            TLLM_LOG_INFO("TRTGptModel maxNumTokens: %d", mMaxNumTokens.value());
        }

        if (optionalParams.enableChunkedContext)
        {
            mMaxInputLen = mMaxSequenceLen - 1;
            TLLM_LOG_INFO(
                "TRTGptModel maxInputLen: %d  = maxSequenceLen - 1 since chunked context is enabled", mMaxInputLen);
        }
        else if (modelConfig.getContextFMHA() && modelConfig.usePackedInput())
        {
            TLLM_CHECK_WITH_INFO(
                mMaxNumTokens, "Max number of tokens has to be set for context FMHA and usePackedInput case.");
            mMaxInputLen = std::min(mMaxSequenceLen - 1, mMaxNumTokens.value());
            TLLM_LOG_INFO(
                "TRTGptModel maxInputLen: %d = min(maxSequenceLen - 1, maxNumTokens) since context FMHA "
                "and usePackedInput are enabled",
                mMaxInputLen);
        }
        else
        {
            mMaxInputLen = modelConfig.getMaxInputLen();
            TLLM_LOG_INFO("TRTGptModel maxInputLen: %d = max_input_len (in trtllm-build args)", mMaxInputLen);
        }

        using tensorrt_llm::common::stl_utils::toString;

        TLLM_LOG_INFO("Capacity Scheduler Policy: %s",
            toString(optionalParams.schedulerConfig.getCapacitySchedulerPolicy()).c_str());
        TLLM_LOG_INFO("Context Chunking Scheduler Policy: %s",
            toString(optionalParams.schedulerConfig.getContextChunkingPolicy()).c_str());
    }

    [[nodiscard]] std::optional<SizeType32> getMaxNumTokens() const
    {
        return mMaxNumTokens;
    }

    [[nodiscard]] SizeType32 getMaxNumSequences() const override
    {
        return mMaxNumSequences;
    }

    [[nodiscard]] SizeType32 getMaxBatchSize() const
    {
        return mMaxBatchSize;
    }

    [[nodiscard]] SizeType32 getMaxInputLen() const override
    {
        return mMaxInputLen;
    }

    [[nodiscard]] virtual SizeType32 getHiddenSize() const override
    {
        return getModelConfig().getHiddenSize();
    };

    [[nodiscard]] SizeType32 getMaxSequenceLen() const override
    {
        return mMaxSequenceLen;
    }

    [[nodiscard]] virtual TrtGptModelType getModelType() const = 0;
    [[nodiscard]] virtual runtime::BufferManager const& getBufferManager() const = 0;
    [[nodiscard]] virtual runtime::ModelConfig const& getModelConfig() const = 0;

    [[nodiscard]] SizeType32 getVocabSizePadded() const override
    {
        return mVocabSizePadded;
    }

    [[nodiscard]] SizeType32 getMaxDraftLen() const override
    {
        return mMaxDraftLen;
    }

    [[nodiscard]] bool computeContextLogits() const override
    {
        return mComputeContextLogits;
    }

    [[nodiscard]] bool computeGenerationLogits() const override
    {
        return mComputeGenerationLogits;
    }

    virtual void setLayerProfiler() = 0;
    [[nodiscard]] virtual std::string getLayerProfileInfo() const = 0;

protected:
    friend class GptManager;

    [[nodiscard]] SizeType32 getMaxBeamWidth() const
    {
        return mMaxBeamWidth;
    }

    [[nodiscard]] SizeType32 getMaxAttentionWindow() const
    {
        return mMaxAttentionWindow;
    }

    [[nodiscard]] SizeType32 getSinkTokenLen() const
    {
        return mSinkTokenLen;
    }

    [[nodiscard]] bool isNormalizeLogProbs() const
    {
        return mNormalizeLogProbs;
    }

    [[nodiscard]] bool isTtrOverlap() const
    {
        return mEnableTrtOverlap;
    }

    [[nodiscard]] virtual std::shared_ptr<kv_cache_manager::KVCacheManager> getKVCacheManager() = 0;
    [[nodiscard]] virtual std::shared_ptr<kv_cache_manager::KVCacheManager const> getKVCacheManager() const = 0;

    [[nodiscard]] virtual std::shared_ptr<BasePeftCacheManager> getPeftCacheManager() = 0;
    [[nodiscard]] virtual std::shared_ptr<BasePeftCacheManager const> getPeftCacheManager() const = 0;

private:
    std::optional<SizeType32> mMaxNumTokens;
    SizeType32 mMaxNumSequences;
    SizeType32 mMaxBatchSize;
    SizeType32 mMaxBeamWidth;
    SizeType32 mMaxInputLen;
    SizeType32 mMaxSequenceLen;
    SizeType32 mMaxDraftLen;

    SizeType32 mVocabSizePadded;
    SizeType32 mMaxAttentionWindow;
    SizeType32 mSinkTokenLen;

    bool mComputeContextLogits;
    bool mComputeGenerationLogits;
    bool mNormalizeLogProbs;
    bool mEnableTrtOverlap;
};

} // namespace tensorrt_llm::batch_manager
