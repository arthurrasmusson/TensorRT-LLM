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

#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/peftCacheManager.h"
#include "tensorrt_llm/batch_manager/runtimeBuffers.h"
#include "tensorrt_llm/batch_manager/sequenceSlotManager.h"
#include "tensorrt_llm/common/optionalRef.h"
#include "tensorrt_llm/runtime/iTensor.h"

namespace tensorrt_llm::batch_manager::utils
{
using SizeType32 = runtime::SizeType32;
using TensorPtr = runtime::ITensor::SharedPtr;

template <typename T>
using OptionalRef = common::OptionalRef<T>;

TensorPtr collectRequestIds(RequestVector const& contextRequests, RequestVector const& generationRequests);

void sortByLoraId(ScheduledRequests& scheduledRequests);

//! @param beforeDecoder    Whether the function is called before the decoder. If it is true, correct the output offset.
//! @param numDroppedTokens The number of dropped tokens for each beam (e.g. when the requests finished early).
//!                         Generation logits for dropped tokens are ignored.
void copyGenerationLogits(RuntimeBuffers const& genRuntimeBuffers, runtime::BufferManager const& bufferManager,
    LlmRequest& llmReq, std::size_t batchIdx, bool beforeDecoder, std::vector<SizeType32> const& numDroppedTokens = {});

void copyAdditionalOutputs(RequestVector const& contextRequests, RequestVector const& generationRequests,
    RuntimeBuffers::TensorMap const& outputMap, runtime::BufferManager const& manager);

void terminateRequest(SequenceSlotManager& seqSlotManager, LlmRequest& llmRequest, SizeType32 maxInputLen,
    OptionalRef<kv_cache_manager::BaseKVCacheManager> kvCacheManager = std::nullopt,
    OptionalRef<kv_cache_manager::BaseKVCacheManager> crossKvCacheManager = std::nullopt,
    OptionalRef<BasePeftCacheManager> peftCacheManager = std::nullopt, bool pause = false);

class CudaGraphExecutor
{
public:
    CudaGraphExecutor() = default;

    ~CudaGraphExecutor()
    {
        try
        {
            clear();
        }
        catch (std::exception& e)
        {
            TLLM_LOG_EXCEPTION(e);
        }
    }

    bool hasInstance() const
    {
        return mInstance != nullptr;
    }

    void clear();
    void prepareNextGraph(std::shared_ptr<runtime::TllmRuntime>& runtime, SizeType32 nextContextId);
    void launch(runtime::CudaStream const& stream);

private:
    void create(cudaGraph_t const& graph);
    bool update(cudaGraph_t const& graph);
    void uploadToStream(runtime::CudaStream const& stream);

    cudaGraphExec_t mInstance;
};

class CudaGraphExecutorCache
{
    /// @brief LRU cache to store cuda graph instances.
public:
    explicit CudaGraphExecutorCache(runtime::SizeType32 capacity)
        : mCapacity(capacity)
    {
    }

    std::optional<std::shared_ptr<CudaGraphExecutor>> get(BatchState const& state);

    void put(BatchState const& state, std::shared_ptr<CudaGraphExecutor> const& value);

private:
    using BatchStateGraphExecutorPair = std::pair<BatchState, std::shared_ptr<CudaGraphExecutor>>;
    using GraphExecutorLruCache = std::list<BatchStateGraphExecutorPair>;
    SizeType32 mCapacity;
    GraphExecutorLruCache mCache;
    std::unordered_map<BatchState, GraphExecutorLruCache::iterator, BatchStateHash> mMap;
};
} // namespace tensorrt_llm::batch_manager::utils
