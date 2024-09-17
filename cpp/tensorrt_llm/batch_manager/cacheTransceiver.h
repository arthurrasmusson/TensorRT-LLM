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
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/common/mpiUtils.h"
#include <future>
#include <map>

namespace tensorrt_llm::batch_manager
{

class CacheTransceiver
{
public:
    enum class CommType : std::uint8_t
    {
        UNKNOWN = 0,
        MPI = 1,
        UCX = 2
    };

    CacheTransceiver(kv_cache_manager::KVCacheManager* cacheManager, CommType commType,
        runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig);

    void respondAndSendAsync(LlmRequest* llmRequest);

    void requestAndReceiveSync(LlmRequest* llmRequest);

    void checkTranferStatus(bool blocking = false);

private:
    void initializeCommState();

    void setContextState(LlmRequest* llmRequest);

    CommType mCommType;
    std::unique_ptr<DataResponder> mDataResponder;
    std::unique_ptr<DataRequester> mDataRequester;
    std::map<LlmRequest*, std::future<void>> mResponderFutures;
    mpi::MpiComm const *mMpiGroupComm{}, *mMpiWorldComm{};
    executor::kv_cache::CommState const* mCommState;
    std::unique_ptr<executor::kv_cache::CacheState> mCacheState;
};

} // namespace tensorrt_llm::batch_manager
