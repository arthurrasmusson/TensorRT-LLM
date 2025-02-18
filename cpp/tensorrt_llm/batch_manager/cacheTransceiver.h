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
#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include <future>
#include <map>
#include <memory>

namespace tensorrt_llm::batch_manager
{

class ContextProgress;

class CacheTransceiver
{
public:
    enum class CommType : std::uint8_t
    {
        UNKNOWN = 0,
        MPI = 1,
        UCX = 2
    };

    CacheTransceiver(kv_cache_manager::BaseKVCacheManager* cacheManager, CommType commType,
        runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig);

    ~CacheTransceiver();

    void respondAndSendAsync(LlmRequest* llmRequest);

    void respondAndSendLayerWise(RequestVector const& requests, std::shared_ptr<ContextProgress> const& progress);

    void requestAndReceiveSync(LlmRequest* llmRequest);
    void requestAndReceiveAsync(LlmRequest* llmRequest);

    void checkContextTransferStatus(bool blocking = false);

    void checkGenTransferStatus(int atLeastRequestNum = 0);

    [[nodiscard]] bool checkGenTransferComplete() const;

private:
    void initializeCommState();

    void setContextState(LlmRequest* llmRequest);

    CommType mCommType;
    std::unique_ptr<DataResponder> mDataResponder;
    std::unique_ptr<DataRequester> mDataRequester;
    std::map<LlmRequest*, std::future<void>> mResponderFutures;
    std::vector<std::pair<LlmRequest*, std::future<void>>> mRequesterFutures;
    mpi::MpiComm const *mMpiGroupComm{}, *mMpiWorldComm{};
    std::shared_ptr<mpi::MpiComm> mMpiGroupTensorParaComm, mMpiGroupPipeParaComm;
    executor::kv_cache::CommState const* mCommState;
    std::unique_ptr<executor::kv_cache::CacheState> mCacheState;

    // library handle to the communicator related features,
    // this is used to defer dependency resolution until needed.
    static std::mutex mDllMutex;
    void* mWrapperLibHandle{nullptr};
};

} // namespace tensorrt_llm::batch_manager
