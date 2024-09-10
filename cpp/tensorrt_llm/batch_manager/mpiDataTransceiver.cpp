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

#include "mpiDataTransceiver.h"
#include "tensorrt_llm/batch_manager/cacheTransceiver.h"
#include "tensorrt_llm/executor/contextPhaseState.h"

namespace tensorrt_llm::batch_manager
{

template class MpiDataSender<executor::kv_cache::CacheState>;

template class MpiDataReceiver<executor::kv_cache::CacheState>;

std::unique_ptr<DataResponder> makeMpiCacheResponder(
    mpi::MpiComm const& comm, kv_cache_manager::KVCacheManager* cacheManager)
{
    using namespace tensorrt_llm::batch_manager::kv_cache_manager;
    return std::make_unique<DataResponder>(std::make_unique<MpiDataSender<executor::kv_cache::CacheState>>(
        comm, std::make_unique<CacheOutputFormatter<MpiComm>>(cacheManager)));
}

std::unique_ptr<DataRequester> makeMpiCacheRequester(
    mpi::MpiComm const& comm, kv_cache_manager::KVCacheManager* cacheManager)
{
    using namespace tensorrt_llm::batch_manager::kv_cache_manager;
    return std::make_unique<DataRequester>(std::make_unique<MpiDataReceiver<executor::kv_cache::CacheState>>(
        comm, std::make_unique<CacheInputFormatter<MpiComm>>(cacheManager)));
}

} // namespace tensorrt_llm::batch_manager
