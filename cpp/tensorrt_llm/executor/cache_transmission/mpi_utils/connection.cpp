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

#include "tensorrt_llm/executor/cache_transmission/mpi_utils/connection.h"

namespace tensorrt_llm::executor::kv_cache
{

MpiConnection::MpiConnection(mpi::MpiComm const* comm, int rank)
    : mComm{comm}
    , mRank{rank}
{
}

void MpiConnection::send(DataContext const& ctx, void const* data, size_t size) const
{
    mComm->send(data, size, mpi::MpiType::kCHAR, mRank, ctx.getTag());
}

void MpiConnection::recv(DataContext const& ctx, void* data, size_t size) const
{
    mComm->recv(data, size, mpi::MpiType::kCHAR, mRank, ctx.getTag());
}

MpiConnectionManager::MpiConnectionManager(mpi::MpiComm const* comm)
    : mComm{comm}
{
    TLLM_CHECK(mComm);
}

MpiConnection const* MpiConnectionManager::recvConnect(DataContext const& ctx, void* data, size_t size)
{
#if ENABLE_MULTI_DEVICE
    MPI_Status status;
    MPI_Recv(data, size, MPI_CHAR, MPI_ANY_SOURCE, ctx.getTag(), static_cast<MPI_Comm>(*mComm), std::addressof(status));
    auto&& [it, success] = mConnections.insert({status.MPI_SOURCE, MpiConnection{mComm, status.MPI_SOURCE}});
    return std::addressof(it->second);
#else
    TLLM_THROW("Multi device support is disabled.");
#endif
}

std::vector<Connection const*> MpiConnectionManager::getConnections(CommState const& state)
{
    std::vector<Connection const*> ret;
    TLLM_CHECK(state.isMpiState());
    for (auto rank : state.getMpiState().mRanks)
    {
        auto&& [it, success] = mConnections.insert({rank, MpiConnection{mComm, rank}});
        ret.emplace_back(&it->second);
    }
    return ret;
}

} // namespace tensorrt_llm::executor::kv_cache
