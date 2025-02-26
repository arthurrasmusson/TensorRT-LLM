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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/executor/cacheCommunicator.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"

namespace tensorrt_llm::executor::kv_cache
{

class MpiConnection : public Connection
{
public:
    MpiConnection(mpi::MpiComm const* comm, int rank);
    void send(DataContext const& ctx, void const* data, size_t size) const override;
    void recv(DataContext const& ctx, void* data, size_t size) const override;

private:
    mpi::MpiComm const* mComm{};
    int mRank{-1};
};

class MpiConnectionManager : public ConnectionManager
{
public:
    MpiConnectionManager(mpi::MpiComm const* comm);
    MpiConnection const* recvConnect(DataContext const& ctx, void* data, size_t size) override;
    std::vector<Connection const*> getConnections(CommState const& state) override;

private:
    mpi::MpiComm const* mComm;
    std::map<int, MpiConnection> mConnections;
};

} // namespace tensorrt_llm::executor::kv_cache
