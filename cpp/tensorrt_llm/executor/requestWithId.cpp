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

#include "requestWithId.h"

#include <istream>
#include <ostream>
#include <sstream>

using namespace tensorrt_llm::executor;

std::vector<char> tensorrt_llm::executor::RequestWithId::serializeReqWithIds(
    std::vector<RequestWithId> const& reqWithIds)
{
    // Compute the size of serialized buffer
    size_t totalSize = 0;
    totalSize += sizeof(size_t);
    for (auto const& reqWithId : reqWithIds)
    {
        totalSize += su::serializedSize(reqWithId.id);
        totalSize += su::serializedSize(reqWithId.req);
    }

    std::vector<char> buffer(totalSize);
    std::stringbuf strbuf{std::ios_base::out | std::ios_base::in};
    strbuf.pubsetbuf(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    std::ostream ostream{&strbuf};

    su::serialize(reqWithIds.size(), ostream);
    for (auto const& reqWithId : reqWithIds)
    {
        su::serialize(reqWithId.id, ostream);
        su::serialize(reqWithId.req, ostream);
    }
    return buffer;
}

std::vector<RequestWithId> tensorrt_llm::executor::RequestWithId::deserializeReqWithIds(std::vector<char>& buffer)
{
    std::vector<RequestWithId> reqWithIds;
    su::VectorWrapBuf<char> strbuf{buffer};
    std::istream istream{&strbuf};
    auto numReq = su::deserialize<std::int64_t>(istream);
    for (int64_t req = 0; req < numReq; ++req)
    {
        auto const id = su::deserialize<std::uint64_t>(istream);
        reqWithIds.emplace_back(RequestWithId{Serialization::deserializeRequest(istream), id});
    }
    return reqWithIds;
}
