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

#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/responseImpl.h"
#include "tensorrt_llm/executor/types.h"

namespace tensorrt_llm::executor
{

Response::Response(IdType requestId, std::string errorMsg)
    : mImpl(std::make_unique<Impl>(requestId, std::move(errorMsg)))
{
}

Response::Response(IdType requestId, Result Result)
    : mImpl(std::make_unique<Impl>(requestId, std::move(Result)))
{
}

Response::~Response() = default;

Response::Response(Response const& other)
    : mImpl(std::make_unique<Impl>(*other.mImpl))
{
}

Response::Response(Response&& other) noexcept = default;

Response& Response::operator=(Response const& other)
{
    if (this != &other)
    {
        mImpl = std::make_unique<Impl>(*other.mImpl);
    }
    return *this;
}

Response& Response::operator=(Response&& other) noexcept = default;

bool Response::hasError() const
{
    return mImpl->hasError();
}

std::string const& Response::getErrorMsg() const
{
    return mImpl->getErrorMsg();
}

IdType Response::getRequestId() const
{
    return mImpl->getRequestId();
}

Result const& Response::getResult() const
{
    return mImpl->getResult();
}

} // namespace tensorrt_llm::executor
