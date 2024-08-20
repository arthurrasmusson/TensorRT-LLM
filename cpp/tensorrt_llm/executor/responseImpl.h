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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/executor/executor.h"
#include <variant>

namespace tensorrt_llm::executor
{

class Response::Impl
{
public:
    Impl(IdType requestId, std::string errorMsg)
        : mRequestId(requestId)
        , mErrOrResult(std::move(errorMsg))
    {
        TLLM_CHECK_WITH_INFO(!std::get<std::string>(mErrOrResult).empty(), "Error message should not be empty");
    }

    Impl(IdType requestId, Result Result)
        : mRequestId(requestId)
        , mErrOrResult(std::move(Result))
    {
    }

    ~Impl() = default;

    [[nodiscard]] bool hasError() const
    {
        return std::holds_alternative<std::string>(mErrOrResult);
    }

    [[nodiscard]] bool hasResult() const
    {
        return std::holds_alternative<Result>(mErrOrResult);
    }

    [[nodiscard]] IdType getRequestId() const
    {
        return mRequestId;
    }

    /// Could throw exception if no result is available
    [[nodiscard]] Result const& getResult() const
    {
        if (hasResult())
        {
            return std::get<Result>(mErrOrResult);
        }
        else
        {
            TLLM_THROW("Cannot get the result for a response with an error");
        }
    }

    [[nodiscard]] std::string const& getErrorMsg() const
    {
        if (hasError())
        {
            return std::get<std::string>(mErrOrResult);
        }
        else
        {
            TLLM_THROW("Cannot get the error message for a response without error");
        }
    }

private:
    friend class Serialization;
    IdType mRequestId;
    std::variant<std::string, Result> mErrOrResult;
};

} // namespace tensorrt_llm::executor
