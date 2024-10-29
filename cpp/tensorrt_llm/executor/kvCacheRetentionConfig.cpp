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

namespace tensorrt_llm::executor
{

KvCacheRetentionConfig::KvCacheRetentionConfig(
    std::vector<KvCacheRetentionConfig::TokenRangeRetentionPriority> const& tokenRangeRetentionPriorities,
    RetentionPriority decodeRetentionPriority)
    : mTokenRangeRetentionPriorities(std::vector<TokenRangeRetentionPriority>(tokenRangeRetentionPriorities))
    , mDecodeRetentionPriority{decodeRetentionPriority}
{

    // The token ranges must be non-overlapping
    // No end token indicates that the range extends to the end of the sequence
    // VALID: [(0, 64), (90, 128)]
    // VALID: [(0, None)]
    // VALID: [(50, 100), (30, 40)]
    // VALID: [(0, 30), (30, 50), (60, 80)]
    // INVALID: [(0, None), (30, 50)]
    // INVALID: [(0, 64), (60, 80)]
    // INVALID: [(0, 128), (64, 256)]

    // Validate that these constraints hold, and throw an error otherwise.

    std::sort(mTokenRangeRetentionPriorities.begin(), mTokenRangeRetentionPriorities.end(),
        [](TokenRangeRetentionPriority const& x, TokenRangeRetentionPriority const& y)
        { return x.tokenStart < y.tokenStart; });

    size_t numRanges = mTokenRangeRetentionPriorities.size();

    for (size_t i = 0; i < numRanges; i++)
    {
        auto entry = mTokenRangeRetentionPriorities[i];
        if (!entry.tokenEnd.has_value() && i != numRanges - 1)
        {
            throw std::invalid_argument("Invalid use of end indicator.");
        }
        if (entry.tokenStart < 0)
        {
            throw std::invalid_argument("Token indices must be non-negative.");
        }
        if (entry.tokenEnd.has_value() && entry.tokenStart >= entry.tokenEnd)
        {
            throw std::invalid_argument("Range must have a positive, non-zero length.");
        }
        if (i != numRanges - 1)
        {
            if (*entry.tokenEnd > mTokenRangeRetentionPriorities[i + 1].tokenStart)
            {
                throw std::invalid_argument("Token ranges must be non-overlapping.");
            }
        }
    }
}

std::vector<KvCacheRetentionConfig::TokenRangeRetentionPriority>
KvCacheRetentionConfig::getTokenRangeRetentionPriorities() const
{
    return mTokenRangeRetentionPriorities;
}

RetentionPriority KvCacheRetentionConfig::getDecodeRetentionPriority() const
{
    return mDecodeRetentionPriority;
}

std::vector<std::optional<RetentionPriority>> KvCacheRetentionConfig::getPerBlockEvictionPolicy(
    SizeType32 blockSize, SizeType32 seqLen)
{
    std::vector<std::optional<RetentionPriority>> perBlockPriorities;

    SizeType32 tokenLoc = 0;
    size_t pointer = 0;
    size_t numRanges = mTokenRangeRetentionPriorities.size();

    // Handle cases where the first range doesn't start at 0
    for (; !mTokenRangeRetentionPriorities.empty() && tokenLoc < mTokenRangeRetentionPriorities[0].tokenStart
         && tokenLoc < seqLen;
         tokenLoc += blockSize)
    {
        perBlockPriorities.emplace_back(std::nullopt);
    }

    while (tokenLoc < seqLen && pointer < numRanges)
    {
        TokenRangeRetentionPriority entry = mTokenRangeRetentionPriorities[pointer];

        if (entry.tokenEnd.has_value() && tokenLoc >= entry.tokenEnd)
        {
            pointer += 1;
        }
        else
        {
            perBlockPriorities.emplace_back(
                tokenLoc < entry.tokenStart ? std::nullopt : std::optional<RetentionPriority>(entry.priority));
            tokenLoc += blockSize;
        }
    }

    for (; tokenLoc < seqLen; tokenLoc += blockSize)
    {
        perBlockPriorities.emplace_back(std::nullopt);
    }

    return perBlockPriorities;
}

} // namespace tensorrt_llm::executor
