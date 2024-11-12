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
    std::vector<KvCacheRetentionConfig::TokenRangeRetentionConfig> const& tokenRangeRetentionPriorities,
    RetentionPriority decodeRetentionPriority, std::optional<std::chrono::milliseconds> decodeDurationMs)
    : mTokenRangeRetentionConfigs(std::vector<TokenRangeRetentionConfig>(tokenRangeRetentionPriorities))
    , mDecodeRetentionPriority{decodeRetentionPriority}
    , mDecodeDurationMs{decodeDurationMs}
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

    std::sort(mTokenRangeRetentionConfigs.begin(), mTokenRangeRetentionConfigs.end(),
        [](TokenRangeRetentionConfig const& x, TokenRangeRetentionConfig const& y)
        { return x.tokenStart < y.tokenStart; });

    size_t numRanges = mTokenRangeRetentionConfigs.size();

    for (size_t i = 0; i < numRanges; i++)
    {
        auto entry = mTokenRangeRetentionConfigs[i];
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
            if (*entry.tokenEnd > mTokenRangeRetentionConfigs[i + 1].tokenStart)
            {
                throw std::invalid_argument("Token ranges must be non-overlapping.");
            }
        }
    }
}

std::vector<KvCacheRetentionConfig::TokenRangeRetentionConfig>
KvCacheRetentionConfig::getTokenRangeRetentionConfigs() const
{
    return mTokenRangeRetentionConfigs;
}

RetentionPriority KvCacheRetentionConfig::getDecodeRetentionPriority() const
{
    return mDecodeRetentionPriority;
}

std::optional<std::chrono::milliseconds> KvCacheRetentionConfig::getDecodeDurationMs() const
{
    return mDecodeDurationMs;
}

std::vector<RetentionPriorityAndDuration> KvCacheRetentionConfig::getPerBlockRetentionPriorityDuration(
    SizeType32 blockSize, SizeType32 seqLen) const
{
    std::vector<RetentionPriorityAndDuration> perBlockRetentions;

    SizeType32 tokenLoc = 0;
    size_t pointer = 0;
    size_t numRanges = mTokenRangeRetentionConfigs.size();

    // Handle cases where the first range doesn't start at 0
    for (; !mTokenRangeRetentionConfigs.empty() && tokenLoc < mTokenRangeRetentionConfigs[0].tokenStart
         && tokenLoc < seqLen;
         tokenLoc += blockSize)
    {
        perBlockRetentions.emplace_back(std::nullopt, std::nullopt);
    }

    while (tokenLoc < seqLen && pointer < numRanges)
    {
        TokenRangeRetentionConfig entry = mTokenRangeRetentionConfigs[pointer];

        if (entry.tokenEnd.has_value() && tokenLoc >= entry.tokenEnd)
        {
            pointer += 1;
        }
        else
        {

            if (tokenLoc < entry.tokenStart)
            {
                perBlockRetentions.emplace_back(std::nullopt, std::nullopt);
            }
            else
            {
                perBlockRetentions.emplace_back(entry.priority, entry.durationMs);
            }

            tokenLoc += blockSize;
        }
    }

    for (; tokenLoc < seqLen; tokenLoc += blockSize)
    {
        perBlockRetentions.emplace_back(std::nullopt, std::nullopt);
    }

    return perBlockRetentions;
}

} // namespace tensorrt_llm::executor
