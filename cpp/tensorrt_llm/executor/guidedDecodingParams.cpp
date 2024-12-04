/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <optional>

namespace tensorrt_llm::executor
{

GuidedDecodingParams::GuidedDecodingParams(GuideType guideType, std::optional<std::string> guide)
    : mGuideType{guideType}
    , mGuide{std::move(guide)}
{
    TLLM_CHECK_WITH_INFO(mGuideType == GuideType::kJSON || mGuide.has_value(),
        "The guide string must be provided unless using GuideType::kJSON.");
}

bool GuidedDecodingParams::operator==(GuidedDecodingParams const& other) const
{
    return mGuideType == other.mGuideType && mGuide == other.mGuide;
}

GuidedDecodingParams::GuideType GuidedDecodingParams::getGuideType() const
{
    return mGuideType;
}

std::optional<std::string> GuidedDecodingParams::getGuide() const
{
    return mGuide;
}

} // namespace tensorrt_llm::executor
