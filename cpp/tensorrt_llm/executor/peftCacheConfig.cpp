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
PeftCacheConfig::PeftCacheConfig(SizeType32 numHostModuleLayer, SizeType32 numDeviceModuleLayer,
    SizeType32 optimalAdapterSize, SizeType32 maxAdapterSize, SizeType32 numPutWorkers, SizeType32 numEnsureWorkers,
    SizeType32 numCopyStreams, SizeType32 maxPagesPerBlockHost, SizeType32 maxPagesPerBlockDevice,
    std::optional<FloatType> const& deviceCachePercent, std::optional<size_t> const& hostCacheSize)
    : mNumHostModuleLayer(numHostModuleLayer)
    , mNumDeviceModuleLayer(numDeviceModuleLayer)
    , mOptimalAdapterSize(optimalAdapterSize)
    , mMaxAdapterSize(maxAdapterSize)
    , mNumPutWorkers(numPutWorkers)
    , mNumEnsureWorkers(numEnsureWorkers)
    , mNumCopyStreams(numCopyStreams)
    , mMaxPagesPerBlockHost(maxPagesPerBlockHost)
    , mMaxPagesPerBlockDevice(maxPagesPerBlockDevice)
    , mDeviceCachePercent(deviceCachePercent)
    , mHostCacheSize(hostCacheSize)
{
}

bool PeftCacheConfig::operator==(PeftCacheConfig const& other) const
{
    return mNumHostModuleLayer == other.mNumHostModuleLayer && mNumDeviceModuleLayer == other.mNumDeviceModuleLayer
        && mOptimalAdapterSize == other.mOptimalAdapterSize && mMaxAdapterSize == other.mMaxAdapterSize
        && mNumPutWorkers == other.mNumPutWorkers && mNumEnsureWorkers == other.mNumEnsureWorkers
        && mNumCopyStreams == other.mNumCopyStreams && mMaxPagesPerBlockHost == other.mMaxPagesPerBlockHost
        && mMaxPagesPerBlockDevice == other.mMaxPagesPerBlockDevice && mDeviceCachePercent == other.mDeviceCachePercent
        && mHostCacheSize == other.mHostCacheSize;
}

SizeType32 PeftCacheConfig::getNumHostModuleLayer() const
{
    return mNumHostModuleLayer;
}

SizeType32 PeftCacheConfig::getNumDeviceModuleLayer() const
{
    return mNumDeviceModuleLayer;
}

SizeType32 PeftCacheConfig::getOptimalAdapterSize() const
{
    return mOptimalAdapterSize;
}

SizeType32 PeftCacheConfig::getMaxAdapterSize() const
{
    return mMaxAdapterSize;
}

SizeType32 PeftCacheConfig::getNumPutWorkers() const
{
    return mNumPutWorkers;
}

SizeType32 PeftCacheConfig::getNumEnsureWorkers() const
{
    return mNumEnsureWorkers;
}

SizeType32 PeftCacheConfig::getNumCopyStreams() const
{
    return mNumCopyStreams;
}

SizeType32 PeftCacheConfig::getMaxPagesPerBlockHost() const
{
    return mMaxPagesPerBlockHost;
}

SizeType32 PeftCacheConfig::getMaxPagesPerBlockDevice() const
{
    return mMaxPagesPerBlockDevice;
}

std::optional<FloatType> PeftCacheConfig::getDeviceCachePercent() const
{
    return mDeviceCachePercent;
}

std::optional<size_t> PeftCacheConfig::getHostCacheSize() const
{
    return mHostCacheSize;
}
} // namespace tensorrt_llm::executor
