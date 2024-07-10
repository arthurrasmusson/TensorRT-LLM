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

#include "tensorrt_llm/batch_manager/namedTensor.h"
#include "tensorrt_llm/runtime/bufferManager.h"

#include <cstring>
#include <vector>

namespace tensorrt_llm::batch_manager
{

NamedTensor::NamedTensor(
    nvinfer1::DataType _type, std::vector<int64_t> const& _shape, std::string _name, void const* _data)
    : Base(std::move(_name))
{
    nvinfer1::Dims dims;
    dims.nbDims = _shape.size();
    for (int32_t i = 0; i < dims.nbDims; ++i)
    {
        dims.d[i] = _shape[i];
    }
    tensor = tensorrt_llm::runtime::BufferManager::pinnedPool(dims, _type);
    if (_data)
    {
        std::memcpy(tensor->data(), _data, tensor->getSizeInBytes());
    }
}

size_t NamedTensor::serializedSize() const
{
    size_t totalSize = 1;

    int n = (name.size() + sizeof(int64_t)) / sizeof(int64_t);
    totalSize += n;

    // memType
    // dataType
    // nbDims
    totalSize += 3;
    totalSize += tensor->getShape().nbDims;

    int m = tensor->getSizeInBytes();
    int mm = (m + sizeof(int64_t) - 1) / sizeof(int64_t);
    totalSize += mm;
    return totalSize;
}

void NamedTensor::serialize(int64_t* vpacked, const size_t totalSize) const
{
    int n = (name.size() + sizeof(int64_t)) / sizeof(int64_t);

    int m = tensor->getSizeInBytes();

    vpacked[0] = name.size();
    std::memcpy(&(vpacked[1]), name.c_str(), name.size());

    int64_t* tensorPtr = &(vpacked[n + 1]);
    *tensorPtr++ = static_cast<int64_t>(tensor->getMemoryType());
    *tensorPtr++ = static_cast<int64_t>(tensor->getDataType());
    *tensorPtr++ = static_cast<int64_t>(tensor->getShape().nbDims);
    for (size_t i = 0; i < static_cast<size_t>(tensor->getShape().nbDims); ++i)
    {
        *tensorPtr++ = static_cast<int64_t>(tensor->getShape().d[i]);
    }
    std::memcpy(tensorPtr, tensor->data(), m);

    tensorPtr += (m + sizeof(int64_t) - 1) / sizeof(int64_t);

    TLLM_CHECK_WITH_INFO(tensorPtr - vpacked == (int64_t) totalSize, "serialize and serializedSize are out of sync");
}

std::vector<int64_t> NamedTensor::serialize() const
{
    size_t totalSize = serializedSize();

    std::vector<int64_t> vpacked(totalSize);
    serialize(vpacked.data(), totalSize);

    return vpacked;
}

NamedTensor NamedTensor::deserialize(int64_t const* packed)
{
    int n = *packed++;
    char const* cname = reinterpret_cast<char const*>(packed);
    int nn = (n + sizeof(int64_t)) / sizeof(int64_t);
    packed += nn;
    ++packed; // tensorrt_llm::runtime::MemoryType
    nvinfer1::DataType trtDType = static_cast<nvinfer1::DataType>(*packed++);
    int64_t nshape = *packed++;
    std::vector<int64_t> shape(nshape);
    memcpy(shape.data(), packed, nshape * sizeof(int64_t));
    packed += nshape;
    return NamedTensor{trtDType, shape, cname, packed};
}

} // namespace tensorrt_llm::batch_manager
