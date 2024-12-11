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

#include <gtest/gtest.h>

#include "tensorrt_llm/batch_manager/inferenceRequest.h"
#include "tensorrt_llm/batch_manager/namedTensor.h"

#include <list>

using namespace tensorrt_llm::batch_manager;

// use anonymous namespace to prevent dual definition error if this file is included in multiple places
namespace
{

template <typename T>
struct SerializeDeserializeTestUtils
{
    static void CompareTensors(NamedTensor const& t1, NamedTensor const& t2)
    {
        EXPECT_EQ(t1.name, t2.name);
        if (t1.tensor == nullptr)
        {
            EXPECT_EQ(t2.tensor, nullptr);
        }
        else
        {
            ASSERT_NE(t2.tensor, nullptr);
            EXPECT_EQ(t1.tensor->getSize(), t2.tensor->getSize());
            EXPECT_TRUE(t1.tensor->shapeEquals(t2.tensor->getShape()));
            auto const data1 = tensorrt_llm::runtime::bufferCast<T>(*t1.tensor);
            auto const data2 = tensorrt_llm::runtime::bufferCast<T>(*t2.tensor);
            for (std::size_t count = 0; count < t1.tensor->getSize(); ++count)
            {
                EXPECT_EQ(data1[count], data2[count]);
            }
        }
    }

    static NamedTensor CreateTensor(std::string const& name, std::vector<int64_t> const& shape, nvinfer1::DataType dt)
    {
        auto const nelems1 = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
        std::vector<T> raw_data(nelems1);
        for (int64_t count = 0; count < nelems1; ++count)
        {
            raw_data[count] = static_cast<T>(count);
        }
        return NamedTensor{dt, shape, name, raw_data.data()};
    }

    static void TestTensors(std::string const& name, std::vector<int64_t> const& shape, nvinfer1::DataType dt)
    {
        // create original tensor
        auto t1 = SerializeDeserializeTestUtils<T>::CreateTensor(name, shape, dt);

        // create copy through deserialize(t1.serialize())
        auto packed = t1.serialize();
        auto t2 = NamedTensor::deserialize(packed.data());

        SerializeDeserializeTestUtils<T>::CompareTensors(t1, t2);
    }

    static void CompareInferenceRequests(InferenceRequest& ir1, InferenceRequest& ir2, std::list<std::string>& names)
    {
        for (auto const& name : names)
        {
            SerializeDeserializeTestUtils<T>::CompareTensors(
                NamedTensor{ir1.getInputTensor(name), name}, NamedTensor{ir2.getInputTensor(name), name});
        }
        EXPECT_EQ(ir1.getRequestId(), ir2.getRequestId());
        EXPECT_EQ(ir1.isStreaming(), ir2.isStreaming());
    }

    static void TestInferenceRequests(int requestId, bool streaming, nvinfer1::DataType dt)
    {
        // create original InferenceRequest
        auto ir1 = InferenceRequest(requestId);

        std::list<std::string> names;

        std::string name1 = inference_request::kInputIdsTensorName;
        auto t1 = SerializeDeserializeTestUtils<T>::CreateTensor(name1, {10, 20, 40}, dt);
        ir1.emplaceInputTensor(std::string(name1), std::move(t1.tensor));
        names.push_back(t1.name);

        std::string name2 = inference_request::kMaxNewTokensTensorName;
        auto t2 = SerializeDeserializeTestUtils<T>::CreateTensor(name2, {6, 2}, dt);
        ir1.emplaceInputTensor(std::string(name2), std::move(t2.tensor));
        names.push_back(t2.name);

        ir1.setIsStreaming(streaming);

        // copy through deserialize(t1.serialize())
        auto packed = ir1.serialize();
        auto ir2 = InferenceRequest::deserialize(packed);

        SerializeDeserializeTestUtils<T>::CompareInferenceRequests(ir1, *ir2, names);
    }
};

} // namespace
