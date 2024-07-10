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

#include <gtest/gtest.h>

#include "serializeDeserializeTestUtils.h"

#include "tensorrt_llm/batch_manager/namedTensor.h"
#include "tensorrt_llm/runtime/iTensor.h"

#include <vector>

class TensorTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
protected:
    void SetUp() override {}

    void TearDown() override
    {
        ;
    }
};

TEST_F(TensorTest, Shape)
{
    std::string input_name = "input";
    std::vector<int64_t> shapev{10, 20, 40};
    auto t = tensorrt_llm::batch_manager::NamedTensor(nvinfer1::DataType::kINT32, shapev, input_name);
    for (int i = 0; i < shapev.size(); ++i)
    {
        EXPECT_EQ(t.tensor->getShape().d[i], shapev[i]);
    }
}

TEST_F(TensorTest, SerializeInt32)
{
    SerializeDeserializeTestUtils<int32_t>::TestTensors("input", {10, 20, 40}, nvinfer1::DataType::kINT32);
}

TEST_F(TensorTest, SerializeUInt32)
{
    SerializeDeserializeTestUtils<uint32_t>::TestTensors("input", {10, 20, 40}, nvinfer1::DataType::kINT32);
}

TEST_F(TensorTest, SerializeUInt64)
{
    SerializeDeserializeTestUtils<uint64_t>::TestTensors("input", {10, 20, 40}, nvinfer1::DataType::kINT64);
}

TEST_F(TensorTest, SerializeFLOAT32)
{
    SerializeDeserializeTestUtils<float>::TestTensors("input", {10, 20, 40}, nvinfer1::DataType::kFLOAT);
}
