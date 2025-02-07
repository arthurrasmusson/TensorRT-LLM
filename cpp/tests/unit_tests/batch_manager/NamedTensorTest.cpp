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

#include <vector>

#include "tensorrt_llm/batch_manager/namedTensor.h"
#include "tests/unit_tests/batch_manager/serializeDeserializeTestUtils.h"

using namespace tensorrt_llm::batch_manager;

class NamedTensorTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
protected:
    void SetUp() override {}

    void TearDown() override
    {
        ;
    }
};

TEST_F(NamedTensorTest, SerializeDeserialize)
{
    std::vector<float> v(60);
    NamedTensor nt(nvinfer1::DataType::kFLOAT, {3, 4, 5}, "t1", v.data());
    std::vector<int64_t> s = nt.serialize();
    NamedTensor nt2 = NamedTensor::deserialize(s.data());
    SerializeDeserializeTestUtils<float>::CompareTensors(nt, nt2);
}
