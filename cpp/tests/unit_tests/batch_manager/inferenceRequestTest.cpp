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
#include "tensorrt_llm/batch_manager/inferenceRequest.h"
#include <gmock/gmock.h>

using namespace tensorrt_llm::batch_manager;

class InferenceRequestTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
protected:
    void SetUp() override {}

    void TearDown() override
    {
        ;
    }
};

TEST_F(InferenceRequestTest, SerializeInt32)
{
    SerializeDeserializeTestUtils<int32_t>::TestInferenceRequests(12345, false, nvinfer1::DataType::kINT32);
    SerializeDeserializeTestUtils<int32_t>::TestInferenceRequests(54321, false, nvinfer1::DataType::kINT32);
    SerializeDeserializeTestUtils<int32_t>::TestInferenceRequests(12345, true, nvinfer1::DataType::kINT32);
    SerializeDeserializeTestUtils<int32_t>::TestInferenceRequests(54321, true, nvinfer1::DataType::kINT32);
}

TEST_F(InferenceRequestTest, SerializeInt8)
{
    SerializeDeserializeTestUtils<int8_t>::TestInferenceRequests(12345, false, nvinfer1::DataType::kINT8);
    SerializeDeserializeTestUtils<int8_t>::TestInferenceRequests(54321, false, nvinfer1::DataType::kINT8);
    SerializeDeserializeTestUtils<int8_t>::TestInferenceRequests(12345, true, nvinfer1::DataType::kINT8);
    SerializeDeserializeTestUtils<int8_t>::TestInferenceRequests(54321, true, nvinfer1::DataType::kINT8);
}

TEST_F(InferenceRequestTest, SerializeUInt8)
{
    SerializeDeserializeTestUtils<uint8_t>::TestInferenceRequests(12345, false, nvinfer1::DataType::kUINT8);
    SerializeDeserializeTestUtils<uint8_t>::TestInferenceRequests(54321, false, nvinfer1::DataType::kUINT8);
    SerializeDeserializeTestUtils<uint8_t>::TestInferenceRequests(12345, true, nvinfer1::DataType::kUINT8);
    SerializeDeserializeTestUtils<uint8_t>::TestInferenceRequests(54321, true, nvinfer1::DataType::kUINT8);
}

TEST_F(InferenceRequestTest, SerializeInt64)
{
    SerializeDeserializeTestUtils<uint64_t>::TestInferenceRequests(12345, false, nvinfer1::DataType::kINT64);
    SerializeDeserializeTestUtils<uint64_t>::TestInferenceRequests(54321, false, nvinfer1::DataType::kINT64);
    SerializeDeserializeTestUtils<uint64_t>::TestInferenceRequests(12345, true, nvinfer1::DataType::kINT64);
    SerializeDeserializeTestUtils<uint64_t>::TestInferenceRequests(54321, true, nvinfer1::DataType::kINT64);
}

TEST_F(InferenceRequestTest, SerializeFLOAT32)
{
    SerializeDeserializeTestUtils<float>::TestInferenceRequests(12345, false, nvinfer1::DataType::kFLOAT);
    SerializeDeserializeTestUtils<float>::TestInferenceRequests(54321, false, nvinfer1::DataType::kFLOAT);
    SerializeDeserializeTestUtils<float>::TestInferenceRequests(12345, true, nvinfer1::DataType::kFLOAT);
    SerializeDeserializeTestUtils<float>::TestInferenceRequests(54321, true, nvinfer1::DataType::kFLOAT);
}

TEST(InferenceRequestLogitsPostProcTest, SerializeWithLogitsPostProcessor)
{
    LlmRequest::LogitsPostProcessor logitsCb
        = [&](uint64_t rId, tensorrt_llm::runtime::ITensor::SharedPtr& logits, LlmRequest::BeamTokens const& tokens,
              tensorrt_llm::runtime::BufferManager::CudaStreamPtr streamPtr, std::optional<uint64_t> cId) {};

    auto request = InferenceRequest(12345);
    request.setLogitsPostProcessor(logitsCb);
    std::string errMsg = "is not supported.";
    // Below show fail since we cannot serialize a request with a logits post processor
    try
    {
        auto packed = request.serialize();
        FAIL() << "Expected failure with " << errMsg;
    }
    catch (tensorrt_llm::common::TllmException const& e)
    {
        EXPECT_THAT(e.what(), testing::HasSubstr(errMsg));
    }
    catch (std::exception const& e)
    {
        FAIL() << "Expected TllmException with " << errMsg << " got " << e.what();
    }
}
