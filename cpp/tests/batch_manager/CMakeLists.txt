# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: NVIDIA TensorRT
# Source Code License Agreement
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this material and related documentation without an express
# license agreement from NVIDIA CORPORATION or its affiliates is strictly
# prohibited.
if(${BUILD_PYT})
  add_gtest(gptManagerTest gptManagerTest.cpp)
  target_link_libraries(gptManagerTest PUBLIC ${TORCH_LIBRARIES})
endif()
add_gtest(batchManagerTensorTest tensorTest.cpp)
add_gtest(batchManagerInferenceRequestTest inferenceRequestTest.cpp)
add_gtest(trtGptModelTest trtGptModelTest.cpp)
add_gtest(trtGptModelRealDecoderTest trtGptModelRealDecoderTest.cpp)
add_gtest(kvCacheManagerTest kvCacheManagerTest.cpp)
add_gtest(capacitySchedulerTest capacitySchedulerTest.cpp)
add_gtest(microBatchSchedulerTest microBatchSchedulerTest.cpp)
add_gtest(llmRequestTest llmRequestTest.cpp)
add_gtest(peftCacheManagerTest peftCacheManagerTest.cpp)
add_gtest(trtEncoderModelTest trtEncoderModelTest.cpp)
