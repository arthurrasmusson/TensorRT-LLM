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

if(NOT Python3_EXECUTABLE)
  find_package(
    Python3
    COMPONENTS Interpreter
    REQUIRED)
endif()

execute_process(
  COMMAND ${Python3_EXECUTABLE} "-c"
          "import pybind11 as pb11; print(pb11.get_cmake_dir(),end='');"
  RESULT_VARIABLE PYBIND_CMAKE_DIR_RET
  OUTPUT_VARIABLE PYBIND_CMAKE_DIR)

if(PYBIND_CMAKE_DIR_RET MATCHES 0)
  list(APPEND CMAKE_PREFIX_PATH "${PYBIND_CMAKE_DIR}")
else()
  message(ERROR "pybind11 CMake directory not found.")
endif()

find_package(pybind11 REQUIRED)
add_library(modelSpec SHARED modelSpec.cpp modelSpecBinding.cpp)
set_target_properties(modelSpec PROPERTIES PREFIX "")
set_target_properties(modelSpec PROPERTIES OUTPUT_NAME "model_spec")
target_link_libraries(modelSpec PRIVATE tensorrt_llm)

# Static lib for other test to link
add_library(modelSpecStatic STATIC modelSpec.cpp)

if(${BUILD_PYT})
  add_gtest(gptManagerTest gptManagerTest.cpp)
  target_link_libraries(gptManagerTest PUBLIC ${TORCH_LIBRARIES})
  target_link_libraries(gptManagerTest PRIVATE modelSpecStatic)
endif()
add_gtest(batchManagerTensorTest tensorTest.cpp)
add_gtest(batchManagerInferenceRequestTest inferenceRequestTest.cpp)
add_gtest(cacheTransceiverTest cacheTransceiverTest.cpp)
add_gtest(trtGptModelTest trtGptModelTest.cpp)
target_link_libraries(trtGptModelTest PRIVATE modelSpecStatic)
add_gtest(trtGptModelRealDecoderTest trtGptModelRealDecoderTest.cpp)
target_link_libraries(trtGptModelRealDecoderTest PRIVATE modelSpecStatic
                                                         testingUtils)
add_gtest(kvCacheManagerTest kvCacheManagerTest.cpp)
add_gtest(kvCacheUtilsTest kvCacheUtilsTest.cpp)
add_gtest(capacitySchedulerTest capacitySchedulerTest.cpp)
add_gtest(microBatchSchedulerTest microBatchSchedulerTest.cpp)
add_gtest(llmRequestTest llmRequestTest.cpp)
add_gtest(peftCacheManagerTest peftCacheManagerTest.cpp)
add_gtest(staticThreadPoolTest staticThreadPoolTest.cpp)
add_gtest(trtEncoderModelTest trtEncoderModelTest.cpp)
if(ENABLE_UCX)
  add_gtest(ucxDataTransceiverTest ucxDataTransceiverTest.cpp)
endif()
