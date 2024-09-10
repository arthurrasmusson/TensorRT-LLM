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
set(BATCH_MANAGER_TARGET_NAME tensorrt_llm_batch_manager)
set(BATCH_MANAGER_STATIC_TARGET ${BATCH_MANAGER_TARGET_NAME}_static)

set(TARGET_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# keep this list sorted alphabetically
set(SRCS
    capacityScheduler.cpp
    dataTransceiver.cpp
    decoderBuffers.cpp
    encoderBuffers.cpp
    GptManager.cpp
    kvCacheUtils.cpp
    mpiDataTransceiver.cpp
    inferenceRequest.cpp
    kvCacheManager.cpp
    kvCacheUtils.cpp
    loraBuffers.cpp
    medusaBuffers.cpp
    microBatchScheduler.cpp
    mpiDataTransceiver.cpp
    namedTensor.cpp
    peftCacheManager.cpp
    promptTuningBuffers.cpp
    rnnStateBuffers.cpp
    rnnStateManager.cpp
    runtimeBuffers.cpp
    sequenceSlotManager.cpp
    transformerBuffers.cpp
    trtEncoderModel.cpp
    trtGptModelInflightBatching.cpp
    trtGptModelV1.cpp
    debugUtils.cpp
    inflightBatchingUtils.cpp)

if(ENABLE_UCX)
  set(SRCS ${SRCS} ucxDataTransceiver.cpp)
endif()

if(NOT WIN32)
  # additional warnings
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
  if(WARNING_IS_ERROR)
    message(STATUS "Treating warnings as errors in GCC compilation")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
  endif()
else() # Windows
  # warning level 4
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
endif()

add_library(${BATCH_MANAGER_STATIC_TARGET} STATIC ${SRCS})
set_target_properties(
  ${BATCH_MANAGER_STATIC_TARGET}
  PROPERTIES CXX_STANDARD "17" CXX_STANDARD_REQUIRED "YES" CXX_EXTENSIONS "NO"
             POSITION_INDEPENDENT_CODE ON)

set_property(TARGET ${BATCH_MANAGER_STATIC_TARGET}
             PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set(TOP_LEVEL_DIR "${PROJECT_SOURCE_DIR}/..")
target_compile_definitions(${BATCH_MANAGER_STATIC_TARGET}
                           PUBLIC TOP_LEVEL_DIR="${TOP_LEVEL_DIR}")
