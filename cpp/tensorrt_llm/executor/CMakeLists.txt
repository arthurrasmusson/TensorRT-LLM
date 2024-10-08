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
set(EXECUTOR_TARGET_NAME tensorrt_llm_executor)
set(EXECUTOR_STATIC_TARGET ${EXECUTOR_TARGET_NAME}_static)

set(TARGET_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# keep this list sorted alphabetically
set(SRCS
    contextPhaseParams.cpp
    debugConfig.cpp
    decodingConfig.cpp
    executor.cpp
    executorConfig.cpp
    executorImpl.cpp
    extendedRuntimePerfKnobConfig.cpp
    jsonSerialization.cpp
    kvCacheConfig.cpp
    logitsPostProcessorConfig.cpp
    loraConfig.cpp
    orchestratorConfig.cpp
    outputConfig.cpp
    parallelConfig.cpp
    peftCacheConfig.cpp
    promptTuningConfig.cpp
    request.cpp
    requestUtils.cpp
    requestWithId.cpp
    response.cpp
    samplingConfig.cpp
    schedulerConfig.cpp
    serialization.cpp
    speculativeDecodingConfig.cpp
    tensor.cpp
    types.cpp
    requestUtils.cpp
    contextPhaseParams.cpp
    cache_transmission/cacheConcatenate.cu)

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

add_library(${EXECUTOR_STATIC_TARGET} STATIC ${SRCS})
set_target_properties(
  ${EXECUTOR_STATIC_TARGET}
  PROPERTIES CXX_STANDARD "17" CXX_STANDARD_REQUIRED "YES" CXX_EXTENSIONS "NO"
             POSITION_INDEPENDENT_CODE ON)

set_property(TARGET ${EXECUTOR_STATIC_TARGET}
             PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set(TOP_LEVEL_DIR "${PROJECT_SOURCE_DIR}/..")
target_compile_definitions(${EXECUTOR_STATIC_TARGET}
                           PUBLIC TOP_LEVEL_DIR="${TOP_LEVEL_DIR}")

target_compile_options(${EXECUTOR_STATIC_TARGET}
                       PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
