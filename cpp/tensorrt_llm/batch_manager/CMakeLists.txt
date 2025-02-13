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
    allocateKvCache.cpp
    assignReqSeqSlots.cpp
    cacheFormatter.cpp
    cacheTransceiver.cpp
    capacityScheduler.cpp
    createNewDecoderRequests.cpp
    contextProgress.cpp
    dataTransceiver.cpp
    decoderBuffers.cpp
    encoderBuffers.cpp
    generateRequestOptions.cpp
    GptManager.cpp
    guidedDecoder.cpp
    handleContextLogits.cpp
    handleGenerationLogits.cpp
    inferenceRequest.cpp
    kvCacheManager.cpp
    kvCacheEventManager.cpp
    kvCacheTransferManager.cpp
    kvCacheUtils.cpp
    llmRequest.cpp
    logitsPostProcessor.cpp
    loraBuffers.cpp
    makeDecodingBatchInputOutput.cpp
    medusaBuffers.cpp
    microBatchScheduler.cpp
    mpiDataTransceiver.cpp
    namedTensor.cpp
    pauseRequests.cpp
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
    utils/debugUtils.cpp
    utils/inflightBatchingUtils.cpp
    utils/logitsThread.cpp
    utils/staticThreadPool.cpp
    evictionPolicy.cpp)

file(GLOB_RECURSE XGRAMMAR_SRCS "${3RDPARTY_DIR}/xgrammar/cpp/*.cc")
list(FILTER XGRAMMAR_SRCS EXCLUDE REGEX
     "${3RDPARTY_DIR}/xgrammar/cpp/pybind/.*\\.cc")
list(APPEND SRCS ${XGRAMMAR_SRCS})

if(NOT WIN32)
  # additional warnings
  #
  # Ignore overloaded-virtual warning. We intentionally change parameters of
  # some methods in derived class.
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
target_include_directories(
  ${BATCH_MANAGER_STATIC_TARGET}
  PUBLIC ${3RDPARTY_DIR}/xgrammar/3rdparty/picojson
         ${3RDPARTY_DIR}/xgrammar/3rdparty/dlpack/include
         ${3RDPARTY_DIR}/xgrammar/include)

set_target_properties(
  ${BATCH_MANAGER_STATIC_TARGET}
  PROPERTIES CXX_STANDARD "17" CXX_STANDARD_REQUIRED "YES" CXX_EXTENSIONS "NO"
             POSITION_INDEPENDENT_CODE ON)

set_property(TARGET ${BATCH_MANAGER_STATIC_TARGET}
             PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set(TOP_LEVEL_DIR "${PROJECT_SOURCE_DIR}/..")
target_compile_definitions(${BATCH_MANAGER_STATIC_TARGET}
                           PUBLIC TOP_LEVEL_DIR="${TOP_LEVEL_DIR}")

# Build UCX data transceiver as shared library for runtime symbol loading, to
# make UCX on-demand runtime dependency.
add_library(${UCX_WRAPPER_TARGET} SHARED ucxDataTransceiver.cpp)
set_target_properties(
  ${UCX_WRAPPER_TARGET}
  PROPERTIES CXX_STANDARD "17" CXX_STANDARD_REQUIRED "YES" CXX_EXTENSIONS "NO"
             POSITION_INDEPENDENT_CODE ON)

set_property(TARGET ${UCX_WRAPPER_TARGET} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS
                                                   ON)
set(TOP_LEVEL_DIR "${PROJECT_SOURCE_DIR}/..")
target_compile_definitions(${UCX_WRAPPER_TARGET}
                           PUBLIC TOP_LEVEL_DIR="${TOP_LEVEL_DIR}")

if(ENABLE_UCX)
  find_package(ucx REQUIRED)
  find_package(ucxx REQUIRED)
  target_include_directories(
    ${UCX_WRAPPER_TARGET}
    PRIVATE $<TARGET_PROPERTY:ucxx::ucxx,INTERFACE_INCLUDE_DIRECTORIES>)
  # link_whole_archive
  if(WIN32)
    target_link_libraries(${UCX_WRAPPER_TARGET}
                          PUBLIC $<TARGET_FILE:ucxx::ucxx>)
    set_target_properties(${UCX_WRAPPER_TARGET}
                          PROPERTIES LINK_FLAGS "/WHOLEARCHIVE:ucxx::ucxx")
  else()
    # Assume everything else is like gcc
    target_link_libraries(
      ${UCX_WRAPPER_TARGET}
      PRIVATE "-Wl,--whole-archive" $<TARGET_FILE:ucxx::ucxx>
              "-Wl,--no-whole-archive")
  endif()
  target_link_libraries(${UCX_WRAPPER_TARGET} PUBLIC ucxx::ucxx ucx::ucs)
  target_link_libraries(${UCX_WRAPPER_TARGET} PUBLIC ${CUDA_RT_LIB})
endif()
