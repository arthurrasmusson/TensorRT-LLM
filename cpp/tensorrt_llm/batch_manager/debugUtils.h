/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/worldConfig.h"

namespace tensorrt_llm::runtime
{
class TllmRuntime;
} // namespace tensorrt_llm::runtime

namespace tensorrt_llm::batch_manager::utils
{

/// @brief Dump the list of request ids to an npy file.
/// @details Files are written to a temp directory determined by `std::filesystem::temp_directory_path`,
/// the exact paths are logged at INFO level.
void dumpRequestIds(executor::IterationType iterCounter, RequestVector const& contextRequests,
    RequestVector const& generationRequests, runtime::WorldConfig const& worldConfig,
    std::shared_ptr<runtime::TllmRuntime> const& runtime);

/// @brief Dump each tensor in the `tensorMap` to an npy file.
/// @details Files are written to a temp directory determined by `std::filesystem::temp_directory_path`,
/// the exact paths are logged at INFO level.
void dumpTensors(executor::IterationType iterCounter, runtime::ITensor::TensorMap const& tensorMap,
    runtime::WorldConfig const& worldConfig, std::shared_ptr<runtime::TllmRuntime> const& runtime);

/// @brief Dump each tensor in `debugTensorNames` to an npy file if it is found in one of the tensor maps.
/// @details Files are written to a temp directory determined by `std::filesystem::temp_directory_path`,
/// the exact paths are logged at INFO level.
void dumpDebugTensors(executor::IterationType iterCounter, std::vector<std::string> const& debugTensorNames,
    runtime::ITensor::TensorMap const& inputMap, runtime::ITensor::TensorMap const& outputMap,
    runtime::WorldConfig const& worldConfig, std::shared_ptr<runtime::TllmRuntime> const& runtime);

} // namespace tensorrt_llm::batch_manager::utils
