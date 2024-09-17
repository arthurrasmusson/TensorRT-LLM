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

#include "debugUtils.h"

#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/runtime/utils/numpyUtils.h"

#include <filesystem>
#include <string>

namespace fs = std::filesystem;

namespace tensorrt_llm::batch_manager::utils
{
using executor::IterationType;
using runtime::ITensor;
using TensorPtr = runtime::ITensor::SharedPtr;
using TensorMap = runtime::ITensor::TensorMap;
using runtime::BufferManager;

namespace
{

fs::path getOutputPath(IterationType iterCounter, runtime::WorldConfig const& worldConfig)
{
    auto tmpPath = fs::temp_directory_path();
    return tmpPath / "tllm_debug"                                        //
        / ("PP_" + std::to_string(worldConfig.getPipelineParallelism())) //
        / ("TP_" + std::to_string(worldConfig.getTensorParallelism()))   //
        / ("iteration_" + std::to_string(iterCounter));
}

void dumpTensor(
    fs::path const& outputPath, std::string const& tensorName, ITensor const& tensor, BufferManager const& manager)
{
    fs::create_directories(outputPath);
    auto const outputFile = outputPath / (tensorName + ".npy");
    TLLM_LOG_INFO("Saving tensor '%s' to '%s'", tensorName.c_str(), outputFile.c_str());
    runtime::utils::saveNpy(manager, tensor, outputFile.string());
}

template <class TensorConsumer>
void forEachDebugTensor(std::vector<std::string> const& debugTensorNames, TensorMap const& inputMap,
    TensorMap const& outputMap, TensorConsumer tensorConsumer)
{
    for (auto const& debugTensorName : debugTensorNames)
    {
        auto foundTensor = false;
        for (auto const& tensorMap : {inputMap, outputMap})
        {
            auto tensorIt = tensorMap.find(debugTensorName);
            if (tensorIt != tensorMap.end())
            {
                auto const& [tensorName, tensor] = *tensorIt;
                tensorConsumer(tensorName, tensor);
                foundTensor = true;
            }
        }
        if (!foundTensor)
        {
            TLLM_LOG_WARNING("Debug tensor with name '%s' not found", debugTensorName.c_str());
        }
    }
}

template <class TensorConsumer>
void forEachTensor(executor::DebugConfig const& debugConfig, TensorPtr const& requestIds, TensorMap const& inputMap,
    TensorMap const& outputMap, TensorConsumer tensorConsumer)
{
    tensorConsumer(std::string("request_ids"), requestIds);

    if (debugConfig.getDebugTensorNames().empty())
    {
        if (debugConfig.getDebugInputTensors())
        {
            for (auto const& [tensorName, tensor] : inputMap)
            {
                tensorConsumer(tensorName, tensor);
            }
        }
        if (debugConfig.getDebugOutputTensors())
        {
            for (auto const& [tensorName, tensor] : outputMap)
            {
                tensorConsumer(tensorName, tensor);
            }
        }
    }
    else
    {
        forEachDebugTensor(debugConfig.getDebugTensorNames(), inputMap, outputMap, tensorConsumer);
    }
}

} // namespace

void dumpTensor(IterationType iterCounter, std::string const& tensorName, ITensor const& tensor,
    runtime::WorldConfig const& worldConfig, BufferManager const& manager)
{
    auto const outputPath = getOutputPath(iterCounter, worldConfig);
    dumpTensor(outputPath, tensorName, tensor, manager);
}

void dumpTensors(IterationType iterCounter, TensorMap const& tensorMap, runtime::WorldConfig const& worldConfig,
    BufferManager const& manager)
{
    auto const outputPath = getOutputPath(iterCounter, worldConfig);

    for (auto const& [tensorName, tensor] : tensorMap)
    {
        dumpTensor(outputPath, tensorName, *tensor, manager);
    }
}

void dumpDebugTensors(IterationType iterCounter, std::vector<std::string> const& debugTensorNames,
    TensorMap const& inputMap, TensorMap const& outputMap, runtime::WorldConfig const& worldConfig,
    BufferManager const& manager)
{
    auto dumpTensorFunc = [outputPath = getOutputPath(iterCounter, worldConfig), &manager](
                              std::string const& tensorName, TensorPtr const& tensor)
    { dumpTensor(outputPath, tensorName, *tensor, manager); };

    forEachDebugTensor(debugTensorNames, inputMap, outputMap, dumpTensorFunc);
}

void dumpIOTensors(executor::DebugConfig const& debugConfig, IterationType iterCounter, TensorPtr const& requestIds,
    TensorMap const& inputMap, TensorMap const& outputMap, runtime::WorldConfig const& worldConfig,
    BufferManager const& manager)
{
    auto dumpTensorFunc = [outputPath = getOutputPath(iterCounter, worldConfig), &manager](
                              std::string const& tensorName, TensorPtr const& tensor)
    { dumpTensor(outputPath, tensorName, *tensor, manager); };

    forEachTensor(debugConfig, requestIds, inputMap, outputMap, dumpTensorFunc);
}

TensorMap storeIOTensors(executor::DebugConfig const& debugConfig, TensorPtr const& requestIds,
    TensorMap const& inputMap, TensorMap const& outputMap, BufferManager const& manager)
{
    TensorMap tensors;

    auto storeTensor = [&tensors, &manager](std::string const& tensorName, TensorPtr const& tensor)
    {
        TensorPtr tensorCopy = manager.copyFrom(*tensor, tensor->getMemoryType());
        tensors.emplace(tensorName, tensorCopy);
    };

    forEachTensor(debugConfig, requestIds, inputMap, outputMap, storeTensor);

    return tensors;
}

} // namespace tensorrt_llm::batch_manager::utils
