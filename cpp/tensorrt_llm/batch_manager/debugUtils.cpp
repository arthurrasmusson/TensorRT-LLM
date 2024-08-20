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

#include "tensorrt_llm/batch_manager/debugUtils.h"
#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"
#include "tensorrt_llm/runtime/utils/numpyUtils.h"

#include <filesystem>
#include <string>

namespace fs = std::filesystem;

namespace tensorrt_llm::batch_manager::utils
{
using executor::IterationType;
using runtime::ITensor;
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

void dumpRequestIds(fs::path const& outputPath, RequestVector const& contextRequests,
    RequestVector const& generationRequests, BufferManager const& manager)
{
    // Collect request IDs
    auto const numRequests = static_cast<ITensor::DimType64>(contextRequests.size() + generationRequests.size());
    auto requestIds
        = runtime::BufferManager::cpu(ITensor::makeShape({numRequests}), runtime::TRTDataType<RequestIdType>::value);
    auto requestIdsRange = runtime::BufferRange<RequestIdType>(*requestIds);
    auto batchIdx{0};
    for (auto const& requests : {contextRequests, generationRequests})
    {
        for (auto const& request : requests)
        {
            requestIdsRange[batchIdx++] = request->mRequestId;
        }
    }

    dumpTensor(outputPath, std::string("request_ids"), *requestIds, manager);
}

} // namespace

void dumpRequestIds(IterationType iterCounter, RequestVector const& contextRequests,
    RequestVector const& generationRequests, runtime::WorldConfig const& worldConfig,
    std::shared_ptr<runtime::TllmRuntime> const& runtime)
{
    auto const& manager = runtime->getBufferManager();
    auto const outputPath = getOutputPath(iterCounter, worldConfig);
    dumpRequestIds(outputPath, contextRequests, generationRequests, manager);
}

void dumpTensors(IterationType iterCounter, TensorMap const& tensorMap, runtime::WorldConfig const& worldConfig,
    std::shared_ptr<runtime::TllmRuntime> const& runtime)
{
    auto const& manager = runtime->getBufferManager();

    auto const outputPath = getOutputPath(iterCounter, worldConfig);

    for (auto const& [tensorName, tensor] : tensorMap)
    {
        dumpTensor(outputPath, tensorName, *tensor, manager);
    }
}

void dumpDebugTensors(IterationType iterCounter, std::vector<std::string> const& debugTensorNames,
    TensorMap const& inputMap, TensorMap const& outputMap, runtime::WorldConfig const& worldConfig,
    std::shared_ptr<runtime::TllmRuntime> const& runtime)

{
    auto const& manager = runtime->getBufferManager();

    auto const outputPath = getOutputPath(iterCounter, worldConfig);

    for (auto const& debugTensorName : debugTensorNames)
    {
        auto foundTensor = false;
        for (auto const& tensorMap : {inputMap, outputMap})
        {
            auto tensorIt = tensorMap.find(debugTensorName);
            if (tensorIt != tensorMap.end())
            {
                auto const& [tensorName, tensor] = *tensorIt;
                dumpTensor(outputPath, tensorName, *tensor, manager);
                foundTensor = true;
            }
        }
        if (!foundTensor)
        {
            TLLM_LOG_WARNING("Debug tensor with name '%s' not found", debugTensorName.c_str());
        }
    }
}

} // namespace tensorrt_llm::batch_manager::utils
