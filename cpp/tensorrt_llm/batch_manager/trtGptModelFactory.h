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

#pragma once

#include "tensorrt_llm/batch_manager/BatchManager.h"

#include "tensorrt_llm/runtime/gptJsonConfig.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/rawEngine.h"
#include "tensorrt_llm/runtime/tllmLogger.h"
#include "tensorrt_llm/runtime/worldConfig.h"
#include "trtGptModelInflightBatching.h"
#include "trtGptModelV1.h"

#include <NvInferPlugin.h>

#include <memory>
#include <optional>

namespace tensorrt_llm::batch_manager
{

class TrtGptModelFactory
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;

    static std::shared_ptr<TrtGptModel> create(std::filesystem::path const& trtEnginePath, TrtGptModelType modelType,
        TrtGptModelOptionalParams const& optionalParams = TrtGptModelOptionalParams())
    {
        auto const jsonConfig = runtime::GptJsonConfig::parse(trtEnginePath / "config.json");
        auto worldConfig = getWorldConfig(jsonConfig, optionalParams.deviceIds);
        auto const enginePath = trtEnginePath / jsonConfig.engineFilename(worldConfig);

        auto const& modelConfig = jsonConfig.getModelConfig();
        return create(runtime::RawEngine(enginePath), modelConfig, worldConfig, modelType, optionalParams);
    }

    static std::shared_ptr<TrtGptModel> create(std::filesystem::path const& trtEnginePath, TrtGptModelType modelType,
        runtime::GptJsonConfig const& jsonConfig, runtime::WorldConfig const& worldConfig,
        TrtGptModelOptionalParams const& optionalParams = TrtGptModelOptionalParams())
    {
        auto const enginePath = trtEnginePath / jsonConfig.engineFilename(worldConfig);
        auto const& modelConfig = jsonConfig.getModelConfig();
        return create(runtime::RawEngine(enginePath), modelConfig, worldConfig, modelType, optionalParams);
    }

    static std::shared_ptr<TrtGptModel> create(runtime::RawEngine const& rawEngine,
        runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig, TrtGptModelType modelType,
        TrtGptModelOptionalParams const& optionalParams = TrtGptModelOptionalParams())
    {
        auto logger = std::make_shared<runtime::TllmLogger>();
        auto const device = worldConfig.getDevice();
        auto const rank = worldConfig.getRank();
        TLLM_LOG_INFO("Rank %d is using GPU %d", rank, device);
        TLLM_CUDA_CHECK(cudaSetDevice(device));

        if (modelType == TrtGptModelType::V1)
        {
            TLLM_LOG_WARNING(
                "TrtGptModelType::V1 is deprecated and will be removed in a future release."
                " Please use TrtGptModelType::InflightBatching or TrtGptModelType::InflightFusedBatching instead.");

            TrtGptModelOptionalParams const& fixedOptionalParams
                = TrtGptModelV1::optionalParamsAreValid(modelConfig, optionalParams)
                ? optionalParams
                : TrtGptModelV1::fixOptionalParams(modelConfig, optionalParams);
            return std::make_shared<TrtGptModelV1>(logger, modelConfig, worldConfig, rawEngine, fixedOptionalParams);
        }
        else if ((modelType == TrtGptModelType::InflightBatching)
            || (modelType == TrtGptModelType::InflightFusedBatching))
        {
            TrtGptModelOptionalParams const& fixedOptionalParams
                = TrtGptModelInflightBatching::optionalParamsAreValid(modelConfig, optionalParams)
                ? optionalParams
                : TrtGptModelInflightBatching::fixOptionalParams(modelConfig, optionalParams);
            return std::make_shared<TrtGptModelInflightBatching>(logger, modelConfig, worldConfig, rawEngine,
                (modelType == TrtGptModelType::InflightFusedBatching), fixedOptionalParams);
        }
        else
        {
            throw std::runtime_error("Invalid modelType in trtGptModelFactory");
        }
    }

private:
    static runtime::WorldConfig getWorldConfig(
        runtime::GptJsonConfig const& json, std::optional<std::vector<SizeType32>> const& deviceIds)
    {
        return runtime::WorldConfig::mpi(json.getGpusPerNode(), json.getTensorParallelism(),
            json.getPipelineParallelism(), json.getContextParallelism(), deviceIds);
    }
};

} // namespace tensorrt_llm::batch_manager
