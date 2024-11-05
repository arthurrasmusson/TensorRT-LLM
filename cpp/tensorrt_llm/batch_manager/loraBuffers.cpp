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

#include "loraBuffers.h"

#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/runtime/loraUtils.h"

namespace tensorrt_llm::batch_manager
{

void LoraBuffers::create(SizeType32 maxBatchSize, SizeType32 maxBeamWidth, runtime::TllmRuntime const& tllmRuntime,
    runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig)
{
    auto const localNbLayers = modelConfig.getNbAttentionLayers(worldConfig.getPipelineParallelism());
    auto const firstLayerId = worldConfig.getPipelineParallelRank() * localNbLayers;

    auto nbModelConfigs = static_cast<SizeType32>(modelConfig.getLoraModules().size());

    auto loraWeightsPtrsShape
        = runtime::ITensor::makeShape({nbModelConfigs, localNbLayers, maxBatchSize * maxBeamWidth, 2});
    auto loraAdapterSizesShape
        = runtime::ITensor::makeShape({nbModelConfigs, localNbLayers, maxBatchSize * maxBeamWidth});

    auto firstModuleName = std::string(modelConfig.getLoraModules().front().name());
    auto ptrsFieldName = firstModuleName + "_lora_weights_pointers_" + std::to_string(firstLayerId);
    auto rankFieldName = firstModuleName + "_lora_ranks_" + std::to_string(firstLayerId);
    auto weightsPtrDtype = tllmRuntime.getEngine().getTensorDataType(ptrsFieldName.c_str());
    auto ranksDtype = tllmRuntime.getEngine().getTensorDataType(rankFieldName.c_str());

    mLoraManager.create(modelConfig);

    mLoraWeightsPointersHost = runtime::BufferManager::pinned(loraWeightsPtrsShape, weightsPtrDtype);
    mLoraAdapterSizesHost = runtime::BufferManager::pinned(loraAdapterSizesShape, ranksDtype);
}

void LoraBuffers::fill(RequestVector const& contextRequests, RequestVector const& genRequests,
    PeftTable const& peftTable, runtime::BufferManager const& manager, runtime::ModelConfig const& modelConfig,
    runtime::WorldConfig const& worldConfig)
{
    manager.setZero(*mLoraWeightsPointersHost);
    manager.setZero(*mLoraAdapterSizesHost);

    SizeType32 batchIdx{0};
    for (auto const& requests : {contextRequests, genRequests})
    {
        for (auto const& llmReq : requests)
        {
            auto const optReqLoraWeights = llmReq->getLoraWeights();
            auto const optReqLoraConfig = llmReq->getLoraConfig();

            auto const isContextRequest = llmReq->isContextInitState();
            auto const beamWidth = isContextRequest ? 1 : llmReq->mSamplingConfig.beamWidth;
            auto const peftIt = peftTable.find(llmReq->mRequestId);
            if (peftIt != peftTable.end())
            {
                auto const& peftValues = peftIt->second;
                if (!peftValues.empty())
                {
                    mLoraManager.fillInputTensors(mLoraWeightsPointersHost, mLoraAdapterSizesHost, peftIt->second,
                        batchIdx, beamWidth, modelConfig, worldConfig);
                }
            }
            ++batchIdx;
        }
    }
}

void LoraBuffers::validate(std::optional<std::uint64_t> const& optTaskId,
    std::optional<TensorPtr> const& optReqLoraWeights, std::optional<TensorPtr> const& optReqLoraConfig,
    runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig)
{
    runtime::lora::loraValidateRequestTensors(optTaskId, optReqLoraWeights, optReqLoraConfig, modelConfig, worldConfig);
}

void LoraBuffers::insertInputTensors(TensorMap& inputTensors, TensorPtr weightsPtrs, TensorPtr adapterSizes,
    runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig) const
{
    mLoraManager.insertInputTensors(inputTensors, weightsPtrs, adapterSizes, modelConfig, worldConfig);
}

void LoraBuffers::reshape(SizeType32 numSequences)
{
    auto weightsPtrsShape = mLoraWeightsPointersHost->getShape();
    weightsPtrsShape.d[2] = numSequences;
    mLoraWeightsPointersHost->reshape(weightsPtrsShape);

    auto adapterSizesShape = mLoraAdapterSizesHost->getShape();
    adapterSizesShape.d[2] = numSequences;
    mLoraAdapterSizesHost->reshape(adapterSizesShape);
}

} // namespace tensorrt_llm::batch_manager
