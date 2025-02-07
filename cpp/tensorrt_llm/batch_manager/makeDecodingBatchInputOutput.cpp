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

#include "tensorrt_llm/batch_manager/makeDecodingBatchInputOutput.h"
#include "tensorrt_llm/batch_manager/decoderBuffers.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/batch_manager/runtimeBuffers.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iGptDecoderBatched.h"
#include "tensorrt_llm/runtime/iTensor.h"

namespace tr = tensorrt_llm::runtime;

namespace tensorrt_llm::batch_manager
{

std::tuple<std::unique_ptr<tr::decoder_batch::Input>, std::unique_ptr<tr::decoder_batch::Output>>
MakeDecodingBatchInputOutput::operator()(RequestVector const& contextRequests, RequestVector const& generationRequests,
    DecoderBuffers& decoderBuffers, RuntimeBuffers const& genRuntimeBuffers, executor::DecodingMode const& decodingMode,
    runtime::ModelConfig const& modelConfig, SizeType32 maxNumSequences) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const active = computeActiveVec(contextRequests, generationRequests, maxNumSequences);
    auto decodingInput = std::make_unique<tr::decoder_batch::Input>(decoderBuffers.logits, active);

    decodingInput->cacheIndirection = decoderBuffers.cacheIndirectionInput;
    if (decodingMode == executor::DecodingMode::BeamSearch())
    {
        auto const scheduledRequestsSize = contextRequests.size() + generationRequests.size();
        decodingInput->seqSlots = tr::BufferManager::pinnedPool(
            tr::ITensor::makeShape({static_cast<tr::ITensor::DimType64>(scheduledRequestsSize)}),
            tr::TRTDataType<SizeType32>::value);
    }

    if (modelConfig.getSpeculativeDecodingMode().hasDraftLogits())
    {
        decodingInput->predictedDraftLogits = decoderBuffers.draftBuffers.predictedDraftLogits;
    }

    if (modelConfig.getSpeculativeDecodingMode().isExplicitDraftTokens())
    {
        // requires mCtxGenFusion == true
        decodingInput->seqSlots = genRuntimeBuffers.seqSlots;
        decodingInput->explicitDraftTokensInputs = genRuntimeBuffers.explicitDraftTokensBuffers->engineOutputs;
        decodingInput->explicitDraftTokensLastInputs = genRuntimeBuffers.explicitDraftTokensBuffers->engineInputs;
    }
    else if (modelConfig.getSpeculativeDecodingMode().isEagle())
    {
        // requires mCtxGenFusion == true
        decodingInput->seqSlots = genRuntimeBuffers.seqSlots;
        decodingInput->eagleInputs = genRuntimeBuffers.eagleBuffers->engineOutputs;
        decodingInput->eagleLastInputs = genRuntimeBuffers.eagleBuffers->engineInputs;
    }

    auto decodingOutput = std::make_unique<tr::decoder_batch::Output>();
    decodingOutput->cacheIndirection = decoderBuffers.cacheIndirectionOutput;
    decodingOutput->sequenceLengths = decoderBuffers.sequenceLengths;

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return {std::move(decodingInput), std::move(decodingOutput)};
}

std::vector<bool> MakeDecodingBatchInputOutput::computeActiveVec(
    RequestVector const& contextRequests, RequestVector const& generationRequests, SizeType32 maxNumSequences) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    std::vector<bool> active(maxNumSequences, false);
    for (auto const& requests : {contextRequests, generationRequests})
    {
        for (auto const& llmReq : requests)
        {
            auto const seqSlot = llmReq->mSeqSlot.value();
            if (llmReq->isGenerationInProgressState() || llmReq->isLastContextChunk())
            {
                active[seqSlot] = true;
            }
        }
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return active;
}

} // namespace tensorrt_llm::batch_manager
