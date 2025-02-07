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

#include "tensorrt_llm/batch_manager/handleGenerationLogits.h"

#include "tensorrt_llm/batch_manager/decoderBuffers.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/batch_manager/medusaBuffers.h"
#include "tensorrt_llm/batch_manager/runtimeBuffers.h"
#include "tensorrt_llm/batch_manager/utils/inflightBatchingUtils.h"
#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"
#include "tensorrt_llm/runtime/utils/debugUtils.h"

namespace tru = tensorrt_llm::runtime::utils;

namespace tensorrt_llm::batch_manager
{

using BufferManager = tensorrt_llm::runtime::BufferManager;
using TensorPtr = runtime::ITensor::SharedPtr;
using ITensor = runtime::ITensor;
using SizeType32 = tensorrt_llm::runtime::SizeType32;

namespace
{

//! @brief Copy logits from generation phase under streaming mode.
void copyStreamingGenerationLogits(BufferManager const& bufferManager, LlmRequest& llmReq)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    // If llmRequest is streaming, directly copy to host.
    // Only one token's logits needs to be copied each time.
    TLLM_CHECK(llmReq.getGenerationLogitsFragmentsSize() == 1);

    SizeType32 numGenerationToken = llmReq.getMaxBeamNumTokens() - llmReq.mPromptLen;
    TensorPtr const& generationLogitsHost
        = llmReq.getGenerationLogitsHost(); // [mMaxNewTokens (or 1), beamWidth, vocabSizePadded]

    TensorPtr hostTensorPtr
        = ITensor::slice(generationLogitsHost, numGenerationToken, 1); // [1, beamWidth, vocabSizePadded]
    TensorPtr deviceTensorPtr = *(llmReq.getGenerationLogitsFragments().begin());

    bufferManager.copy(*deviceTensorPtr, *hostTensorPtr);
    llmReq.clearGenerationLogitsFragments();
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void setupMedusaLogits(std::vector<TensorPtr>& medusaLogitsHeads, TensorPtr const& medusaLogitsDevice,
    SizeType32 medusaHeads, SizeType32 logitsIndex, SizeType32 numLogits)
{
    for (SizeType32 hi = 0; hi < medusaHeads; ++hi)
    {
        TensorPtr logitsHead = ITensor::slice(medusaLogitsDevice, hi, 1);
        logitsHead->squeeze(0);
        medusaLogitsHeads[hi] = ITensor::slice(logitsHead, logitsIndex, numLogits);
    }
}

} // namespace

void HandleGenerationLogits::operator()(SizeType32 logitsIndex, RequestVector const& contextRequests,
    RequestVector const& generationRequests, RuntimeBuffers const& genRuntimeBuffers, DecoderBuffers& decoderBuffers,
    tr::ModelConfig const& modelConfig, runtime::TllmRuntime const& runtime) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(HandleGenerationLogits);

    auto const& manager = runtime.getBufferManager();
    auto batchIdx = contextRequests.size();

    for (auto const& llmReq : generationRequests)
    {
        auto const reqBeamWidth = llmReq->mSamplingConfig.beamWidth;
        auto const seqSlot = llmReq->mSeqSlot.value();

        auto const draftLength = llmReq->getNumDraftTokens();
        auto const numLogits = draftLength + reqBeamWidth;

        TLLM_CHECK(draftLength == 0 || reqBeamWidth == 1);

        TLLM_LOG_DEBUG("logitsIndex: %d", logitsIndex);
        TLLM_LOG_DEBUG("draftLength: %d", draftLength);
        TLLM_LOG_DEBUG("reqBeamWidth: %d", reqBeamWidth);

        // genRuntimeBuffers.logits shape: [numGen*reqBeamWidth, vocabSize]
        // logitsView shape: [numLogits, vocabSize]
        TensorPtr logitsView = ITensor::slice(genRuntimeBuffers.logits, logitsIndex, numLogits);
        TLLM_CHECK_DEBUG_WITH_INFO(tru::tensorHasInvalid<float>(*logitsView, manager, "logits") == false,
            "Found invalid number (NaN or Inf) in logits");
        auto& decoderLogits = decoderBuffers.logits.at(seqSlot);
        auto const logitsViewShape = logitsView->getShape();
        if (reqBeamWidth > 1)
        {
            decoderLogits = logitsView;
            decoderLogits->unsqueeze(0);
        }
        else
        {
            decoderLogits
                = ITensor::view(logitsView, ITensor::makeShape({logitsViewShape.d[0], 1, logitsViewShape.d[1]}));
        }

        if (llmReq->getReturnGenerationLogits())
        {
            TLLM_CHECK_WITH_INFO(modelConfig.getSpeculativeDecodingMode().isNone()
                    || modelConfig.getSpeculativeDecodingMode().isDraftTokensExternal(),
                "Only speculative decoding with external draft tokens supports returning generation logits");

            // Push into fragments vector
            llmReq->addGenerationLogitsFragment(logitsView);
            TLLM_CHECK(llmReq->getGenerationLogitsFragmentsSize() <= GENERATION_LOGITS_BUFFER_LENGTH);
            if (llmReq->isStreaming())
            {
                copyStreamingGenerationLogits(manager, *llmReq);
            }
            // Copy back to host for every GENERATION_LOGITS_BUFFER_LENGTH steps to mitigate GPU memory pressure
            else if (llmReq->getGenerationLogitsFragmentsSize() == GENERATION_LOGITS_BUFFER_LENGTH)
            {
                auto constexpr beforeDecoder = true;
                utils::copyGenerationLogits(genRuntimeBuffers, manager, *llmReq, batchIdx, beforeDecoder);
            }
        }
        if (modelConfig.getSpeculativeDecodingMode().hasDraftLogits())
        {
            auto& medusaLogitsHeads = decoderBuffers.draftBuffers.predictedDraftLogits.at(seqSlot);
            setupMedusaLogits(medusaLogitsHeads, genRuntimeBuffers.medusaBuffers->medusaLogitsDevice,
                modelConfig.getSpeculativeDecodingModule().getMaxDraftPathLen(), logitsIndex, draftLength);
        }
        logitsIndex += numLogits;
        ++batchIdx;
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

} // namespace tensorrt_llm::batch_manager
