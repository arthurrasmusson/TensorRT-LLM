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

#include "tensorrt_llm/batch_manager/handleContextLogits.h"

#include "tensorrt_llm/batch_manager/decoderBuffers.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/batch_manager/medusaBuffers.h"
#include "tensorrt_llm/batch_manager/runtimeBuffers.h"
#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"
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

//! @brief Copy logits from context phase to beginning of generation logits.
//! @details Usually, this concerns logits of 1 token. In speculative decoding this concerns draftLen + 1 tokens.
void copyLastContextLogits(TensorPtr const& contextLogits, LlmRequest& llmReq, BufferManager const& bufferManager)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const numLogits = contextLogits->getShape().d[0];
    for (int beam = 0; beam < llmReq.mSamplingConfig.beamWidth; beam++)
    {
        // [beamWidth, mMaxNewTokens, vocabSizePadded] -> [numLogits, vocabSizePadded]
        auto beamHostTensorPtr = ITensor::slice(llmReq.getGenerationLogitsHost(), {beam, 0}, numLogits);
        bufferManager.copy(*contextLogits, *beamHostTensorPtr);
    }
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

SizeType32 HandleContextLogits::operator()(RequestVector const& contextRequests,
    RuntimeBuffers const& contextRuntimeBuffers, DecoderBuffers& decoderBuffers, tr::ModelConfig const& modelConfig,
    runtime::TllmRuntime const& runtime) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(HandleContextLogits);

    auto const& manager = runtime.getBufferManager();
    auto const& stream = runtime.getStream();

    SizeType32 batchIndex{0};
    SizeType32 logitsIndex{0};
    // Copy logits into decoderBuffers.logits
    for (auto const& llmReq : contextRequests)
    {
        auto const reqBeamWidth = llmReq->mSamplingConfig.beamWidth;
        auto const numContextLogits = contextRuntimeBuffers.numContextLogits.at(batchIndex);
        auto const draftLength = llmReq->isLastContextChunk() ? llmReq->getNumDraftTokens() : 0;

        TLLM_LOG_DEBUG("logitsIndex: %d", logitsIndex);
        TLLM_LOG_DEBUG("numContextLogits %d", numContextLogits);
        TLLM_LOG_DEBUG("draftLength: %d", draftLength);

        if (modelConfig.computeContextLogits())
        {
            // Since the computational graph has been modified, only the last token is needed.
            TLLM_CHECK_WITH_INFO(!modelConfig.getSpeculativeDecodingMode().isMedusa()
                    && !modelConfig.getSpeculativeDecodingMode().isLookaheadDecoding(),
                "Return context logits is not supported with Medusa and Lookahead decoding");

            if (llmReq->getReturnContextLogits())
            {
                if (llmReq->getPrepopulatedPromptLen() > 0)
                {
                    TLLM_LOG_WARNING(
                        "Because of KV cache reuse, not all context logits could be produced for request %lu.",
                        llmReq->mRequestId);
                }
                TensorPtr contextLogitsDeviceView
                    = ITensor::slice(contextRuntimeBuffers.logits, logitsIndex, numContextLogits);
                TensorPtr contextLogitsHostView = ITensor::slice(
                    llmReq->getContextLogitsHost(), llmReq->getContextCurrentPosition(), numContextLogits);
                // Copy to host directly
                manager.copy(*contextLogitsDeviceView, *contextLogitsHostView);
            }
        }
        logitsIndex += numContextLogits + draftLength;

        // Get the logits from the last context token and draft tokens
        auto const numDecoderLogits = 1 + draftLength;
        auto const seqSlot = llmReq->mSeqSlot.value();
        auto& decoderLogits = decoderBuffers.logits.at(seqSlot);
        TensorPtr logitsView
            = ITensor::slice(contextRuntimeBuffers.logits, logitsIndex - numDecoderLogits, numDecoderLogits);

        if (modelConfig.getSpeculativeDecodingMode().hasDraftLogits())
        {
            auto& medusaLogitsHeads = decoderBuffers.draftBuffers.predictedDraftLogits.at(seqSlot);
            setupMedusaLogits(medusaLogitsHeads, contextRuntimeBuffers.medusaBuffers->medusaLogitsDevice,
                modelConfig.getSpeculativeDecodingModule().getMaxDraftPathLen(), logitsIndex - numDecoderLogits,
                numDecoderLogits);
        }

        // Save the last token logits of context into generation logits or
        // save the accepted token logits from target model
        if (llmReq->getReturnGenerationLogits())
        {
            copyLastContextLogits(logitsView, *llmReq, manager);
        }

        TLLM_CHECK_DEBUG_WITH_INFO(tru::tensorHasInvalid<float>(*logitsView, manager, "logits") == false,
            "Found invalid number (NaN or Inf) in logits");
        // Scatter the output logits to the decoderLogits
        if (reqBeamWidth > 1)
        {
            // Tile logits of context requests
            auto const logitsShape = logitsView->getShape();
            auto const logitsType = logitsView->getDataType();
            decoderLogits = manager.gpu(ITensor::makeShape({reqBeamWidth, logitsShape.d[1]}), logitsType);
            tensorrt_llm::runtime::kernels::tileTensor(*decoderLogits, *logitsView, reqBeamWidth, stream);
            decoderLogits->unsqueeze(0);
        }
        else
        {
            auto const logitsViewShape = logitsView->getShape();
            decoderLogits
                = ITensor::view(logitsView, ITensor::makeShape({logitsViewShape.d[0], 1, logitsViewShape.d[1]}));
        }

        ++batchIndex;
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return logitsIndex;
}

} // namespace tensorrt_llm::batch_manager
