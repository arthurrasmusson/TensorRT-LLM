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

#include "trtGptModelInflightBatching.h"

#include "runtimeBuffers.h"
#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/batch_manager/peftCacheManager.h"
#include "tensorrt_llm/batch_manager/requestScheduler.h"
#include "tensorrt_llm/batch_manager/rnnStateManager.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/mpiUtils.h"
#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/gptDecoderBatched.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/ipcUtils.h"
#include "tensorrt_llm/runtime/lookaheadModule.h"
#include "tensorrt_llm/runtime/memoryCounters.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"
#include "tensorrt_llm/runtime/tllmLogger.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"
#include "tensorrt_llm/runtime/utils/debugUtils.h"
#include "tensorrt_llm/runtime/utils/sessionUtils.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace tensorrt_llm::runtime;
namespace tc = tensorrt_llm::common;
namespace tk = tensorrt_llm::kernels;
namespace tru = tensorrt_llm::runtime::utils;
namespace texe = tensorrt_llm::executor;

namespace tensorrt_llm::batch_manager
{
namespace
{
using TensorPtr = runtime::ITensor::SharedPtr;
using TensorConstPtr = runtime::ITensor::SharedConstPtr;
using KVCacheManager = kv_cache_manager::KVCacheManager;

void setupMedusaLogits(std::vector<TensorPtr>& medusaLogitsHeads, TensorPtr& medusaLogitsDevice, SizeType32 medusaHeads,
    SizeType32 logitsIndex, SizeType32 numLogits)
{
    for (SizeType32 hi = 0; hi < medusaHeads; ++hi)
    {
        TensorPtr logitsHead = ITensor::slice(medusaLogitsDevice, hi, 1);
        logitsHead->squeeze(0);
        medusaLogitsHeads[hi] = ITensor::slice(logitsHead, logitsIndex, numLogits);
    }
}

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

//! @param beforeDecoder    Whether the function is called before the decoder. If it is true, correct the output offset.
//! @param numDroppedTokens The number of dropped tokens for each beam (e.g. when the requests finished early).
//!                         Generation logits for dropped tokens are ignored.
void copyGenerationLogits(RuntimeBuffers const& genRuntimeBuffers, BufferManager const& bufferManager,
    LlmRequest& llmReq, std::size_t batchIdx, bool beforeDecoder, std::vector<SizeType32> const& numDroppedTokens = {})
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    TLLM_CHECK_WITH_INFO(
        !beforeDecoder || numDroppedTokens.empty(), "numDroppedTokens are only possible after decoder.");

    auto const reqBeamWidth = llmReq.mSamplingConfig.beamWidth;
    TLLM_CHECK_WITH_INFO(numDroppedTokens.empty() || numDroppedTokens.size() == static_cast<size_t>(reqBeamWidth),
        "Dropped tokens have to be defined for all beams.");

    auto const fragmentSize = llmReq.getGenerationLogitsFragmentsSize();

    // Merge logits fragments on device
    TensorPtr transposeBufferPtr = ITensor::slice(genRuntimeBuffers.cacheTransposedGenerationLogits, batchIdx, 1);
    transposeBufferPtr->squeeze(0); // [beamWidth, GENERATION_LOGITS_BUFFER_LENGTH, vocabSize]
    auto cachePointerDevice = ITensor::slice(genRuntimeBuffers.cacheGenerationFragmentPointerDevice, batchIdx, 1);
    auto cachePointerHost = ITensor::slice(genRuntimeBuffers.cacheGenerationFragmentPointerHost, batchIdx, 1);
    tensorrt_llm::runtime::kernels::mergeLogitsFragments(bufferManager, *transposeBufferPtr,
        llmReq.getGenerationLogitsFragments(), *cachePointerDevice, *cachePointerHost, 0, 1, reqBeamWidth,
        bufferManager.getStream(), 0);
    llmReq.clearGenerationLogitsFragments();

    // Copy logits to host
    for (SizeType32 beam = 0; beam < reqBeamWidth; beam++)
    {
        auto const droppedSize = !numDroppedTokens.empty() ? numDroppedTokens.at(beam) : 0;
        // Ignore logits of dropped tokens
        auto const beamFragmentSize = fragmentSize - droppedSize;
        // If this function is called before the decoder, the request does not contain the generated token of the
        // current iteration, so we add 1 to the number of tokens.
        auto const numGenerationToken
            = static_cast<SizeType32>(beforeDecoder) + llmReq.getNumTokens(beam) - llmReq.mPromptLen;
        auto const hostOffset = numGenerationToken - beamFragmentSize;

        // [beamWidth, GENERATION_LOGITS_BUFFER_LENGTH, vocabSizePadded] -> [beamFragmentSize, vocabSizePadded]
        auto beamDeviceTensorPtr = ITensor::slice(transposeBufferPtr, {beam, 0}, beamFragmentSize);
        // [beamWidth, mMaxNewTokens, vocabSizePadded] -> [beamFragmentSize, vocabSizePadded]
        auto beamHostTensorPtr = ITensor::slice(llmReq.getGenerationLogitsHost(), {beam, hostOffset}, beamFragmentSize);
        bufferManager.copy(*beamDeviceTensorPtr, *beamHostTensorPtr);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void allocateKvCache(ScheduledRequests const& scheduledRequests, kv_cache_manager::KVCacheManager* kvCacheManagerPtr,
    kv_cache_manager::KVCacheManager* crossKvCacheManagerPtr)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TLLM_CHECK(kvCacheManagerPtr);
    auto& kvCacheManager = *kvCacheManagerPtr;

    for (auto const& llmReq : scheduledRequests.contextRequests)
    {
        if (llmReq->isFirstContextChunk())
        {
            // Get slot of the current sequence in the KV cache
            auto const seqSlot = llmReq->mSeqSlot.value();
            auto const promptLen = llmReq->mPromptLen;
            auto const reqBeamWidth = llmReq->mSamplingConfig.beamWidth;
            auto const draftLength = llmReq->getNumDraftTokens();

            // Allocate/Reuse KV cache
            kvCacheManager.addSequence(seqSlot, promptLen, reqBeamWidth, llmReq);

            // Allocate more KV cache for speculative decoding
            if (draftLength > 0)
            {
                for (SizeType32 di = 0; di < draftLength; ++di)
                {
                    kvCacheManager.addToken(seqSlot);
                }
            }

            if (crossKvCacheManagerPtr != nullptr)
            {
                crossKvCacheManagerPtr->addSequence(seqSlot, llmReq->getEncoderLen(), reqBeamWidth, llmReq);
            }

            auto const prepopulatedPromptLen = llmReq->getPrepopulatedPromptLen();
            TLLM_CHECK(prepopulatedPromptLen < promptLen);

            if (prepopulatedPromptLen > 0)
            {
                // Currently, the runtime process is to apply for cache first and then determine prepopulation.
                // Use the prepopulated length to advance the context position and decrease chunk size if necessary.
                if (llmReq->isFullContextRequest())
                {
                    llmReq->setContextCurrentPosition(prepopulatedPromptLen);
                    llmReq->setContextChunkSize(promptLen);
                }
                else
                {
                    auto chunkSize = llmReq->getContextChunkSize();
                    if (prepopulatedPromptLen + chunkSize < promptLen)
                    {
                        // make sure to end at block boundary after current chunk
                        auto const flooredEndPosition = (prepopulatedPromptLen + chunkSize)
                            / kvCacheManager.getTokensPerBlock() * kvCacheManager.getTokensPerBlock();
                        chunkSize = flooredEndPosition - prepopulatedPromptLen;
                        TLLM_CHECK(chunkSize <= llmReq->getContextChunkSize());
                    }
                    llmReq->setContextCurrentPosition(prepopulatedPromptLen);
                    llmReq->setContextChunkSize(chunkSize);
                }
                if (!llmReq->isLastContextChunk())
                {
                    TLLM_CHECK_WITH_INFO((llmReq->getContextCurrentPosition() + llmReq->getContextChunkSize())
                                % kvCacheManager.getTokensPerBlock()
                            == 0,
                        "To prevent cache fragmentation, the context position after current chunk should be divisible "
                        "by the number of tokens per block, except for the last chunk.");
                }
            }
        }
    }

    for (auto const& llmReq : scheduledRequests.generationRequests)
    {
        // Get slot of the current sequence in the KV cache
        auto const seqSlot = llmReq->mSeqSlot.value();
        auto const draftLength = llmReq->getNumDraftTokens();

        for (SizeType32 di = 0; di < draftLength + 1; ++di)
        {
            kvCacheManager.addToken(seqSlot);
        }
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

} // namespace

TrtGptModelInflightBatching::TrtGptModelInflightBatching(std::shared_ptr<nvinfer1::ILogger> logger,
    ModelConfig const& modelConfig, WorldConfig const& worldConfig, RawEngine const& rawEngine, bool ctxGenFusion,
    TrtGptModelOptionalParams const& optionalParams)
    : TrtGptModel(modelConfig, worldConfig, optionalParams)
    , mModelConfig(modelConfig)
    , mWorldConfig(worldConfig)
    , mDecodingConfig{optionalParams.decodingConfig}
    , mDevice{runtime::utils::initDevice(worldConfig)}
    , mLogger{logger ? std::move(logger) : std::make_shared<TllmLogger>()}
    , mRuntime{std::make_shared<TllmRuntime>(
          rawEngine, mLogger.get(), optionalParams.gpuWeightsPercent, modelConfig.useShapeInference())}
    , mMicroBatchId(0)
    , mCtxGenFusion(ctxGenFusion)
    , mExtendedRuntimePerfKnobConfig{optionalParams.extendedRuntimePerfKnobConfig}
    , mOperatingBeamWidth{getMaxBeamWidth()}
    , mCopyBufferManager{std::make_shared<CudaStream>()}
    , mLastIterationStatsIFB(-1)
    , mDecStepAsyncSndHdl(nullptr)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    if (!(mModelConfig.supportsInflightBatching()))
    {
        throw std::runtime_error(
            "TrtGptModelInflightBatching requires GPT attention/Mamba Conv 1d plugin with "
            "packed input and paged KV cache.");
    }

    if (mWorldConfig.isPipelineParallel())
    {
        mNumMicroBatches = mWorldConfig.getPipelineParallelism();
    }
    else
    {
        mNumMicroBatches = isTtrOverlap() ? 2 : 1;
    }

    mNumBuffers = (mCtxGenFusion ? 1 : 2) * mNumMicroBatches;

    if (optionalParams.kvCacheConfig.enableBlockReuse)
    {
        TLLM_CHECK_WITH_INFO(mModelConfig.getPagedContextFMHA(),
            "When KV cache block reuse is set, model has to be built with paged context FMHA support");
    }

    if (!optionalParams.kvCacheConfig.onboardBlocks)
    {
        TLLM_CHECK_WITH_INFO(!mModelConfig.getPagedContextFMHA() && !mModelConfig.useXQA(),
            "KV cache blocks need to be onboarded if context FMHA or XAQ kernels are used");
    }

    if (mModelConfig.getSpeculativeDecodingMode().isDraftTokensExternal())
    {
        TLLM_CHECK_WITH_INFO(optionalParams.kvCacheConfig.enableBlockReuse,
            "KV cache block reuse must be enabled for speculative decoding target model");
    }

    if (mCtxGenFusion)
    {
        TLLM_CHECK_WITH_INFO(!mModelConfig.isRnnBased(), "RNN based model doesn't support context generation fusion.");
        TLLM_CHECK_WITH_INFO(
            mModelConfig.isTransformerBased(), "Only transformer based model support context generation fusion now.");
    }

    if (mWorldConfig.isTensorParallel())
    {
        createCustomAllReduceWorkspace();
    }

    setupSpeculativeDecodingModule(mDecodingConfig);

    createRuntimeContexts();

    auto& memCounter = MemoryCounters::getInstance();
    auto const gpuUsage1 = memCounter.getGpu();
    createBuffers(mDecodingConfig, mExtendedRuntimePerfKnobConfig);
    auto const gpuUsage2 = memCounter.getGpu();
    TLLM_LOG_INFO("[MemUsageChange] Allocated %s GPU memory for runtime buffers.",
        memCounter.bytesToString(gpuUsage2 - gpuUsage1).c_str());

    createDecoder(mDecodingConfig.getDecodingMode());
    auto const gpuUsage3 = memCounter.getGpu();
    TLLM_LOG_INFO("[MemUsageChange] Allocated %s GPU memory for decoder.",
        memCounter.bytesToString(gpuUsage3 - gpuUsage2).c_str());

    if (mModelConfig.isRnnBased())
    {
        createRnnStateManager();
    }
    if (mModelConfig.isTransformerBased())
    {
        mKvCacheManager = createKvCacheManager(optionalParams.kvCacheConfig, KvCacheType::kSELF);
        if (mModelConfig.useCrossAttention())
        {
            // assume encoder and decoder configs are the same
            mCrossKvCacheManager = createKvCacheManager(optionalParams.kvCacheConfig, KvCacheType::kCROSS);
        }
    }

    if (mWorldConfig.isPipelineParallel())
    {
        auto const& commSession = COMM_SESSION;
        mMpiCommPipelinePara = std::make_shared<tensorrt_llm::mpi::MpiComm>(
            commSession.split(mWorldConfig.getTensorParallelRank(), mWorldConfig.getPipelineParallelRank()));
        mDecSlotAsyncSndHdls.reserve(getMaxBatchSize());
    }

    // TODO: Get from backend
    // Sequence is considered idle if not update for 180 seconds
    uint64_t maxSeqIdleMicroseconds = 180 * 1000 * 1000;

    mSeqSlotManager = std::make_shared<SequenceSlotManager>(getMaxNumSequences(), maxSeqIdleMicroseconds);

    if (mModelConfig.useLoraPlugin())
    {
        mPeftCacheManager = std::make_shared<PeftCacheManager>(
            optionalParams.peftCacheManagerConfig, mModelConfig, mWorldConfig, mRuntime->getBufferManager());
    }
    else
    {
        mPeftCacheManager = std::make_shared<NoOpPeftCacheManager>();
    }

    mMicroBatchScheduledRequests.resize(mNumMicroBatches);
    mDecoderWaitEvents.resize(mNumMicroBatches);
    mPeftTables.resize(mNumMicroBatches);

    if (modelConfig.isRnnBased())
    {
        TLLM_CHECK_WITH_INFO(modelConfig.getMaxBeamWidth() == 1, "RNN based model doesn't support beam search now.");
        TLLM_CHECK_WITH_INFO(
            !optionalParams.enableChunkedContext, "RNN based model doesn't support Chunked Context now.");
        TLLM_CHECK_WITH_INFO(
            modelConfig.getSpeculativeDecodingMode().isNone(), "RNN based model doesn't support speculative decoding.");
    }

    std::optional<batch_scheduler::ContextChunkingConfig> ctxChunkConfig;
    if (optionalParams.enableChunkedContext)
    {
        TLLM_CHECK_WITH_INFO(mModelConfig.getPagedContextFMHA(),
            "Chunked context requires the engine to be built with paged FMHA enabled.");
        ctxChunkConfig
            = batch_scheduler::ContextChunkingConfig{optionalParams.schedulerConfig.getContextChunkingPolicy().value_or(
                                                         texe::ContextChunkingPolicy::kFIRST_COME_FIRST_SERVED),
                mKvCacheManager->getTokensPerBlock()};
    }

    auto maxNumTokens = getMaxNumTokens();
    TLLM_CHECK_WITH_INFO(maxNumTokens, "Max number of tokens is not set in model config.");

    // For chunked context the max chunk size is limited by max_num_tokens.
    // For context FMHA the max context size is limited by max_num_tokens.
    // Else it is limited by the model.
    auto const maxContextLength = (optionalParams.enableChunkedContext || mModelConfig.getContextFMHA())
        ? maxNumTokens
        : std::make_optional<SizeType32>(mModelConfig.getMaxInputLen());

    mRequestScheduler = std::make_shared<batch_scheduler::RequestScheduler>(getMaxBatchSize(), mNumMicroBatches,
        mKvCacheManager, mCrossKvCacheManager, mPeftCacheManager, optionalParams.schedulerConfig, maxNumTokens,
        ctxChunkConfig, maxContextLength);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

TrtGptModelInflightBatching::~TrtGptModelInflightBatching()
{
    if (mMpiWaitThread)
    {
        mMpiWaitThread->join();
        mMpiWaitThread.reset(nullptr);
    }
}

void TrtGptModelInflightBatching::setupSpeculativeDecodingModule(executor::DecodingConfig const& decodingConfig)
{
    if (mModelConfig.getSpeculativeDecodingMode().isExplicitDraftTokens())
    {
        TLLM_CHECK_WITH_INFO(mCtxGenFusion, "Current speculative decoding mode requires context-gen fusion IFB");
    }

    if (mModelConfig.getSpeculativeDecodingMode().isLookaheadDecoding())
    {
        // FIXME(nkorobov) choose defaults
        auto maxLookaheadConfig = decodingConfig.getLookaheadDecodingConfig().value();

        SizeType32 maxDraftTokens, maxDraftPathLen;
        std::tie(std::ignore, std::ignore, maxDraftTokens, maxDraftPathLen)
            = maxLookaheadConfig.calculateSpeculativeResource();
        TLLM_CHECK(maxDraftTokens <= mModelConfig.getMaxDecodingDraftTokens());
        mModelConfig.getSpeculativeDecodingModulePtr()->setMaxDraftTokens(maxDraftTokens);
        mModelConfig.getSpeculativeDecodingModulePtr()->setMaxDraftPathLen(maxDraftPathLen);

        auto LookaheadModulePtr
            = std::dynamic_pointer_cast<runtime::LookaheadModule>(mModelConfig.getSpeculativeDecodingModulePtr());
        LookaheadModulePtr->setExecutionConfig(maxLookaheadConfig);
    }
}

void TrtGptModelInflightBatching::reshapeKvTensors(KVCacheManager const& kvCacheManager)
{
    auto const kvCacheType = kvCacheManager.isCrossKv() ? KvCacheType::kCROSS : KvCacheType::kSELF;
    auto const maxBlocksPerSeq = kvCacheManager.getMaxBlocksPerSeq();

    TLLM_CHECK(mBuffers.size() == static_cast<size_t>(mNumBuffers));
    for (auto& buffers : mBuffers)
    {
        TLLM_CHECK(buffers->transformerBuffers);
        // any method that operates on transformerBuffers must distinguish between self and cross cache, because
        // transformerBuffers is not managed by KVCacheManager same rule applies to kv pool pointers below
        buffers->transformerBuffers->reshapeKvTensors(
            getMaxBatchSize(), mOperatingBeamWidth, maxBlocksPerSeq, *mRuntime, kvCacheType);
    }
}

std::shared_ptr<KVCacheManager> TrtGptModelInflightBatching::createKvCacheManager(
    KvCacheConfig const& kvCacheConfig, KvCacheType kvCacheType)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    TLLM_CHECK_WITH_INFO(
        mModelConfig.isTransformerBased(), "KvCacheManager is only needed by transformer based model.");

    auto const tokensPerBlock = mModelConfig.getTokensPerBlock();
    auto const kvDtype = mModelConfig.getKvDataType();

    TLLM_CHECK_WITH_INFO(getMaxBeamWidth() == 1 || getMaxSequenceLen() == getMaxAttentionWindow(),
        "Can't support cyclic kv cache with beam search.");

    auto const [blocksInPrimaryPool, blocksInSecondaryPool] = KVCacheManager::calculateMaxNumBlocks(
        kvCacheConfig, kvDtype, mModelConfig, mWorldConfig, mRuntime->getBufferManager());

    auto const useOneMoreBlock = false;

    // init KV cache block manager
    auto const localNbLayers = mModelConfig.getNbAttentionLayers(mWorldConfig.getPipelineParallelism());
    auto const nbKvHeads = mModelConfig.getNbKvHeads();
    auto const sizePerHead = mModelConfig.getSizePerHead();

    // now we check if maxAttentionWindow is too large for at least one sequence to fit in kvCache
    // this can happen if maxSeqLen is deduced from the model and is too large
    // and user also either didn't provide maxAttentionWindow, which leads it to be equal to maxSeqLen
    if (kvCacheType == KvCacheType::kSELF)
    {
        auto const maxAttentionWindowUpperBound = KVCacheManager::getMaxAttentionWindowUpperBound(
            blocksInPrimaryPool, tokensPerBlock, getMaxBeamWidth(), getSinkTokenLen(), useOneMoreBlock);

        if (maxAttentionWindowUpperBound < getMaxAttentionWindow())
        {
            TLLM_LOG_WARNING(
                "maxAttentionWindow and maxSequenceLen are too large for at least one sequence to fit in kvCache. "
                "they are reduced to %d",
                maxAttentionWindowUpperBound);
            setMaxAttentionWindow(maxAttentionWindowUpperBound);
            setMaxSequenceLen(maxAttentionWindowUpperBound);
            if (getMaxInputLen() > getMaxSequenceLen() - 1)
            {
                setMaxInputLen(getMaxSequenceLen() - 1);
                TLLM_LOG_WARNING("maxInputLen is reduced to %d", getMaxInputLen());
            }
        }
    }

    auto const maxKvCacheLength
        = kvCacheType == KvCacheType::kSELF ? getMaxAttentionWindow() : mModelConfig.getMaxEncoderLen();
    auto kvCacheManager = std::make_shared<KVCacheManager>(localNbLayers, nbKvHeads, sizePerHead, tokensPerBlock,
        blocksInPrimaryPool, blocksInSecondaryPool, getMaxNumSequences(), getMaxBeamWidth(), maxKvCacheLength,
        getSinkTokenLen(), useOneMoreBlock, mRuntime->getStreamPtr(), kvCacheConfig.enableBlockReuse,
        kvCacheConfig.onboardBlocks, kvCacheType);

    reshapeKvTensors(*kvCacheManager);

    kvCacheManager->allocatePools(kvDtype, kvCacheConfig.useUvm);

    for (auto& buffers : mBuffers)
    {
        buffers->transformerBuffers->setKvPoolPointers(*kvCacheManager);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);

    return kvCacheManager;
}

void TrtGptModelInflightBatching::createRnnStateManager()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    TLLM_CHECK_WITH_INFO(mModelConfig.isRnnBased(), "RnnStateManager is only needed by RNN based model.");

    mRnnStateManager
        = std::make_shared<RnnStateManager>(mModelConfig, mWorldConfig, mRuntime->getStreamPtr(), getMaxNumSequences());

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtGptModelInflightBatching::createCustomAllReduceWorkspace()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const& manager = mRuntime->getBufferManager();
    auto const hiddenSize = mModelConfig.getHiddenSize();

    mAllReduceBuffers = std::make_shared<AllReduceBuffers>(
        getMaxBatchSize(), getMaxBeamWidth(), getMaxSequenceLen(), hiddenSize, manager, mWorldConfig);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtGptModelInflightBatching::terminateRequest(std::shared_ptr<LlmRequest> const& llmReq, bool pause)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    // If a sequence slot is associated with this request id, free it
    mSeqSlotManager->freeSequenceSlot(llmReq->mRequestId);
    // Remove the sequence from kvCacheManager
    auto const seqSlot = llmReq->mSeqSlot;
    if (mKvCacheManager && seqSlot)
    {
        mKvCacheManager->removeSequence(seqSlot.value(), llmReq);
    }
    if (mCrossKvCacheManager && seqSlot)
    {
        mCrossKvCacheManager->removeSequence(seqSlot.value(), llmReq);
    }
    if (pause && !llmReq->isGenerationCompleteState())
    {
        llmReq->pause(getMaxInputLen());
    }
    else
    {
        TLLM_LOG_DEBUG("terminated: id %lu, paused: %d", llmReq->mRequestId, pause);
    }

    mPeftCacheManager->markRequestDone(llmReq, pause);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

TrtGptModelInflightBatching::IterationStatsIFB TrtGptModelInflightBatching::fillIterationStats(
    ScheduledRequests const& scheduledRequests, RequestVector const& requestsToPause)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(fillIterationStats);

    IterationStatsIFB iterationStatsIfb{mMicroBatchId};
    iterationStatsIfb.numCtxRequests = scheduledRequests.contextRequests.size();
    iterationStatsIfb.numGenRequests = scheduledRequests.generationRequests.size();
    iterationStatsIfb.avgNumDecodedTokensPerIter = 0;

    auto const contextBufferId = mCtxGenFusion ? getFusedBufferId() : getContextBufferId();
    auto const& buffers = mBuffers.at(contextBufferId);
    iterationStatsIfb.numCtxTokens = buffers->getNumContextTokens();

    for (auto const& llmReq : scheduledRequests.contextRequests)
    {
        iterationStatsIfb.scheduledRequests.insert(llmReq->mRequestId);
    }
    for (auto const& llmReq : scheduledRequests.generationRequests)
    {
        iterationStatsIfb.scheduledRequests.insert(llmReq->mRequestId);
        iterationStatsIfb.avgNumDecodedTokensPerIter += llmReq->getAvgDecodedTokensPerIter();
    }
    if (iterationStatsIfb.numGenRequests > 0)
    {
        iterationStatsIfb.avgNumDecodedTokensPerIter /= iterationStatsIfb.numGenRequests;
    }
    for (auto const& llmReq : requestsToPause)
    {
        iterationStatsIfb.pausedRequests.insert(llmReq->mRequestId);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return iterationStatsIfb;
}

void TrtGptModelInflightBatching::forwardSync()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(forwardSync);

    auto const device = mWorldConfig.getDevice();
    TLLM_CUDA_CHECK(cudaSetDevice(device));

    if (!mWorldConfig.isLastPipelineParallelRank())
    {
        if (mMpiWaitThread)
        {
            mMpiWaitThread->join();
            mMpiWaitThread.reset(nullptr);
        }
    }

    auto& currRequests = mMicroBatchScheduledRequests.at(mMicroBatchId);
    auto& decoderWaitEvent = mDecoderWaitEvents.at(mMicroBatchId);

    if (!currRequests.empty())
    {
        if (!mWorldConfig.isPipelineParallel() || !mWorldConfig.isLastPipelineParallelRank())
        {
            TLLM_CHECK_WITH_INFO(mDecStepAsyncSndHdl.get() == nullptr, "decoderSync handle must be nullptr.");
            // Wait for decoding for requests in flight for the current micro batch
            mDecStepAsyncSndHdl = decoderSync(currRequests, decoderWaitEvent);
            if (!mWorldConfig.isLastPipelineParallelRank())
            {
                mMpiWaitThread = std::make_unique<std::thread>(
                    [this]()
                    {
                        auto const device = mWorldConfig.getDevice();
                        TLLM_CUDA_CHECK(cudaSetDevice(device));
                        mDecStepAsyncSndHdl.reset();
                        mDecSlotAsyncSndHdls.clear();
                    });
            }
        }
        else
        {
            for (auto const& requests : {currRequests.contextRequests, currRequests.generationRequests})
            {
                for (auto const& llmReq : requests)
                {
                    if (llmReq->mState == REQUEST_STATE_GENERATION_TO_COMPLETE)
                    {
                        llmReq->mState = REQUEST_STATE_GENERATION_COMPLETE;
                        terminateRequest(llmReq);
                    }
                }
            }
        }

        NVTX3_SCOPED_RANGE(pauseFlaggedCurrRequests);
        for (auto const& requests : {currRequests.contextRequests, currRequests.generationRequests})
        {
            for (auto const& llmReq : requests)
            {
                auto const reqId = llmReq->mRequestId;
                mInflightReqIds.erase(reqId);
                TLLM_LOG_DEBUG("request ID %u removed from DECODER inflight set", reqId);

                // If a request in this context had been flagged to be paused, pause it right away
                if (mReqIdsToPause.find(reqId) != mReqIdsToPause.end())
                {
                    terminateRequest(llmReq, true);
                    mReqIdsToPause.erase(reqId);
                }
            }
        }
    }
    // report profile data
    auto const bufferId = getFusedBufferId();
    auto const contextId = mBuffers[bufferId]->getContextIndex();
    if (mRuntime->hasLayerProfiler(contextId))
    {
        mRuntime->reportToProfiler(contextId);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtGptModelInflightBatching::forwardAsync(RequestList const& activeRequests)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(ModelForwardAsync);

    auto const device = mWorldConfig.getDevice();
    TLLM_CUDA_CHECK(cudaSetDevice(device));

    try
    {
        verifyRequests(activeRequests);

        auto& currRequests = mMicroBatchScheduledRequests.at(mMicroBatchId);
        auto& decoderWaitEvent = mDecoderWaitEvents.at(mMicroBatchId);

        // Get a new set of requests for that context
        // The scheduler will not include any requests that are (i) still in encoder state if encoder-decoder models OR
        // (ii) already in flight for decoder models
        TLLM_LOG_DEBUG("Running DECODER request scheduler");
        RequestVector requestsToPause;
        std::tie(currRequests.contextRequests, currRequests.generationRequests, requestsToPause)
            = mRequestScheduler->scheduleRequests(activeRequests, mInflightReqIds);

        assert(currRequests.size() <= static_cast<size_t>(getMaxBatchSize()));

        {
            NVTX3_SCOPED_RANGE(pauseRequestsFlaggedByScheduler);
            // Loop over requests flagged to be paused, and if not in flight pause it right away
            for (auto const& llmReq : requestsToPause)
            {
                auto const reqId = llmReq->mRequestId;
                if (mInflightReqIds.find(reqId) == mInflightReqIds.end())
                {
                    // Not in flight, can terminate right away
                    terminateRequest(llmReq, true);
                }
                else
                {
                    // In flight, add to set for pausing later
                    mReqIdsToPause.insert(reqId);
                }
            }
        }

        if (!currRequests.empty())
        {
            TLLM_LOG_DEBUG("Running DECODER model with batch size: %u", currRequests.size());
            {
                NVTX3_SCOPED_RANGE(updateInflightReqIds);
                // Add to set of requests in flight
                for (auto const& requests : {currRequests.contextRequests, currRequests.generationRequests})
                {
                    for (auto const& llmReq : requests)
                    {
                        TLLM_LOG_DEBUG("request ID %u added to DECODER inflight set", llmReq->mRequestId);
                        mInflightReqIds.insert(llmReq->mRequestId);
                    }
                }
            }

            assignReqSeqSlots(currRequests);

            if (mKvCacheManager)
            {
                allocateKvCache(currRequests, mKvCacheManager.get(), mCrossKvCacheManager.get());
            }
            mPeftTables.at(mMicroBatchId) = mPeftCacheManager->ensureBatch(currRequests, true);
            executeBatch(currRequests);

            sync_check_cuda_error();

            if (!currRequests.contextRequests.empty())
            {
                TLLM_LOG_DEBUG(
                    "request ID: %u finishes decoder ctx phase", currRequests.contextRequests[0]->mRequestId);
            }

            // Postpone decoder setup if model does not need to setup buffers for the context phase.
            if (!mModelConfig.getSpeculativeDecodingMode().needsDecoderPrologue())
            {
                setupDecoderStep(currRequests.contextRequests);
            }

            sync_check_cuda_error();

            decoderWaitEvent = mWorldConfig.isLastPipelineParallelRank() ? decoderStepAsync(currRequests) : TokenPtr();

            mLastIterationStatsIFB = fillIterationStats(currRequests, requestsToPause);

            for (auto const& requests : {currRequests.contextRequests, currRequests.generationRequests})
            {
                for (auto const& llmReq : requests)
                {
                    if (llmReq->isContextInitState())
                    {
                        llmReq->moveToNextContextChunk();
                        if (llmReq->getContextRemainingLength() == 0)
                        {
                            llmReq->mState = REQUEST_STATE_GENERATION_IN_PROGRESS;

                            // for encoder-decoder models, free encoder output buffers after decoder context phase is
                            // completed
                            if (llmReq->getEncoderTokens().has_value())
                            {
                                llmReq->freeEncoderOutputBuffers();
                            }
                        }
                    }
                    else if (llmReq->isGenerationInProgressState())
                    {
                        TLLM_LOG_DEBUG("request ID: %u forwards a step in decoder gen phase", llmReq->mRequestId);
                    }
                }
            }
        }
        else
        {
            mLastIterationStatsIFB = IterationStatsIFB{mMicroBatchId};
        }

        if (mWorldConfig.isPipelineParallel() && mWorldConfig.isLastPipelineParallelRank())
        {
            if (mMpiWaitThread)
            {
                mMpiWaitThread->join();
                mMpiWaitThread.reset(nullptr);
            }

            if (!currRequests.empty())
            {
                TLLM_CHECK_WITH_INFO(mDecStepAsyncSndHdl.get() == nullptr, "decoderSync handle must be nullptr.");
                // Wait for decoding for requests in flight for the current micro batch
                mDecStepAsyncSndHdl = decoderSync(currRequests, decoderWaitEvent);
                mMpiWaitThread = std::make_unique<std::thread>(
                    [this]()
                    {
                        auto const device = mWorldConfig.getDevice();
                        TLLM_CUDA_CHECK(cudaSetDevice(device));
                        mDecStepAsyncSndHdl.reset();
                        mDecSlotAsyncSndHdls.clear();
                    });
            }
        }

        // Update the micro batch ID
        mMicroBatchId = (mMicroBatchId + 1) % mNumMicroBatches;
    }
    // In case of error, we need to free the batch slot associated with those requests
    catch (std::exception const& e)
    {
        for (auto const& llmReq : activeRequests)
        {
            terminateRequest(llmReq);
        }
        throw;
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtGptModelInflightBatching::updatePeftCache(std::shared_ptr<LlmRequest> const& llmRequest)
{
    mPeftCacheManager->addRequestPeft(llmRequest, true);
}

runtime::BufferManager const& TrtGptModelInflightBatching::getBufferManager() const
{
    return mRuntime->getBufferManager();
}

BufferManager::CudaStreamPtr TrtGptModelInflightBatching::getRuntimeStreamPtr() const
{
    return mRuntime->getStreamPtr();
}

void TrtGptModelInflightBatching::executeContext(SizeType32 runtimeContextId)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(executeContext);
    auto enqueueSuccessful = mRuntime->executeContext(runtimeContextId);
    if (!enqueueSuccessful)
    {
        throw std::runtime_error("Executing TRT engine failed!");
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtGptModelInflightBatching::setLayerProfiler()
{
    mRuntime->setLayerProfiler();
}

std::string TrtGptModelInflightBatching::getLayerProfileInfo() const
{
    return mRuntime->getLayerProfileInfo();
}

void TrtGptModelInflightBatching::verifyRequests(RequestList const& activeRequests)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(verifyRequests);

    if (activeRequests.empty())
    {
        return;
    }

    auto const& firstRequest = activeRequests.front();
    auto const firstRequestId = firstRequest->mRequestId;
    auto const firstBeamWidth = firstRequest->mSamplingConfig.beamWidth;

    for (auto const& llmReq : activeRequests)
    {
        auto const beamWidth = llmReq->mSamplingConfig.beamWidth;
        auto const draftLength = llmReq->getNumDraftTokens();
        auto const maxDraftLength = mModelConfig.getMaxDecodingDraftTokens();

        TLLM_CHECK_WITH_INFO(beamWidth == 1 || draftLength == 0, "Can't use speculative decoding with beam search.");
        TLLM_CHECK_WITH_INFO(draftLength <= maxDraftLength,
            "Number of draft tokens (%d) is larger than maximum number of draft tokens (%d)", draftLength,
            maxDraftLength);

        // FIXME: Remove this check when varying beam width is supported
        {
            TLLM_CHECK_WITH_INFO(beamWidth == firstBeamWidth,
                "All active requests must have same beam width, "
                "but request %lu with beam width %d differs from first request %lu with beam width %d",
                llmReq->mRequestId, beamWidth, firstRequestId, firstBeamWidth);
        }
    }

    if (firstBeamWidth != mOperatingBeamWidth)
    {
        changeBeamWidth(firstBeamWidth);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtGptModelInflightBatching::executeBatch(ScheduledRequests const& scheduledRequests)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(executeBatch);

    if (!mCtxGenFusion)
    {
        if (!scheduledRequests.contextRequests.empty())
        {
            auto const bufferId = getContextBufferId();
            executeStep(scheduledRequests.contextRequests, {}, bufferId);
        }
        if (!scheduledRequests.generationRequests.empty())
        {
            auto const bufferId = getGenerationBufferId();
            executeStep({}, scheduledRequests.generationRequests, bufferId);
        }
    }
    else
    {
        auto const bufferId = getFusedBufferId();
        executeStep(scheduledRequests.contextRequests, scheduledRequests.generationRequests, bufferId);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtGptModelInflightBatching::createRuntimeContexts()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    mRuntime->clearContexts();
    auto const numProfiles = mRuntime->getNbProfiles();
    for (auto i = 0; i < numProfiles; ++i)
    {
        mRuntime->addContext(i);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

namespace
{
// TODO(rkobus): move this somewhere else?
executor::DecodingMode getDecodingMode(SpeculativeDecodingMode specDecodingMode,
    std::optional<executor::DecodingMode> const& decodingModeOpt, runtime::SizeType32 beamWidth)
{
    auto getDefaultDecodingMode = [beamWidth](std::optional<executor::DecodingMode> const& decodingModeOpt)
    {
        if (!decodingModeOpt.has_value() || decodingModeOpt->isAuto())
        {
            if (beamWidth == 1)
            {
                return executor::DecodingMode::TopKTopP();
            }
            else
            {
                return executor::DecodingMode::BeamSearch();
            }
        }
        return decodingModeOpt.value();
    };

    auto decodingMode = getDefaultDecodingMode(decodingModeOpt);
    // Overwrite decoding mode when beam width is one.
    if (beamWidth == 1 && decodingMode.isBeamSearch())
    {
        TLLM_LOG_WARNING(
            "Beam width is set to 1, but decoding mode is BeamSearch. Overwriting decoding mode to TopKTopP.");
        decodingMode = executor::DecodingMode::TopKTopP();
    }
    // Overwrite decoding mode when Medusa is used.
    if (specDecodingMode.isMedusa() && !decodingMode.isMedusa())
    {
        TLLM_LOG_WARNING("Model is Medusa, but decoding mode is not Medusa. Overwriting decoding mode to Medusa.");
        decodingMode = executor::DecodingMode::Medusa();
    }
    // Overwrite decoding mode when Medusa is not used.
    if (!specDecodingMode.isMedusa() && decodingMode.isMedusa())
    {
        TLLM_LOG_WARNING("Model is not Medusa, but decoding mode is Medusa. Overwriting decoding mode.");
        decodingMode = getDefaultDecodingMode(decodingModeOpt);
    }
    // Overwrite decoding mode when lookahead decoding is not used.
    if (!specDecodingMode.isLookaheadDecoding() && decodingMode.isLookahead())
    {
        TLLM_LOG_WARNING(
            "Model is not built with Lookahead decoding, but decoding mode is Lookahead. Overwriting decoding "
            "mode.");
        decodingMode = getDefaultDecodingMode(decodingModeOpt);
    }
    // Overwrite decoding mode when 'explicit draft tokens' is used.
    if (specDecodingMode.isExplicitDraftTokens() && !decodingMode.isExplicitDraftTokens())
    {
        TLLM_LOG_WARNING(
            "Model is built with 'explicit draft tokens' decoding, but decoding mode is something else. Overwriting "
            "decoding mode.");
        decodingMode = executor::DecodingMode::ExplicitDraftTokens();
    }
    // Overwrite decoding mode when 'explicit draft tokens' is not used.
    if (!specDecodingMode.isExplicitDraftTokens() && decodingMode.isExplicitDraftTokens())
    {
        TLLM_LOG_WARNING(
            "Model is not built with 'explicit draft tokens' decoding, but decoding mode is set to it. Overwriting "
            "decoding "
            "mode to default.");
        decodingMode = getDefaultDecodingMode(decodingModeOpt);
    }
    return decodingMode;
}
} // namespace

void TrtGptModelInflightBatching::createDecoder(std::optional<executor::DecodingMode> const& decodingModeOpt)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    if (mWorldConfig.isLastPipelineParallelRank())
    {
        auto decoderType = mRuntime->getEngine().getTensorDataType("logits");

        auto const decodingMode
            = getDecodingMode(mModelConfig.getSpeculativeDecodingMode(), decodingModeOpt, mOperatingBeamWidth);
        if (decodingMode.isExplicitDraftTokens())
        {
            decoderType = mModelConfig.getDataType();
        }
        mDecoder = std::make_shared<runtime::GptDecoderBatched>(mModelConfig.getVocabSize(),
            mModelConfig.getVocabSizePadded(mWorldConfig.getSize()), mRuntime->getStreamPtr(),
            mModelConfig.getSpeculativeDecodingMode(), decoderType);
        mDecoder->setup(decodingMode, getMaxNumSequences(), mOperatingBeamWidth, getMaxAttentionWindow(),
            getSinkTokenLen(), getMaxSequenceLen(), mModelConfig.getMaxDecodingTokens(), decoderType, mModelConfig);

        if (decodingMode.isExplicitDraftTokens())
        {
            mDecoder->setupExplicitDraftTokens(mDecoderBuffers->explicitDraftTokensBuffers);
        }
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtGptModelInflightBatching::createBuffers(executor::DecodingConfig const& decodingConfig,
    executor::ExtendedRuntimePerfKnobConfig const& extendedRuntimePerfKnobConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto allReduceCommPtrs = mAllReduceBuffers ? mAllReduceBuffers->mAllReduceCommPtrs : TensorPtr{};

    mBuffers.clear();
    for (SizeType32 i = 0; i < mNumBuffers; ++i)
    {
        mBuffers.emplace_back(std::make_shared<RuntimeBuffers>(getMaxBatchSize(), mOperatingBeamWidth,
            getMaxAttentionWindow(), getSinkTokenLen(), extendedRuntimePerfKnobConfig, allReduceCommPtrs, *mRuntime,
            mModelConfig, mWorldConfig, decodingConfig, getMaxNumTokens()));
    }

    mDecodingInputs.resize(mNumMicroBatches);

    mDecoderBuffers
        = std::make_shared<DecoderBuffers>(getMaxNumSequences(), mOperatingBeamWidth, getMaxAttentionWindow(),
            getMaxSequenceLen(), mModelConfig.getMaxDecodingTokens(), *mRuntime, mModelConfig, mWorldConfig);

    mSlotDecoderBuffers.clear();
    for (SizeType32 i = 0; i < getMaxNumSequences(); ++i)
    {
        mSlotDecoderBuffers.emplace_back(
            std::make_shared<SlotDecoderBuffers>(mOperatingBeamWidth, getMaxSequenceLen(), *mRuntime));
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtGptModelInflightBatching::assignReqSeqSlots(ScheduledRequests const& scheduledRequests)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(assignReqSeqSlots);

    mSeqSlotManager->freeIdleSequenceSlots();
    for (auto const& requests : {scheduledRequests.contextRequests, scheduledRequests.generationRequests})
    {
        for (auto const& llmReq : requests)
        {
            auto const isReqNew = llmReq->isContextInitState() && llmReq->isFirstContextChunk();
            auto const reqSeqSlot = mSeqSlotManager->getSequenceSlot(isReqNew, llmReq->mRequestId);
            TLLM_CHECK_WITH_INFO(reqSeqSlot, "Unable to get batch slot for reqId");
            llmReq->mSeqSlot = reqSeqSlot;
        }
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtGptModelInflightBatching::executeStep(
    RequestVector const& contextRequests, RequestVector const& generationRequests, SizeType32 bufferId)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(executeStep);

    auto& runtimeBuffers = *mBuffers[bufferId];

    auto [optProfileId, inputMap, outputMap]
        = runtimeBuffers.prepareStep(contextRequests, generationRequests, mOperatingBeamWidth, getMaxAttentionWindow(),
            *mDecoderBuffers, mKvCacheManager.get(), mCrossKvCacheManager.get(), mRnnStateManager.get(),
            mPeftTables[mMicroBatchId], *mRuntime, mModelConfig, mWorldConfig);

    // Do decoder setup before context phase if model needs to setup buffers for the context phase.
    if (mModelConfig.getSpeculativeDecodingMode().needsDecoderPrologue())
    {
        setupDecoderStep(contextRequests);

        mBuffers[bufferId]->prepareExplicitDraftTokenBuffers(*mDecoderBuffers, *mRuntime, mModelConfig, mWorldConfig);
    }

    mRuntime->setInputTensors(optProfileId, inputMap);
    mRuntime->setOutputTensors(optProfileId, outputMap);

    executeContext(optProfileId);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtGptModelInflightBatching::setupDecoderStep(RequestVector const& contextRequests)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(setupDecoderStep);

    auto const contextBufferId = mCtxGenFusion ? getFusedBufferId() : getContextBufferId();

    auto& manager = mRuntime->getBufferManager();
    auto& buffers = *mBuffers.at(contextBufferId);
    auto* decoderInputLengthsHost = bufferCast<SizeType32>(*buffers.decoderInputLengthsHost);
    TensorPtr inputIdsFlatView = ITensor::view(buffers.decoderInputsIds);
    TLLM_CHECK(inputIdsFlatView->getShape().nbDims == 1);

    std::vector<SizeType32> seqSlots;
    std::vector<decoder_batch::Request> decoderRequests;
    std::vector<SamplingConfig> samplingConfigs;
    SizeType32 inputOffset{0};
    SizeType32 batchIdx{0};
    for (auto const& llmReq : contextRequests)
    {
        if (!llmReq->isLastContextChunk())
        {
            continue;
        }

        auto const decoderInputLength = decoderInputLengthsHost[batchIdx];
        if (mWorldConfig.isLastPipelineParallelRank())
        {
            auto const seqSlot = llmReq->mSeqSlot.value();
            TensorPtr inputView = ITensor::slice(inputIdsFlatView, inputOffset, decoderInputLength);
            auto decoderRequest
                = decoder_batch::Request{inputView, decoderInputLength, llmReq->mMaxNewTokens, llmReq->mEndId};

            auto const& draftTokens = llmReq->getDraftTokens();
            llmReq->mSamplingConfig.normalizeLogProbs = isNormalizeLogProbs();
            if (mModelConfig.getSpeculativeDecodingMode().isDraftTokensExternal())
            {
                decoderRequest.draftTokens = manager.copyFrom(*draftTokens, MemoryType::kPINNED);
                auto const& draftLogits = llmReq->getDraftLogits();
                if (draftLogits.has_value())
                {
                    decoderRequest.draftLogits = manager.copyFrom(*draftLogits.value(), MemoryType::kPINNED);
                }
                decoderRequest.generatedTokensPerEngineStep = draftTokens->size() + 1;
            }
            else if (!mModelConfig.getSpeculativeDecodingMode().isNone())
            {
                decoderRequest.generatedTokensPerEngineStep = mModelConfig.getMaxDecodingTokens();
            }
            if (mModelConfig.getSpeculativeDecodingMode().isMedusa())
            {
                llmReq->mSamplingConfig.topKMedusaHeads = {buffers.medusaBuffers->mTopKs};
                decoderRequest.medusaPaths = ITensor::slice(buffers.medusaBuffers->medusaPathsDevice, seqSlot, 1);
                decoderRequest.medusaTreeIds = ITensor::slice(buffers.medusaBuffers->medusaTreeIdsDevice, seqSlot, 1);
            }
            if (llmReq->getEmbeddingBias().has_value())
            {
                auto embeddingBias = llmReq->getEmbeddingBias().value();
                // Check that embedding bias type is same as logits type
                auto const logitsType = mRuntime->getEngine().getTensorDataType("logits");
                if (embeddingBias->getDataType() != logitsType)
                {
                    TLLM_THROW("Embedding bias data type must be same as model logits type.");
                }
                decoderRequest.embeddingBias = embeddingBias;
            }
            if (llmReq->getBadWordsList().has_value())
            {
                // Move to GPU and remove leading bs1 dimension since this is what decoderRequest expects
                decoderRequest.badWordsList = manager.copyFrom(*llmReq->getBadWordsList().value(), MemoryType::kGPU);
                decoderRequest.badWordsList->squeeze(0);
            }
            if (llmReq->getStopWordsList().has_value())
            {
                decoderRequest.stopWordsList = manager.copyFrom(*llmReq->getStopWordsList().value(), MemoryType::kGPU);
                decoderRequest.stopWordsList->squeeze(0);
            }
            seqSlots.push_back(seqSlot);
            decoderRequests.push_back(decoderRequest);
            samplingConfigs.push_back(llmReq->mSamplingConfig);
        }

        inputOffset += decoderInputLength;
        ++batchIdx;
    }
    if (decoderRequests.size())
    {
        NVTX3_SCOPED_RANGE(decoderNewRequests);
        mDecoder->newRequests(seqSlots, decoderRequests, samplingConfigs);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtGptModelInflightBatching::postProcessRequest(
    LlmRequest& llmReq, SizeType32 batchIdx, std::vector<SizeType32> const& numDroppedTokens)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const seqSlot = llmReq.mSeqSlot.value();
    auto const reqBeamWidth = llmReq.mSamplingConfig.beamWidth;
    auto const& bufferManager = getBufferManager();

    if (llmReq.getReturnGenerationLogits() && llmReq.getGenerationLogitsFragments().size() > 0)
    {
        auto const genBufferId = mCtxGenFusion ? getFusedBufferId() : getGenerationBufferId();
        auto const& genRuntimeBuffers = *mBuffers.at(genBufferId);

        auto constexpr beforeDecoder = false;
        copyGenerationLogits(genRuntimeBuffers, bufferManager, llmReq, batchIdx, beforeDecoder, numDroppedTokens);

        bufferManager.getStream().synchronize();
    }

    if (mWorldConfig.isPipelineParallel())
    {
        // Send context logits from last to first PP rank
        if (llmReq.getReturnContextLogits())
        {
            if (mWorldConfig.isLastPipelineParallelRank())
            {
                mMpiCommPipelinePara->send(*(llmReq.getContextLogitsHost()), 0, 1);
            }
            else if (mWorldConfig.isFirstPipelineParallelRank())
            {
                mMpiCommPipelinePara->recv(
                    *(llmReq.getContextLogitsHost()), mWorldConfig.getPipelineParallelism() - 1, 1);
            }
        }

        // Send generation logits from last to first PP rank
        if (llmReq.getReturnGenerationLogits())
        {
            if (mWorldConfig.isLastPipelineParallelRank())
            {
                mMpiCommPipelinePara->send(*(llmReq.getGenerationLogitsHost()), 0, 2);
            }
            else if (mWorldConfig.isFirstPipelineParallelRank())
            {
                mMpiCommPipelinePara->recv(
                    *(llmReq.getGenerationLogitsHost()), mWorldConfig.getPipelineParallelism() - 1, 2);
            }
        }
    }

    if (reqBeamWidth == 1)
    {
        return;
    }

    // Update mDecoderBuffers->slotOutputIdsHost and synchronize
    getDecoderSlotHostOutputs(seqSlot, llmReq.returnLogProbs(), llmReq.mSamplingConfig);

    auto const* outputIdsHostData = bufferCast<TokenIdType>(*mSlotDecoderBuffers[seqSlot]->outputIdsHost);
    auto const* sequenceLengthsHostData = bufferCast<SizeType32>(*mSlotDecoderBuffers[seqSlot]->sequenceLengthsHost);
    auto const* cumLogProbsHostData = bufferCast<float>(*mSlotDecoderBuffers[seqSlot]->cumLogProbsHost);
    auto logProbsHost = mSlotDecoderBuffers[seqSlot]->logProbsHost;
    auto const* logProbsHostData = bufferCast<float>(*logProbsHost);

    auto const& outputIdsShape = mSlotDecoderBuffers[seqSlot]->outputIdsHost->getShape();
    auto const maxSeqLength = outputIdsShape.d[1];

    std::vector<std::vector<TokenIdType>> generatedTokens(reqBeamWidth);
    for (SizeType32 beam = 0; beam < reqBeamWidth; ++beam)
    {
        auto const* const begin = outputIdsHostData + tc::flat_index2(beam, llmReq.mPromptLen, maxSeqLength);
        auto const generatedLength = sequenceLengthsHostData[beam] - llmReq.mPromptLen;
        auto const* const end = begin + generatedLength;
        generatedTokens[beam].assign(begin, end);

        if (llmReq.returnLogProbs())
        {
            llmReq.setCumLogProb(cumLogProbsHostData[beam], beam);

            auto const beginLogProbsOffset = reqBeamWidth == 1 ? llmReq.mPromptLen : 0;
            auto const* const begin = logProbsHostData + beam * logProbsHost->getShape().d[1] + beginLogProbsOffset;
            auto const* const end = begin + generatedLength;
            LlmRequest::VecLogProbs logProbs(begin, end);
            llmReq.setLogProbs(logProbs, beam);
        }
    }
    llmReq.setGeneratedTokens(generatedTokens);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtGptModelInflightBatching::getDecoderSlotHostOutputs(
    SizeType32 seqSlot, bool returnLogProbs, SamplingConfig const& samplingConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TensorPtr outputIdsView = mSlotDecoderBuffers[seqSlot]->outputIds;
    // Sequence Length is already computed on device
    TensorPtr sequenceLengthView = ITensor::slice(mDecoderBuffers->sequenceLengths, seqSlot, 1);
    TensorPtr cumLogProbsView = nullptr;
    TensorPtr logProbsView = nullptr;
    if (returnLogProbs)
    {
        cumLogProbsView = mSlotDecoderBuffers[seqSlot]->cumLogProbs;
        logProbsView = mSlotDecoderBuffers[seqSlot]->logProbs;
    }

    if (mWorldConfig.isLastPipelineParallelRank())
    {
        auto event = mDecoder->finalize(seqSlot, samplingConfig);
        // Make sure that postprocessing is done before copying outputIds
        mCopyBufferManager.getStream().wait(event.get());

        auto outputIds = mDecoder->getOutputIds(seqSlot);
        auto cumLogProbs = mDecoder->getCumLogProbs(seqSlot);
        auto logProbs = mDecoder->getLogProbs(seqSlot);

        runtime::CudaEvent beforeEvent{};
        mRuntime->getStreamPtr()->record(beforeEvent);
        mCopyBufferManager.getStream().wait(beforeEvent);
        mCopyBufferManager.copy(*outputIds, *outputIdsView);
        if (returnLogProbs)
        {
            mCopyBufferManager.copy(*cumLogProbs, *cumLogProbsView);
            mCopyBufferManager.copy(*logProbs, *logProbsView);
        }

        if (mWorldConfig.isPipelineParallel())
        {
            // Make sure that postprocessing is done before sending outputIds
            event.synchronize();

            auto const peerSend = 0;
            mDecSlotAsyncSndHdls.emplace_back(mSlotDecoderBuffers[seqSlot]->asyncSend(
                mMpiCommPipelinePara, outputIds, sequenceLengthView, cumLogProbs, logProbs, returnLogProbs, peerSend));
        }
    }
    else
    {
        auto const peerRecv = mWorldConfig.getPipelineParallelRank() == 0 ? mWorldConfig.getPipelineParallelism() - 1
                                                                          : mWorldConfig.getPipelineParallelRank() - 1;
        mSlotDecoderBuffers[seqSlot]->recv(mMpiCommPipelinePara, sequenceLengthView, returnLogProbs, peerRecv);

        auto const peerSend = mWorldConfig.getPipelineParallelRank() + 1;
        if (peerSend != mWorldConfig.getPipelineParallelism() - 1)
        {
            mDecSlotAsyncSndHdls.emplace_back(mSlotDecoderBuffers[seqSlot]->asyncSend(
                mMpiCommPipelinePara, sequenceLengthView, returnLogProbs, peerSend));
        }
    }
    sync_check_cuda_error();

    // Here copy stream is synchronized after receiving decoderSlotOutputIdsView either by copy or by receive
    // before copying to host on copy stream
    runtime::CudaEvent beforeEvent{};
    mRuntime->getStreamPtr()->record(beforeEvent);
    mCopyBufferManager.getStream().wait(beforeEvent);
    mCopyBufferManager.copy(*outputIdsView, *mSlotDecoderBuffers[seqSlot]->outputIdsHost);
    mCopyBufferManager.copy(*sequenceLengthView, *mSlotDecoderBuffers[seqSlot]->sequenceLengthsHost);

    if (returnLogProbs)
    {
        mCopyBufferManager.copy(*cumLogProbsView, *mSlotDecoderBuffers[seqSlot]->cumLogProbsHost);
        mCopyBufferManager.copy(*logProbsView, *mSlotDecoderBuffers[seqSlot]->logProbsHost);
    }

    // Make sure copy is done before continuing on host
    mCopyBufferManager.getStream().synchronize();

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

TrtGptModelInflightBatching::TokenPtr TrtGptModelInflightBatching::decoderStepAsync(
    ScheduledRequests const& scheduledRequests)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(decoderStepAsync);

    auto& manager = mRuntime->getBufferManager();
    auto const& stream = mRuntime->getStream();

    auto const contextBufferId = mCtxGenFusion ? getFusedBufferId() : getContextBufferId();
    auto& contextRuntimeBuffers = *mBuffers.at(contextBufferId);
    SizeType32 batchIndex{0};
    SizeType32 logitsIndex{0};
    // Copy logits into mDecoderBuffers->logits
    for (auto const& llmReq : scheduledRequests.contextRequests)
    {
        auto const reqBeamWidth = llmReq->mSamplingConfig.beamWidth;
        auto const numContextLogits = contextRuntimeBuffers.numContextLogits.at(batchIndex);
        auto const draftLength = llmReq->getNumDraftTokens();

        TLLM_LOG_DEBUG("logitsIndex: %d", logitsIndex);
        TLLM_LOG_DEBUG("numContextLogits %d", numContextLogits);
        TLLM_LOG_DEBUG("draftLength: %d", draftLength);

        if (mModelConfig.computeContextLogits())
        {
            // Since the computational graph has been modified, only the last token is needed.
            TLLM_CHECK_WITH_INFO(!mModelConfig.getSpeculativeDecodingMode().isMedusa()
                    && !mModelConfig.getSpeculativeDecodingMode().isLookaheadDecoding(),
                "Return context logits is not supported with Medusa and Lookahead decoding");

            if (llmReq->getReturnContextLogits())
            {
                TensorPtr contextLogitsDeviceView
                    = ITensor::slice(contextRuntimeBuffers.logits, logitsIndex, numContextLogits);
                if (llmReq->isFullContextRequest())
                {
                    // Output at the back of context logits (skip reused part)
                    auto const offset = llmReq->mPromptLen - numContextLogits;
                    if (offset != 0)
                    {
                        TLLM_LOG_WARNING(
                            "Because of KV cache reuse, not all context logits could be produced for request %lu.",
                            llmReq->mRequestId);
                    }
                    TensorPtr contextLogitsHostView
                        = ITensor::slice(llmReq->getContextLogitsHost(), offset, numContextLogits);
                    // Copy to host directly
                    manager.copy(*contextLogitsDeviceView, *contextLogitsHostView);
                }
                else
                {
                    // For chunked context, output at the position of the chunk
                    TensorPtr contextLogitsHostView = ITensor::slice(
                        llmReq->getContextLogitsHost(), llmReq->getContextCurrentPosition(), numContextLogits);
                    // Copy to host directly
                    manager.copy(*contextLogitsDeviceView, *contextLogitsHostView);
                }
            }
        }
        logitsIndex += numContextLogits + draftLength;

        // Get the logits from the last context token and draft tokens
        auto const numDecoderLogits = 1 + draftLength;
        auto const seqSlot = llmReq->mSeqSlot.value();
        auto& decoderLogits = mDecoderBuffers->logits.at(seqSlot);
        TensorPtr logitsView
            = ITensor::slice(contextRuntimeBuffers.logits, logitsIndex - numDecoderLogits, numDecoderLogits);

        if (mModelConfig.getSpeculativeDecodingMode().hasDraftLogits())
        {
            auto& medusaLogitsHeads = mDecoderBuffers->draftBuffers.predictedDraftLogits.at(seqSlot);
            setupMedusaLogits(medusaLogitsHeads, contextRuntimeBuffers.medusaBuffers->medusaLogitsDevice,
                mModelConfig.getSpeculativeDecodingModule().getMaxDraftPathLen(), logitsIndex - numDecoderLogits,
                numDecoderLogits);
        }

        // Save the last token logits of context into generation logits or
        // save the accepted token logits from target model
        if (llmReq->getReturnGenerationLogits())
        {
            copyLastContextLogits(logitsView, *llmReq, manager);
        }

        TLLM_CHECK_DEBUG_WITH_INFO(
            tru::tensorHasNan<float>(*logitsView, manager, "logits") == false, "Found Nan in logits");
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

    // Slice logits of generation requests
    logitsIndex = mCtxGenFusion ? logitsIndex : 0;
    auto batchIdx = scheduledRequests.contextRequests.size();
    auto const genBufferId = mCtxGenFusion ? getFusedBufferId() : getGenerationBufferId();
    auto& genRuntimeBuffers = *mBuffers.at(genBufferId);
    for (auto const& llmReq : scheduledRequests.generationRequests)
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
        TLLM_CHECK_DEBUG_WITH_INFO(
            tru::tensorHasNan<float>(*logitsView, manager, "logits") == false, "Found Nan in logits");
        auto& decoderLogits = mDecoderBuffers->logits.at(seqSlot);
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
            TLLM_CHECK_WITH_INFO(mModelConfig.getSpeculativeDecodingMode().isNone()
                    || mModelConfig.getSpeculativeDecodingMode().isDraftTokensExternal(),
                "Only speculative decoding with external draft tokens supports returning generation logits");

            // Push into fragments vector
            llmReq->addGenerationLogitsFragment(logitsView);
            TLLM_CHECK(llmReq->getGenerationLogitsFragmentsSize() <= GENERATION_LOGITS_BUFFER_LENGTH);
            // Copy back to host for every GENERATION_LOGITS_BUFFER_LENGTH steps to mitigate GPU memory pressure
            if (llmReq->getGenerationLogitsFragmentsSize() == GENERATION_LOGITS_BUFFER_LENGTH)
            {
                auto constexpr beforeDecoder = true;
                copyGenerationLogits(genRuntimeBuffers, manager, *llmReq, batchIdx, beforeDecoder);
            }
        }
        if (mModelConfig.getSpeculativeDecodingMode().hasDraftLogits())
        {
            auto& medusaLogitsHeads = mDecoderBuffers->draftBuffers.predictedDraftLogits.at(seqSlot);
            setupMedusaLogits(medusaLogitsHeads, genRuntimeBuffers.medusaBuffers->medusaLogitsDevice,
                mModelConfig.getSpeculativeDecodingModule().getMaxDraftPathLen(), logitsIndex, draftLength);
        }
        logitsIndex += numLogits;
        ++batchIdx;
    }

    // Copy indirection output into input
    // TODO: Could we avoid this by modifying batchDecoder to take a vector of tensors instead?
    copyCacheIndirectionFromOutputsToInputs(scheduledRequests, genBufferId);

    for (auto const& requests : {scheduledRequests.contextRequests, scheduledRequests.generationRequests})
    {
        for (auto const& llmReq : requests)
        {
            if (llmReq->mLogitsPostProcessor)
            {
                auto& logits = mDecoderBuffers->logits.at(llmReq->mSeqSlot.value());
                llmReq->mLogitsPostProcessor.value()(
                    llmReq->mRequestId, logits, llmReq->getTokens(), mRuntime->getStreamPtr(), llmReq->mClientId);
            }
        }
    }

    std::vector<LlmRequest::RequestIdType> reqIdsVec;
    std::vector<LlmRequest::TensorPtr> logitsVec;
    std::vector<std::reference_wrapper<LlmRequest::BeamTokens const>> beamTokensVec;
    std::vector<std::optional<LlmRequest::RequestIdType>> clientIdsVec;

    for (auto const& requests : {scheduledRequests.contextRequests, scheduledRequests.generationRequests})
    {
        for (auto const& llmReq : requests)
        {
            if (llmReq->mApplyLogitsPostProcessorBatched)
            {
                reqIdsVec.push_back(llmReq->mRequestId);

                auto& logits = mDecoderBuffers->logits.at(llmReq->mSeqSlot.value());
                logitsVec.push_back(logits);

                beamTokensVec.emplace_back(llmReq->getTokens());
                clientIdsVec.push_back(llmReq->mClientId);
            }
        }
    }
    if (!reqIdsVec.empty())
    {
        mLogitsPostProcessorBatched.value()(
            reqIdsVec, logitsVec, beamTokensVec, mRuntime->getStreamPtr(), clientIdsVec);
    }

    auto& decodingInput = mDecodingInputs.at(mMicroBatchId);
    auto const active = computeActiveVec(scheduledRequests);
    decodingInput = std::make_shared<decoder_batch::Input>(mDecoderBuffers->logits, active);

    decodingInput->cacheIndirection = mDecoderBuffers->cacheIndirectionInput;
    if (mDecoder->getDecodingMode() == executor::DecodingMode::BeamSearch())
    {
        decodingInput->seqSlots
            = BufferManager::pinnedPool(ITensor::makeShape({static_cast<ITensor::DimType64>(scheduledRequests.size())}),
                TRTDataType<SizeType32>::value);
    }

    if (mModelConfig.getSpeculativeDecodingMode().hasDraftLogits())
    {
        decodingInput->predictedDraftLogits = mDecoderBuffers->draftBuffers.predictedDraftLogits;
    }

    if (mModelConfig.getSpeculativeDecodingMode().isExplicitDraftTokens())
    {
        // requires mCtxGenFusion == true
        decodingInput->seqSlots = genRuntimeBuffers.seqSlots;
        decodingInput->explicitDraftTokensInputs = genRuntimeBuffers.explicitDraftTokensBuffers->engineOutputs;
        decodingInput->explicitDraftTokensLastInputs = genRuntimeBuffers.explicitDraftTokensBuffers->engineInputs;
    }

    mDecodingOutput = std::make_shared<decoder_batch::Output>();
    mDecodingOutput->cacheIndirection = mDecoderBuffers->cacheIndirectionOutput;
    mDecodingOutput->sequenceLengths = mDecoderBuffers->sequenceLengths;

    TokenPtr decoderToken = mDecoder->forwardAsync(*mDecodingOutput, *decodingInput);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return decoderToken;
}

void TrtGptModelInflightBatching::copyCacheIndirectionFromOutputsToInputs(
    ScheduledRequests const& scheduledRequests, SizeType32 genBufferId)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(copyCacheIndirectionFromOutputsToInputs);

    auto& genRuntimeBuffers = *mBuffers.at(genBufferId);
    auto* srcOffsetsPtr = bufferCast<SizeType32>(*genRuntimeBuffers.cacheIndirDecoderIOBatchedCopySrcOffsets);
    auto* dstOffsetsPtr = bufferCast<SizeType32>(*genRuntimeBuffers.cacheIndirDecoderIOBatchedCopyDstOffsets);
    auto* copySizesPtr = bufferCast<SizeType32>(*genRuntimeBuffers.cacheIndirDecoderIOBatchedCopySizes);

    auto const& cacheIndirShape = mDecoderBuffers->cacheIndirectionOutput->getShape();

    SizeType32 batchIdx{0};
    SizeType32 maxCopySize{0};
    auto& manager = mRuntime->getBufferManager();
    for (auto const& requests : {scheduledRequests.contextRequests, scheduledRequests.generationRequests})
    {
        for (auto const& llmReq : requests)
        {
            auto const reqBeamWidth = llmReq->mSamplingConfig.beamWidth;
            auto const seqSlot = llmReq->mSeqSlot.value();

            auto const copySize = static_cast<SizeType32>(cacheIndirShape.d[2]) * reqBeamWidth;
            srcOffsetsPtr[batchIdx] = seqSlot * copySize;
            dstOffsetsPtr[batchIdx] = seqSlot * copySize;
            copySizesPtr[batchIdx] = copySize;
            maxCopySize = std::max(maxCopySize, copySize);

            batchIdx++;
        }
    }
    if (batchIdx != 0)
    {
        auto const srcOffsetsSlice
            = ITensor::slice(genRuntimeBuffers.cacheIndirDecoderIOBatchedCopySrcOffsets, 0, batchIdx);
        auto const srcOffsetsSliceDeviceSlice
            = ITensor::slice(genRuntimeBuffers.mCacheIndirDecoderIOBatchedCopySrcOffsetsSliceDevice, 0, batchIdx);
        manager.copy(srcOffsetsSlice->data(), *srcOffsetsSliceDeviceSlice,
            runtime::MemoryType::kGPU); // Explicitly move to device for faster access.
        auto const dstOffsetsSlice
            = ITensor::slice(genRuntimeBuffers.cacheIndirDecoderIOBatchedCopyDstOffsets, 0, batchIdx);
        auto const dstOffsetsSliceDeviceSlice
            = ITensor::slice(genRuntimeBuffers.mCacheIndirDecoderIOBatchedCopyDstOffsetsSliceDevice, 0, batchIdx);
        manager.copy(dstOffsetsSlice->data(), *dstOffsetsSliceDeviceSlice,
            runtime::MemoryType::kGPU); // Explicitly move to device for faster access.
        auto const sizesSlice = ITensor::slice(genRuntimeBuffers.cacheIndirDecoderIOBatchedCopySizes, 0, batchIdx);
        auto const copySizesDeviceSlice
            = ITensor::slice(genRuntimeBuffers.mCacheIndirDecoderIOBatchedCopyCopySizesDevice, 0, batchIdx);
        manager.copy(sizesSlice->data(), *copySizesDeviceSlice); // Explicitly move to device for faster access.
        runtime::kernels::invokeCopyBatch(*mDecoderBuffers->cacheIndirectionOutput,
            *mDecoderBuffers->cacheIndirectionInput, *srcOffsetsSliceDeviceSlice, *dstOffsetsSliceDeviceSlice,
            *copySizesDeviceSlice, maxCopySize, manager.getStream());
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

std::vector<bool> TrtGptModelInflightBatching::computeActiveVec(ScheduledRequests const& scheduledRequests)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    std::vector<bool> active(getMaxNumSequences(), false);
    for (auto const& requests : {scheduledRequests.contextRequests, scheduledRequests.generationRequests})
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

std::unique_ptr<DecoderStepAsyncSend> TrtGptModelInflightBatching::decoderSync(
    ScheduledRequests const& scheduledRequests, TokenPtr const& decoderToken)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(decoderSync);

    // If one of the request needs log probs, need to get from decoder and communicate
    bool returnLogProbs{false};
    for (auto const& requests : {scheduledRequests.contextRequests, scheduledRequests.generationRequests})
    {
        for (auto const& llmReq : requests)
        {
            if (llmReq->returnLogProbs())
            {
                returnLogProbs = true;
                break;
            }
        }
    }

    std::unique_ptr<DecoderStepAsyncSend> asyncHandle = nullptr;
    if (mWorldConfig.isLastPipelineParallelRank())
    {
        // Wait for the decoder event for that context
        mDecoder->forwardSync(*decoderToken);

        // Get finished
        auto const decoderFinished = mDecoder->getFinished();
        std::copy(decoderFinished.begin(), decoderFinished.end(), bufferCast<std::uint8_t>(*mDecoderBuffers->finished));

        mDecoderBuffers->newOutputTokens = mDecoder->getAllNewTokens();

        // Here host is already synchronized with forwardSync event, so can trigger copy
        // Using a different stream
        runtime::CudaEvent beforeEvent{};
        mRuntime->getStreamPtr()->record(beforeEvent);
        mCopyBufferManager.getStream().wait(beforeEvent);
        mCopyBufferManager.copy(*mDecoderBuffers->newOutputTokens, *mDecoderBuffers->newOutputTokensHost);
        mCopyBufferManager.copy(*mDecoderBuffers->sequenceLengths, *mDecoderBuffers->sequenceLengthsHost);

        if (returnLogProbs)
        {
            mDecoderBuffers->cumLogProbs = mDecoder->getCumLogProbs();
            mDecoderBuffers->logProbs = mDecoder->getLogProbs();
            mCopyBufferManager.copy(*mDecoderBuffers->cumLogProbs, *mDecoderBuffers->cumLogProbsHost);
            mCopyBufferManager.copy(*mDecoderBuffers->logProbs, *mDecoderBuffers->logProbsHost);
        }

        if (mModelConfig.getSpeculativeDecodingMode().predictsDraftTokens())
        {
            mDecoderBuffers->draftBuffers.nextDraftTokensDevice = mDecoder->getNextDraftTokens();
            mCopyBufferManager.copy(*mDecoderBuffers->draftBuffers.nextDraftTokensDevice,
                *mDecoderBuffers->draftBuffers.nextDraftTokensHost);

            if (mModelConfig.getSpeculativeDecodingMode().variableDraftLength())
            {
                mDecoderBuffers->draftBuffers.nextDraftTokensLengthsDevice = mDecoder->getNextDraftTokensLengths();
                mDecoderBuffers->draftBuffers.prevDraftTokensLengthsDevice = mDecoder->getPrevDraftTokensLengths();
                mCopyBufferManager.copy(*mDecoderBuffers->draftBuffers.nextDraftTokensLengthsDevice,
                    *mDecoderBuffers->draftBuffers.nextDraftTokensLengthsHost);
                mCopyBufferManager.copy(*mDecoderBuffers->draftBuffers.prevDraftTokensLengthsDevice,
                    *mDecoderBuffers->draftBuffers.prevDraftTokensLengthsHost);
            }
        }

        if (mModelConfig.getSpeculativeDecodingMode().needsKVCacheRewind())
        {
            mDecoderBuffers->draftBuffers.acceptedLengthsCumSumDevice = mDecoder->getAcceptedLengthsCumSum();
            mDecoderBuffers->draftBuffers.acceptedPackedPathsDevice = mDecoder->getAcceptedPackedPaths();
        }

        // Make sure host only continues when copy is done
        runtime::CudaEvent copyEvent{};
        mCopyBufferManager.getStream().record(copyEvent);
        copyEvent.synchronize();

        if (mWorldConfig.isPipelineParallel())
        {
            auto const peerSend = 0;
            asyncHandle = mDecoderBuffers->asyncSend(mMpiCommPipelinePara, returnLogProbs, mOperatingBeamWidth,
                mModelConfig.getSpeculativeDecodingMode().needsKVCacheRewind(), peerSend);
        }
    }
    else
    {
        auto const peerRecv = mWorldConfig.isFirstPipelineParallelRank() ? mWorldConfig.getPipelineParallelism() - 1
                                                                         : mWorldConfig.getPipelineParallelRank() - 1;
        mDecoderBuffers->recv(mMpiCommPipelinePara, returnLogProbs, mOperatingBeamWidth,
            mModelConfig.getSpeculativeDecodingMode().needsKVCacheRewind(), peerRecv);
        auto const peerSend = mWorldConfig.getPipelineParallelRank() + 1;
        if (peerSend != mWorldConfig.getPipelineParallelism() - 1)
        {
            asyncHandle = mDecoderBuffers->asyncSend(mMpiCommPipelinePara, returnLogProbs, mOperatingBeamWidth,
                mModelConfig.getSpeculativeDecodingMode().needsKVCacheRewind(), peerSend);
        }
    }

    auto const hostNewOutputTokensShape = mDecoderBuffers->newOutputTokensHost->getShape();
    auto const* const hostNewOutputTokensData = bufferCast<TokenIdType const>(*mDecoderBuffers->newOutputTokensHost);
    auto const* const sequenceLengthsHostData = bufferCast<SizeType32 const>(*mDecoderBuffers->sequenceLengthsHost);
    auto const* const decoderFinishedPtr = bufferCast<std::uint8_t const>(*mDecoderBuffers->finished);
    auto const* const cumLogProbsPtr = bufferCast<float const>(*mDecoderBuffers->cumLogProbsHost);
    auto const* const logProbsPtr = bufferCast<float const>(*mDecoderBuffers->logProbsHost);
    auto const* const nextDraftTokensHostData = mModelConfig.getSpeculativeDecodingMode().predictsDraftTokens()
        ? bufferCast<TokenIdType const>(*mDecoderBuffers->draftBuffers.nextDraftTokensHost)
        : nullptr;
    auto const* const nextDraftTokensLengthsHostData = mModelConfig.getSpeculativeDecodingMode().predictsDraftTokens()
            && mModelConfig.getSpeculativeDecodingMode().variableDraftLength()
        ? bufferCast<SizeType32 const>(*mDecoderBuffers->draftBuffers.nextDraftTokensLengthsHost)
        : nullptr;

    SizeType32 batchIdx{0};
    SizeType32 numSequences{0};
    // Update the request table tokens
    for (auto const& requests : {scheduledRequests.contextRequests, scheduledRequests.generationRequests})
    {
        for (auto const& llmReq : requests)
        {
            auto const reqBeamWidth = llmReq->mSamplingConfig.beamWidth;
            if (llmReq->isContextInitState())
            {
                ++batchIdx;
                numSequences += reqBeamWidth;
                continue;
            }
            auto const seqSlot = llmReq->mSeqSlot.value();
            auto const numGeneratedTokens = llmReq->getNumDraftTokens() + 1;
            auto const currentNumOfTokens = llmReq->getMaxBeamNumTokens();

            // Save the accepted token logits from target model
            if (llmReq->getReturnGenerationLogits()
                && mModelConfig.getSpeculativeDecodingMode().isDraftTokensExternal())
            {
                TLLM_CHECK_WITH_INFO(reqBeamWidth == 1, "Speculative decoding only works for beam width == 1");

                SizeType32 numAcceptedTokens
                    = sequenceLengthsHostData[seqSlot * mOperatingBeamWidth + 0] - llmReq->getMaxBeamNumTokens();

                auto const& generationLogitsHost = llmReq->getGenerationLogitsHost();
                auto shape = generationLogitsHost->getShape();
                shape.d[1] = numAcceptedTokens;
                generationLogitsHost->reshape(shape);
            }

            std::vector<SizeType32> numDroppedTokens(reqBeamWidth);
            for (SizeType32 beam = 0; beam < reqBeamWidth; ++beam)
            {
                auto const seqLen = sequenceLengthsHostData[seqSlot * mOperatingBeamWidth + beam];
                // The content of newOutputTokens might not be accurate in the case where
                // the sequence has finished early due to end token, so only add new tokens
                // to llmReq if decoder seq length is greater than current number of tokens
                auto const numNewTokens = std::min(numGeneratedTokens, seqLen - llmReq->getNumTokens(beam));
                numDroppedTokens[beam] = numGeneratedTokens - numNewTokens;
                for (SizeType32 step = 0; step < numNewTokens; ++step)
                {
                    auto const newTokenIdx = tc::flat_index(hostNewOutputTokensShape.d, step, seqSlot, beam);
                    auto const newToken = hostNewOutputTokensData[newTokenIdx];
                    llmReq->addNewToken(newToken, beam);

                    if (llmReq->returnLogProbs())
                    {
                        auto const cumLogProb = cumLogProbsPtr[seqSlot * mOperatingBeamWidth + beam];
                        llmReq->setCumLogProb(cumLogProb, beam);

                        auto const beginLogProbsOffset = reqBeamWidth == 1 ? llmReq->mPromptLen : 0;
                        SizeType32 offset
                            = (seqSlot * mOperatingBeamWidth + beam) * getMaxSequenceLen() + beginLogProbsOffset;
                        auto const generatedLength = seqLen - llmReq->mPromptLen;
                        std::vector<float> logProbs(logProbsPtr + offset, logProbsPtr + offset + generatedLength);
                        llmReq->setLogProbs(logProbs, beam);
                    }
                }
            }

            // Set number of tokens predicted per runtime iteration. Will be > 1 for speculative decoding.
            llmReq->setNumTokensPerIteration(llmReq->getMaxBeamNumTokens() - currentNumOfTokens);

            // Fill new draft tokens for the next step
            // FIXME(nkorobov): remove this when lookahead is supported
            if (!mModelConfig.getSpeculativeDecodingMode().isLookaheadDecoding() && decoderFinishedPtr[seqSlot] == 0U
                && (mModelConfig.getSpeculativeDecodingMode().predictsDraftTokens()
                    || mModelConfig.getSpeculativeDecodingMode().needsKVCacheRewind()))
            {
                auto const maxDraftTokensLen = mModelConfig.getMaxDecodingDraftTokens();
                auto const prevDraftTokensLen = llmReq->getNumDraftTokens();
                auto nextDraftTokensLen = mModelConfig.getSpeculativeDecodingModule().getMaxDecodingDraftTokens();
                if (mModelConfig.getSpeculativeDecodingMode().variableDraftLength())
                {
                    nextDraftTokensLen = nextDraftTokensLengthsHostData[seqSlot];
                }
                TLLM_CHECK(nextDraftTokensLen <= maxDraftTokensLen);

                auto draftTokensShared
                    = std::make_shared<std::vector<TokenIdType>>(nextDraftTokensHostData + seqSlot * maxDraftTokensLen,
                        nextDraftTokensHostData + seqSlot * maxDraftTokensLen + nextDraftTokensLen);

                llmReq->setDraftTokens(draftTokensShared);

                // For all phases except context that does not have draft tokens
                if (prevDraftTokensLen != 0 && mModelConfig.getSpeculativeDecodingMode().needsKVCacheRewind())
                {
                    // -1 here is for current 'main' token
                    auto const acceptedTokensLen = llmReq->getMaxBeamNumTokens() - currentNumOfTokens - 1;
                    TLLM_CHECK(0 <= acceptedTokensLen && acceptedTokensLen <= prevDraftTokensLen);
                    TLLM_LOG_DEBUG("Request %d accepted %d draft tokens", llmReq->mRequestId, acceptedTokensLen);

                    auto const rewindLength = prevDraftTokensLen - acceptedTokensLen;
                    // At this point, KV cache rows are already gathered and moved to the right location.
                    // We can safely rewind (draft - accepted) tokens
                    mKvCacheManager->rewindKVCache(seqSlot, rewindLength);
                }
            }

            // Terminate if request has finished or if it is speculative decoding target model
            // FIXME(nkorobov): remove this when lookahead is supported
            if (decoderFinishedPtr[seqSlot] != 0U || mModelConfig.getSpeculativeDecodingMode().isLookaheadDecoding()
                || mModelConfig.getSpeculativeDecodingMode().isDraftTokensExternal())
            {
                postProcessRequest(*llmReq, batchIdx, numDroppedTokens);
                if (!mWorldConfig.isPipelineParallel() || !mWorldConfig.isLastPipelineParallelRank())
                {
                    llmReq->mState = REQUEST_STATE_GENERATION_COMPLETE;
                    terminateRequest(llmReq);
                }
                else
                {
                    llmReq->mState = REQUEST_STATE_GENERATION_TO_COMPLETE;
                }
            }
            else
            {
                llmReq->mState = REQUEST_STATE_GENERATION_IN_PROGRESS;
            }
            ++batchIdx;
            numSequences += reqBeamWidth;

            llmReq->advanceDecodingIter();
        }
    }
    // FIXME(nkorobov): remove this when lookahead is supported
    if (!mModelConfig.getSpeculativeDecodingMode().isLookaheadDecoding()
        && mModelConfig.getSpeculativeDecodingMode().needsKVCacheRewind())
    {
        TLLM_CHECK_WITH_INFO(mCtxGenFusion, "Current speculative decoding mode requires context-gen fusion IFB");
        rewindKVCacheBlocks(numSequences);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return asyncHandle;
}

void TrtGptModelInflightBatching::rewindKVCacheBlocks(SizeType32 numSequences)
{
    auto const bufferId = getFusedBufferId();
    auto& runtimeBuffers = *mBuffers.at(bufferId);

    auto const localNbLayers = mModelConfig.getNbAttentionLayers(mWorldConfig.getPipelineParallelism());
    auto const numKvHeads = mModelConfig.getNbKvHeads();
    auto const tokensPerBlock = mModelConfig.getTokensPerBlock();
    auto const elemSize = BufferDataType(mModelConfig.getKvDataType()).getSize();
    auto const sizeInBytesPerKVHead = mModelConfig.getSizePerHead() * elemSize;
    auto const maxBlocksPerSeq = mKvCacheManager->getMaxBlocksPerSeq();

    auto* const* pointerArrayPtr = bufferCast<void*>(*runtimeBuffers.transformerBuffers->kvCacheBlockPoolPointers);
    auto const* offsetArrayPtr
        = bufferCast<tk::KVCacheIndex>(*runtimeBuffers.transformerBuffers->kvCacheBlockOffsetsDevice);

    auto commonRewindLen = mModelConfig.getSpeculativeDecodingModule().getMaxDecodingDraftTokens();
    SizeType32 const* rewindLens = nullptr;
    if (mModelConfig.getSpeculativeDecodingMode().variableDraftLength())
    {
        commonRewindLen = 0;
        rewindLens = bufferCast<SizeType32 const>(*mDecoderBuffers->draftBuffers.prevDraftTokensLengthsHost);
    }

    tensorrt_llm::runtime::kernels::invokeUpdateKVBlockArrayDraftTokenLocation(
        *mDecoderBuffers->draftBuffers.acceptedLengthsCumSumDevice,
        *mDecoderBuffers->draftBuffers.acceptedPackedPathsDevice, *runtimeBuffers.sequenceLengthsDevice,
        pointerArrayPtr, offsetArrayPtr, localNbLayers, numSequences, numKvHeads, sizeInBytesPerKVHead, commonRewindLen,
        rewindLens, *runtimeBuffers.seqSlotRemappingDevice, *runtimeBuffers.sortedSeqSlots, getMaxAttentionWindow(),
        maxBlocksPerSeq, tokensPerBlock, mRuntime->getStreamPtr()->get());

    sync_check_cuda_error();
}

nvinfer1::DataType TrtGptModelInflightBatching::getLogitDataType() const
{
    return mModelConfig.getLogitsDtype();
}

void TrtGptModelInflightBatching::changeBeamWidth(SizeType32 beamWidth)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TLLM_CHECK(mInflightReqIds.empty());

    TLLM_CHECK_WITH_INFO(beamWidth <= getMaxBeamWidth(),
        "Requested beam width %d is larger than configured max beam width %d", beamWidth, getMaxBeamWidth());
    TLLM_LOG_INFO("Changing operating beam width from %d to %d", mOperatingBeamWidth, beamWidth);
    mOperatingBeamWidth = beamWidth;

    createBuffers(mDecodingConfig, mExtendedRuntimePerfKnobConfig);
    createDecoder(mDecodingConfig.getDecodingMode());

    if (mKvCacheManager)
    {
        reshapeKvTensors(*mKvCacheManager);

        for (auto& buffers : mBuffers)
        {
            buffers->transformerBuffers->setKvPoolPointers(*mKvCacheManager);
        }
    }
    if (mCrossKvCacheManager)
    {
        reshapeKvTensors(*mCrossKvCacheManager);

        for (auto& buffers : mBuffers)
        {
            buffers->transformerBuffers->setKvPoolPointers(*mCrossKvCacheManager);
        }
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtGptModelInflightBatching::getCurrentIterationStats(executor::IterationStats& stats) const
{
    // KVCacheManager statistics
    auto const& kvCacheManager = getKVCacheManager();
    if (kvCacheManager)
    {
        executor::KvCacheStats kvStats;
        auto kvCacheStats = kvCacheManager->getKvCacheStats();
        kvStats.maxNumBlocks = kvCacheStats.maxNumBlocks;
        kvStats.freeNumBlocks = kvCacheStats.freeNumBlocks;
        kvStats.usedNumBlocks = kvCacheStats.usedNumBlocks;
        kvStats.tokensPerBlock = kvCacheStats.toksPerBlock;
        kvStats.allocTotalBlocks = kvCacheStats.allocTotalBlocks;
        kvStats.allocNewBlocks = kvCacheStats.allocNewBlocks;
        kvStats.reusedBlocks = kvCacheStats.reusedBlocks;
        stats.kvCacheStats = kvStats;
    }
    auto const& crossKvCacheManager = getCrossKVCacheManager();
    if (crossKvCacheManager)
    {
        executor::KvCacheStats kvStats;
        auto kvCacheStats = crossKvCacheManager->getKvCacheStats();
        kvStats.maxNumBlocks = kvCacheStats.maxNumBlocks;
        kvStats.freeNumBlocks = kvCacheStats.freeNumBlocks;
        kvStats.usedNumBlocks = kvCacheStats.usedNumBlocks;
        kvStats.tokensPerBlock = kvCacheStats.toksPerBlock;
        kvStats.allocTotalBlocks = kvCacheStats.allocTotalBlocks;
        kvStats.allocNewBlocks = kvCacheStats.allocNewBlocks;
        kvStats.reusedBlocks = kvCacheStats.reusedBlocks;
        stats.crossKvCacheStats = kvStats;
    }
    executor::InflightBatchingStats modelStats;
    modelStats.numScheduledRequests = mLastIterationStatsIFB.scheduledRequests.size();
    modelStats.numContextRequests = mLastIterationStatsIFB.numCtxRequests;
    modelStats.numGenRequests = mLastIterationStatsIFB.numGenRequests;
    modelStats.numPausedRequests = mLastIterationStatsIFB.pausedRequests.size();
    modelStats.avgNumDecodedTokensPerIter = mLastIterationStatsIFB.avgNumDecodedTokensPerIter;
    modelStats.numCtxTokens = mLastIterationStatsIFB.numCtxTokens;
    modelStats.microBatchId = mLastIterationStatsIFB.microBatchId;
    stats.inflightBatchingStats = modelStats;
}

void TrtGptModelInflightBatching::getCurrentRequestStats(executor::RequestStatsPerIteration& stats) const
{
    for (auto& requestStat : stats.requestStats)
    {
        requestStat.scheduled
            = mLastIterationStatsIFB.scheduledRequests.count(static_cast<RequestIdType>(requestStat.id));
        requestStat.paused = mLastIterationStatsIFB.pausedRequests.count(static_cast<RequestIdType>(requestStat.id));
    }
}

} // namespace tensorrt_llm::batch_manager
