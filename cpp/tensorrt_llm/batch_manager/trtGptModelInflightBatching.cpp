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
#include "tensorrt_llm/batch_manager/allocateKvCache.h"
#include "tensorrt_llm/batch_manager/assignReqSeqSlots.h"
#include "tensorrt_llm/batch_manager/capacityScheduler.h"
#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/contextProgress.h"
#include "tensorrt_llm/batch_manager/guidedDecoder.h"
#include "tensorrt_llm/batch_manager/kvCacheConfig.h"
#include "tensorrt_llm/batch_manager/kvCacheEventManager.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/batch_manager/microBatchScheduler.h"
#include "tensorrt_llm/batch_manager/pauseRequests.h"
#include "tensorrt_llm/batch_manager/peftCacheManager.h"
#include "tensorrt_llm/batch_manager/rnnStateManager.h"
#include "tensorrt_llm/batch_manager/utils/debugUtils.h"
#include "tensorrt_llm/batch_manager/utils/inflightBatchingUtils.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/mpiUtils.h"
#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/common/timestampUtils.h"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/layers/defaultDecodingParams.h"
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
#include <utility>
#include <vector>

using namespace tensorrt_llm::runtime;
namespace tc = tensorrt_llm::common;
namespace tk = tensorrt_llm::kernels;
namespace tru = tensorrt_llm::runtime::utils;
namespace texe = tensorrt_llm::executor;

namespace tensorrt_llm::batch_manager
{

bool TrtGptModelInflightBatching::optionalParamsAreValid(
    ModelConfig const& modelConfig, TrtGptModelOptionalParams const& optionalParams)
{
    // Make sure logic in this function matches fixOptionalParams
    if (optionalParams.kvCacheConfig.enableBlockReuse)
    {
        if (!modelConfig.getPagedContextFMHA())
        {
            return false;
        }
    }
    return true;
}

TrtGptModelOptionalParams TrtGptModelInflightBatching::fixOptionalParams(
    ModelConfig const& modelConfig, TrtGptModelOptionalParams const& optionalParams)
{
    // Make sure logic in this function matches optionalParamsAreValid
    auto fixedOptionalParams = TrtGptModelOptionalParams(optionalParams);
    if (fixedOptionalParams.kvCacheConfig.enableBlockReuse)
    {
        if (!modelConfig.getPagedContextFMHA())
        {
            TLLM_LOG_WARNING(
                "Fix optionalParams : KV cache reuse disabled because model was not built with paged context FMHA "
                "support");
            fixedOptionalParams.kvCacheConfig.enableBlockReuse = false;
        }
    }
    return fixedOptionalParams;
}

TrtGptModelInflightBatching::TrtGptModelInflightBatching(std::shared_ptr<nvinfer1::ILogger> logger,
    ModelConfig const& modelConfig, WorldConfig const& worldConfig, RawEngine const& rawEngine, bool ctxGenFusion,
    TrtGptModelOptionalParams const& optionalParams)
    : TrtGptModel(modelConfig, worldConfig, optionalParams)
    , mModelConfig(modelConfig)
    , mWorldConfig(worldConfig)
    , mDevice{runtime::utils::initDevice(worldConfig)}
    , mDecodingConfig{optionalParams.decodingConfig}
    , mExtendedRuntimePerfKnobConfig{optionalParams.extendedRuntimePerfKnobConfig}
    , mDebugConfig{optionalParams.debugConfig}
    , mAdditionalOutputNames{optionalParams.additionalOutputNames}
    , mLogger{logger ? std::move(logger) : std::make_shared<TllmLogger>()}
    , mRuntime{std::make_shared<TllmRuntime>(
          rawEngine, mLogger.get(), optionalParams.gpuWeightsPercent, modelConfig.useShapeInference())}
    , mCopyBufferManager{std::make_shared<CudaStream>()}
    , mCtxGenFusion(ctxGenFusion)
    , mOperatingBeamWidth{getMaxBeamWidth()}
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    if (!(mModelConfig.supportsInflightBatching()))
    {
        throw std::runtime_error(
            "TrtGptModelInflightBatching requires GPT attention/Mamba Conv 1d plugin with "
            "packed input and paged KV cache.");
    }
    if (mWorldConfig.isTensorParallel())
    {
        mRuntime->initializeUserBuffer(mWorldConfig.getTensorParallelism(), mModelConfig.getMaxBatchSize(),
            mModelConfig.getMaxBeamWidth(), mModelConfig.getMaxSequenceLen(), mModelConfig.getHiddenSize(),
            getMaxNumTokens());
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

    if (!optionalParams.kvCacheConfig.onboardBlocks)
    {
        TLLM_CHECK_WITH_INFO(
            !mModelConfig.getPagedContextFMHA(), "KV cache blocks need to be onboarded if context FMHA.");
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

    if (mModelConfig.getSpeculativeDecodingMode().isLookaheadDecoding())
    {
        mSeamlessLADMaxDraftLen = modelConfig.getMaxDecodingDraftTokens();
        // TODO: enable it when speculativeDecodingMode == None and run with '--lookahead_config'
        mUseSeamlessLookahead = false;
    }

    setupSpeculativeDecodingModule(mDecodingConfig);

    if (mModelConfig.getSpeculativeDecodingMode().needsKVCacheRewind())
    {
        TLLM_CHECK_WITH_INFO(
            mModelConfig.isKVCacheEnabled(), "When needsKVCacheRewind() returns true, KV cache needs to be enabled.");
    }

    if (mWorldConfig.isLastPipelineParallelRank() && optionalParams.guidedDecodingConfig)
    {
        mGuidedDecoder = std::make_unique<GuidedDecoder>(optionalParams.guidedDecodingConfig.value(),
            getMaxNumSequences(), mModelConfig.getVocabSizePadded(mWorldConfig.getSize()),
            mModelConfig.getLogitsDtype(), mRuntime->getBufferManager());
    }

    createRuntimeContexts();

    if (mWorldConfig.isTensorParallel())
    {
        createCustomAllReduceWorkspace();
    }

    if (mModelConfig.isTransformerBased())
    {
        createRuntimePerfKnobsTensor(mExtendedRuntimePerfKnobConfig);
    }

    auto& memCounter = MemoryCounters::getInstance();
    auto const gpuUsage1 = memCounter.getGpu();
    createBuffers(mDecodingConfig, mAdditionalOutputNames);
    auto const gpuUsage2 = memCounter.getGpu();
    TLLM_LOG_INFO("[MemUsageChange] Allocated %s GPU memory for runtime buffers.",
        memCounter.bytesToString(gpuUsage2 - gpuUsage1).c_str());

    createDecoder(mDecodingConfig.getDecodingMode());
    auto const gpuUsage3 = memCounter.getGpu();
    TLLM_LOG_INFO("[MemUsageChange] Allocated %s GPU memory for decoder.",
        memCounter.bytesToString(gpuUsage3 - gpuUsage2).c_str());

    if (modelConfig.getManageWeightsType() != ModelConfig::ManageWeightsType::kDisabled)
    {
        mRuntime->loadManagedWeights(rawEngine, worldConfig.getLocalRank());
    }

    if (mModelConfig.useLoraPlugin())
    {
        mPeftCacheManager = std::make_shared<PeftCacheManager>(
            optionalParams.peftCacheManagerConfig, mModelConfig, mWorldConfig, mRuntime->getBufferManager());
    }
    else
    {
        mPeftCacheManager = std::make_shared<NoOpPeftCacheManager>();
    }

    if (mModelConfig.isRnnBased())
    {
        createRnnStateManager();
    }
    if (mModelConfig.isTransformerBased() && modelConfig.isKVCacheEnabled())
    {
        auto const [blocksInPrimaryPool, blocksInSecondaryPool]
            = BaseKVCacheManager::calculateMaxNumBlocks(optionalParams.kvCacheConfig, mModelConfig.getKvDataType(),
                mModelConfig, mWorldConfig, mRuntime->getBufferManager());
        if (mModelConfig.useCrossAttention())
        {
            TLLM_CHECK_WITH_INFO(optionalParams.kvCacheConfig.crossKvCacheFraction.has_value(),
                "Must set crossKvCacheFraction for encoder-decoder model");
            auto const crossKvCacheFraction = optionalParams.kvCacheConfig.crossKvCacheFraction.value();
            auto selfCacheSizePerToken
                = kv_cache_manager::KVCacheManager::calculateCacheSizePerToken(mModelConfig, mWorldConfig, false);
            auto crossCacheSizePerToken
                = kv_cache_manager::KVCacheManager::calculateCacheSizePerToken(mModelConfig, mWorldConfig, true);
            mKvCacheManager = createKvCacheManager(optionalParams.kvCacheConfig,
                blocksInPrimaryPool * (1.0f - crossKvCacheFraction),
                blocksInSecondaryPool * (1.0f - crossKvCacheFraction), KvCacheType::kSELF);
            auto const numCrossBlocks
                = (float) blocksInPrimaryPool * crossKvCacheFraction * selfCacheSizePerToken / crossCacheSizePerToken;
            auto const numCrossSecondaryBlocks
                = (float) blocksInSecondaryPool * crossKvCacheFraction * selfCacheSizePerToken / crossCacheSizePerToken;
            mCrossKvCacheManager = createKvCacheManager(
                optionalParams.kvCacheConfig, numCrossBlocks, numCrossSecondaryBlocks, KvCacheType::kCROSS);
            TLLM_LOG_INFO("This is an Encoder-Decoder model, set %0.1f cross KV cache fraction based on the config.",
                crossKvCacheFraction);
            TLLM_LOG_INFO("Number of blocks in self KV cache primary pool: %d, in cross KV cache primary pool: %d",
                (SizeType32) (blocksInPrimaryPool * (1.0f - crossKvCacheFraction)), (SizeType32) (numCrossBlocks));
            TLLM_LOG_INFO("Number of blocks in self KV cache secondary pool: %d, in cross KV cache secondary pool: %d",
                (SizeType32) (blocksInSecondaryPool * (1.0f - crossKvCacheFraction)),
                (SizeType32) (numCrossSecondaryBlocks));
        }
        else
        {
            TLLM_CHECK_WITH_INFO(!optionalParams.kvCacheConfig.crossKvCacheFraction.has_value(),
                "Do not set crossKvCacheFraction for decoder-only model");
            mKvCacheManager = createKvCacheManager(
                optionalParams.kvCacheConfig, blocksInPrimaryPool, blocksInSecondaryPool, KvCacheType::kSELF);
        }
        if (common::getEnvUseUCXKvCache())
        {
            TLLM_LOG_INFO("Enable UCX KV cache transport.");
            mCacheTransceiver = std::make_unique<CacheTransceiver>(
                mKvCacheManager.get(), CacheTransceiver::CommType::UCX, mModelConfig, mWorldConfig);
        }
        else
        {
            TLLM_LOG_INFO("Enable MPI KV cache transport.");
            mCacheTransceiver = std::make_unique<CacheTransceiver>(
                mKvCacheManager.get(), CacheTransceiver::CommType::MPI, mModelConfig, mWorldConfig);
        }
    }

    if (mWorldConfig.isPipelineParallel())
    {
        mAsyncSendWaitThread = std::make_unique<tensorrt_llm::mpi::MpiWaitThread>(
            "asyncSendWaitThread",
            [this]()
            {
                mDecStepAsyncSndHdls.clear();
                mDecSlotAsyncSndHdls.clear();
            },
            [this]() { TLLM_CUDA_CHECK(cudaSetDevice(mWorldConfig.getDevice())); });
    }
    if (mWorldConfig.isPipelineParallel())
    {
        auto const& commSession = COMM_SESSION;
        mMpiCommPipelinePara = std::make_shared<tensorrt_llm::mpi::MpiComm>(
            commSession.split(mWorldConfig.getTensorParallelRank(), mWorldConfig.getPipelineParallelRank()));
        mDecSlotAsyncSndHdls.reserve(getMaxBatchSize());
    }
    if (mWorldConfig.isTensorParallel())
    {
        auto const& commSession = COMM_SESSION;
        mMpiCommTensorPara = std::make_shared<tensorrt_llm::mpi::MpiComm>(
            commSession.split(mWorldConfig.getPipelineParallelRank(), mWorldConfig.getTensorParallelRank()));
    }

    mSeqSlotManager
        = std::make_shared<SequenceSlotManager>(getMaxNumSequences(), optionalParams.maxSeqIdleMicroseconds);

    mMicroBatchScheduledRequests.resize(mNumMicroBatches);
    mDecoderFinishedEvents.resize(mNumMicroBatches);
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
        TLLM_CHECK_WITH_INFO(modelConfig.isKVCacheEnabled() && mModelConfig.getPagedContextFMHA(),
            "Chunked context requires context FMHA, paged kv_cache and paged context FMHA all enabled at the same "
            "time.");
        SizeType32 chunkUnitSize = mKvCacheManager->getTokensPerBlock();
        // If sliding window attention is used, then make sure the unit size aligns with the paged context fmha's kv
        // step size.
        if (getMaxInputLen() > getMaxAttentionWindow())
        {
            chunkUnitSize = std::max(/* maxKvStepSizeInFmha */ 256, chunkUnitSize);
            TLLM_LOG_INFO("ChunkUnitSize is set to %d as sliding window attention is used.", chunkUnitSize);
        }
        ctxChunkConfig
            = batch_scheduler::ContextChunkingConfig{optionalParams.schedulerConfig.getContextChunkingPolicy().value_or(
                                                         texe::ContextChunkingPolicy::kFIRST_COME_FIRST_SERVED),
                chunkUnitSize};
    }

    auto maxNumTokens = getMaxNumTokens();
    TLLM_CHECK_WITH_INFO(maxNumTokens, "Max number of tokens is not set in model config.");

    // For chunked context the max chunk size is limited by max_num_tokens.
    // For context FMHA the max context size is limited by max_num_tokens.
    // Else it is limited by the model.
    auto const maxContextLength = (optionalParams.enableChunkedContext || mModelConfig.getContextFMHA())
        ? maxNumTokens
        : std::make_optional<SizeType32>(mModelConfig.getMaxInputLen());

    mMaxBatchSizeTunerRecommended = 0;
    mMaxBatchSizeRuntime = getMaxBatchSize();
    mMaxNumTokensStatic = maxNumTokens;
    mMaxNumTokensTunerRecommended = 0;
    mMaxNumTokensRuntime = maxNumTokens;

    if (mKvCacheManager && ctxChunkConfig)
    {
        TLLM_CHECK_WITH_INFO(ctxChunkConfig.value().chunkUnitSize % mKvCacheManager->getTokensPerBlock() == 0,
            "To prevent cache fragmentation, the context chunk unit size (%d) should be divisible by the number of "
            "tokens per kv-cache block (%d).",
            ctxChunkConfig.value().chunkUnitSize, mKvCacheManager->getTokensPerBlock());
    }

    mCapacityScheduler = std::make_unique<CapacityScheduler>(getMaxBatchSize() * mNumMicroBatches,
        optionalParams.schedulerConfig.getCapacitySchedulerPolicy(), mKvCacheManager != nullptr, mNumMicroBatches > 1);

    mMicroBatchScheduler = std::make_unique<MicroBatchScheduler>(ctxChunkConfig, maxContextLength);

    if (ctxChunkConfig)
    {
        if (maxContextLength)
        {
            ctxChunkConfig.value().chunkUnitSize
                = std::min(ctxChunkConfig.value().chunkUnitSize, maxContextLength.value());
        }
        TLLM_CHECK_WITH_INFO(ctxChunkConfig.value().chunkUnitSize > 0,
            "Context chunk size (%d) must be a positive integer.", maxContextLength.value());
    }
    else
    {
        if (maxContextLength && maxNumTokens)
        {
            TLLM_CHECK_WITH_INFO(maxContextLength.value() <= maxNumTokens.value(),
                "Without enabling chunked context, the max context length (%d) needs to be less than or equal to the "
                "max number of tokens (%d).",
                maxContextLength.value(), maxNumTokens.value());
        }
    }

    mPauseRequests = std::make_unique<PauseRequests>(getMaxInputLen());
    mAssignReqSeqSlots = std::make_unique<AssignReqSeqSlots>();
    mAllocateKvCache = std::make_unique<AllocateKvCache>();

    if (isCudaGraphMode())
    {
        // Limit cuda graph cache size. Depending on the model one graph is 4-10MB of GPU memory.
        SizeType32 cudaGraphCacheSize
            = std::min(getMaxBatchSize(), std::max(mExtendedRuntimePerfKnobConfig.getCudaGraphCacheSize(), 1));
        // We can't have common cache for all microbatches as cuda graph is tied to the memory pointers of the runtime
        // buffers.
        mCudaGraphExecutorCaches.resize(mNumBuffers, utils::CudaGraphExecutorCache(cudaGraphCacheSize));
    }

    mSpeculativeDecodingFastLogits = optionalParams.speculativeDecodingConfig.has_value()
        && optionalParams.speculativeDecodingConfig.value().fastLogits;
    mIsLeaderInOrchMode = optionalParams.isLeaderInOrchMode;
    if (mSpeculativeDecodingFastLogits && modelConfig.getSpeculativeDecodingMode().isNone() && mIsLeaderInOrchMode)
    {
        mDraftModelSendLogitsThread
            = std::make_unique<std::thread>(&TrtGptModelInflightBatching::draftModelSendLogitsThread, this);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

TrtGptModelInflightBatching::~TrtGptModelInflightBatching()
{
    if (mCacheTransceiver)
    {
        mCacheTransceiver->checkContextTransferStatus(true);
        TLLM_CHECK_WITH_INFO(mCacheTransceiver->checkGenTransferComplete(), "Genearation transfer not complete");
    }
    if (mAsyncSendWaitThread)
    {
        mAsyncSendWaitThread.reset(nullptr);
    }
    if (mDraftModelSendLogitsThread)
    {
        mDraftModelThreadShouldExit = true;
        mDraftModelSendLogitsThread->join();
        mDraftModelSendLogitsThread.reset(nullptr);
    }
}

void TrtGptModelInflightBatching::setupSpeculativeDecodingModule(executor::DecodingConfig const& decodingConfig)
{
    if (mModelConfig.getSpeculativeDecodingMode().isExplicitDraftTokens()
        || mModelConfig.getSpeculativeDecodingMode().isEagle())
    {
        TLLM_CHECK_WITH_INFO(mCtxGenFusion, "Current speculative decoding mode requires context-gen fusion IFB");
    }

    if (mModelConfig.getSpeculativeDecodingMode().isLookaheadDecoding() && decodingConfig.getLookaheadDecodingConfig())
    {
        // FIXME(nkorobov) choose defaults
        auto maxLookaheadConfig = decodingConfig.getLookaheadDecodingConfig().value();

        SizeType32 maxDraftTokens{0};
        SizeType32 maxDraftPathLen{0};
        std::tie(std::ignore, std::ignore, maxDraftTokens, maxDraftPathLen)
            = maxLookaheadConfig.calculateSpeculativeResource();
        TLLM_CHECK(maxDraftTokens <= mModelConfig.getMaxDecodingDraftTokens());
        mModelConfig.getSpeculativeDecodingModulePtr()->setMaxDraftTokens(maxDraftTokens);
        mModelConfig.getSpeculativeDecodingModulePtr()->setMaxDraftPathLen(maxDraftPathLen);

        auto lookaheadModulePtr
            = std::dynamic_pointer_cast<runtime::LookaheadModule>(mModelConfig.getSpeculativeDecodingModulePtr());
        lookaheadModulePtr->setExecutionConfig(maxLookaheadConfig);
    }
}

void TrtGptModelInflightBatching::reshapeKvTensors(
    SizeType32 maxBlocksPerSeq, kv_cache_manager::CacheType kvCacheType, SizeType32 numPools)
{
    TLLM_CHECK(mBuffers.size() == static_cast<size_t>(mNumBuffers));
    auto const& manager = mRuntime->getBufferManager();
    for (auto& buffers : mBuffers)
    {
        TLLM_CHECK(buffers->transformerBuffers);
        // any method that operates on transformerBuffers must distinguish between self and cross cache, because
        // transformerBuffers is not managed by KVCacheManager same rule applies to kv pool pointers below
        buffers->transformerBuffers->reshapeKvTensors(
            getMaxBatchSize(), mOperatingBeamWidth, maxBlocksPerSeq, kvCacheType, numPools, manager);
    }
}

void TrtGptModelInflightBatching::draftModelSendLogitsThread()
{
#if ENABLE_MULTI_DEVICE
    TLLM_CUDA_CHECK(cudaSetDevice(mDevice));
    auto const& worldComm = tensorrt_llm::mpi::MpiComm::world();

    bool msgReady{false};
    MPI_Message msg = nullptr;
    MPI_Status status;

    while (true)
    {
        msgReady = worldComm.improbe(MPI_ANY_SOURCE, kMPI_SPEC_DEC_ID_TAG, &msg, &status);

        if (!msgReady)
        {
            if (mDraftModelThreadShouldExit)
            {
                TLLM_LOG_INFO("Draft model sender thread exiting");
                break;
            }

            continue;
        }

        int const source_rank = status.MPI_SOURCE;

        int32_t count = 0;
        MPICHECK(MPI_Get_count(&status, MPI_UINT64_T, &count));
        TLLM_CHECK(count == 1);

        FastLogitsMpiId mpiId{};
        MPICHECK(MPI_Mrecv(&mpiId, count, MPI_UINT64_T, &msg, &status));

        TLLM_CHECK(mpiId == FastLogitsMpiId::ASK_TENSOR);

        worldComm.mprobe(MPI_ANY_SOURCE, kMPI_SPEC_DEC_DATA_TAG, &msg, &status);
        MPICHECK(MPI_Get_count(&status, MPI_UINT64_T, &count));
        TLLM_CHECK(count == 1);

        uint64_t draftRequestId = 0;
        MPICHECK(MPI_Mrecv(&draftRequestId, count, MPI_UINT64_T, &msg, &status));

        auto const findDraftRequest = [this, draftRequestId]() -> std::shared_ptr<LlmRequest>
        {
            for (auto it = mDraftRequestsWaitingToSendLogits.begin(); it != mDraftRequestsWaitingToSendLogits.end();
                 ++it)
            {
                auto req = *it;
                if (req->mRequestId == draftRequestId)
                {
                    mDraftRequestsWaitingToSendLogits.erase(it);
                    return req;
                }
            }
            return nullptr;
        };

        std::shared_ptr<LlmRequest> draftRequest = findDraftRequest();
        TLLM_CHECK(draftRequest != nullptr);

        auto draftLogits = ITensor::slice(draftRequest->getGenerationLogitsHost(), {0, 0});

        auto const shape = draftLogits->getShape();
        TLLM_CHECK(shape.nbDims == 2);

        FastLogitsMpiId constexpr id{FastLogitsMpiId::SEND_TENSOR};
        worldComm.send(&id, 1, mpi::MpiType::kUINT64, source_rank, kMPI_SPEC_DEC_ID_TAG);

        worldComm.send(shape.d, 2, mpi::MpiType::kINT64, source_rank, kMPI_SPEC_DEC_DATA_TAG);
        worldComm.send(draftLogits->data(), draftLogits->getSizeInBytes(), mpi::MpiType::kUINT8, source_rank,
            kMPI_SPEC_DEC_DATA_TAG);

        terminateRequest(draftRequest);
    }
#endif // ENABLE_MULTI_DEVICE
}

std::optional<TrtGptModelInflightBatching::TensorPtr> TrtGptModelInflightBatching::targetModelReceiveLogits(
    executor::SpeculativeDecodingFastLogitsInfo const& fastLogitsInfo)
{
#if ENABLE_MULTI_DEVICE
    auto const& worldComm = tensorrt_llm::mpi::MpiComm::world();

    FastLogitsMpiId mpiId{FastLogitsMpiId::ASK_TENSOR};
    worldComm.send(&mpiId, 1, mpi::MpiType::kUINT64, fastLogitsInfo.draftParticipantId, kMPI_SPEC_DEC_ID_TAG);
    worldComm.send(&fastLogitsInfo.draftRequestId, 1, mpi::MpiType::kUINT64, fastLogitsInfo.draftParticipantId,
        kMPI_SPEC_DEC_DATA_TAG);

    MPI_Message msg = nullptr;
    MPI_Status status;
    worldComm.mprobe(fastLogitsInfo.draftParticipantId, kMPI_SPEC_DEC_ID_TAG, &msg, &status);

    int32_t count = 0;
    MPICHECK(MPI_Get_count(&status, MPI_UINT64_T, &count));
    TLLM_CHECK(count == 1);

    MPICHECK(MPI_Mrecv(&mpiId, count, MPI_UINT64_T, &msg, &status));
    TLLM_CHECK(mpiId == FastLogitsMpiId::SEND_TENSOR);

    worldComm.mprobe(fastLogitsInfo.draftParticipantId, kMPI_SPEC_DEC_DATA_TAG, &msg, &status);

    MPICHECK(MPI_Get_count(&status, MPI_INT64_T, &count));
    TLLM_CHECK(count == 2);

    int64_t dims[2];
    MPICHECK(MPI_Mrecv(&dims, count, MPI_INT64_T, &msg, &status));

    auto const logitsDtype = mModelConfig.getLogitsDtype();

    auto tensor = tensorrt_llm::runtime::BufferManager::pinnedPool(ITensor::makeShape({dims[0], dims[1]}), logitsDtype);

    worldComm.mprobe(fastLogitsInfo.draftParticipantId, kMPI_SPEC_DEC_DATA_TAG, &msg, &status);

    MPICHECK(MPI_Get_count(&status, MPI_UINT8_T, &count));

    TLLM_CHECK((uint64_t) count == dims[0] * dims[1] * tc::getDTypeSize(logitsDtype));

    MPICHECK(MPI_Mrecv(tensor->data(), count, MPI_UINT8_T, &msg, &status));

    return tensor;
#else
    return std::nullopt;
#endif // ENABLE_MULTI_DEVICE
}

void TrtGptModelInflightBatching::adjustMaxAttentionWindow(SizeType32 numPrimaryBlocks, SizeType32 numTokensPerBlock)
{
    // At this point, we can only validate that the cheapest sequence in terms of kv-cache resources still fits. More
    // validation is needed on a per-request basis, once the prompt / output lengths and the actual beam width are
    // known.
    auto const promptLength = getMaxInputLen();
    auto const outputLength
        = getMaxSequenceLen() - promptLength; // This makes it the best case scenario, as context tokens are 'cheaper'
                                              // in terms of kv-cache resources on average.
    auto const sinkTokenLength = getSinkTokenLen();
    auto const maxAttentionWindow = getMaxAttentionWindow();
    auto const maxBeamWidth = getMaxBeamWidth();
    auto const bestCaseBlockRequirements = kv_cache_manager::KVCacheManager::calculateMaxBlockRequirements(
        promptLength, outputLength, sinkTokenLength, maxAttentionWindow, maxBeamWidth, numTokensPerBlock);
    if (bestCaseBlockRequirements > numPrimaryBlocks)
    {
        auto const newMaxAttentionWindow = KVCacheManager::calculateMaxAttentionWindow(
            promptLength, outputLength, sinkTokenLength, numPrimaryBlocks, maxBeamWidth, numTokensPerBlock);
        TLLM_LOG_WARNING(
            "maxAttentionWindow and maxSequenceLen are too large for at least one sequence to fit in kvCache. "
            "they are reduced to %d",
            newMaxAttentionWindow);
        setMaxAttentionWindow(newMaxAttentionWindow);
        setMaxSequenceLen(newMaxAttentionWindow);
        if (getMaxInputLen() > getMaxSequenceLen() - 1)
        {
            setMaxInputLen(getMaxSequenceLen() - 1);
            TLLM_LOG_WARNING("maxInputLen is reduced to %d", getMaxInputLen());
        }
    }
}

std::shared_ptr<kv_cache_manager::KVCacheManager> TrtGptModelInflightBatching::createKvCacheManager(
    KvCacheConfig const& kvCacheConfig, SizeType32 blocksInPrimaryPool, SizeType32 blocksInSecondaryPool,
    KvCacheType kvCacheType)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    bool isCrossAttention = kvCacheType == KvCacheType::kCROSS;
    TLLM_CHECK_WITH_INFO(
        mModelConfig.isTransformerBased(), "KvCacheManager is only needed by transformer based model.");

    auto const tokensPerBlock = mModelConfig.getTokensPerBlock();
    auto const kvDtype = mModelConfig.getKvDataType();

    bool enableCyclicKvCache = false;
    for (SizeType32 maxAttenWin : getMaxAttentionWindowVec())
    {
        if (maxAttenWin != getMaxSequenceLen())
        {
            enableCyclicKvCache = true;
            break;
        }
    }
    TLLM_CHECK_WITH_INFO(
        getMaxBeamWidth() == 1 || !enableCyclicKvCache, "Can't support cyclic kv cache with beam search.");

    // init KV cache block manager
    auto [numKvHeadsPerLayerBegin, numKvHeadsPerLayerEnd] = mModelConfig.getNumKvHeadsPerLayerLocalRange(
        mWorldConfig.getPipelineParallelism(), mWorldConfig.getPipelineParallelRank(), isCrossAttention);
    auto numKvHeadsPerLayer = std::vector<SizeType32>(numKvHeadsPerLayerBegin, numKvHeadsPerLayerEnd);
    auto const sizePerHead = mModelConfig.getSizePerHead();

    // now we check if maxAttentionWindow is too large for at least one sequence to fit in kvCache
    // this can happen if maxSeqLen is deduced from the model and is too large
    // and user also either didn't provide maxAttentionWindow, which leads it to be equal to maxSeqLen
    if (kvCacheType == KvCacheType::kSELF)
    {
        adjustMaxAttentionWindow(blocksInPrimaryPool, tokensPerBlock);
    }

    auto const maxKvCacheLength
        = kvCacheType == KvCacheType::kSELF ? getMaxAttentionWindow() : mModelConfig.getMaxEncoderLen();
    // Only needed when sliding window attention + paged context fmha are used together.
    // In that case, a temporary kv cache buffer with maximum chunk size (maxNumTokens) is needed.
    // TODO: There are several things that can be improved later.
    //  1. a dynamic temporary kv cache allocation based on real chunk size might be needed.
    //  2. reuse the same temporary kv cache buffer among all layers in the same pool.
    SizeType32 temporaryKvCacheLength{0};
    if (mModelConfig.getPagedContextFMHA() && (getMaxInputLen() > getMaxAttentionWindow()))
    {
        TLLM_CHECK_WITH_INFO(getMaxNumTokens(), "Max number of tokens is not set in model config.");
        temporaryKvCacheLength = std::min(getMaxNumTokens().value(), getMaxInputLen() - getMaxAttentionWindow());
        TLLM_LOG_INFO("TemporaryKvCacheLength for sliding window attention: %d", temporaryKvCacheLength);
    }

    if (kvCacheType == KvCacheType::kCROSS && kvCacheConfig.enableBlockReuse)
    {
        TLLM_LOG_INFO(
            "Cross KV cache does not support reuse because cross attention depends on encoder and decoder input ids. "
            "Thus, KV cache reuse is disabled for cross KV cache.");
    }
    auto const enableBlockReuse = kvCacheType == KvCacheType::kSELF ? kvCacheConfig.enableBlockReuse : false;

    auto kvCacheManager = std::make_shared<KVCacheManager>(numKvHeadsPerLayer, sizePerHead, tokensPerBlock,
        blocksInPrimaryPool, blocksInSecondaryPool, getMaxNumSequences(), getMaxBeamWidth(), maxKvCacheLength,
        temporaryKvCacheLength, getSinkTokenLen(), mRuntime->getStreamPtr(), std::nullopt, enableBlockReuse,
        kvCacheConfig.onboardBlocks, kvCacheType, kvCacheConfig.secondaryOffloadMinPriority,
        kvCacheConfig.eventBufferMaxSize > 0
            ? std::make_unique<kv_cache_manager::KVCacheEventManager>(kvCacheConfig.eventBufferMaxSize)
            : nullptr);

    auto const& blockManager = kvCacheManager->getBlockManager();

    reshapeKvTensors(kvCacheManager->getMaxBlocksPerSeq(), blockManager.getCacheType(), blockManager.getNumPools());

    kvCacheManager->allocatePools(kvDtype, kvCacheConfig.useUvm);

    TensorMap inputBuffers;
    TensorPtr poolPointers = kvCacheManager->getBlockPoolPointers();
    TensorPtr poolMapping = kvCacheManager->getLayerToPoolMapping();

    if (kvCacheType == KvCacheType::kSELF)
    {
        inputBuffers.insert_or_assign("host_kv_cache_pool_pointers", std::move(poolPointers));
        inputBuffers.insert_or_assign("host_kv_cache_pool_mapping", std::move(poolMapping));
    }
    else
    {
        inputBuffers.insert_or_assign("host_cross_kv_cache_pool_pointers", std::move(poolPointers));
        inputBuffers.insert_or_assign("host_cross_kv_cache_pool_mapping", std::move(poolMapping));
    }
    mRuntime->setStaticInputTensors(inputBuffers);

    // Emit the `created` event
    kvCacheManager->flushIterationEvents();

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return kvCacheManager;
}

void TrtGptModelInflightBatching::createRnnStateManager()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    TLLM_CHECK_WITH_INFO(mModelConfig.isRnnBased(), "RnnStateManager is only needed by RNN based model.");

    mRnnStateManager = std::make_shared<RnnStateManager>(
        getMaxNumSequences(), mModelConfig, mWorldConfig, mRuntime->getBufferManager());

    TensorMap inputBuffers;
    mRnnStateManager->getPtrBuffers(inputBuffers, mModelConfig, mWorldConfig);
    mRuntime->setStaticInputTensors(inputBuffers);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtGptModelInflightBatching::createCustomAllReduceWorkspace()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TLLM_CHECK(mWorldConfig.isTensorParallel());

    auto const& manager = mRuntime->getBufferManager();
    auto const hiddenSize = mModelConfig.getHiddenSize();

    mAllReduceBuffers = std::make_shared<AllReduceBuffers>(getMaxBatchSize(), getMaxBeamWidth(), getMaxSequenceLen(),
        hiddenSize, manager, mWorldConfig, mRuntime->isUserBufferEnabled());

    TensorMap inputBuffers;
    inputBuffers.insert_or_assign("all_reduce_workspace", mAllReduceBuffers->mAllReduceCommPtrs);
    mRuntime->setStaticInputTensors(inputBuffers);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtGptModelInflightBatching::createRuntimePerfKnobsTensor(
    executor::ExtendedRuntimePerfKnobConfig const& extendedRuntimePerfKnobConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    SizeType32 constexpr perfKnobSize{16};
    mExtendedRuntimePerfKnobsHost = BufferManager::cpu(ITensor::makeShape({perfKnobSize}), nvinfer1::DataType::kINT64);
    auto* runtimePerfKnobsHostPtr = bufferCast<int64_t>(*mExtendedRuntimePerfKnobsHost);
    std::fill_n(runtimePerfKnobsHostPtr, perfKnobSize, -1);
    SizeType32 multiBlockModeVal = extendedRuntimePerfKnobConfig.getMultiBlockMode() ? 1 : 0;
    SizeType32 enableContextFMHAFP32AccVal = extendedRuntimePerfKnobConfig.getEnableContextFMHAFP32Acc() ? 1 : 0;
    runtimePerfKnobsHostPtr[0] = multiBlockModeVal;
    runtimePerfKnobsHostPtr[1] = enableContextFMHAFP32AccVal;

    TensorMap inputBuffers;
    inputBuffers.insert_or_assign("host_runtime_perf_knobs", mExtendedRuntimePerfKnobsHost);
    mRuntime->setStaticInputTensors(inputBuffers);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtGptModelInflightBatching::terminateRequest(LlmRequestPtr const& llmReq, bool pause)
{
    utils::terminateRequest(
        *mSeqSlotManager, *llmReq, getMaxInputLen(), mKvCacheManager, mCrossKvCacheManager, mPeftCacheManager, pause);
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
        TLLM_LOG_DEBUG(
            "iterationStatsIfb.avgNumDecodedTokensPerIter = %.2f", iterationStatsIfb.avgNumDecodedTokensPerIter);
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
    NVTX3_SCOPED_RANGE_WITH_NAME(range, "TrtGptModelInflightBatching::forwardSync");

    auto const device = mWorldConfig.getDevice();
    TLLM_CUDA_CHECK(cudaSetDevice(device));

    if (!mWorldConfig.isLastPipelineParallelRank())
    {
        mAsyncSendWaitThread->waitStop();
    }

    auto& currRequests = mMicroBatchScheduledRequests.at(mMicroBatchId);
    auto& decoderWaitEvent = mDecoderFinishedEvents.at(mMicroBatchId);

    if (!currRequests.empty())
    {
        if (!mWorldConfig.isPipelineParallel() || !mWorldConfig.isLastPipelineParallelRank())
        {
            for (auto& hdl : mDecStepAsyncSndHdls)
            {
                TLLM_CHECK_WITH_INFO(hdl.get() == nullptr, "decoderSync handle must be nullptr.");
            }
            // Wait for decoding for requests in flight for the current micro batch
            mDecStepAsyncSndHdls = decoderSync(currRequests, decoderWaitEvent);
            if (!mWorldConfig.isLastPipelineParallelRank())
            {
                mAsyncSendWaitThread->notifyStart();
            }
        }
        else
        {
            for (auto const& requests : {currRequests.contextRequests, currRequests.generationRequests})
            {
                for (auto const& llmReq : requests)
                {
                    for (SizeType32 beam = 0; beam < llmReq->mSamplingConfig.beamWidth; ++beam)
                    {
                        llmReq->setNumPreDecodedTokens(0, beam);
                    }
                    if (llmReq->mState == LlmRequestState::kGENERATION_TO_COMPLETE)
                    {
                        llmReq->mState = LlmRequestState::kGENERATION_COMPLETE;
                        terminateRequest(llmReq);
                    }
                }
            }
        }

        (*mPauseRequests)(currRequests.contextRequests, mInflightReqIds, mReqIdsToPause, true, *mSeqSlotManager,
            mKvCacheManager, mCrossKvCacheManager, mPeftCacheManager);
        (*mPauseRequests)(currRequests.generationRequests, mInflightReqIds, mReqIdsToPause, true, *mSeqSlotManager,
            mKvCacheManager, mCrossKvCacheManager, mPeftCacheManager);

        for (auto const& llmReq : currRequests.contextRequests)
        {
            // If a context-only request is finished, send its KV cache and mark it.
            if (llmReq->isContextOnlyRequest()
                && (llmReq->isGenerationInProgressState()
                    || llmReq->mState == LlmRequestState::kDISAGG_CONTEXT_INIT_AND_TRANS))
            {
                // TODO: skip if sending layer-wise
                {
                    mCacheTransceiver->respondAndSendAsync(llmReq.get());
                }
                mSeqSlotManager->freeSequenceSlot(llmReq->mRequestId);
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
    if (mCacheTransceiver)
    {
        mCacheTransceiver->checkContextTransferStatus();
    }
    ++mIterCounter;

    if (mKvCacheManager)
    {
        mKvCacheManager->flushIterationEvents();
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtGptModelInflightBatching::storeContextBlocks(std::shared_ptr<LlmRequest> const& llmReq)
{
    // TMJ - Note
    // Make context blocks reusable immediately after context phase finishes.
    // For chunked contexts, this occurs in step that processes last context chunk.
    // isLastContextChunk() is always true for non-chunked contexts.
    // This check is made in code that calls storeContextBlocks, so omitted here.
    if (mKvCacheManager)
    {
        mKvCacheManager->storeContextBlocks(*llmReq);
    }
    if (mCrossKvCacheManager)
    {
        mCrossKvCacheManager->storeContextBlocks(*llmReq);
    }
}

void TrtGptModelInflightBatching::resetIterationStats()
{
    mLastIterationStatsIFB = IterationStatsIFB{mMicroBatchId};
}

void TrtGptModelInflightBatching::forwardAsync(RequestList const& activeRequests)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE_WITH_NAME(range, "TrtGptModelInflightBatching::ForwardAsync");

    auto const device = mWorldConfig.getDevice();
    TLLM_CUDA_CHECK(cudaSetDevice(device));

    try
    {
        verifyRequests(activeRequests);
        if (mModelConfig.isTransformerBased() && getKVCacheManager())
        {
            prepareDistGenInitRequests(activeRequests);
        }
        auto& currRequests = mMicroBatchScheduledRequests.at(mMicroBatchId);
        auto& decoderFinishedEvent = mDecoderFinishedEvents.at(mMicroBatchId);

        // Get a new set of requests for that context
        // The scheduler will not include any requests that are (i) still in encoder state if encoder-decoder models OR
        // (ii) already in flight for decoder models
        TLLM_LOG_DEBUG("Running DECODER request scheduler");
        auto [fittingRequests, requestsToPause]
            = (*mCapacityScheduler)(activeRequests, mKvCacheManager, mPeftCacheManager, mCrossKvCacheManager);
        if (fittingRequests.empty())
        {
            TLLM_LOG_WARNING(
                "CapacityScheduler pick up no requests, probably because of insufficient resources such as kvCache "
                ",will try wait for kvCache transfer to complete");
            if (mCacheTransceiver)
            {
                mCacheTransceiver->checkContextTransferStatus(true);
                // will free kvCache in next iteration.
            }
        }
        std::tie(currRequests.contextRequests, currRequests.generationRequests)
            = (*mMicroBatchScheduler)(fittingRequests, mInflightReqIds, mMaxBatchSizeRuntime, mMaxNumTokensRuntime);

        TLLM_CHECK(currRequests.size() <= static_cast<size_t>(getMaxBatchSize()));

        (*mPauseRequests)(requestsToPause, mInflightReqIds, mReqIdsToPause, false, *mSeqSlotManager, mKvCacheManager,
            mCrossKvCacheManager, mPeftCacheManager);

        if (mUseSeamlessLookahead)
        {
            changeSpecDecMode(currRequests);
        }

        if (!currRequests.empty())
        {
            TLLM_LOG_DEBUG("Running DECODER model with batch size: %lu", currRequests.size());
            {
                NVTX3_SCOPED_RANGE(updateInflightReqIds);
                // Add to set of requests in flight
                for (auto const& requests : {currRequests.contextRequests, currRequests.generationRequests})
                {
                    for (auto const& llmReq : requests)
                    {
                        TLLM_LOG_DEBUG("request %lu added to DECODER model inflight set", llmReq->mRequestId);
                        mInflightReqIds.insert(llmReq->mRequestId);
                    }
                }
            }

            (*mAssignReqSeqSlots)(*mSeqSlotManager, currRequests.contextRequests, currRequests.generationRequests);

            if (mKvCacheManager)
            {
                (*mAllocateKvCache)(*mKvCacheManager, currRequests.contextRequests, currRequests.generationRequests,
                    mModelConfig, mCrossKvCacheManager);
            }

            mPeftTables.at(mMicroBatchId)
                = mPeftCacheManager->ensureBatch(currRequests.contextRequests, currRequests.generationRequests, true);

            prepareDistGenBufferAndDecoder(currRequests.generationRequests);
            executeBatch(currRequests);

            if (mWorldConfig.isLastPipelineParallelRank() && mGuidedDecoder)
            {
                // XGrammar: build maskcache for context requests and perform maskgen for all requests
                // These need to be overlapped with the kernel execution of forward step
                mGuidedDecoder->build(currRequests);
            }

            sync_check_cuda_error();

            // Postpone decoder setup if model does not need to setup buffers for the context phase.
            if (!mModelConfig.getSpeculativeDecodingMode().needsDecoderPrologue())
            {
                setupDecoderStep(currRequests.contextRequests);
            }

            sync_check_cuda_error();

            decoderFinishedEvent = mWorldConfig.isLastPipelineParallelRank() ? decoderStepAsync(currRequests)
                                                                             : DecoderFinishedEventPtr();

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
                            TLLM_LOG_DEBUG("[RANK %d] request %lu finishes decoder ctx phase", COMM_SESSION.getRank(),
                                llmReq->mRequestId);

                            llmReq->mState = LlmRequestState::kGENERATION_IN_PROGRESS;

                            // for encoder-decoder models, free encoder output buffers after decoder context phase is
                            // completed
                            if (llmReq->getEncoderTokens().has_value())
                            {
                                llmReq->freeEncoderOutputBuffers();
                            }
                            storeContextBlocks(llmReq);
                        }
                    }
                    else if (llmReq->isGenerationInProgressState())
                    {
                        TLLM_LOG_DEBUG("request %lu forwards a step in decoder gen phase", llmReq->mRequestId);
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
            mAsyncSendWaitThread->waitStop();
            if (!currRequests.empty())
            {
                for (auto& hdl : mDecStepAsyncSndHdls)
                {
                    TLLM_CHECK_WITH_INFO(hdl.get() == nullptr, "decoderSync handle must be nullptr.");
                }
                // Wait for decoding for requests in flight for the current micro batch
                mDecStepAsyncSndHdls = decoderSync(currRequests, decoderFinishedEvent);
                mAsyncSendWaitThread->notifyStart();
            }
        }

        // Update the micro batch ID
        mMicroBatchId = getNextMicroBatchId(mMicroBatchId);
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

void TrtGptModelInflightBatching::setRuntimeBatchSize(SizeType32 runtimeMaxBatchSize)
{
    mMaxBatchSizeTunerRecommended = runtimeMaxBatchSize;
    mMaxBatchSizeRuntime = std::min(getMaxBatchSize(), runtimeMaxBatchSize);
}

SizeType32 TrtGptModelInflightBatching::getRuntimeBatchSize() const
{
    return mMaxBatchSizeRuntime;
}

void TrtGptModelInflightBatching::setRuntimeMaxNumTokens(SizeType32 runtimeMaxNumTokens)
{
    mMaxNumTokensTunerRecommended = runtimeMaxNumTokens;
    mMaxNumTokensRuntime
        = (mMaxNumTokensStatic) ? std::min(mMaxNumTokensStatic.value(), runtimeMaxNumTokens) : runtimeMaxNumTokens;
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

void TrtGptModelInflightBatching::executeContext(SizeType32 runtimeContextId, SizeType32 bufferId)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(executeContext);

    auto const& currBatchState = mBuffers[bufferId]->getBatchState();

    bool hasCudaGraph = false;
    // If batch state is context only, do not capture/launch graph and execute the engine as is.
    if (isCudaGraphMode() && !currBatchState.isAnyContext())
    {
        auto cudaGraphOpt = mCudaGraphExecutorCaches[bufferId].get(currBatchState);
        // If graph exists for current batch state, launch it.
        if (cudaGraphOpt.has_value())
        {
            hasCudaGraph = true;
        }
    }

    // If there is no graph for current state, execute the engine.
    if (!hasCudaGraph)
    {
        auto enqueueSuccessful = mRuntime->executeContext(runtimeContextId);
        if (!enqueueSuccessful)
        {
            throw std::runtime_error("Executing TRT engine failed!");
        }
    }
    else
    {
        // Launch graph.
        auto cudaGraphOpt = mCudaGraphExecutorCaches[bufferId].get(currBatchState);
        cudaGraphOpt.value()->launch(mRuntime->getStream());
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

            return executor::DecodingMode::BeamSearch();
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
    // Overwrite decoding mode when lookahead decoding is used.
    if (specDecodingMode.isLookaheadDecoding() && !decodingMode.isLookahead())
    {
        TLLM_LOG_WARNING(
            "Model is Lookahead, but decoding mode is not Lookahead. Overwriting decoding mode to Lookahead.");
        decodingMode = executor::DecodingMode::Lookahead();
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
    // Overwrite decoding mode when EAGLE is used.
    if (specDecodingMode.isEagle() && !decodingMode.isEagle())
    {
        TLLM_LOG_WARNING("Model is Eagle, but decoding mode is not Eagle. Overwriting decoding mode to Eagle.");
        decodingMode = executor::DecodingMode::Eagle();
    }
    // Overwrite decoding mode when Eagle is not used.
    if (!specDecodingMode.isEagle() && decodingMode.isEagle())
    {
        TLLM_LOG_WARNING("Model is not Eagle, but decoding mode is Eagle. Overwriting decoding mode.");
        decodingMode = getDefaultDecodingMode(decodingModeOpt);
    }
    if (specDecodingMode.isDraftTokensExternal())
    {
        TLLM_LOG_WARNING("Overwriting decoding mode to external draft token");
        decodingMode = executor::DecodingMode::ExternalDraftTokens();
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
            // There are no logits in Explicit draft tokens model.
            decoderType = mModelConfig.getDataType();
            // Decoder is not instantiated for bf16. We use half to get the same data size
            // and explicitly pass dtype to redrafter that has bf16 kernels.
            if (decoderType == nvinfer1::DataType::kBF16)
            {
                decoderType = nvinfer1::DataType::kHALF;
            }
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
        if (decodingMode.isLookahead())
        {
            mDecoder->setupLookahead(mDecoderBuffers->lookaheadBuffers.value());
        }
        if (decodingMode.isEagle())
        {
            mDecoder->setupEagle(mDecoderBuffers->eagleBuffers);
        }
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtGptModelInflightBatching::createBuffers(executor::DecodingConfig const& decodingConfig,
    std::optional<std::vector<std::string>> const& additionalOutputNames)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    mBuffers.clear();
    for (SizeType32 i = 0; i < mNumBuffers; ++i)
    {
        mBuffers.emplace_back(std::make_shared<RuntimeBuffers>(getMaxBatchSize(), mOperatingBeamWidth,
            getMaxAttentionWindowVec(), getMaxAttentionWindow(), getSinkTokenLen(), *mRuntime, mModelConfig,
            mWorldConfig, decodingConfig, getMaxNumTokens(), additionalOutputNames));
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

RequestVector TrtGptModelInflightBatching::scheduleDistGenInitRequests(RequestList const& activeRequests)
{
    RequestVector scheduledRequests;

    // Now check if we can add pending requests
    auto const numFreeBlocks = mKvCacheManager->getNumFreeBlocks();
    auto const maxPeftCachePages = mPeftCacheManager->getMaxDevicePages();

    // If a request is already in progress, include it
    // If it's been allocated, it had resource to run to completion
    // Also keep track of blocks needed to drive all in-progress requests to completion
    SizeType32 reservedBlocks{0};
    SizeType32 claimedPeftPages{0};
    std::unordered_set<uint64_t> uniqTaskIds{};
    for (auto const& req : activeRequests)
    {
        if (req->isGenerationInProgressState())
        {
            reservedBlocks += mKvCacheManager->getRemainingBlocksToCompletion(*req);

            bool const reqHasLora = req->getLoraTaskId().has_value();
            bool const isNewTask = reqHasLora && !uniqTaskIds.count(req->getLoraTaskId().value());
            if (isNewTask)
            {
                claimedPeftPages += mPeftCacheManager->determineNumPages(req);
                uniqTaskIds.insert(req->getLoraTaskId().value());
            }
        }
    }

    auto availableBlocks = numFreeBlocks - reservedBlocks;
    auto availablePeftPages = maxPeftCachePages - claimedPeftPages;
    for (auto const& req : activeRequests)
    {
        // if request cannot be scheduled yet or request should no longer be scheduled, skip

        if (req->isDisaggGenerationInitState())
        {
            auto neededBlocks = mKvCacheManager->getRemainingBlocksToCompletion(*req);
            bool reqHasLora = req->getLoraTaskId().has_value();
            bool isNewTask = reqHasLora && !uniqTaskIds.count(req->getLoraTaskId().value());
            auto neededPeftPages = isNewTask ? mPeftCacheManager->determineNumPages(req) : 0;

            if (neededBlocks <= availableBlocks && neededPeftPages <= availablePeftPages)
            {
                scheduledRequests.emplace_back(req);
                availableBlocks -= neededBlocks;
                availablePeftPages -= neededPeftPages;
                if (isNewTask)
                {
                    uniqTaskIds.insert(req->getLoraTaskId().value());
                }
            }
            else if (neededBlocks > availableBlocks)
            {
                // If one requests fails to be scheduled, break
                break;
            }
        }
    }
    return scheduledRequests;
}

void TrtGptModelInflightBatching::prepareDistGenInitRequests(RequestList const& activeRequests)
{
    NVTX3_SCOPED_RANGE(prepareDistGenInitRequests);
    auto const toPrepare = std::any_of(activeRequests.begin(), activeRequests.end(),
        [](auto const& req) { return req->isDisaggGenerationInitState(); });

    if (!toPrepare)
    {

        if (common::getEnvDisableKVCacheTransferOverlap())
        {
            return;
        }
        if (mCacheTransceiver)
        {
            auto timeStart = std::chrono::steady_clock::now();

            auto const needCheck = std::any_of(activeRequests.begin(), activeRequests.end(),
                [](auto const& req) { return req->isDisaggGenerationTransmissionInProgress(); });
            auto const needCheckOne = std::all_of(activeRequests.begin(), activeRequests.end(),
                [](auto const& req) { return req->isDisaggGenerationTransmissionInProgress(); });

            if (needCheck)
            {
                int atLeastNum = needCheckOne ? 1 : 0;
                TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
                    "noPreppared requests, checkGenTransferStatus atLeastNum:%d", atLeastNum);

                mCacheTransceiver->checkGenTransferStatus(atLeastNum);

                auto timeEnd = std::chrono::steady_clock::now();
                auto duration = std::chrono::duration<float, std::milli>(timeEnd - timeStart).count();
                TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
                    "no Prepare prepareDistGenInitRequests time:%f ms, "
                    "needCheckOne:%d,needCheck:%ld,activeRequests.size():%ld",
                    duration, needCheckOne, needCheck, activeRequests.size());
            }
        }
        return;
    }
    auto timeStart = std::chrono::steady_clock::now();

    auto const genInitReqNum = std::count_if(activeRequests.begin(), activeRequests.end(),
        [](auto const& req) { return req->isDisaggGenerationInitState(); });
    auto newGenReqs = scheduleDistGenInitRequests(activeRequests);
    for (auto& newGenReq : newGenReqs)
    {
        if (newGenReq->isDisaggGenerationInitState())
        {
            auto const requestId = newGenReq->mRequestId;
            mKvCacheManager->addSequence(
                requestId, newGenReq->mPromptLen, newGenReq->mSamplingConfig.beamWidth, newGenReq);
            if (common::getEnvDisableKVCacheTransferOverlap())
            {
                mCacheTransceiver->requestAndReceiveSync(newGenReq.get());
            }
            else
            {
                mCacheTransceiver->requestAndReceiveAsync(newGenReq.get());
            }
        }
    }
    if (common::getEnvDisableKVCacheTransferOverlap())
    {
        return;
    }
    if (mCacheTransceiver)
    {
        auto const blockTransfer = std::all_of(activeRequests.begin(), activeRequests.end(),
            [](auto const& req) { return req->isDisaggGenerationTransmissionInProgress(); });
        TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
            "newGenReqs.size():%ld requests, activeRequests.size():%ld checkGenTransferStatus :%d original "
            "gen_only_requests_num:%ld",
            newGenReqs.size(), activeRequests.size(), blockTransfer, genInitReqNum);
        mCacheTransceiver->checkGenTransferStatus(blockTransfer ? 1 : 0);
        auto timeEnd = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration<float, std::milli>(timeEnd - timeStart).count();
        TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
            "prepareDistGenInitRequests time:%f ms, "
            "blockTransfer:%d,genInitReqNum:%ld,newGenReqs.size():%ld,activeRequests.size():%ld",
            duration, blockTransfer, genInitReqNum, newGenReqs.size(), activeRequests.size());
    }
}

void TrtGptModelInflightBatching::prepareDistGenBufferAndDecoder(RequestVector const& generationRequests)
{

    if (!mModelConfig.getSpeculativeDecodingMode().needsDecoderPrologue())
    {
        // set decoderStep for disagg_generation
        RequestVector cacheTransCompleteRequests;
        for (auto const& request : generationRequests)
        {
            if (request->isDisaggGenerationTransmissionComplete())
            {
                cacheTransCompleteRequests.push_back((request));
            }
        }
        if (!cacheTransCompleteRequests.empty())
        {
            auto timeStart = std::chrono::steady_clock::now();
            auto const bufferId = getFusedBufferId();
            auto& runtimeBuffers = *mBuffers[bufferId];
            runtimeBuffers.prepareStep(cacheTransCompleteRequests, {}, getMaxBeamWidth(), getMaxAttentionWindow(),
                *mDecoderBuffers, mKvCacheManager.get(), mCrossKvCacheManager.get(), mRnnStateManager.get(),
                mPeftTables[mMicroBatchId], *mRuntime, mModelConfig, mWorldConfig);
            setupDecoderStep(cacheTransCompleteRequests);
            sync_check_cuda_error();
            auto timeEnd = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration<float, std::milli>(timeEnd - timeStart).count();
            TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
                "prepareDistGenBufferAndDecoder time:%f ms , cacheTransCompleteRequests.size():%ld", duration,
                cacheTransCompleteRequests.size());
        }
        for (auto& request : cacheTransCompleteRequests)
        {
            request->mState = LlmRequestState::kGENERATION_IN_PROGRESS;
            request->setContextCurrentPosition(request->mPromptLen);
            request->setDecodingIter(1);
            auto const reqBeamWidth = request->mSamplingConfig.beamWidth;
            auto firstGenTokens = request->getContextPhaseParams().value().getFirstGenTokens();
            for (SizeType32 beam = 0; beam < reqBeamWidth; ++beam)
            {
                request->addNewToken(firstGenTokens.at(beam), beam);
            }
        }
    }
}

void TrtGptModelInflightBatching::debugIOTensors(RequestVector const& contextRequests,
    RequestVector const& generationRequests, TensorMap const& inputMap, TensorMap const& outputMap)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TLLM_CHECK(mDebugConfig);

    auto const& manager = mRuntime->getBufferManager();
    auto requestIds = utils::collectRequestIds(contextRequests, generationRequests);

    if (mDebugConfig->getDebugTensorsMaxIterations() > 0)
    {
        mLastIterationDebugTensors.clear();
        mLastIterationDebugTensors = utils::storeIOTensors(*mDebugConfig, requestIds, inputMap, outputMap, manager);
    }
    else
    {
        utils::dumpIOTensors(*mDebugConfig, mIterCounter, requestIds, inputMap, outputMap, mWorldConfig, manager);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

std::tuple<SizeType32, runtime::StringPtrMap<runtime::ITensor> const&, runtime::StringPtrMap<runtime::ITensor>&>
TrtGptModelInflightBatching::prepareBuffers(
    RequestVector const& contextRequests, RequestVector const& generationRequests, SizeType32 bufferId)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(prepareBuffers);

    auto& runtimeBuffers = *mBuffers[bufferId];

    auto [optProfileId, inputMap, outputMap]
        = runtimeBuffers.prepareStep(contextRequests, generationRequests, mOperatingBeamWidth, getMaxAttentionWindow(),
            *mDecoderBuffers, mKvCacheManager.get(), mCrossKvCacheManager.get(), mRnnStateManager.get(),
            mPeftTables[bufferId], *mRuntime, mModelConfig, mWorldConfig);

    // Do decoder setup before context phase if model needs to setup buffers for the context phase.
    if (mModelConfig.getSpeculativeDecodingMode().needsDecoderPrologue())
    {
        setupDecoderStep(contextRequests);
        if (mModelConfig.getSpeculativeDecodingMode().isExplicitDraftTokens())
        {
            mBuffers[bufferId]->prepareExplicitDraftTokenBuffers(
                *mDecoderBuffers, *mRuntime, mModelConfig, mWorldConfig);
        }
        if (mModelConfig.getSpeculativeDecodingMode().isEagle())
        {
            mBuffers[bufferId]->prepareEagleBuffers(
                contextRequests, generationRequests, *mDecoderBuffers, *mRuntime, mModelConfig, mWorldConfig);
        }
    }

    mRuntime->setInputTensors(optProfileId, inputMap);
    mRuntime->setOutputTensors(optProfileId, outputMap);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return {optProfileId, inputMap, outputMap};
}

void TrtGptModelInflightBatching::prepareGraph(SizeType32 bufferId, SizeType32 optProfileId)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(prepareGraph);

    auto const nextBatchState = mBuffers[bufferId]->getBatchState();
    auto cudaGraphOpt = mCudaGraphExecutorCaches[bufferId].get(nextBatchState);
    // If graph is not found in the cache, capture it.
    if (!cudaGraphOpt.has_value())
    {
        // We need to prepare some tensors once again to properly set values for graph capture.
        // Graph capture requires setting some tensors (e.g. past_kv_len)
        // to the round_up(max_kv_cache_len, kKV_CACHE_LEN_CUDA_GRAPH_ROUND_SIZE)
        // in order to capture the kernels with the large enough grid.
        mBuffers[bufferId]->prepareBuffersForCudaGraph(getMaxSequenceLen());

        auto cudaGraph = std::make_shared<utils::CudaGraphExecutor>();
        cudaGraph->prepareNextGraph(mRuntime, optProfileId);
        mCudaGraphExecutorCaches[bufferId].put(nextBatchState, cudaGraph);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtGptModelInflightBatching::copyAdditionalOutputs(
    RequestVector const& contextRequests, RequestVector const& generationRequests, TensorMap const& outputMap)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const& manager = mRuntime->getBufferManager();
    SizeType32 srcTensorIndex{0}; // One index shared across all output tensors
    for (auto const& llmReq : contextRequests)
    {
        auto numContextTokens = llmReq->getContextChunkSize();
        for (auto const& outputTensor : llmReq->getAdditionalContextOutputs())
        {
            auto const& outputTensorName = outputTensor.first;
            auto tensorIt = outputMap.find(outputTensorName);
            TLLM_CHECK_WITH_INFO(
                tensorIt != outputMap.end(), "Additional context output tensor not found %s", outputTensorName.c_str());

            auto srcView = ITensor::slice(tensorIt->second, srcTensorIndex, numContextTokens);
            auto dstView = ITensor::slice(outputTensor.second, llmReq->getContextCurrentPosition(), numContextTokens);
            manager.copy(*srcView, *dstView);
        }
        srcTensorIndex += numContextTokens;
    }

    for (auto const& llmReq : generationRequests)
    {
        auto const reqBeamWidth = llmReq->mSamplingConfig.beamWidth;
        for (auto const& outputTensor : llmReq->getAdditionalGenerationOutputs())
        {
            auto const& outputTensorName = outputTensor.first;
            auto tensorIt = outputMap.find(outputTensorName);
            TLLM_CHECK_WITH_INFO(tensorIt != outputMap.end(), "Additional generation output tensor not found %s",
                outputTensorName.c_str());

            for (SizeType32 beam = 0; beam < reqBeamWidth; beam++)
            {
                auto generatedLength = llmReq->getNumTokens(beam) - llmReq->getPromptLen();
                TLLM_CHECK(generatedLength >= 1);
                auto srcView = ITensor::slice(tensorIt->second, srcTensorIndex + beam, 1);
                auto dstView = ITensor::slice(outputTensor.second, {beam, generatedLength - 1}, 1);
                manager.copy(*srcView, *dstView);
            }
        }
        srcTensorIndex += reqBeamWidth;
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtGptModelInflightBatching::executeStep(
    RequestVector const& contextRequests, RequestVector const& generationRequests, SizeType32 bufferId)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE_WITH_NAME(range,
        "executeStep: " + std::to_string(contextRequests.size()) + " ctx reqs, "
            + std::to_string(generationRequests.size()) + " gen reqs");

    auto [optProfileId, inputMap, outputMap] = prepareBuffers(contextRequests, generationRequests, bufferId);

    if (mBuffers[bufferId]->transformerBuffers.has_value())
    {
        // Creation of context progress, or remains nullptr if not needed
        std::shared_ptr<ContextProgress> progress = nullptr;
        RequestVector layerWiseRequests;
        if (common::getEnvDisaggLayerwise())
        {
            for (auto const& request : contextRequests)
            {
                bool enableLayerWise = request->isContextOnlyRequest() && request->isLastContextChunk();
                if (enableLayerWise)
                {
                    layerWiseRequests.push_back(request);
                }
            }
        }
        // TODO: support layer-wise cross kv cache in encoder-decoder models
        if (!layerWiseRequests.empty() && !mModelConfig.useCrossAttention())
        {
            int const numLayers = mModelConfig.getNbAttentionLayers(mWorldConfig.getPipelineParallelism());
            progress = std::make_shared<ContextProgress>(numLayers);
        }
        bufferCast<void*>(*mBuffers[bufferId]->transformerBuffers->contextProgressHost)[0] = progress.get();
        if (progress)
        {
            mCacheTransceiver->respondAndSendLayerWise(layerWiseRequests, progress);
        }
    }

    executeContext(optProfileId, bufferId);

    // If batch state has any context request, do not capture this graph.
    if (isCudaGraphMode() && contextRequests.empty())
    {
        // Capture graph of current batch state during engine execution.
        // This is based on the assumptions that
        // a) We can hide CPU graph capture behind the GPU engine execution.
        // b) Batch size in the next iterations won't change and we can reuse the graph multiple times.
        prepareGraph(bufferId, optProfileId);
    }

    if (mDebugConfig)
    {
        debugIOTensors(contextRequests, generationRequests, inputMap, outputMap);
    }

    if (mAdditionalOutputNames.has_value() && !mAdditionalOutputNames.value().empty())
    {
        copyAdditionalOutputs(contextRequests, generationRequests, outputMap);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

TrtGptModelInflightBatching::TensorPtr TrtGptModelInflightBatching::retrieveDraftLogits(TensorPtr const& tensor)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const& manager = mRuntime->getBufferManager();

    if (!mSpeculativeDecodingFastLogits)
    {
        TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
        return manager.copyFrom(*tensor, MemoryType::kPINNEDPOOL);
    }

    if (mIsLeaderInOrchMode)
    {
        executor::SpeculativeDecodingFastLogitsInfo fastLogitsInfo;
        std::memcpy(&fastLogitsInfo, tensor->data(), sizeof(fastLogitsInfo));
        auto logits = targetModelReceiveLogits(fastLogitsInfo).value();

        // Broadcast to other ranks if needed
        if (mWorldConfig.isTensorParallel())
        {
            auto const& commSession = COMM_SESSION;
            auto shape = logits->getShape();
            commSession.bcastValue(shape.d[0], 0);
            commSession.bcastValue(shape.d[1], 0);
            commSession.bcast(logits->data(), logits->getSizeInBytes(), mpi::MpiType::kUINT8, 0);
        }
        TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
        return logits;
    }

    // Get logits from leader rank
    auto const& commSession = COMM_SESSION;
    int64_t dims[2];
    commSession.bcastValue(dims[0], 0);
    commSession.bcastValue(dims[1], 0);
    auto logits = tensorrt_llm::runtime::BufferManager::pinnedPool(
        ITensor::makeShape({dims[0], dims[1]}), nvinfer1::DataType::kFLOAT);
    commSession.bcast(logits->data(), logits->getSizeInBytes(), mpi::MpiType::kUINT8, 0);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return logits;
}

void TrtGptModelInflightBatching::setupDecoderStep(RequestVector const& contextRequests)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(setupDecoderStep);

    if (mWorldConfig.isLastPipelineParallelRank())
    {
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
                ++batchIdx;
                continue;
            }

            auto const decoderInputLength = decoderInputLengthsHost[batchIdx];
            TensorPtr inputView = ITensor::slice(inputIdsFlatView, inputOffset, decoderInputLength);
            auto decoderRequest
                = decoder_batch::Request{inputView, decoderInputLength, llmReq->mMaxNewTokens, llmReq->mEndId};

            llmReq->mSamplingConfig.normalizeLogProbs = isNormalizeLogProbs();
            if (mModelConfig.getSpeculativeDecodingMode().isDraftTokensExternal())
            {
                if (llmReq->hasDraftTokens())
                {
                    auto const& draftTokens = llmReq->getDraftTokens();
                    decoderRequest.draftTokens = manager.copyFrom(*draftTokens, MemoryType::kPINNEDPOOL);
                    auto const& draftLogits = llmReq->getDraftLogits();
                    if (draftLogits.has_value())
                    {
                        decoderRequest.draftLogits = retrieveDraftLogits(draftLogits.value());
                    }
                    decoderRequest.generatedTokensPerEngineStep = draftTokens->size() + 1;
                }
                else
                {
                    decoderRequest.generatedTokensPerEngineStep = 1;
                }
            }
            else if (!mModelConfig.getSpeculativeDecodingMode().isNone())
            {
                decoderRequest.generatedTokensPerEngineStep = mModelConfig.getMaxDecodingTokens();
            }
            if (mModelConfig.getSpeculativeDecodingMode().isMedusa())
            {
                llmReq->mSamplingConfig.topKMedusaHeads = {buffers.medusaBuffers->mTopKs};
                // FIXME: we must set medusa paths and tree ids not from seq slot, but from llmRequest?
                // When multiple microbatches buffers are used, runtime buffers can not be addressed with seqSlot.
                decoderRequest.medusaPaths = ITensor::slice(buffers.medusaBuffers->medusaPathsDevice, 0, 1);
                decoderRequest.medusaTreeIds = ITensor::slice(buffers.medusaBuffers->medusaTreeIdsDevice, 0, 1);
            }
            else if (mModelConfig.getSpeculativeDecodingMode().isLookaheadDecoding())
            {
                decoderRequest.lookaheadRuntimeConfig = llmReq->getLookaheadConfig()
                    ? llmReq->getLookaheadConfig()
                    : mDecodingConfig.getLookaheadDecodingConfig();
            }
            else if (mModelConfig.getSpeculativeDecodingMode().isExplicitDraftTokens())
            {
                // Only Explicit draft tokens model needs dtype to WAR the lack of bf16 decoder.
                decoderRequest.dtype = mModelConfig.getDataType();
            }
            else if (mModelConfig.getSpeculativeDecodingMode().isEagle())
            {
                decoderRequest.eagleConfig
                    = llmReq->getEagleConfig() ? llmReq->getEagleConfig() : mDecodingConfig.getEagleConfig();
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
            seqSlots.push_back(llmReq->mSeqSlot.value());
            decoderRequests.push_back(decoderRequest);
            samplingConfigs.push_back(llmReq->mSamplingConfig);

            inputOffset += decoderInputLength;
            ++batchIdx;
        }
        if (!decoderRequests.empty())
        {
            NVTX3_SCOPED_RANGE(decoderNewRequests);
            mDecoder->newRequests(seqSlots, decoderRequests, samplingConfigs, mModelConfig);
        }
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

    if (llmReq.getReturnGenerationLogits() && !llmReq.getGenerationLogitsFragments().empty())
    {
        TLLM_CHECK(!llmReq.isStreaming());
        auto const genBufferId = mCtxGenFusion ? getFusedBufferId() : getGenerationBufferId();
        auto const& genRuntimeBuffers = *mBuffers.at(genBufferId);

        auto constexpr beforeDecoder = false;
        utils::copyGenerationLogits(
            genRuntimeBuffers, bufferManager, llmReq, batchIdx, beforeDecoder, numDroppedTokens);

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
        TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
        return;
    }

    // Update mDecoderBuffers->slotOutputIdsHost and synchronize
    getDecoderSlotHostOutputs(seqSlot, llmReq.returnLogProbs(), llmReq.mSamplingConfig, llmReq.isStreaming());

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

    // store the generated tokens into the mTokensGathered buffer
    llmReq.setGeneratedTokens(generatedTokens);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtGptModelInflightBatching::getDecoderSlotHostOutputs(
    SizeType32 seqSlot, bool returnLogProbs, SamplingConfig const& samplingConfig, bool streaming)
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
        auto event = mDecoder->finalize(seqSlot, samplingConfig, streaming);
        // Make sure that postprocessing is done before copying outputIds
        mCopyBufferManager.getStream().wait(event.get());

        auto outputIds = mDecoder->getGatheredIds(seqSlot);
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

TrtGptModelInflightBatching::DecoderFinishedEventPtr TrtGptModelInflightBatching::decoderStepAsync(
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
        auto const draftLength = llmReq->isLastContextChunk() ? llmReq->getNumDraftTokens() : 0;

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
        auto& decoderLogits = mDecoderBuffers->logits.at(seqSlot);
        TensorPtr logitsView
            = ITensor::slice(contextRuntimeBuffers.logits, logitsIndex - numDecoderLogits, numDecoderLogits);

        if (mModelConfig.getSpeculativeDecodingMode().hasDraftLogits())
        {
            auto& medusaLogitsHeads = mDecoderBuffers->draftBuffers.predictedDraftLogits.at(seqSlot);
            utils::setupMedusaLogits(medusaLogitsHeads, contextRuntimeBuffers.medusaBuffers->medusaLogitsDevice,
                mModelConfig.getSpeculativeDecodingModule().getMaxDraftPathLen(), logitsIndex - numDecoderLogits,
                numDecoderLogits);
        }

        // Save the last token logits of context into generation logits or
        // save the accepted token logits from target model
        if (llmReq->getReturnGenerationLogits())
        {
            utils::copyLastContextLogits(logitsView, *llmReq, manager);
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
        TLLM_CHECK_DEBUG_WITH_INFO(tru::tensorHasInvalid<float>(*logitsView, manager, "logits") == false,
            "Found invalid number (NaN or Inf) in logits");
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
            if (llmReq->isStreaming())
            {
                utils::copyStreamingGenerationLogits(manager, *llmReq);
            }
            // Copy back to host for every GENERATION_LOGITS_BUFFER_LENGTH steps to mitigate GPU memory pressure
            else if (llmReq->getGenerationLogitsFragmentsSize() == GENERATION_LOGITS_BUFFER_LENGTH)
            {
                auto constexpr beforeDecoder = true;
                utils::copyGenerationLogits(genRuntimeBuffers, manager, *llmReq, batchIdx, beforeDecoder);
            }
        }
        if (mModelConfig.getSpeculativeDecodingMode().hasDraftLogits())
        {
            auto& medusaLogitsHeads = mDecoderBuffers->draftBuffers.predictedDraftLogits.at(seqSlot);
            utils::setupMedusaLogits(medusaLogitsHeads, genRuntimeBuffers.medusaBuffers->medusaLogitsDevice,
                mModelConfig.getSpeculativeDecodingModule().getMaxDraftPathLen(), logitsIndex, draftLength);
        }
        logitsIndex += numLogits;
        ++batchIdx;
    }

    // Copy indirection output into input
    // TODO: Could we avoid this by modifying batchDecoder to take a vector of tensors instead?
    copyCacheIndirectionFromOutputsToInputs(scheduledRequests, genBufferId);

    invokeLogitsPostProcessors(scheduledRequests);

    if (mGuidedDecoder)
    {
        mGuidedDecoder->execute(scheduledRequests, mRuntime->getBufferManager(), mDecoderBuffers->logits);
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
    else if (mModelConfig.getSpeculativeDecodingMode().isEagle())
    {
        // requires mCtxGenFusion == true
        decodingInput->seqSlots = genRuntimeBuffers.seqSlots;
        decodingInput->eagleInputs = genRuntimeBuffers.eagleBuffers->engineOutputs;
        decodingInput->eagleLastInputs = genRuntimeBuffers.eagleBuffers->engineInputs;
    }

    mDecodingOutput = std::make_shared<decoder_batch::Output>();
    mDecodingOutput->cacheIndirection = mDecoderBuffers->cacheIndirectionOutput;
    mDecodingOutput->sequenceLengths = mDecoderBuffers->sequenceLengths;

    DecoderFinishedEventPtr decoderFinishEvent = mDecoder->forwardAsync(*mDecodingOutput, *decodingInput);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return decoderFinishEvent;
}

void TrtGptModelInflightBatching::copyCacheIndirectionFromOutputsToInputs(
    ScheduledRequests const& scheduledRequests, SizeType32 genBufferId)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(copyCacheIndirectionFromOutputsToInputs);

    auto& genRuntimeBuffers = *mBuffers.at(genBufferId);
    auto* srcOffsetsPtr = bufferCast<SizeType64>(*genRuntimeBuffers.cacheIndirDecoderIOBatchedCopySrcOffsets);
    auto* dstOffsetsPtr = bufferCast<SizeType64>(*genRuntimeBuffers.cacheIndirDecoderIOBatchedCopyDstOffsets);
    auto* copySizesPtr = bufferCast<SizeType64>(*genRuntimeBuffers.cacheIndirDecoderIOBatchedCopySizes);

    auto const& cacheIndirShape = mDecoderBuffers->cacheIndirectionOutput->getShape();

    SizeType32 batchIdx{0};
    SizeType64 maxCopySize{0};
    auto& manager = mRuntime->getBufferManager();
    for (auto const& requests : {scheduledRequests.contextRequests, scheduledRequests.generationRequests})
    {
        for (auto const& llmReq : requests)
        {
            auto const reqBeamWidth = llmReq->mSamplingConfig.beamWidth;
            auto const seqSlot = llmReq->mSeqSlot.value();

            auto const copySize = static_cast<SizeType64>(cacheIndirShape.d[2]) * reqBeamWidth;
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

void TrtGptModelInflightBatching::updateDecoderBuffers(bool returnLogProbs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(updateDecoderBuffers);

    // Get finished
    auto const decoderFinished = mDecoder->getFinished();
    std::copy(decoderFinished.begin(), decoderFinished.end(), bufferCast<std::uint8_t>(*mDecoderBuffers->finished));

    mDecoderBuffers->newOutputTokens = mDecoder->getAllNewTokens();
    auto const finishReasonsDevice = mDecoder->getFinishReasons();

    // Here host is already synchronized with forwardSync event, so can trigger copy
    // Using a different stream
    runtime::CudaEvent beforeEvent{};
    mRuntime->getStreamPtr()->record(beforeEvent);
    mCopyBufferManager.getStream().wait(beforeEvent);
    mCopyBufferManager.copy(*mDecoderBuffers->newOutputTokens, *mDecoderBuffers->newOutputTokensHost);
    mCopyBufferManager.copy(*mDecoderBuffers->sequenceLengths, *mDecoderBuffers->sequenceLengthsHost);
    mCopyBufferManager.copy(*finishReasonsDevice, *mDecoderBuffers->finishReasonsHost);

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
        mCopyBufferManager.copy(
            *mDecoderBuffers->draftBuffers.nextDraftTokensDevice, *mDecoderBuffers->draftBuffers.nextDraftTokensHost);

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

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

std::vector<std::unique_ptr<DecoderStepAsyncSend>> TrtGptModelInflightBatching::communicateDecoderBuffers(
    bool returnLogProbs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(communicateDecoderBuffers);

    std::vector<std::unique_ptr<DecoderStepAsyncSend>> asyncHandles;
    if (mWorldConfig.isLastPipelineParallelRank())
    {
        if (broadcastPostDecoder())
        {
            mDecoderBuffers->bcast(mMpiCommTensorPara, returnLogProbs, mOperatingBeamWidth,
                mModelConfig.getSpeculativeDecodingMode().needsKVCacheRewind(), 0);
        }

        if (mWorldConfig.isPipelineParallel())
        {
            auto const peerSend = 0;
            asyncHandles.emplace_back(mDecoderBuffers->asyncSend(mMpiCommPipelinePara, returnLogProbs,
                mOperatingBeamWidth, mModelConfig.getSpeculativeDecodingMode().needsKVCacheRewind(), peerSend));
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
            asyncHandles.emplace_back(mDecoderBuffers->asyncSend(mMpiCommPipelinePara, returnLogProbs,
                mOperatingBeamWidth, mModelConfig.getSpeculativeDecodingMode().needsKVCacheRewind(), peerSend));
        }
    }
    TLLM_CHECK_WITH_INFO(asyncHandles.size() <= 2, "Up to two decoder step async handles expected");

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return asyncHandles;
}

void TrtGptModelInflightBatching::updateRequests(ScheduledRequests const& scheduledRequests)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(updateRequests);

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
    auto const* const finishReasonsHostData = bufferCast<kernels::FinishedState>(*mDecoderBuffers->finishReasonsHost);

    SizeType32 batchIdx{0};
    SizeType32 numSequences{0};
    // Update the request table tokens
    // instead of copy-pasting the loop for context and generation requests,
    // just run the inner loop for both containers
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

            std::vector<SizeType32> numNewTokens(reqBeamWidth);
            std::vector<SizeType32> numDroppedTokens(reqBeamWidth);
            for (SizeType32 beam = 0; beam < reqBeamWidth; ++beam)
            {
                auto const seqLen = sequenceLengthsHostData[seqSlot * mOperatingBeamWidth + beam];
                // The content of newOutputTokens might not be accurate in the case where
                // the sequence has finished early due to end token, so only add new tokens
                // to llmReq if decoder seq length is greater than current number of tokens
                numNewTokens[beam] = std::min(numGeneratedTokens, seqLen - llmReq->getNumTokens(beam));
                numDroppedTokens[beam] = numGeneratedTokens - numNewTokens[beam];
                for (SizeType32 step = 0; step < numNewTokens[beam]; ++step)
                {
                    auto const newTokenIdx = tc::flat_index(hostNewOutputTokensShape.d, step, seqSlot, beam);
                    auto const newToken = hostNewOutputTokensData[newTokenIdx];
                    llmReq->addNewToken(newToken, beam);
                    TLLM_LOG_DEBUG("RequestTokens beam=%d, newToken=%d", beam, newToken);

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

                auto const finishReason = finishReasonsHostData[seqSlot * mOperatingBeamWidth + beam];
                llmReq->setFinishedReason(finishReason.toFinishReason(), beam);

                TLLM_LOG_DEBUG("[RANK %d] decoderSync: request %lu beam %d tokens %s finished %d",
                    COMM_SESSION.getRank(), llmReq->mRequestId, beam, common::vec2str(llmReq->getTokens(beam)).c_str(),
                    static_cast<int>(finishReason.toFinishReason()));
            }

            // Set number of tokens predicted per runtime iteration. Will be > 1 for speculative decoding.
            llmReq->setNumTokensPerIteration(llmReq->getMaxBeamNumTokens() - currentNumOfTokens);

            // Fill new draft tokens for the next step
            if (decoderFinishedPtr[seqSlot] == 0U
                && (mModelConfig.getSpeculativeDecodingMode().predictsDraftTokens()
                    || mModelConfig.getSpeculativeDecodingMode().needsKVCacheRewind()))
            {
                auto const maxDraftTokensLen = mModelConfig.getMaxDecodingDraftTokens();
                auto prevDraftTokensLen = llmReq->getNumDraftTokens();

                // We overallocate KV cache for EAGLE to the maxDecodingTokens + maxPathLen in order to fit both
                // Base model verification (needs up to maxDecodingTokens) and
                // Drafter (needs up to maxPathLen of accepted tokens and maxDecodingDraftTokens for new draft tokens).
                if (mModelConfig.getSpeculativeDecodingMode().isEagle())
                {
                    prevDraftTokensLen = mModelConfig.getSpeculativeDecodingModule().getMaxDecodingTokens()
                        + mModelConfig.getSpeculativeDecodingModule().getMaxPathLen() - 1;
                }

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
                if (!llmReq->isGenerationCompleteState() && prevDraftTokensLen != 0
                    && mModelConfig.getSpeculativeDecodingMode().needsKVCacheRewind())
                {
                    // -1 here is for current 'main' token
                    auto const acceptedTokensLen = llmReq->getMaxBeamNumTokens() - currentNumOfTokens - 1;
                    TLLM_LOG_DEBUG("REQ(%d, %d, %d) acceptedTokensLen=%d, prevDraftTokensLen=%d", llmReq->mRequestId,
                        llmReq->mSeqSlot.value(), seqSlot, acceptedTokensLen, prevDraftTokensLen);
                    TLLM_CHECK(0 <= acceptedTokensLen && acceptedTokensLen <= prevDraftTokensLen);

                    auto const rewindLength = prevDraftTokensLen - acceptedTokensLen;
                    TLLM_LOG_DEBUG("Request %d accepted %d draft tokens, rewind %d tokens", llmReq->mRequestId,
                        acceptedTokensLen, rewindLength);

                    // At this point, KV cache rows are already gathered and moved to the right location.
                    // We can safely rewind (draft - accepted) tokens
                    mKvCacheManager->rewindKVCache(llmReq->mRequestId, rewindLength);
                }
            }

            // Terminate if request has finished or if it is speculative decoding target model
            if (decoderFinishedPtr[seqSlot] != 0U || mModelConfig.getSpeculativeDecodingMode().isDraftTokensExternal())
            {
                postProcessRequest(*llmReq, batchIdx, numDroppedTokens);

                if (!mWorldConfig.isPipelineParallel() || !mWorldConfig.isLastPipelineParallelRank())
                {
                    if (llmReq->getReturnGenerationLogits() && mSpeculativeDecodingFastLogits && mIsLeaderInOrchMode)
                    {
                        mDraftRequestsWaitingToSendLogits.push_back(llmReq);
                    }
                    else
                    {
                        terminateRequest(llmReq);
                    }
                    llmReq->mState = LlmRequestState::kGENERATION_COMPLETE;
                }
                else
                {
                    llmReq->mState = LlmRequestState::kGENERATION_TO_COMPLETE;
                }
            }
            else
            {
                // gather tokens in the case of streaming and beam search
                if (llmReq->isStreaming() && llmReq->mSamplingConfig.beamWidth > 1)
                {
                    postProcessRequest(*llmReq, batchIdx, numDroppedTokens);
                }
                if (llmReq->isContextInitState())
                {
                    llmReq->mState = LlmRequestState::kGENERATION_IN_PROGRESS;
                }
            }
            ++batchIdx;
            numSequences += reqBeamWidth;

            if (llmReq->getReturnPerfMetrics())
            {
                llmReq->updatePerfMetrics(mIterCounter);
            }

            llmReq->advanceDecodingIter();

            if (mWorldConfig.isPipelineParallel() && mWorldConfig.isLastPipelineParallelRank())
            {
                for (SizeType32 beam = 0; beam < reqBeamWidth; ++beam)
                {
                    llmReq->setNumPreDecodedTokens(numNewTokens[beam], beam);
                }
            }
        }
    }

    if (mModelConfig.getSpeculativeDecodingMode().needsKVCacheRewind())
    {
        TLLM_CHECK_WITH_INFO(mCtxGenFusion, "Current speculative decoding mode requires context-gen fusion IFB");
        rewindKVCacheBlocks(numSequences);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

std::vector<std::unique_ptr<DecoderStepAsyncSend>> TrtGptModelInflightBatching::decoderSync(
    ScheduledRequests const& scheduledRequests, DecoderFinishedEventPtr const& decoderFinishEvent)
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

    if (mWorldConfig.isLastPipelineParallelRank())
    {
        // Wait for the decoder event for that context
        mDecoder->forwardSync(*decoderFinishEvent);

        updateDecoderBuffers(returnLogProbs);
    }

    auto asyncHandles = communicateDecoderBuffers(returnLogProbs);

    updateRequests(scheduledRequests);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return asyncHandles;
}

void TrtGptModelInflightBatching::rewindKVCacheBlocks(SizeType32 numSequences)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TLLM_CHECK_WITH_INFO(mKvCacheManager->getBlockManager().getNumPools() == 1,
        "Rewinding KV cache blocks for models with multiple pools is not supported");
    auto const bufferId = getFusedBufferId();
    auto& runtimeBuffers = *mBuffers.at(bufferId);

    auto localNbLayers = mModelConfig.getNbAttentionLayers(
        mWorldConfig.getPipelineParallelism(), mWorldConfig.getPipelineParallelRank());
    if (mWorldConfig.isLastPipelineParallelRank() && mModelConfig.getSpeculativeDecodingMode().isEagle())
    {
        // Do not correct the last kv caches, which are for EagleNet drafter. Those KV caches are managed separately.
        auto eagleModulePtr
            = std::dynamic_pointer_cast<runtime::EagleModule>(mModelConfig.getSpeculativeDecodingModulePtr());
        localNbLayers -= eagleModulePtr->getNumTransformerLayers();
    }

    // TODO: VGQA is not supported, assume all layers have the same num_kv_heads
    auto const numKvHeads = mModelConfig.getNbKvHeads(0);
    auto const tokensPerBlock = mModelConfig.getTokensPerBlock();
    auto const elemSize = BufferDataType(mModelConfig.getKvDataType()).getSize();
    auto const sizeInBytesPerKVHead = mModelConfig.getSizePerHead() * elemSize;
    auto const maxBlocksPerSeq = mKvCacheManager->getMaxBlocksPerSeq();
    auto const useOneMoreBlock = mKvCacheManager->isUseOneMoreBlock();

    auto const poolPointers = mKvCacheManager->getBlockPoolPointers();
    auto* const* pointerArrayPtr = bufferCast<void*>(*poolPointers);
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
        maxBlocksPerSeq, tokensPerBlock, useOneMoreBlock, mRuntime->getStreamPtr()->get());

    sync_check_cuda_error();

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtGptModelInflightBatching::invokeLogitsPostProcessors(ScheduledRequests const& scheduledRequests)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    // Arguments for batched processor
    std::vector<LlmRequest::RequestIdType> reqIdsVec;
    std::vector<LlmRequest::TensorPtr> logitsVec;
    std::vector<std::reference_wrapper<LlmRequest::BeamTokens const>> beamTokensVec;
    std::vector<std::optional<LlmRequest::RequestIdType>> clientIdsVec;

    mLogitsPostProcessorIsApplied = false;
    for (auto const& requests : {scheduledRequests.contextRequests, scheduledRequests.generationRequests})
    {
        for (auto const& llmReq : requests)
        {
            if (llmReq->isContextInitState() ? llmReq->isLastContextChunk() : llmReq->isGenerationInProgressState())
            {
                // Invoke non-batched processor or collect arguments for batched processor
                if (llmReq->mLogitsPostProcessor)
                {
                    mLogitsPostProcessorIsApplied = true;
                    if (mReplicateLogitsPostProcessor || mWorldConfig.isFirstTensorParallelRank())
                    {
                        auto& logits = mDecoderBuffers->logits.at(llmReq->mSeqSlot.value());
                        llmReq->mLogitsPostProcessor.value()(llmReq->mRequestId, logits, llmReq->getTokens(),
                            mRuntime->getStreamPtr(), llmReq->mClientId);
                    }
                }
                else if (llmReq->mApplyLogitsPostProcessorBatched)
                {
                    reqIdsVec.push_back(llmReq->mRequestId);

                    auto& logits = mDecoderBuffers->logits.at(llmReq->mSeqSlot.value());
                    logitsVec.push_back(logits);

                    beamTokensVec.emplace_back(llmReq->getTokens());
                    clientIdsVec.push_back(llmReq->mClientId);
                }
            }
        }
    }

    // Invoke batched processor
    if (!reqIdsVec.empty())
    {
        mLogitsPostProcessorIsApplied = true;
        if (mReplicateLogitsPostProcessor || mWorldConfig.isFirstTensorParallelRank())
        {
            mLogitsPostProcessorBatched.value()(
                reqIdsVec, logitsVec, beamTokensVec, mRuntime->getStreamPtr(), clientIdsVec);
        }
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
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

    createBuffers(mDecodingConfig, mAdditionalOutputNames);
    createDecoder(mDecodingConfig.getDecodingMode());

    if (static_cast<bool>(mKvCacheManager))
    {
        auto const& blockManager = mKvCacheManager->getBlockManager();
        reshapeKvTensors(
            mKvCacheManager->getMaxBlocksPerSeq(), blockManager.getCacheType(), blockManager.getNumPools());
    }
    if (static_cast<bool>(mCrossKvCacheManager))
    {
        auto const& blockManager = mCrossKvCacheManager->getBlockManager();
        reshapeKvTensors(
            mCrossKvCacheManager->getMaxBlocksPerSeq(), blockManager.getCacheType(), blockManager.getNumPools());
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtGptModelInflightBatching::changeSpecDecMode(ScheduledRequests const& scheduledRequests)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    if ((!mModelConfig.getSpeculativeDecodingMode().isLookaheadDecoding()
            && !mModelConfig.getSpeculativeDecodingMode().isNone())
        || scheduledRequests.empty() || mSeamlessLADMaxDraftLen == 0 || mModelConfig.computeGenerationLogits()
        || mModelConfig.isRnnBased())
    {
        return;
    }

    bool canUseLookahead = false;
    auto maxNumRequestForLad = mDecodingConfig.getLookaheadDecodingMaxNumRequest();
    SizeType32 numRequests = scheduledRequests.contextRequests.size() + scheduledRequests.generationRequests.size();
    if (numRequests > maxNumRequestForLad)
    {
        if (mModelConfig.getSpeculativeDecodingMode().isLookaheadDecoding())
        {
            canUseLookahead = false;
        }
        else
        {
            return;
        }
    }
    {
        bool useTopKTopP = false;
        bool useBanWords = false;
        bool useTempAccVocabPenalties = false; // use temperature and penalties that need to accumulate #vocab.
        SizeType32 beamWidth = 1;
        for (auto const& requests : {scheduledRequests.contextRequests, scheduledRequests.generationRequests})
        {
            for (auto const& llmReq : requests)
            {
                useTopKTopP |= !(llmReq->mSamplingConfig.useDefaultValues(
                                     llmReq->mSamplingConfig.topK, layers::DefaultDecodingParams::getTopK())
                    || llmReq->mSamplingConfig.useDefaultValues(llmReq->mSamplingConfig.topK, 1));
                useTopKTopP |= !llmReq->mSamplingConfig.useDefaultValues(
                    llmReq->mSamplingConfig.topP, layers::DefaultDecodingParams::getTopP());
                useBanWords |= llmReq->getBadWordsList().has_value();
                useBanWords |= !llmReq->mSamplingConfig.useDefaultValues(
                    llmReq->mSamplingConfig.noRepeatNgramSize, layers::DefaultDecodingParams::getNoRepeatNgramSize());
                useTempAccVocabPenalties |= !llmReq->mSamplingConfig.useDefaultValues(
                    llmReq->mSamplingConfig.temperature, layers::DefaultDecodingParams::getTemperature());
                useTempAccVocabPenalties |= !llmReq->mSamplingConfig.useDefaultValues(
                    llmReq->mSamplingConfig.repetitionPenalty, layers::DefaultDecodingParams::getRepetitionPenalty());
                useTempAccVocabPenalties |= !llmReq->mSamplingConfig.useDefaultValues(
                    llmReq->mSamplingConfig.presencePenalty, layers::DefaultDecodingParams::getPresencePenalty());
                useTempAccVocabPenalties |= !llmReq->mSamplingConfig.useDefaultValues(
                    llmReq->mSamplingConfig.frequencyPenalty, layers::DefaultDecodingParams::getFrequencyPenalty());
                beamWidth = llmReq->mSamplingConfig.beamWidth;
                if (useTopKTopP || useBanWords || useTempAccVocabPenalties || beamWidth > 1)
                {
                    break;
                }
            }
            canUseLookahead = !(useTopKTopP || useBanWords || useTempAccVocabPenalties || beamWidth > 1);
        }
    }

    // Change speculative decoding mode
    auto const bufferId = mCtxGenFusion
        ? getFusedBufferId()
        : (!scheduledRequests.contextRequests.empty() ? getContextBufferId() : getGenerationBufferId());
    // TODO: enable lookahead for generation requests.
    bool canChangeToLookahead = scheduledRequests.generationRequests.empty();
    if (mModelConfig.getSpeculativeDecodingMode().isNone() && canUseLookahead && canChangeToLookahead)
    {
        // None -> Lookahead
        mModelConfig.enableSeamlessLookaheadDecoding(mSeamlessLADMaxDraftLen);
        mDecodingConfig.enableSeamlessLookaheadDecoding();
        setupSpeculativeDecodingModule(mDecodingConfig);
        mBuffers[bufferId]->lookaheadBuffers->enableLookaheadDecoding(
            getMaxBatchSize(), mModelConfig.getMaxDecodingTokens());
        mDecoderBuffers->enableLookaheadDecoding(getMaxNumSequences(), mModelConfig.getMaxDecodingTokens());
        createDecoder(mDecodingConfig.getDecodingMode());
    }
    else if (mModelConfig.getSpeculativeDecodingMode().isLookaheadDecoding()
        && (!canUseLookahead || numRequests > maxNumRequestForLad))
    {
        // Lookahead -> None
        mModelConfig.disableSeamlessLookaheadDecoding();
        mDecodingConfig.setDecodingMode(executor::DecodingMode::Auto());
        mBuffers[bufferId]->lookaheadBuffers->disableLookaheadDecoding();
        mDecoderBuffers->disableLookaheadDecoding(getMaxNumSequences());
        mDecoder->disableLookahead(getMaxNumSequences(), scheduledRequests.generationRequests);
        for (auto const& llmReq : scheduledRequests.generationRequests)
        {
            if (llmReq->getNumDraftTokens() > 0)
            {
                llmReq->discardDraftTokens(llmReq->getNumDraftTokens());
            }
        }
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtGptModelInflightBatching::getCurrentIterationStats(executor::IterationStats& stats) const
{
    stats.iter = mIterCounter;

    // Max batch size and max num tokens can be tuned at runtime
    stats.maxBatchSizeStatic = getMaxBatchSize();
    stats.maxBatchSizeTunerRecommended = mMaxBatchSizeTunerRecommended;
    stats.maxBatchSizeRuntime = mMaxBatchSizeRuntime;
    stats.maxNumTokensStatic = mMaxNumTokensStatic.value_or(0);
    stats.maxNumTokensTunerRecommended = mMaxNumTokensTunerRecommended;
    stats.maxNumTokensRuntime = mMaxNumTokensRuntime.value_or(0);

    // KVCacheManager statistics
    auto const& kvCacheManager = getKVCacheManager();
    if (kvCacheManager)
    {
        executor::KvCacheStats kvStats{};
        auto kvCacheStats = kvCacheManager->getKvCacheStats();
        kvStats.maxNumBlocks = kvCacheStats.maxNumBlocks;
        kvStats.freeNumBlocks = kvCacheStats.freeNumBlocks;
        kvStats.usedNumBlocks = kvCacheStats.usedNumBlocks;
        kvStats.tokensPerBlock = kvCacheStats.toksPerBlock;
        kvStats.allocTotalBlocks = kvCacheStats.allocTotalBlocks;
        kvStats.allocNewBlocks = kvCacheStats.allocNewBlocks;
        kvStats.reusedBlocks = kvCacheStats.reusedBlocks;
        kvStats.missedBlocks = kvCacheStats.missedBlocks;
        kvStats.cacheHitRate = kvCacheStats.cacheHitRate;
        stats.kvCacheStats = kvStats;
    }
    auto const& crossKvCacheManager = getCrossKVCacheManager();
    if (crossKvCacheManager)
    {
        executor::KvCacheStats kvStats{};
        auto kvCacheStats = crossKvCacheManager->getKvCacheStats();
        kvStats.maxNumBlocks = kvCacheStats.maxNumBlocks;
        kvStats.freeNumBlocks = kvCacheStats.freeNumBlocks;
        kvStats.usedNumBlocks = kvCacheStats.usedNumBlocks;
        kvStats.tokensPerBlock = kvCacheStats.toksPerBlock;
        kvStats.allocTotalBlocks = kvCacheStats.allocTotalBlocks;
        kvStats.allocNewBlocks = kvCacheStats.allocNewBlocks;
        kvStats.reusedBlocks = kvCacheStats.reusedBlocks;
        kvStats.missedBlocks = kvCacheStats.missedBlocks;
        kvStats.cacheHitRate = kvCacheStats.cacheHitRate;
        stats.crossKvCacheStats = kvStats;
    }
    executor::InflightBatchingStats modelStats{};
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
    stats.iter = mIterCounter;
    for (auto& requestStat : stats.requestStats)
    {
        requestStat.scheduled
            = mLastIterationStatsIFB.scheduledRequests.count(static_cast<RequestIdType>(requestStat.id));
        requestStat.paused = mLastIterationStatsIFB.pausedRequests.count(static_cast<RequestIdType>(requestStat.id));
    }
}

executor::DebugTensorsPerIteration TrtGptModelInflightBatching::getCurrentDebugTensors() const
{
    executor::DebugTensorsPerIteration debugTensors;
    debugTensors.iter = mIterCounter;

    for (auto const& [name, tensor] : mLastIterationDebugTensors)
    {
        debugTensors.debugTensors.emplace(name, executor::detail::ofITensor(tensor));
    }

    return debugTensors;
}

nvinfer1::DataType TrtGptModelInflightBatching::getTensorDataType(std::string const& name) const
{
    auto const& engine = mRuntime->getEngine();
    return engine.getTensorDataType(name.c_str());
}

nvinfer1::Dims TrtGptModelInflightBatching::getTensorShape(std::string const& name) const
{
    auto const& engine = mRuntime->getEngine();
    return engine.getTensorShape(name.c_str());
}

} // namespace tensorrt_llm::batch_manager
