/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: NVIDIA TensorRT
 * Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "../dataTransceiverState.h"
#include "cacheConcatenate.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaFp8Utils.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/common/mpiUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/executor/tensor.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include <NvInferRuntimeBase.h>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace tensorrt_llm::executor::kv_cache
{

// inputBlockNums:[ outputBlockNum , inputRanks.size]
// [PP,TP]
TargetRanksInfo targetIRanks(
    kv_cache::CacheState const& iCacheState, kv_cache::CacheState const& oCacheState, int oRank)
{
    int iPPNum = iCacheState.getParallelConfig().mPipelineParallelism; // TODO:
    int oPPNum = oCacheState.getParallelConfig().mPipelineParallelism;
    int oNbKvHeads = oCacheState.getModelConfig().mNbKvHeadsPerLayer[0];
    int oNbLayers = oCacheState.getModelConfig().mNbKvHeadsPerLayer.size() / oPPNum;
    int iNbKvHeads = iCacheState.getModelConfig().mNbKvHeadsPerLayer[0];
    int iNbLayers = iCacheState.getModelConfig().mNbKvHeadsPerLayer.size() / iPPNum;
    int oTpRank = oRank % oCacheState.getParallelConfig().mTensorParallelism;
    int oPpRank = oRank / oCacheState.getParallelConfig().mTensorParallelism;
    int startHeadId = oTpRank * oNbKvHeads;
    int endHeadId = (oTpRank + 1) * oNbKvHeads;
    int startLayerId = oPpRank * oNbLayers;
    int endLayerId = (oPpRank + 1) * oNbLayers;
    int iTpRankStart = startHeadId / iNbKvHeads;
    int iTpRankEndInclude = (endHeadId - 1) / iNbKvHeads;
    int iPpRankStart = startLayerId / iNbLayers;
    int iPpRankEndInclude = (endLayerId - 1) / iNbLayers;

    int iTPNum = iCacheState.getParallelConfig().mTensorParallelism;
    std::vector<int> retRanks;

    for (int i = iTpRankStart; i <= iTpRankEndInclude; i++)
    {
        for (int j = iPpRankStart; j <= iPpRankEndInclude; j++)
        {
            int irank = j * iTPNum + i;
            retRanks.push_back(irank);
        }
    }
    // [tp ,pp]  order
    int mDomainPPSize = iPpRankEndInclude - iPpRankStart + 1;
    int mDomainTPSize = iTpRankEndInclude - iTpRankStart + 1;
    return {mDomainPPSize, mDomainTPSize, std::move(retRanks)};
}

template <typename T>
struct BlockInfo
{

    T* data;

    int startTokenId;
    int tokensPerBlock;

    int startHeadId;
    int headsPerBlock;

    int startLayerId;
    int layersPerBlock;

    int dimsPerHead;
    size_t offset; // (data-offset)[idx]

    __forceinline__ __device__ __host__ T* getKblockPtr(int layerid)
    {
        // return layerid- startLayerId
        return data + (layerid * 2) * headsPerBlock * tokensPerBlock * dimsPerHead;
    }

    __forceinline__ __device__ __host__ T* getVblockPtr(int layerid)
    {
        return data + (layerid * 2 + 1) * headsPerBlock * tokensPerBlock * dimsPerHead;
    }

    __forceinline__ __device__ __host__ T* getKDimsPtr(int layerid, int headid, int tokenid)
    {
        return data + (layerid * 2) * headsPerBlock * tokensPerBlock * dimsPerHead
            + headid * tokensPerBlock * dimsPerHead + tokenid * dimsPerHead;
    }

    __forceinline__ __device__ __host__ T const* getKDimsPtr(int layerid, int headid, int tokenid) const
    {
        return data + (layerid * 2) * headsPerBlock * tokensPerBlock * dimsPerHead
            + headid * tokensPerBlock * dimsPerHead + tokenid * dimsPerHead;
    }

    __forceinline__ __device__ __host__ T* getVDimsPtr(int layerid, int headid, int tokenid)
    {
        return data + (layerid * 2 + 1) * headsPerBlock * tokensPerBlock * dimsPerHead
            + headid * tokensPerBlock * dimsPerHead + tokenid * dimsPerHead;
    }

    __forceinline__ __device__ __host__ T const* getVDimsPtr(int layerid, int headid, int tokenid) const
    {
        return data + (layerid * 2 + 1) * headsPerBlock * tokensPerBlock * dimsPerHead
            + headid * tokensPerBlock * dimsPerHead + tokenid * dimsPerHead;
    }

    std::string to_string()
    {
        std::stringstream ss;
        ss << "{data ptr: " << data << "startTokenId: " << startTokenId << "tokensPerBlock:  " << tokensPerBlock
           << " startHeadId: " << startHeadId << "headsPerBlock: " << headsPerBlock << "startLayerId:" << startLayerId
           << "layersPerBlock: " << layersPerBlock << "dimsPerHead: " << dimsPerHead << " offset: " << offset << "}";
        return ss.str();
    }
};

// refer blockPtr

// Block shape [ head,tokens,dimsPerHead]
//  CacheBlock [numLayers,2,mBlockSize] . BlockSize[

// kV  and copy

// note k and v not continuous

__forceinline__ __device__ int getInputBlockId(int outputBlockId, int headId, int layerId, int inputBlockNumEachOutput,
    int headNumPerBlock, int layerNumPerBlock, int headNumInputModel, int layerNumInputModel)
{

    int offset = outputBlockId * inputBlockNumEachOutput;

    int layerOffset = layerId / layerNumPerBlock;

    int headOffset = headId / headNumPerBlock;

    int headBlockNum = headNumInputModel / headNumPerBlock;
    return offset + layerOffset * headBlockNum + headOffset;
}

// subWarpSize*subWarpGroupSize
template <typename T, int subWarpSize, int subWarpNumInGroup, int vecSizeByte>
__global__ void splitAndConcatenateBlocksKernel(BlockInfo<T> const* iBlockInfo, BlockInfo<T>* oBlockInfo, int iBlockNum,
    int iNumBlockEachO, int oBlockNum, int headNumInputModel, int layerNumInputModel, int iHeadsPerBlock,
    int iLayersPerBlock)
{

    // for blockDim.y for output_blockNum
    // blockDim.x for layer

    // wraps for heads*tokens
    // threads for dimsPerHead

    // input_id can be decided by outputid,layerid,headid
    // cuda blockNum layers*oBlockNum

    int const subWarpId = threadIdx.x / subWarpSize;
    int const laneId = threadIdx.x % subWarpSize;
    int const subWarpNum = blockDim.x / subWarpSize;
    int const subWarpGroupId = subWarpId / subWarpNumInGroup; //
    int const subWarpGroupNum = subWarpNum / subWarpNumInGroup;
    int const subWarpIdInGroup = subWarpId % subWarpNumInGroup;
    static_assert(vecSizeByte >= sizeof(T));
    int constexpr numElePerThread = vecSizeByte / sizeof(T);
    // using VecType = typename common::packed_as<T,numElePerThread>::type;
    using VecType = typename common::BytesToType<vecSizeByte>::type;
#pragma unroll 1
    for (int oBlockId = blockIdx.y; oBlockId < oBlockNum; oBlockId += gridDim.y)
    {
        int oLayerNum = oBlockInfo[oBlockId].layersPerBlock;
        int headNum = oBlockInfo[oBlockId].headsPerBlock;
        int tokenNum = oBlockInfo[oBlockId].tokensPerBlock;
        int dimsPerHead = oBlockInfo[oBlockId].dimsPerHead;
#pragma unroll 1

        for (int layerid = blockIdx.x; layerid < oLayerNum; layerid += gridDim.x)
        {
#pragma unroll 1

            for (int headId = subWarpGroupId; headId < headNum; headId += subWarpGroupNum)
            {
                int const targetHeadId = oBlockInfo[oBlockId].startHeadId + headId;
                int const targetLayerId = oBlockInfo[oBlockId].startLayerId + layerid;

                int const iBlockId = getInputBlockId(oBlockId, targetHeadId, targetLayerId, iNumBlockEachO,
                    iHeadsPerBlock, iLayersPerBlock, headNumInputModel, layerNumInputModel);
                int const iLayerId = targetLayerId % iLayersPerBlock;
                int const iHeadId = targetHeadId % iHeadsPerBlock;
#pragma unroll 1

                for (int tokenId = subWarpIdInGroup; tokenId < tokenNum; tokenId += subWarpNumInGroup)
                {

                    T* oKPtr = oBlockInfo[oBlockId].getKDimsPtr(layerid, headId, tokenId);
                    T const* iKPtr = iBlockInfo[iBlockId].getKDimsPtr(iLayerId, iHeadId, tokenId);
                    T* oVPtr = oBlockInfo[oBlockId].getVDimsPtr(layerid, headId, tokenId);
                    T const* iVPtr = iBlockInfo[iBlockId].getVDimsPtr(iLayerId, iHeadId, tokenId);
#pragma unroll 1

                    for (int channelId = laneId * numElePerThread; channelId < dimsPerHead;
                         channelId += (subWarpSize * numElePerThread))
                    {

                        common::copy<vecSizeByte>(iKPtr + channelId, oKPtr + channelId);
                        common::copy<vecSizeByte>(iVPtr + channelId, oVPtr + channelId);
                    }
                }
            }
        }
    }
}

template <typename T>
void concatenateKVCache(runtime::ITensor::SharedPtr* inputBlocks, int inputBlockNum, std::vector<int> const& inputRanks,
    kv_cache::CacheState const& iCacheState, runtime::ITensor::SharedPtr* outputBlocks, int outputBlockNum, int oRank,
    kv_cache::CacheState const& oCacheState, runtime::BufferManager const& bufferManager)

{

    TLLM_CHECK_WITH_INFO(!inputRanks.empty(), "input should not be empty!");
    TLLM_CHECK_WITH_INFO(
        inputBlockNum == outputBlockNum * inputRanks.size(), "inputBlockNum==outputBlockNum*inputRanks.size()");

    TLLM_CHECK(inputRanks == targetIRanks(iCacheState, oCacheState, oRank).mIRanks);
    int const inputAllRankNum
        = iCacheState.getParallelConfig().mPipelineParallelism * iCacheState.getParallelConfig().mTensorParallelism;
    std::vector<BlockInfo<T>> blockInfos(outputBlockNum * inputAllRankNum + outputBlockNum);

    auto fillBlockInfo = [](kv_cache::CacheState const& cacheState, runtime::ITensor::SharedPtr buffer, int rank)
    {
        int tpRank = rank % cacheState.getParallelConfig().mTensorParallelism;
        int ppRank = rank / cacheState.getParallelConfig().mTensorParallelism;
        int ppNum = cacheState.getParallelConfig().mPipelineParallelism;
        int headsPerBlock = cacheState.getModelConfig().mNbKvHeadsPerLayer[0];
        int layersPerBlock = cacheState.getModelConfig().mNbKvHeadsPerLayer.size() / ppNum; //  TODO:need  / PPSize?

        int tokensPerBlock = cacheState.getModelConfig().mTokensPerBlock;
        int dimsPerBlock = cacheState.getModelConfig().mSizePerHead;
        int startHead = tpRank * headsPerBlock;
        int startLayer = ppRank * layersPerBlock;
        // TODO:just ignore start Tokenid
        int startTokenId = 0;
        T* data = static_cast<T*>(buffer->data());
        return BlockInfo<T>{
            data, startTokenId, tokensPerBlock, startHead, headsPerBlock, startLayer, layersPerBlock, dimsPerBlock, 0};
    };
    // fill blcokInfo from CacheState and inputBlocks
    for (int oi = 0; oi < outputBlockNum; oi++)
    {
        int iRankNum = inputRanks.size();
        for (int i = 0; i < iRankNum; i++)
        {
            int iRank = inputRanks[i];
            blockInfos[oi * inputAllRankNum + iRank]
                = fillBlockInfo(iCacheState, inputBlocks[oi * iRankNum + i], iRank);
        }

        blockInfos[outputBlockNum * inputAllRankNum + oi] = fillBlockInfo(oCacheState, outputBlocks[oi], oRank);
    }
    runtime::BufferManager::IBufferPtr blockInfosDeviceBuffer
        = bufferManager.gpu(sizeof(BlockInfo<T>) * (blockInfos.size()), nvinfer1::DataType::kUINT8);
    bufferManager.copy((blockInfos.data()), *blockInfosDeviceBuffer, runtime::MemoryType::kCPU);

    BlockInfo<T>* iBlockInfoDevice = static_cast<BlockInfo<T>*>(blockInfosDeviceBuffer->data());

    BlockInfo<T>* oBlockInfoDevice = iBlockInfoDevice + outputBlockNum * inputAllRankNum;

    constexpr int subWarpSize = 8;
    constexpr int subWarpNumInGroup = 8;
    int blockDimx = 128;
    int oPpNum = oCacheState.getParallelConfig().mPipelineParallelism;
    int iPpNum = iCacheState.getParallelConfig().mPipelineParallelism;
    unsigned int gridDimx = oCacheState.getModelConfig().mNbKvHeadsPerLayer.size() / oPpNum;
    unsigned int gridDimy = outputBlockNum;

    dim3 gridDim{gridDimx, gridDimy};
    int const headsInputModel
        = iCacheState.getModelConfig().mNbKvHeadsPerLayer[0] * iCacheState.getParallelConfig().mTensorParallelism;
    int const layersInputModel = iCacheState.getModelConfig().mNbKvHeadsPerLayer.size();
    int const iHeadsPerBlock = iCacheState.getModelConfig().mNbKvHeadsPerLayer[0];
    int const iLayersPerBlock = iCacheState.getModelConfig().mNbKvHeadsPerLayer.size() / iPpNum;
    int const sizePerHead = oCacheState.getModelConfig().mSizePerHead;
    int const remainder = sizePerHead * sizeof(T) % 16;
    switch (remainder)
    {
    case 0:
    {
        splitAndConcatenateBlocksKernel<T, subWarpSize, subWarpNumInGroup, 16>
            <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(iBlockInfoDevice, oBlockInfoDevice,
                outputBlockNum * inputAllRankNum, inputAllRankNum, outputBlockNum, headsInputModel, layersInputModel,
                iHeadsPerBlock, iLayersPerBlock);
        break;
    }
    case 8:
    {
        splitAndConcatenateBlocksKernel<T, subWarpSize, subWarpNumInGroup, 8>
            <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(iBlockInfoDevice, oBlockInfoDevice,
                outputBlockNum * inputAllRankNum, inputAllRankNum, outputBlockNum, headsInputModel, layersInputModel,
                iHeadsPerBlock, iLayersPerBlock);
        break;
    }
    case 4:
    case 12:
    {
        if constexpr (sizeof(T) <= 4)
        {
            splitAndConcatenateBlocksKernel<T, subWarpSize, subWarpNumInGroup, 4>
                <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(iBlockInfoDevice, oBlockInfoDevice,
                    outputBlockNum * inputAllRankNum, inputAllRankNum, outputBlockNum, headsInputModel,
                    layersInputModel, iHeadsPerBlock, iLayersPerBlock);
            break;
        }
    }
    case 2:
    case 6:
    case 10:
    case 14:
    {
        if constexpr (sizeof(T) <= 2)
        {

            splitAndConcatenateBlocksKernel<T, subWarpSize, subWarpNumInGroup, 2>
                <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(iBlockInfoDevice, oBlockInfoDevice,
                    outputBlockNum * inputAllRankNum, inputAllRankNum, outputBlockNum, headsInputModel,
                    layersInputModel, iHeadsPerBlock, iLayersPerBlock);
            break;
        }
    }
    default:
    {
        if constexpr (sizeof(T) <= 1)
        {
            splitAndConcatenateBlocksKernel<T, subWarpSize, subWarpNumInGroup, 1>
                <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(iBlockInfoDevice, oBlockInfoDevice,
                    outputBlockNum * inputAllRankNum, inputAllRankNum, outputBlockNum, headsInputModel,
                    layersInputModel, iHeadsPerBlock, iLayersPerBlock);
        }
        else
        {
            TLLM_THROW(" concatenateKVCacheDispatch no support data type error");
        }
    }
    }
}

void concatenateKVCacheDispatch(runtime::ITensor::SharedPtr* inputBlocks, int inputBlockNum,
    std::vector<int> const& inputRanks, kv_cache::CacheState const& iCacheState,
    runtime::ITensor::SharedPtr* outputBlocks, int outputBlockNum, int oRanks, kv_cache::CacheState const& oCacheState,
    runtime::BufferManager const& bufferManager)
{
    auto dataType = outputBlocks[0]->getDataType();
    int dataSize = tensorrt_llm::common::getDTypeSize(dataType);

    switch (dataSize)
    {
    case 8:
    {
        concatenateKVCache<double>(inputBlocks, inputBlockNum, inputRanks, iCacheState, outputBlocks, outputBlockNum,
            oRanks, oCacheState, bufferManager);
        break;
    }
    case 4:
    {
        concatenateKVCache<float>(inputBlocks, inputBlockNum, inputRanks, iCacheState, outputBlocks, outputBlockNum,
            oRanks, oCacheState, bufferManager);
        break;
    }
    case 2:
    {

        concatenateKVCache<half>(inputBlocks, inputBlockNum, inputRanks, iCacheState, outputBlocks, outputBlockNum,
            oRanks, oCacheState, bufferManager);
        break;
    }

    case 1:
    {

        concatenateKVCache<uint8_t>(inputBlocks, inputBlockNum, inputRanks, iCacheState, outputBlocks, outputBlockNum,
            oRanks, oCacheState, bufferManager);
        break;
    }

    default:
    {
        TLLM_THROW(" concatenateKVCacheDispatch no support");
    }
    }
}

nvinfer1::Dims makeShapeFromCacheState(kv_cache::CacheState const& cacheState)
{

    long blockSize = cacheState.getModelConfig().mNbKvHeadsPerLayer[0] * cacheState.getModelConfig().mTokensPerBlock
        * cacheState.getModelConfig().mSizePerHead;
    int PpNum = cacheState.getParallelConfig().mPipelineParallelism;
    return runtime::ITensor::makeShape(
        {static_cast<long>(cacheState.getModelConfig().mNbKvHeadsPerLayer.size() / PpNum), 2, blockSize});
}

// Block shape [ head,tokens,dimsPerHead]
//  CacheBlock [numLayers,2,mBlockSize] .

//[outputSplitCaches,numLayers,2,head,tokens_per_block,dimsPerHead]
// tokens maybe large, so We

// subWarpSize*subWarpGroupSize
template <typename T, int subWarpSize, int subWarpNumInGroup, int vecSizeByte>
__global__ void splitKVCacheKernel(T const** __restrict__ inputBlocks, T** __restrict__ outputCaches,
    int tokensPerBlock, int numLayers, int headNum, int dimsPerHead, int inputBlockNum, int DomainPPSize,
    int DomainTPSize, int layerNumDomainPP, int headNumDomainTP)
{

    int const subWarpId = threadIdx.x / subWarpSize;
    int const laneId = threadIdx.x % subWarpSize;
    int const subWarpNum = blockDim.x / subWarpSize;
    int const subWarpGroupId = subWarpId / subWarpNumInGroup; //
    int const subWarpGroupNum = subWarpNum / subWarpNumInGroup;
    int const subWarpIdInGroup = subWarpId % subWarpNumInGroup;
    static_assert(vecSizeByte >= sizeof(T));
    int constexpr numElePerThread = vecSizeByte / sizeof(T);
    using VecType = typename common::BytesToType<vecSizeByte>::type;
#pragma unroll 1

    for (int blockId = blockIdx.y; blockId < inputBlockNum; blockId += gridDim.y)
    {
#pragma unroll 1

        for (int layerId = blockIdx.x; layerId < numLayers; layerId += gridDim.x)
        {
#pragma unroll 1

            for (int headId = subWarpGroupId; headId < headNum; headId += subWarpGroupNum)
            {

                T const* inputBlockPtr = inputBlocks[blockId];
                T const* kInputPtr = inputBlockPtr + layerId * 2 * headNum * tokensPerBlock * dimsPerHead
                    + headId * tokensPerBlock * dimsPerHead;
                T const* vInputPtr = inputBlockPtr + (layerId * 2 + 1) * headNum * tokensPerBlock * dimsPerHead
                    + headId * tokensPerBlock * dimsPerHead;

                int outputCacheIdx = headId / headNumDomainTP * DomainPPSize + layerId / layerNumDomainPP;
                T* outputCachePtr = outputCaches[outputCacheIdx];
                int layerIdInDomainPP = layerId % layerNumDomainPP;

                int headIdInDomainTP = headId % headNumDomainTP;
                T* kOutputPtr = outputCachePtr
                    + blockId * (layerNumDomainPP * 2 * headNumDomainTP * tokensPerBlock * dimsPerHead)
                    + layerIdInDomainPP * 2 * headNumDomainTP * tokensPerBlock * dimsPerHead
                    + headIdInDomainTP * tokensPerBlock * dimsPerHead;

                T* vOutputPtr = kOutputPtr + headNumDomainTP * tokensPerBlock * dimsPerHead;
#pragma unroll 1

                for (int tokenId = subWarpIdInGroup; tokenId < tokensPerBlock; tokenId += subWarpNumInGroup)
                {
                    T const* iKPtr = kInputPtr + tokenId * dimsPerHead;
                    T const* iVPtr = vInputPtr + tokenId * dimsPerHead;
                    T* oKPtr = kOutputPtr + tokenId * dimsPerHead;
                    T* oVPtr = vOutputPtr + tokenId * dimsPerHead;
#pragma unroll 1

                    for (int channelId = laneId * numElePerThread; channelId < dimsPerHead;
                         channelId += (subWarpSize * numElePerThread))
                    {
                        common::copy<vecSizeByte>(iKPtr + channelId, oKPtr + channelId);
                        common::copy<vecSizeByte>(iVPtr + channelId, oVPtr + channelId);
                    }
                }
            }
        }
    }
}

template <typename T, int subWarpSize, int subWarpNumInGroup, int vecSizeByte>
__global__ void concatenateKVCacheKernel(T const** __restrict__ inputCaches, T** __restrict__ outputBlocks,
    int tokensPerBlock, int numLayers, int headNum, int dimsPerHead, int outputBlockNum, int DomainPPSize,
    int DomainTPSize, int layerNumDomainPP, int headNumDomainTP)
{
    int const subWarpId = threadIdx.x / subWarpSize;
    int const laneId = threadIdx.x % subWarpSize;
    int const subWarpNum = blockDim.x / subWarpSize;
    int const subWarpGroupId = subWarpId / subWarpNumInGroup; //
    int const subWarpGroupNum = subWarpNum / subWarpNumInGroup;
    int const subWarpIdInGroup = subWarpId % subWarpNumInGroup;
    static_assert(vecSizeByte >= sizeof(T));
    int constexpr numElePerThread = vecSizeByte / sizeof(T);
    using VecType = typename common::BytesToType<vecSizeByte>::type;
#pragma unroll 1
    for (int blockId = blockIdx.y; blockId < outputBlockNum; blockId += gridDim.y)
    {
#pragma unroll 1
        for (int layerId = blockIdx.x; layerId < numLayers; layerId += gridDim.x)
        {

#pragma unroll 1
            for (int headId = subWarpGroupId; headId < headNum; headId += subWarpGroupNum)
            {

                T* outputBlockPtr = outputBlocks[blockId];
                T* kOutputPtr = outputBlockPtr + layerId * 2 * headNum * tokensPerBlock * dimsPerHead
                    + headId * tokensPerBlock * dimsPerHead;
                T* vOutputPtr = outputBlockPtr + (layerId * 2 + 1) * headNum * tokensPerBlock * dimsPerHead
                    + headId * tokensPerBlock * dimsPerHead;

                int inputCacheIdx = headId / headNumDomainTP * DomainPPSize + layerId / layerNumDomainPP;
                T const* inputCachePtr = inputCaches[inputCacheIdx];
                int layerIdInDomainPP = layerId % layerNumDomainPP;

                int headIdInDomainTP = headId % headNumDomainTP;
                T const* kInputPtr = inputCachePtr
                    + blockId * (layerNumDomainPP * 2 * headNumDomainTP * tokensPerBlock * dimsPerHead)
                    + layerIdInDomainPP * 2 * headNumDomainTP * tokensPerBlock * dimsPerHead
                    + headIdInDomainTP * tokensPerBlock * dimsPerHead;

                T const* vInputPtr = kInputPtr + headNumDomainTP * tokensPerBlock * dimsPerHead;
#pragma unroll 1
                for (int tokenId = subWarpIdInGroup; tokenId < tokensPerBlock; tokenId += subWarpNumInGroup)
                {
                    T const* iKPtr = kInputPtr + tokenId * dimsPerHead;
                    T const* iVPtr = vInputPtr + tokenId * dimsPerHead;
                    T* oKPtr = kOutputPtr + tokenId * dimsPerHead;
                    T* oVPtr = vOutputPtr + tokenId * dimsPerHead;

#pragma unroll 1
                    for (int channelId = laneId * numElePerThread; channelId < dimsPerHead;
                         channelId += (subWarpSize * numElePerThread))
                    {
                        common::copy<vecSizeByte>(iKPtr + channelId, oKPtr + channelId);
                        common::copy<vecSizeByte>(iVPtr + channelId, oVPtr + channelId);
                    }
                }
            }
        }
    }
}

template <typename T>
void splitKVCache(std::vector<runtime::ITensor::SharedPtr> const& kVCacheBlocks,
    std::vector<runtime::ITensor::SharedPtr>& outputSplitBlocks, kv_cache::CacheState const& iCacheState,
    kv_cache::CacheState const& oCacheState, int selfIdx, runtime::BufferManager const& bufferManager)
{

    auto inputBlockNum = kVCacheBlocks.size();
    auto targetRankInfo = targetIRanks(iCacheState, oCacheState, selfIdx);
    TLLM_CHECK(targetRankInfo.mIRanks.size()
        == (static_cast<size_t>(targetRankInfo.mDomainPPSize * targetRankInfo.mDomainTPSize)));
    auto outputCacheNum = targetRankInfo.mIRanks.size();
    TLLM_CHECK(outputCacheNum == outputSplitBlocks.size());
    TLLM_CHECK(inputBlockNum > 0);
    auto cacheBlockSize = kVCacheBlocks.at(0)->getSize();
    auto cacheDataType = kVCacheBlocks.at(0)->getDataType();
    std::vector<T*> CachePtrs;

    for (auto&& kvCacheBlock : kVCacheBlocks)
    {
        TLLM_CHECK(kvCacheBlock->getDataType() == cacheDataType);
        TLLM_CHECK(kvCacheBlock->getSize() == cacheBlockSize);
        CachePtrs.push_back(reinterpret_cast<T*>(kvCacheBlock->data()));
    }

    for (auto&& outputSplitBlock : outputSplitBlocks)
    {
        TLLM_CHECK(outputSplitBlock->getDataType() == cacheDataType);
        TLLM_CHECK(outputSplitBlock->getSize() == cacheBlockSize * inputBlockNum / outputCacheNum);
        CachePtrs.push_back(reinterpret_cast<T*>(outputSplitBlock->data()));
    }
    runtime::BufferManager::IBufferPtr PtrsDeviceBuffer
        = bufferManager.gpu(CachePtrs.size(), nvinfer1::DataType::kINT64);
    TLLM_CHECK(PtrsDeviceBuffer->getSizeInBytes() == CachePtrs.size() * sizeof(T*));
    bufferManager.copy(CachePtrs.data(), *PtrsDeviceBuffer, runtime::MemoryType::kCPU);

    constexpr int subWarpSize = 8;
    constexpr int subWarpNumInGroup = 8;
    int blockDimx = 128;

    int oPpNum = oCacheState.getParallelConfig().mPipelineParallelism;
    // layers
    unsigned int gridDimx = oCacheState.getModelConfig().mNbKvHeadsPerLayer.size() / oPpNum;
    // blockNum
    unsigned int gridDimy = inputBlockNum;

    dim3 gridDim{gridDimx, gridDimy};

    int const sizePerHead = oCacheState.getModelConfig().mSizePerHead;
    T const** inputBlockPtrsDev = reinterpret_cast<T const**>(PtrsDeviceBuffer->data());
    T** outputCachePtrsDev = reinterpret_cast<T**>(PtrsDeviceBuffer->data()) + inputBlockNum;
    int tokensPerBlock = oCacheState.getModelConfig().mTokensPerBlock;
    int numLayers = oCacheState.getModelConfig().mNbKvHeadsPerLayer.size() / oPpNum;
    int headNum = oCacheState.getModelConfig().mNbKvHeadsPerLayer[0];
    int dimsPerHead = oCacheState.getModelConfig().mSizePerHead;
    int DomainPPSize = targetRankInfo.mDomainPPSize;
    int DomainTPSize = targetRankInfo.mDomainTPSize;
    int iPPNum = iCacheState.getParallelConfig().mPipelineParallelism;
    int iTPNum = iCacheState.getParallelConfig().mTensorParallelism;
    int oTPNum = oCacheState.getParallelConfig().mTensorParallelism;
    int layerNumDomainPP = numLayers / DomainPPSize;
    int headNumDomainTP = headNum / DomainTPSize;

    TLLM_LOG_DEBUG(
        "splitKVCache: numLayers: %d, headNum: %d, DomainPPSize:%d, DomainTPSize:%d, layerNumDomainPP:%d, "
        "headNumDomainTP:%d",
        numLayers, headNum, DomainPPSize, DomainTPSize, layerNumDomainPP, headNumDomainTP);

    int const remainder = sizePerHead * sizeof(T) % 16;
    switch (remainder)
    {
    case 0:
    {
        splitKVCacheKernel<T, subWarpSize, subWarpNumInGroup, 16>
            <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputBlockPtrsDev, outputCachePtrsDev,
                tokensPerBlock, numLayers, headNum, dimsPerHead, inputBlockNum, DomainPPSize, DomainTPSize,
                layerNumDomainPP, headNumDomainTP);
        break;
    }
    case 8:
    {
        splitKVCacheKernel<T, subWarpSize, subWarpNumInGroup, 8>
            <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputBlockPtrsDev, outputCachePtrsDev,
                tokensPerBlock, numLayers, headNum, dimsPerHead, inputBlockNum, DomainPPSize, DomainTPSize,
                layerNumDomainPP, headNumDomainTP);
        break;
    }
    case 4:
    case 12:
    {
        if constexpr (sizeof(T) <= 4)
        {
            splitKVCacheKernel<T, subWarpSize, subWarpNumInGroup, 4>
                <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputBlockPtrsDev, outputCachePtrsDev,
                    tokensPerBlock, numLayers, headNum, dimsPerHead, inputBlockNum, DomainPPSize, DomainTPSize,
                    layerNumDomainPP, headNumDomainTP);
            break;
        }
    }

    case 2:
    case 6:
    case 10:
    case 14:
    {
        if constexpr (sizeof(T) <= 2)
        {
            splitKVCacheKernel<T, subWarpSize, subWarpNumInGroup, 2>
                <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputBlockPtrsDev, outputCachePtrsDev,
                    tokensPerBlock, numLayers, headNum, dimsPerHead, inputBlockNum, DomainPPSize, DomainTPSize,
                    layerNumDomainPP, headNumDomainTP);
            break;
        }
    }
    default:
    {
        if constexpr (sizeof(T) <= 1)
        {
            splitKVCacheKernel<T, subWarpSize, subWarpNumInGroup, 1>
                <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputBlockPtrsDev, outputCachePtrsDev,
                    tokensPerBlock, numLayers, headNum, dimsPerHead, inputBlockNum, DomainPPSize, DomainTPSize,
                    layerNumDomainPP, headNumDomainTP);
            break;
        }
        else
        {
            TLLM_THROW(" splitKVCacheDispatch no support data type error");
        }
    }
    }
}

void splitKVCacheDispatch(std::vector<runtime::ITensor::SharedPtr> const& kVCacheBlocks,
    std::vector<runtime::ITensor::SharedPtr>& ouputSplitBlocks, kv_cache::CacheState const& iCacheState,
    kv_cache::CacheState const& oCacheState, int selfIdx, runtime::BufferManager const& bufferManager)
{
    auto dataType = kVCacheBlocks.at(0)->getDataType();
    auto dataSize = tensorrt_llm::common::getDTypeSize(dataType);
    switch (dataSize)
    {
    case 8:
    {
        splitKVCache<double>(kVCacheBlocks, ouputSplitBlocks, iCacheState, oCacheState, selfIdx, bufferManager);
        break;
    }
    case 4:
    {
        splitKVCache<float>(kVCacheBlocks, ouputSplitBlocks, iCacheState, oCacheState, selfIdx, bufferManager);
        break;
    }
    case 2:
    {
        splitKVCache<half>(kVCacheBlocks, ouputSplitBlocks, iCacheState, oCacheState, selfIdx, bufferManager);
        break;
    }
    case 1:
    {
        splitKVCache<uint8_t>(kVCacheBlocks, ouputSplitBlocks, iCacheState, oCacheState, selfIdx, bufferManager);
        break;
    }
    default:
    {
        TLLM_THROW(" splitKVCacheDispatch no support data type error");
    }
    }
}

template <typename T>
void concatenateKVCache(std::vector<runtime::ITensor::SharedPtr> const& inputSplitBlocks,
    std::vector<runtime::ITensor::SharedPtr>& outputKvCacheBlocks,

    kv_cache::CacheState const& iCacheState, kv_cache::CacheState const& oCacheState, int selfIdx,
    runtime::BufferManager const& bufferManager)
{

    auto outputBlockNum = outputKvCacheBlocks.size();
    auto targetRankInfo = targetIRanks(iCacheState, oCacheState, selfIdx);
    TLLM_CHECK(targetRankInfo.mIRanks.size()
        == (static_cast<size_t>(targetRankInfo.mDomainPPSize * targetRankInfo.mDomainTPSize)));

    auto inputCacheNum = targetRankInfo.mIRanks.size();
    TLLM_CHECK(inputCacheNum == inputSplitBlocks.size());
    TLLM_CHECK(outputBlockNum > 0);
    auto cacheBlockSize = outputKvCacheBlocks.at(0)->getSize();
    auto cacheDataType = outputKvCacheBlocks.at(0)->getDataType();
    std::vector<T*> CachePtrs;
    for (auto&& kvCacheBlock : outputKvCacheBlocks)
    {
        TLLM_CHECK(kvCacheBlock->getDataType() == cacheDataType);
        TLLM_CHECK(kvCacheBlock->getSize() == cacheBlockSize);
        CachePtrs.push_back(reinterpret_cast<T*>(kvCacheBlock->data()));
    }
    for (auto&& inputSplitBlock : inputSplitBlocks)
    {
        TLLM_CHECK(inputSplitBlock->getDataType() == cacheDataType);
        TLLM_CHECK(inputSplitBlock->getSize() == cacheBlockSize * outputBlockNum / inputCacheNum);
        CachePtrs.push_back(reinterpret_cast<T*>(inputSplitBlock->data()));
    }
    runtime::BufferManager::IBufferPtr PtrsDeviceBuffer
        = bufferManager.gpu(CachePtrs.size(), nvinfer1::DataType::kINT64);
    TLLM_CHECK(PtrsDeviceBuffer->getSizeInBytes() == CachePtrs.size() * sizeof(T*));
    bufferManager.copy(CachePtrs.data(), *PtrsDeviceBuffer, runtime::MemoryType::kCPU);

    constexpr int subWarpSize = 8;
    constexpr int subWarpNumInGroup = 8;
    int blockDimx = 128;

    int oPpNum = oCacheState.getParallelConfig().mPipelineParallelism;
    // layers
    unsigned int gridDimx = oCacheState.getModelConfig().mNbKvHeadsPerLayer.size() / oPpNum;
    // blockNum
    unsigned int gridDimy = outputBlockNum;

    dim3 gridDim{gridDimx, gridDimy};
    int const sizePerHead = oCacheState.getModelConfig().mSizePerHead;
    int endLayerId = oCacheState.getModelConfig().mNbKvHeadsPerLayer.size() / oPpNum;
    T** ouptutBlockPtrsDev = reinterpret_cast<T**>(PtrsDeviceBuffer->data());
    T const** inputSplitBlockPtrsDev = reinterpret_cast<T const**>(PtrsDeviceBuffer->data()) + outputBlockNum;
    int tokensPerBlock = oCacheState.getModelConfig().mTokensPerBlock;
    int numLayers = oCacheState.getModelConfig().mNbKvHeadsPerLayer.size() / oPpNum;
    int headNum = oCacheState.getModelConfig().mNbKvHeadsPerLayer[0];
    int dimsPerHead = oCacheState.getModelConfig().mSizePerHead;
    int DomainPPSize = targetRankInfo.mDomainPPSize;
    int DomainTPSize = targetRankInfo.mDomainTPSize;
    int iPPNum = iCacheState.getParallelConfig().mPipelineParallelism;
    int iTPNum = iCacheState.getParallelConfig().mTensorParallelism;
    int oTPNum = oCacheState.getParallelConfig().mTensorParallelism;
    int layerNumDomainPP = numLayers / DomainPPSize;
    int headNumDomainTP = headNum / DomainTPSize;

    TLLM_LOG_DEBUG(
        "concatenateKVCache: numLayers: %d, headNum: %d, DomainPPSize:%d, DomainTPSize:%d, layerNumDomainPP:%d, "
        "headNumDomainTP:%d",
        numLayers, headNum, DomainPPSize, DomainTPSize, layerNumDomainPP, headNumDomainTP);
    int const remainder = sizePerHead * sizeof(T) % 16;

    switch (remainder)
    {
    case 0:
    {
        concatenateKVCacheKernel<T, subWarpSize, subWarpNumInGroup, 16>
            <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputSplitBlockPtrsDev, ouptutBlockPtrsDev,
                tokensPerBlock, numLayers, headNum, dimsPerHead, outputBlockNum, DomainPPSize, DomainTPSize,
                layerNumDomainPP, headNumDomainTP);

        break;
    }
    case 8:
    {
        concatenateKVCacheKernel<T, subWarpSize, subWarpNumInGroup, 8>
            <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputSplitBlockPtrsDev, ouptutBlockPtrsDev,
                tokensPerBlock, numLayers, headNum, dimsPerHead, outputBlockNum, DomainPPSize, DomainTPSize,
                layerNumDomainPP, headNumDomainTP);

        break;
    }
    case 4:
    case 12:
    {
        if constexpr (sizeof(T) <= 4)
        {
            concatenateKVCacheKernel<T, subWarpSize, subWarpNumInGroup, 4>
                <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputSplitBlockPtrsDev, ouptutBlockPtrsDev,
                    tokensPerBlock, numLayers, headNum, dimsPerHead, outputBlockNum, DomainPPSize, DomainTPSize,
                    layerNumDomainPP, headNumDomainTP);

            break;
        }
    }
    case 2:
    case 6:
    case 10:
    case 14:
    {
        if constexpr (sizeof(T) <= 2)
        {
            concatenateKVCacheKernel<T, subWarpSize, subWarpNumInGroup, 2>
                <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputSplitBlockPtrsDev, ouptutBlockPtrsDev,
                    tokensPerBlock, numLayers, headNum, dimsPerHead, outputBlockNum, DomainPPSize, DomainTPSize,
                    layerNumDomainPP, headNumDomainTP);

            break;
        }
    }
    default:
    {
        if constexpr (sizeof(T) <= 1)
        {
            concatenateKVCacheKernel<T, subWarpSize, subWarpNumInGroup, 1>
                <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputSplitBlockPtrsDev, ouptutBlockPtrsDev,
                    tokensPerBlock, numLayers, headNum, dimsPerHead, outputBlockNum, DomainPPSize, DomainTPSize,
                    layerNumDomainPP, headNumDomainTP);

            break;
        }
        else
        {
            TLLM_THROW(" concatenateKVCache no support data type error");
        }
    }
    }
}

void concatenateKvCacheV2Dispatch(std::vector<runtime::ITensor::SharedPtr> const& inputSplitBlocks,
    std::vector<runtime::ITensor::SharedPtr>& outputKvCacheBlocks, kv_cache::CacheState const& iCacheState,
    kv_cache::CacheState const& oCacheState, int selfIdx, runtime::BufferManager const& bufferManager)
{

    auto dataType = outputKvCacheBlocks.at(0)->getDataType();
    auto dataSize = tensorrt_llm::common::getDTypeSize(dataType);
    switch (dataSize)
    {
    case 8:
    {
        concatenateKVCache<double>(
            inputSplitBlocks, outputKvCacheBlocks, iCacheState, oCacheState, selfIdx, bufferManager);
        break;
    }
    case 4:
    {
        concatenateKVCache<float>(
            inputSplitBlocks, outputKvCacheBlocks, iCacheState, oCacheState, selfIdx, bufferManager);
        break;
    }
    case 2:
    {
        concatenateKVCache<half>(
            inputSplitBlocks, outputKvCacheBlocks, iCacheState, oCacheState, selfIdx, bufferManager);
        break;
    }
    case 1:
    {
        concatenateKVCache<uint8_t>(
            inputSplitBlocks, outputKvCacheBlocks, iCacheState, oCacheState, selfIdx, bufferManager);
        break;
    }
    default:
    {
        TLLM_THROW(" concatenateKVCache no support data type error");
    }
    }
}

} // namespace tensorrt_llm::executor::kv_cache
