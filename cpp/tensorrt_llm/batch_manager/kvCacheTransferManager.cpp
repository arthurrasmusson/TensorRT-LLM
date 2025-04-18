/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

 #include <cstdio>
 #include <cstring>
 #include <string>
 #include <vector>
 #include <optional>
 #include <fstream>
 #include <stdexcept>
 
 // CUDA includes for device pointer queries
 #include <cuda_runtime_api.h>
 #include <cuda.h>
 
 // For YAML parsing
 #include <yaml-cpp/yaml.h>
 
 // For GDS API usage
 #include "gds_client_api.h"
 
 #include "tensorrt_llm/batch_manager/kvCacheTransferManager.h"
 #include "tensorrt_llm/batch_manager/kvCacheEventManager.h"
 #include "tensorrt_llm/batch_manager/kvCacheManager.h"
 #include "tensorrt_llm/kernels/kvCachePartialCopy.h"
 #include "tensorrt_llm/runtime/bufferManager.h"
 #include "tensorrt_llm/runtime/cudaEvent.h"
 #include "tensorrt_llm/runtime/cudaStream.h"
 #include "tensorrt_llm/executor/types.h"
 
 #ifndef _WIN32
 #  include <fcntl.h>
 #  include <sys/stat.h>
 #  include <sys/types.h>
 #  include <unistd.h>
 #endif
 
 namespace tr = tensorrt_llm::runtime;
 namespace tk = tensorrt_llm::kernels;
 
 namespace tensorrt_llm {
 namespace batch_manager {
 namespace kv_cache_manager
 {
 
 /*!
  * \brief Helper to retrieve the PCI bus ID for a device pointer.
  */
 static std::string getGpuPciBusId(const void* dptr)
 {
     if (!dptr) return "";
 
     cudaPointerAttributes attrs;
     auto err = cudaPointerGetAttributes(&attrs, dptr);
     if (err != cudaSuccess)
     {
         TLLM_LOG_ERROR("cudaPointerGetAttributes failed, err=%d", static_cast<int>(err));
         return "";
     }
     int deviceIndex = attrs.device;
 
     char pciBusId[128];
     auto err2 = cudaDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), deviceIndex);
     if (err2 != cudaSuccess)
     {
         TLLM_LOG_ERROR("cudaDeviceGetPCIBusId failed, err=%d", static_cast<int>(err2));
         return "";
     }
 
     // Typically "0000:1b:00.0"
     return std::string(pciBusId);
 }
 
 /*!
  * \brief Minimal struct for GPU–NIC pairing from YAML
  */
 struct GpuNicPair
 {
     std::string gpuPci;
     std::string nicPci;
     std::string nicIp;
 };
 
 /*!
  * \brief Parse GPU–NIC pairs from a YAML file - TODO: move this out of kvCacheTransferManager to utils.
  *        We need to pass in a path from KvCacheRetentionConfig via gpuNicYaml parameter.
  */
 static std::vector<GpuNicPair> parseGpuNicPairsFromYaml(const std::string& yamlPath)
 {
     std::vector<GpuNicPair> result;
     YAML::Node root = YAML::LoadFile(yamlPath);
     if (!root["pairs"])
     {
         TLLM_LOG_ERROR("YAML file has no 'pairs' node: %s", yamlPath.c_str());
         return result;
     }
     for (auto const& it : root["pairs"])
     {
         GpuNicPair gp;
         if (it["gpu"] && it["gpu"]["pci"])
         {
             gp.gpuPci = it["gpu"]["pci"].as<std::string>();
         }
         if (it["nic"] && it["nic"]["pci"])
         {
             gp.nicPci = it["nic"]["pci"].as<std::string>();
         }
         if (it["nic"] && it["nic"]["ip"])
         {
             gp.nicIp = it["nic"]["ip"].as<std::string>();
         }
         result.push_back(gp);
     }
     return result;
 }
 
 /*!
  * \brief Function that registers each GPU–NIC pair with GDS.
  */
 static void registerGpuNicPairs(std::vector<GpuNicPair> const& pairs)
 {
     for (auto const& p : pairs)
     {
         int rc = gds_api_add_gpu_nic_pair(
             const_cast<char*>(p.gpuPci.c_str()),
             const_cast<char*>(p.nicPci.c_str()),
             const_cast<char*>(p.nicIp.c_str()));
         if (rc != 0)
         {
             TLLM_LOG_ERROR("Failed gds_api_add_gpu_nic_pair() for GPU='%s' NIC='%s' IP='%s'",
                            p.gpuPci.c_str(), p.nicPci.c_str(), p.nicIp.c_str());
         }
         else
         {
             TLLM_LOG_INFO("Registered GPU='%s' NIC='%s' IP='%s' with GDS", 
                           p.gpuPci.c_str(), p.nicPci.c_str(), p.nicIp.c_str());
         }
     }
 }
 
 /*!
  * \brief Implementation of the KVCacheTransferManager
  */
 KVCacheTransferManager::KVCacheTransferManager(tr::BufferManager const& bufferManager)
     : mBufferManager{bufferManager}
     , mOnboardManager(std::make_shared<tr::CudaStream>())
     , mOffloadManager(std::make_shared<tr::CudaStream>())
 {
 }
 
 /*!
  * \brief Parse a YAML, then register GPU–NIC pairs. 
  *        TODO: pass gpuNicYaml parameter from KvCacheRetentionConfig instead.
  */
 void KVCacheTransferManager::initWekaGdsPairs(std::string const& yamlPath)
 {
     auto pairs = parseGpuNicPairsFromYaml(yamlPath);
     registerGpuNicPairs(pairs);
 }
 
 tr::ITensor::SharedPtr KVCacheTransferManager::computeBlockPointer(
     BlockPtr const& block, std::vector<KVCacheBlockPool> const& pools, size_t poolIdx)
 {
     TLLM_CHECK_WITH_INFO(!pools.empty(), "Pool index %lu is out of bounds", poolIdx);
     auto const& pool = pools.at(poolIdx);
     auto ptr = block->isPrimary() ? pool.primaryPtr : pool.secondaryPtr;
     auto const blockOffset = block->getMemoryPoolBlockIndex();
     tr::ITensor::SharedPtr blockTensor{tr::ITensor::slice(ptr, blockOffset, 1)};
     return blockTensor;
 }
 
 /*!
  * \brief Main copy logic. We handle partial copy if numTokensToCopy>0, 
  *        or do a full copy otherwise. This function is used for both 
  *        offload (GPU->host/disk) and onboard (host/disk->GPU).
  *
  * \param isOffload If true, direction is GPU->host; if false, host->GPU
  * \param mode e.g. DRAM, GDS, POSIX
  * \param directory The directory to store or read from if file-based
  */
 void KVCacheTransferManager::copyBlock(
     BlockPtr const& src,
     BlockPtr const& dst,
     std::vector<KVCacheBlockPool> const& pools,
     bool isOffload,
     int numTokensToCopy,
     executor::KvCacheTransferMode mode,
     std::optional<std::string> directory)
 {
     TLLM_LOG_DEBUG(
         "copyBlock entered: srcId=%d, dstId=%d, isOffload=%s, mode=%d",
         src->getBlockId(),
         dst->getBlockId(),
         (isOffload ? "GPU->host" : "host->GPU"),
         static_cast<int>(mode));
 
     // Identify the GPU pointer for TODO: Test
     {
         auto srcPtrSample = computeBlockPointer(src, pools, 0 /*just 1st pool sample*/);
         void* dptr = srcPtrSample ? srcPtrSample->data() : nullptr;
         auto pciId = getGpuPciBusId(dptr);
         TLLM_LOG_INFO("GPU pointer is on PCI BDF: %s", pciId.c_str());
     }
 
     // If the user wants DRAM-based copy, we do normal host-based approach
     if (mode == executor::KvCacheTransferMode::DRAM)
     {
         TLLM_LOG_INFO("Using DRAM-based approach (no GDS).");
         // block spans multiple pools
         for (size_t poolIdx = 0; poolIdx < pools.size(); ++poolIdx)
         {
             auto srcPtr = computeBlockPointer(src, pools, poolIdx);
             auto dstPtr = computeBlockPointer(dst, pools, poolIdx);
 
             // partial copy logic:
             if (numTokensToCopy <= 0
                 || srcPtr->getDataType() == nvinfer1::DataType::kINT4
                 || srcPtr->getDataType() == nvinfer1::DataType::kFP4)
             {
                 (isOffload ? mOffloadManager : mOnboardManager).copy(*srcPtr, *dstPtr);
             }
             else
             {
                 int const tokensPerBlock = pools[poolIdx].tokensPerBlock;
                 if (numTokensToCopy >= tokensPerBlock)
                 {
                     (isOffload ? mOffloadManager : mOnboardManager).copy(*srcPtr, *dstPtr);
                 }
                 else
                 {
                     auto stream = (isOffload ? mOffloadManager : mOnboardManager).getStream().get();
                     int const numLayers   = pools[poolIdx].numLayers;
                     int const numHeads    = pools[poolIdx].numKvHeads;
                     int const sizePerHead = pools[poolIdx].sizePerHead;
                     auto shape = srcPtr->getShape();
 
                     TLLM_CHECK_WITH_INFO(
                         shape.nbDims == 4,
                         "Expected KVCache block to have 4 dims, got %d",
                         shape.nbDims);
 
                     tk::kvCacheBlockPartialCopy(
                         *dstPtr, *srcPtr,
                         numLayers, numHeads, tokensPerBlock, sizePerHead,
                         numTokensToCopy,
                         stream);
                 }
             }
         }
 
         return; // done DRAM path
     }
 
 #ifndef _WIN32
     // If GDS or POSIX_DEBUG_FALLBACK
     TLLM_CHECK_WITH_INFO(directory.has_value(),
         "Expected directory path for KVCache offload, but none was provided.");
 
     for (size_t poolIdx = 0; poolIdx < pools.size(); ++poolIdx)
     {
         auto srcPtr = computeBlockPointer(src, pools, poolIdx);
         auto dstPtr = computeBlockPointer(dst, pools, poolIdx);
 
         // Create a file name based on blockId, poolIdx
         int size = std::snprintf(nullptr, 0,
             "%s/block_%d_pool_%zu.bin",
             directory.value().c_str(),
             src->getBlockId(),
             poolIdx);
         std::string filename(size + 1, '\0');
         std::snprintf(filename.data(), filename.size(),
             "%s/block_%d_pool_%zu.bin",
             directory.value().c_str(),
             src->getBlockId(),
             poolIdx);
 
         // Replace cuFileWrite/cuFileRead with gds_api_memcpy_device_to_host / host_to_device
         if (mode == executor::KvCacheTransferMode::POSIX_DEBUG_FALLBACK)
         {
             // I do a POSIX read/write with a staging buffer, but GPU copy by gds_api calls
             TLLM_LOG_INFO("Using POSIX_DEBUG_FALLBACK for file: %s", filename.c_str());
 
             int fd = (isOffload ? ::open(filename.c_str(), O_CREAT | O_WRONLY, 0664)
                                 : ::open(filename.c_str(), O_RDONLY));
             if (fd < 0)
             {
                 TLLM_LOG_ERROR("open('%s') failed, fallback to something else", filename.c_str());
                 continue;
             }
 
             ssize_t numBytes = static_cast<ssize_t>(srcPtr->getSizeInBytes());
             std::vector<uint8_t> hostBuffer(numBytes);
 
             if (isOffload)
             {
                 // GPU -> Host
                 gds_api_memcpy_device_to_host(reinterpret_cast<unsigned long>(srcPtr->data()), 
                                               hostBuffer.data(), numBytes);
                 // Then do a write
                 ssize_t written = ::write(fd, hostBuffer.data(), numBytes);
                 if (written < 0)
                 {
                     TLLM_LOG_ERROR("POSIX write error = %zd", written);
                 }
             }
             else
             {
                 // read -> host, host->GPU
                 ssize_t readCount = ::read(fd, hostBuffer.data(), numBytes);
                 if (readCount < 0)
                 {
                     TLLM_LOG_ERROR("POSIX read error = %zd", readCount);
                 }
                 else
                 {
                     gds_api_memcpy_host_to_device(reinterpret_cast<unsigned long>(dstPtr->data()),
                                                   hostBuffer.data(), static_cast<unsigned long>(readCount));
                 }
             }
 
             ::close(fd);
             continue;
         }
         else if (mode == executor::KvCacheTransferMode::GDS)
         {
 
             TLLM_LOG_INFO("Using GDS mode to do device <-> file with gds_api_memcpy_* as a stand-in for cuFile.");
 
             int openFlags = isOffload ? (O_CREAT | O_WRONLY) : O_RDONLY;
             int fd = ::open(filename.c_str(), openFlags, 0664);
             if (fd < 0)
             {
                 TLLM_LOG_ERROR("open('%s') failed for GDS fallback", filename.c_str());
                 continue;
             }
 
             ssize_t numBytes = static_cast<ssize_t>(srcPtr->getSizeInBytes());
             std::vector<uint8_t> hostBuf(numBytes);
 
             if (isOffload)
             {
                 // GPU->Host
                 gds_api_memcpy_device_to_host(reinterpret_cast<unsigned long>(srcPtr->data()),
                                               hostBuf.data(), numBytes);
                 // Then POSIX write to file:
                 auto wr = ::write(fd, hostBuf.data(), numBytes);
                 if (wr < 0)
                 {
                     TLLM_LOG_ERROR("write to file failed, wr=%zd", wr);
                 }
             }
             else
             {
                 // read from file to host:
                 auto rd = ::read(fd, hostBuf.data(), numBytes);
                 if (rd < 0)
                 {
                     TLLM_LOG_ERROR("read from file failed, rd=%zd", rd);
                 }
                 else
                 {
                     // host->device
                     gds_api_memcpy_host_to_device(reinterpret_cast<unsigned long>(dstPtr->data()),
                                                   hostBuf.data(),
                                                   static_cast<unsigned long>(rd));
                 }
             }
 
             ::close(fd);
         }
     }
 #else
     // Windows fallback
     TLLM_LOG_WARN("GDS or POSIX fallback not supported on Windows build.");
 #endif // !_WIN32
 }
 
 void KVCacheTransferManager::onboard(
     BlockPtr const& offloadBlock,
     BlockPtr const& block,
     std::vector<KVCacheBlockPool> const& pools,
     int numTokensToCopy,
     executor::KvCacheTransferMode mode,
     std::optional<std::string> directory)
 {
     // Wait for any pending offload event that wrote offloadBlock
     if (mPendingOffloads.find(offloadBlock->getBlockId()) != mPendingOffloads.end())
     {
         mOnboardManager.getStream().wait(mPendingOffloads[offloadBlock->getBlockId()]);
     }
     // Do the copy
     copyBlock(offloadBlock, block, pools, false, numTokensToCopy, mode, directory);
 }
 
 void KVCacheTransferManager::offload(
     BlockPtr const& block,
     BlockPtr const& offloadBlock,
     std::vector<KVCacheBlockPool> const& pools,
     int numTokensToCopy,
     executor::KvCacheTransferMode mode,
     std::optional<std::string> directory)
 {
     // Record event so that if something else tries to use offloadBlock, we can wait
     mPendingOffloads[block->getBlockId()] = tr::CudaEvent();
     copyBlock(block, offloadBlock, pools, true, numTokensToCopy, mode, directory);
     // record event
     mOffloadManager.getStream().record(mPendingOffloads[block->getBlockId()]);
 }
 
 void KVCacheTransferManager::syncTransfers()
 {
     tr::CudaEvent offloadEvent;
     mOffloadManager.getStream().record(offloadEvent);
 
     tr::CudaEvent onboardEvent;
     mOnboardManager.getStream().record(onboardEvent);
 
     mBufferManager.getStream().wait(offloadEvent);
     mBufferManager.getStream().wait(onboardEvent);
 
     // Once we sync, clear our list of pending thransfers.
     mPendingOffloads.clear();
 }
 
 } // namespace kv_cache_manager
 } // namespace batch_manager
 } // namespace tensorrt_llm
 