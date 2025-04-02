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

 #include <cstdint>
 #include <cuda_runtime.h>
 #include <openssl/aes.h>  // For AES key generation and management
 
 #include "tensorrt_llm/batch_manager/kvCacheTransferManager.h"
 #include "tensorrt_llm/batch_manager/kvCacheEventManager.h"
 #include "tensorrt_llm/batch_manager/kvCacheManager.h"
 #include "tensorrt_llm/kernels/kvCachePartialCopy.h"
 #include "tensorrt_llm/runtime/bufferManager.h"
 #include "tensorrt_llm/runtime/cudaEvent.h"
 #include "tensorrt_llm/runtime/cudaStream.h"
 
 // For GPUDirect Storage (cuFile)
 // Add condition (CMake doesn't find cufile -> build without GDS & generate tensorrt_llm.so with POSIX IO instead)
 #include <cufile.h>
 #include <fcntl.h>
 #include <unistd.h>
 #include <cstdio>
 #include <cstring>
 #include <sys/types.h>
 #include <sys/stat.h>
 
 // Performant CUDA_AES
 // To Do add this code.
 // https://github.com/cihangirtezcan/CUDA_AES/blob/gh-pages/256-es.cuh 
 // https://github.com/cihangirtezcan/CUDA_AES/blob/gh-pages/256-ctr.cuh
 extern "C" void cudaAESEncrypt(const void* input, void* output, size_t size, const unsigned char* key, cudaStream_t stream);
 extern "C" void cudaAESDecrypt(const void* input, void* output, size_t size, const unsigned char* key, cudaStream_t stream);
 
 namespace tr = tensorrt_llm::runtime;
 namespace tk = tensorrt_llm::kernels;
 
 namespace tensorrt_llm::batch_manager::kv_cache_manager
 {
 
 KVCacheTransferManager::KVCacheTransferManager(tr::BufferManager const& bufferManager)
     : mBufferManager{bufferManager}
     , mOnboardManager(std::make_shared<tr::CudaStream>())
     , mOffloadManager(std::make_shared<tr::CudaStream>())
 {
     /* 
     /  There are a few ways we can do the key.
     /  One option is that we have a "cluster shared key"
     /  which could give all nodes the same encryption key.
     /  
     /  Another approach is we could have per session keys,
     /  which would wrap the LlmRequest object containing 
     /  the user's input tokens with an AES key the user
     /  of LlmRequest passes from TRT's Python or C++ API. 
    */

    // Initialize AES key (256-bit key as our example)
     unsigned char aesKey[AES_KEY_SIZE_256] = { /* User's input token key here */ };
     memcpy(mAesKey, aesKey, AES_KEY_SIZE_256);
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
 
 void KVCacheTransferManager::copyBlock(
     BlockPtr const& src,
     BlockPtr const& dst,
     std::vector<KVCacheBlockPool> const& pools,
     bool isOffload,
     bool DRAMDestination = true)
 {
     printf("ENTERED COPY BLOCK: isOffload=%s, DRAMDestination=%s\n",
         isOffload ? "true" : "false",
         DRAMDestination ? "true" : "false");
 
     auto const numPools = pools.size();
     auto stream = isOffload ? mOffloadManager.getStream() : mOnboardManager.getStream();
 
     for (size_t poolIdx = 0; poolIdx < numPools; poolIdx++)
     {
         auto srcPtr = computeBlockPointer(src, pools, poolIdx);
         auto dstPtr = computeBlockPointer(dst, pools, poolIdx);
         size_t bufferSize = srcPtr->getSizeInBytes();
         void* srcData = srcPtr->data();
 
         if (DRAMDestination)
         {
             printf("[INFO] DRAMDestination is true; using original GPU-to-GPU copy\n");
             (isOffload ? mOffloadManager : mOnboardManager).copy(*srcPtr, *dstPtr);
             continue;
         }
 
         // Build filename
         // TODO: define this in kv_cache_config
         char filename[256];
         std::snprintf(filename, sizeof(filename), "/mnt/weka/block_%d_pool_%zu.bin", src->getBlockId(), poolIdx);
 
         int openFlags = isOffload ? (O_CREAT | O_WRONLY) : O_RDONLY;
         int fd = ::open(filename, openFlags, 0664);
         if (fd < 0)
         {
             printf("[ERROR] Failed to open '%s' for %s\n", filename, isOffload ? "writing" : "reading");
             continue;
         }
 
         if (isOffload)
         {
             // Allocate GPU buffer for encrypted data
             void* encryptedBuffer;
             cudaMalloc(&encryptedBuffer, bufferSize);
 
             // Encrypt the raw KV data
             cudaAESEncrypt(srcData, encryptedBuffer, bufferSize, mAesKey, stream.get());
 
             // GDS or POSIX write
             if (!mDebugNeverGDS)
             {
                 CUfileDescr_t cufileDesc{};
                 cufileDesc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
                 cufileDesc.handle.fd = fd;
                 CUfileHandle_t cufileHandle;
                 CUfileError_t status = cuFileHandleRegister(&cufileHandle, &cufileDesc);
 
                 if (status.err == CU_FILE_SUCCESS)
                 {
                     printf("[DEBUG] Using GDS mode for file: %s\n", filename);
                     ssize_t bytesWritten = cuFileWrite(cufileHandle, encryptedBuffer, bufferSize, 0, 0);
                     if (bytesWritten < 0)
                         printf("[ERROR] cuFileWrite error=%zd\n", bytesWritten);
                     else
                         printf("[DEBUG] Wrote %zd encrypted bytes to %s\n", bytesWritten, filename);
                     cuFileHandleDeregister(cufileHandle);
                 }
                 else
                 {
                     printf("[WARN] cuFileHandleRegister failed (err=%d). Falling back to POSIX\n", status.err);
                     fallbackPosixIOEncrypted(encryptedBuffer, nullptr, fd, filename, true, bufferSize);
                 }
             }
             else
             {
                 printf("[INFO] mDebugNeverGDS=true; forcing POSIX fallback\n");
                 fallbackPosixIOEncrypted(encryptedBuffer, nullptr, fd, filename, true, bufferSize);
             }
 
             cudaFree(encryptedBuffer);
         }
         else // Onboard (decrypt)
         {
             // Allocate GPU buffer for encrypted data
             void* encryptedBuffer;
             cudaMalloc(&encryptedBuffer, bufferSize);
 
             // Read encrypted data
             if (!mDebugNeverGDS)
             {
                 CUfileDescr_t cufileDesc{};
                 cufileDesc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
                 cufileDesc.handle.fd = fd;
                 CUfileHandle_t cufileHandle;
                 CUfileError_t status = cuFileHandleRegister(&cufileHandle, &cufileDesc);
 
                 if (status.err == CU_FILE_SUCCESS)
                 {
                     printf("[DEBUG] Using GDS mode for file: %s\n", filename);
                     ssize_t bytesRead = cuFileRead(cufileHandle, encryptedBuffer, bufferSize, 0, 0);
                     if (bytesRead < 0)
                         printf("[ERROR] cuFileRead error=%zd\n", bytesRead);
                     else
                         printf("[DEBUG] Read %zd encrypted bytes from %s\n", bytesRead, filename);
                     cuFileHandleDeregister(cufileHandle);
                 }
                 else
                 {
                     printf("[WARN] cuFileHandleRegister failed (err=%d). Falling back to POSIX\n", status.err);
                     fallbackPosixIOEncrypted(nullptr, encryptedBuffer, fd, filename, false, bufferSize);
                 }
             }
             else
             {
                 printf("[INFO] mDebugNeverGDS=true; forcing POSIX fallback\n");
                 fallbackPosixIOEncrypted(nullptr, encryptedBuffer, fd, filename, false, bufferSize);
             }
 
             // Decrypt to dstPtr
             cudaAESDecrypt(encryptedBuffer, dstPtr->data(), bufferSize, mAesKey, stream.get());
             cudaFree(encryptedBuffer);
         }
 
         ::close(fd);
     }
 
     printf("[DEBUG] Exiting copyBlock: srcId=%d, dstId=%d, isOffload=%s\n\n",
         src->getBlockId(), dst->getBlockId(), isOffload ? "true" : "false");
 }
 
 // New helper function for POSIX fallback with encrypted data
 void KVCacheTransferManager::fallbackPosixIOEncrypted(void* srcData, void* dstData, int fd, const char* filename, bool isWrite, size_t size)
 {
     if (isWrite)
     {
         void* hostBuffer;
         cudaMallocHost(&hostBuffer, size);
         cudaMemcpy(hostBuffer, srcData, size, cudaMemcpyDeviceToHost);
         ssize_t bytesWritten = ::write(fd, hostBuffer, size);
         if (bytesWritten < 0)
             printf("[ERROR] POSIX write error=%zd\n", bytesWritten);
         else
             printf("[DEBUG] POSIX wrote %zd encrypted bytes to %s\n", bytesWritten, filename);
         cudaFreeHost(hostBuffer);
     }
     else
     {
         void* hostBuffer;
         cudaMallocHost(&hostBuffer, size);
         ssize_t bytesRead = ::read(fd, hostBuffer, size);
         if (bytesRead < 0)
             printf("[ERROR] POSIX read error=%zd\n", bytesRead);
         else
         {
             printf("[DEBUG] POSIX read %zd encrypted bytes from %s\n", bytesRead, filename);
             cudaMemcpy(dstData, hostBuffer, size, cudaMemcpyHostToDevice);
         }
         cudaFreeHost(hostBuffer);
     }
 }
 
 void KVCacheTransferManager::onboard(BlockPtr const& offloadBlock, BlockPtr const& block,
     std::vector<KVCacheBlockPool> const& pools, int numTokensToCopy)
 {
     if (mPendingOffloads.find(offloadBlock->getBlockId()) != mPendingOffloads.end())
     {
         mOnboardManager.getStream().wait(mPendingOffloads[offloadBlock->getBlockId()]);
     }
     copyBlock(offloadBlock, block, pools, false, numTokensToCopy);
 }
 
 void KVCacheTransferManager::offload(BlockPtr const& block, BlockPtr const& offloadBlock,
     std::vector<KVCacheBlockPool> const& pools, int numTokensToCopy)
 {
     mPendingOffloads[block->getBlockId()] = tr::CudaEvent();
     copyBlock(block, offloadBlock, pools, true, numTokensToCopy);
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
 
     mPendingOffloads.clear();
 }
 
 } // namespace tensorrt_llm::batch_manager::kv_cache_manager