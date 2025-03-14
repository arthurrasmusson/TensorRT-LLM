#!/bin/bash

TLLM_LOG_LEVEL=DEBUG trtllm-serve \
   /mnt/weka/Models/Engines/llama-3.1-8b-engine \
   --tokenizer /mnt/weka/Models/Safetensors/Meta-Llama-3.1-8B-Instruct \
   --port 8000
