#!/bin/bash

docker run --gpus all -it --network=host --ipc=host \
    -v /mnt/weka:/mnt/weka \
    -v "$(pwd)":/code/tensorrt_llm \
    -v /usr/local/cuda-12.6/targets/x86_64-linux/lib:/usr/local/cuda-12.6/targets/x86_64-linux/lib \
    -w /code/tensorrt_llm \
    tensorrt_llm/devel:latest
