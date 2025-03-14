#!/bin/bash

docker run --gpus all -it --network=host --ipc=host \
    -v /mnt/weka:/mnt/weka \
    -v "$(pwd)":/code/tensorrt_llm \
    -w /code/tensorrt_llm \
    tensorrt_llm/devel:latest
