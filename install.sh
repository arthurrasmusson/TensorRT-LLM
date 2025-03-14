#!/bin/bash

#pip3 uninstall tensorrt_llm && python3 ./scripts/build_wheel.py --disable-mpi --fast_build --use_ccache --job_count 112 --benchmarks --cuda_architectures "90-real" --trt_root /usr/local/tensorrt && pip install ./build/tensorrt_llm*.whl
# Re-enabled MPI to see if this fixes test code.
pip3 uninstall tensorrt_llm && python3 ./scripts/build_wheel.py --fast_build --use_ccache --job_count 112 --benchmarks --cuda_architectures "90-real" --trt_root /usr/local/tensorrt && cd cpp/build/ && make modelSpec && cd /code/tensorrt_llm && pip install ./build/tensorrt_llm*.whl
