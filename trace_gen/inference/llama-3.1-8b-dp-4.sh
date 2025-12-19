#!/bin/bash

export VLLM_TORCH_PROFILER_DIR=/pscratch/sd/b/bck/kineto_out
export VLLM_TORCH_PROFILER_RECORD_SHAPES=0
export VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY=0
export VLLM_TORCH_PROFILER_WITH_STACK=0
export VLLM_TORCH_PROFILER_WITH_FLOPS=0
export VLLM_RPC_TIMEOUT=1800000

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B \
    --swap-space 16 \
    --disable-log-requests \
    --enforce-eager \
    --enable-chunked-prefill \
    --max-num-batched-tokens 512 \
    --max-num-seqs 512 \
    --disable-sliding-window \
    --data-parallel-size 4