#!/bin/bash

python -m vllm.entrypoints.cli.main bench serve \
  --backend vllm \
  --base-url http://127.0.0.1:8000 \
  --model meta-llama/Llama-3.1-8B \
  --dataset-name random \
  --random-input-len 2048 \
  --random-output-len 512 \
  --num-prompts 50 \
  --request-rate 10 \
  --profile