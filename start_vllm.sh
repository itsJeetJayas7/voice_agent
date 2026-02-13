#!/bin/bash
source /root/voxtral-env/bin/activate
export VLLM_DISABLE_COMPILE_CACHE=1
vllm serve mistralai/Voxtral-Mini-4B-Realtime-2602 \
  --compilation_config '{"cudagraph_mode":"PIECEWISE"}' \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.75
