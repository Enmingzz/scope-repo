#!/usr/bin/env bash
set -euo pipefail
# Run in the separate judge environment, without the training PYTHONPATH.
: "${JUDGE_MODEL_PATH:?Set the local Qwen3.6-27B-FP8 snapshot directory}"
: "${JUDGE_MODEL_REVISION:?Record the pinned snapshot revision}"
unset PYTHONPATH PYTHONHOME
export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
exec vllm serve "$JUDGE_MODEL_PATH" \
  --served-model-name Qwen/Qwen3.6-27B-FP8 \
  --host 127.0.0.1 --port "${JUDGE_PORT:-8000}" \
  --tensor-parallel-size "${JUDGE_GPUS:-1}" \
  --max-model-len "${JUDGE_CONTEXT:-8192}" \
  --gpu-memory-utilization "${JUDGE_MEMORY_FRACTION:-0.90}" \
  --limit-mm-per-prompt '{"image": 24, "video": 0}' \
  --mm-processor-kwargs '{"min_pixels": 3136, "max_pixels": 1003520}' \
  --max-num-seqs "${JUDGE_MAX_SEQS:-2}"
