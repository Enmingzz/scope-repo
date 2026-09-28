#!/usr/bin/env bash
# Source from an already activated Python environment. No cluster-specific modules.
_scopd_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
_scopd_mode=${1:-train}
case "$_scopd_mode" in train|eval) ;; *) printf 'Expected train or eval\n' >&2; return 2 ;; esac
if [[ ! -f "$_scopd_root/.runtime/$_scopd_mode/src/transformers/__init__.py" ]]; then
  printf 'Run python runtime/prepare.py first\n' >&2
  return 1
fi
export PATCHED_TRANSFORMERS_SRC="$_scopd_root/.runtime/$_scopd_mode/src"
export VISIONZIP_QWEN25VL_ROOT="$_scopd_root/third_party/VisionZip/Qwen2_5_VL"
export PYTHONPATH="$_scopd_root:$PATCHED_TRANSFORMERS_SRC:$_scopd_root/third_party/VLMEvalKit:$_scopd_root/third_party/VLMEvalKit/internvl${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-${SCRATCH:-$_scopd_root/.cache}/scopd/hf}"
export TMPDIR="${TMPDIR:-${SCRATCH:-$_scopd_root/.cache}/scopd/tmp}"
export LMUData="${LMUData:-${SCRATCH:-$_scopd_root/.cache}/scopd/benchmarks}"
export MMSTAR_QWEN_JUDGE=0 VLMEVAL_SAVE_RAW_PREDICTION=1 VLMEVAL_STRICT_ERRORS=1
mkdir -p "$HF_HOME" "$TMPDIR" "$LMUData"
unset _scopd_root _scopd_mode

