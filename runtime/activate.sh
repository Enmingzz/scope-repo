#!/usr/bin/env bash
# Source from an already activated Python environment. No cluster-specific modules.
_scope_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
_scope_mode=${1:-train}
case "$_scope_mode" in train|eval) ;; *) printf 'Expected train or eval\n' >&2; return 2 ;; esac
if [[ ! -f "$_scope_root/.runtime/$_scope_mode/src/transformers/__init__.py" ]]; then
  printf 'Run python runtime/prepare.py first\n' >&2
  return 1
fi
export PATCHED_TRANSFORMERS_SRC="$_scope_root/.runtime/$_scope_mode/src"
export VISIONZIP_QWEN25VL_ROOT="$_scope_root/third_party/VisionZip/Qwen2_5_VL"
export PYTHONPATH="$_scope_root:$PATCHED_TRANSFORMERS_SRC:$_scope_root/third_party/VLMEvalKit:$_scope_root/third_party/VLMEvalKit/internvl${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-${SCRATCH:-$_scope_root/.cache}/scope/hf}"
export TMPDIR="${TMPDIR:-${SCRATCH:-$_scope_root/.cache}/scope/tmp}"
export LMUData="${LMUData:-${SCRATCH:-$_scope_root/.cache}/scope/benchmarks}"
export MMSTAR_QWEN_JUDGE=0 VLMEVAL_SAVE_RAW_PREDICTION=1 VLMEVAL_STRICT_ERRORS=1
mkdir -p "$HF_HOME" "$TMPDIR" "$LMUData"
unset _scope_root _scope_mode

