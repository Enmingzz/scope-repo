#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
VENV_DIR=${VENV_DIR:-"$ROOT/.venv-eval"}
"${PYTHON:-python3}" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Use Python 3.11"'
"${PYTHON:-python3}" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install 'pip>=24' 'setuptools>=68' wheel packaging ninja
python -m pip install torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-eval.txt
# Native VisionZip needs the patched FlashAttention attention statistic.
# A source build requires nvcc and matching PyTorch/CUDA headers. On 8.x GPUs,
# FA's sm80 target also supports Ada/L40S; omit the other GPU families to save time.
if ! python -c 'import torch, flash_attn; assert flash_attn.__version__ == "2.8.3"' 2>/dev/null; then
  command -v nvcc >/dev/null || { printf 'CUDA toolkit with nvcc is required to build flash-attn.\n' >&2; exit 1; }
  if [[ -z "${FLASH_ATTN_CUDA_ARCHS:-}" ]] && python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 8' 2>/dev/null; then
    export FLASH_ATTN_CUDA_ARCHS=80
  fi
  MAX_JOBS=${MAX_JOBS:-1} python -m pip install --force-reinstall --no-deps \
    flash-attn==2.8.3 --no-build-isolation
fi
python runtime/prepare.py
python -m pip install --no-deps -e .
python -m pip check
python -c 'import torch, flash_attn, PIL; print("torch", torch.__version__, "flash-attn", flash_attn.__version__, "Pillow", PIL.__version__)'
printf 'Ready. Activate with: source %q/bin/activate\n' "$VENV_DIR"
printf 'Then: source runtime/activate.sh eval\n'
