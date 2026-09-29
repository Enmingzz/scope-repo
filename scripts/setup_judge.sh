#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
VENV_DIR=${JUDGE_VENV_DIR:-"$ROOT/.venv-judge"}
"${PYTHON:-python3}" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Use Python 3.11"'
"${PYTHON:-python3}" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install 'pip>=24' 'setuptools>=68' wheel
python -m pip install -r requirements-judge.txt
python -m pip check
printf 'Judge environment ready: %q\n' "$VENV_DIR"
