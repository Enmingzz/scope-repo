"""Select the pinned evaluator sources before importing Transformers/VLMEvalKit."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
VLM = ROOT / 'third_party/VLMEvalKit'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def activate():
    src = ROOT / '.runtime/eval/src'
    manifest = json.loads((ROOT / 'runtime/manifest.json').read_text())
    for relative, expected in manifest['overlays']['eval'].items():
        if not (src / relative).is_file() or sha256(src / relative) != expected:
            raise RuntimeError('Run python runtime/prepare.py; evaluation runtime hash mismatch: ' + relative)
    if 'transformers' in sys.modules:
        loaded = Path(sys.modules['transformers'].__file__).resolve()
        if not loaded.is_relative_to(src):
            raise RuntimeError('A different Transformers was already imported')
    for path in reversed((ROOT, src, VLM, VLM / 'internvl')):
        sys.path.insert(0, str(path))
    os.environ['MMSTAR_QWEN_JUDGE'] = '0'
    os.environ['VLMEVAL_SAVE_RAW_PREDICTION'] = '1'
    os.environ['VLMEVAL_STRICT_ERRORS'] = '1'
