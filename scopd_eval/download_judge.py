"""Download the public visual judge and record its immutable revision."""
import argparse
import json
from pathlib import Path

from .protocol import DEFAULT_MODEL


def main():
    from huggingface_hub import HfApi, snapshot_download
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True, help='Small JSON manifest, not the model directory')
    p.add_argument('--revision', default='main')
    a = p.parse_args()
    if a.output.exists():
        previous = json.loads(a.output.read_text())
        if previous['model'] != DEFAULT_MODEL or a.revision not in ('main', previous['revision']):
            p.error('Judge manifest already pins a different model/revision')
        revision = previous['revision']
    else:
        revision = HfApi().model_info(DEFAULT_MODEL, revision=a.revision).sha
    path = snapshot_download(DEFAULT_MODEL, revision=revision,
                             allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', '*.model'],
                             max_workers=2)
    result = {'model': DEFAULT_MODEL, 'revision': revision, 'path': path}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = a.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(a.output)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
