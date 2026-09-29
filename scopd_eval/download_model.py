"""Download a released Qwen2.5 adapter and its pinned base; no credentials needed."""
import argparse
import json
from pathlib import Path

from .prepare_data import file_sha256

RELEASE = 'enmingzhangzz/SCOPD'


def download(variant, output, revision='main'):
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / 'download_manifest.json'
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous['variant'] != variant or revision not in ('main', previous['release_revision']):
            raise ValueError('Different download already exists; choose another output directory')
        revision = previous['release_revision']
    else:
        revision = HfApi().model_info(RELEASE, revision=revision).sha
    provenance_path = hf_hub_download(RELEASE, f'models/{variant}/provenance.json', revision=revision)
    provenance = json.loads(Path(provenance_path).read_text())
    if provenance['base_model'] != 'Qwen/Qwen2.5-VL-7B-Instruct':
        raise ValueError('This runner supports Qwen2.5-VL-7B; use the release Qwen3 bundle for Qwen3')
    snapshot_download(RELEASE, revision=revision, local_dir=output / 'release',
                      allow_patterns=[f'models/{variant}/*'], max_workers=2)
    adapter = output / 'release/models' / variant
    if file_sha256(adapter / 'adapter_model.safetensors') != provenance['adapter_sha256']:
        raise ValueError('Adapter hash differs from its public provenance')
    base = snapshot_download(provenance['base_model'], revision=provenance['base_revision'],
                             allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', '*.model'],
                             max_workers=2)
    result = {'variant': variant, 'release_revision': revision, 'base_model': provenance['base_model'],
              'base_revision': provenance['base_revision'], 'base_path': str(base),
              'adapter_path': str(adapter), 'adapter_sha256': provenance['adapter_sha256']}
    temporary = manifest.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(manifest)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', default='scopd-plus',
                   choices=['scopd', 'scopd-plus', 'scopd-plus-top20', 'scopd-plus-top40',
                            'scopd-plus-top60', 'scopd-plus-reverse-kl', 'scopd-plus-jsd',
                            'scopd-offpolicy', 'sft', 'epic', 'grpo'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--revision', default='main', help='Resolved to an immutable commit and saved on first use')
    a = p.parse_args()
    print(json.dumps(download(a.variant, a.output, a.revision), indent=2))


if __name__ == '__main__':
    main()
