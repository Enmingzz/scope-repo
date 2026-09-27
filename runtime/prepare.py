"""Reconstruct the exact patched Transformers source without altering site-packages."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('train', 'eval', 'both'), default='both')
    p.add_argument('--wheel', type=Path)
    args = p.parse_args()
    manifest = json.loads((ROOT / 'runtime/manifest.json').read_text())
    cache = ROOT / '.runtime/downloads'
    cache.mkdir(parents=True, exist_ok=True)
    wheel = args.wheel or cache / 'transformers-4.57.0-py3-none-any.whl'
    if not wheel.exists():
        subprocess.run([sys.executable, '-m', 'pip', 'download', '--no-deps', '--dest', str(cache),
                        'transformers==4.57.0'], check=True)
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != manifest['wheel_sha256']:
        raise RuntimeError('Transformers wheel hash differs from the audited source baseline')
    modes = ('train', 'eval') if args.mode == 'both' else (args.mode,)
    with ZipFile(wheel) as archive:
        for mode in modes:
            target = ROOT / '.runtime' / mode / 'src'
            target.mkdir(parents=True, exist_ok=True)
            for member in archive.infolist():
                if member.filename.startswith('transformers/') and not member.is_dir():
                    if '..' in Path(member.filename).parts:
                        raise ValueError('Unsafe wheel member')
                    dest = target / member.filename
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(archive.read(member))
            for relative, expected in manifest['overlays'][mode].items():
                src = ROOT / 'runtime/overlays' / mode / relative
                if hashlib.sha256(src.read_bytes()).hexdigest() != expected:
                    raise RuntimeError(f'Changed runtime overlay: {relative}')
                dest = target / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dest)
            print(f'{mode}: prepared and verified {len(manifest["overlays"][mode])} patched source files')


if __name__ == '__main__':
    main()

