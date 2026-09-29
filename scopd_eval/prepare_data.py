"""Prepare checksum-pinned public benchmark data without private caches."""
import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import shutil


_MIRROR = {
    'repo_id': 'mm-eval/VLMEvalKit',
    'revision': '772639df87331ba19da76ea9cc5fa77836a15a18',
}
# The mirror files were checked against the original bundled VLMEvalKit MD5s.
# SHA256 also pins the downloaded content, independently of the mirror revision.
SOURCES = {
    key: dict(_MIRROR, filename=filename, md5=md5, sha256=sha256)
    for key, filename, md5, sha256 in [
        ('mmstar', 'MMStar.tsv', 'e1ecd2140806c1b1bbf54b43372efb9e',
         '5b581d485fcfbb0f4c8fc0440fc4ecea17fb4ba1750c6446fd20833070552e95'),
        ('mme', 'MME.tsv', 'b36b43c3f09801f5d368627fb92187c3',
         '3b6c914445710ce582ffb51ef8c6c410783ea52e8c517b0a33dbdc26c64ed680'),
        ('mathvista', 'MathVista_MINI.tsv', 'f199b98e178e5a2a20e7048f5dcb0464',
         '876bc02fce10e4d2a13c1cf4fc02226daf8c039599f85725c05c5ef4996dc455'),
        ('logicvista', 'LogicVista.tsv', '41c5d33adf33765c399e0e6ae588c061',
         'f5e797e904e875843e823556e54e91d95d427426ad160321a06f034539ad9795'),
        ('mathverse', 'MathVerse_MINI_Vision_Only.tsv', '68a11d4680014ac881fa37adeadea3a4',
         'e4280ca3f27563c3bc3559e6d27bc3af0586d690356318354c160346313e52ed'),
        ('mathvision', 'MathVision_MINI.tsv', '060fe4fa5d868987ce179307bd5f8a33',
         'b8ae07d465d1b8fa15190de5248b9a76d72dd4f6a4f79ff9a3e5caf60a87da8f'),
        ('realworldqa', 'RealWorldQA.tsv', '4de008f55dc4fd008ca9e15321dc44b7',
         '77ed7d5adff8522a3b6c28d3a3df5830473901c9c2c0768ad5b0b26151813816'),
        ('visonly', 'VisOnlyQA-VLMEvalKit.tsv', 'cf460a31d2acb8d3a7cecd0e69298bfa',
         '65d28db36d94af10a31537aade87639bc702e28165560a034a21c6b9f1900a21'),
        ('mmvp', 'MMVP.tsv', '8cb732b141a0cba5b42159df2839e557',
         '225204bbb168fc963c59dca4fdd916d63ebe22471ab803a84aa96fdd7e57162c'),
        ('blink', 'BLINK.tsv', '3b6649b6a662184ea046908e5506260e',
         '8f59140f615d0cf0e471e13306969e8ef3d10c2a744cd8cd326443ff4733d644'),
        ('pope', 'POPE.tsv', 'c12f5acb142f2ef1f85a26ba2fbe41d5',
         '837c674037e0f0dfefe837918159820f8cb7b0abb171162c044b3affdc2f4d6d'),
    ]
}
SOURCES['hallusion'] = {
    'repo_id': 'OMG-Research/VLM',
    'revision': '64612316e1f9009e3a8e58253b0ac186c7e330c1',
    'filename': 'eval/HallusionBench/HallusionBench.tsv',
    'md5': '0c23ac0dc9ef46832d7a24504f2a0c7c',
    'sha256': '8f92be804abf146f24bfa1a98d8fcb1cba335f6b0dc69fc27ea698fa446df3b8',
}
SOURCES.update({
    'hr4k': {
        'repo_id': 'DreamMr/HR-Bench',
        'revision': '83b9013d6293b85dc507e87199ca52517536939c',
        'filename': 'hr_bench_4k.tsv', 'local_filename': 'HRBench4K.tsv',
        'md5': 'f6b041b03d49543494b8a56d2e35be65',
        'sha256': 'a2147a5525fd6c7246e1891b738e1987d16d6ce2df2ee0c4f96946a959e76e18',
    },
    'mmereal_lite': {
        'repo_id': 'yifanzhang114/MME-RealWorld-Base64',
        'revision': '93310bbe383ad164ef4afadd329090679c52d7c4',
        'filename': 'mme_realworld_lite.tsv', 'local_filename': 'MME-RealWorld-Lite.tsv',
        'md5': '4c17057d7d3b6c4a0d4397c3dae0881c',
        'sha256': 'd66abee0758232d75d1586c27a39a19059d7e9bb4c855d3ecceab283a7488065',
    },
})


def file_sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def source_metadata(key):
    if key == 'mmmupro':
        from .prepare_mmmupro import SOURCE
        return SOURCE
    return SOURCES.get(key)


def prepare_dataset(key, root=None):
    if key == 'mmmupro':
        from .prepare_mmmupro import prepare
        return prepare(root)
    from huggingface_hub import hf_hub_download
    source = SOURCES[key]
    root = Path(root or os.environ.get('LMUData', Path.home() / 'LMUData'))
    root.mkdir(parents=True, exist_ok=True)
    target = root / source.get('local_filename', Path(source['filename']).name)
    with (root / f'.{key}-download.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists():
            if file_sha256(target) != source['sha256']:
                raise ValueError(f'Unexpected {key} checksum: {target}. Use a clean LMUData directory.')
            return target
        cached = hf_hub_download(repo_id=source['repo_id'], revision=source['revision'],
                                 filename=source['filename'], repo_type='dataset')
        if file_sha256(cached) != source['sha256']:
            raise ValueError(f'Downloaded {key} checksum mismatch')
        temporary = target.with_suffix('.tsv.tmp')
        shutil.copyfile(cached, temporary)
        temporary.replace(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=sorted([*SOURCES, 'mmmupro']), default='mmstar')
    parser.add_argument('--data-root', type=Path)
    args = parser.parse_args()
    print(prepare_dataset(args.dataset, args.data_root))


if __name__ == '__main__':
    main()
