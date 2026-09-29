"""Materialize official MMMU-Pro four-option data for the bundled evaluator."""
import ast
import base64
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import string

SOURCE = {
    'repo_id': 'MMMU/MMMU_Pro',
    'revision': '563f3e84bb3b90893083a1f039cfa13077f2302b',
    'config': 'standard (4 options)',
    'split': 'test',
    'conversion': 'official-images-options-v1',
    'files': {
        'standard (4 options)/test-00000-of-00002.parquet':
            'a2c31b693f7387af2ebd1e1dff6451be62ed6fdb705316cde8f7244b1494f1a9',
        'standard (4 options)/test-00001-of-00002.parquet':
            '4231c7c7ac3256b507d4eede525a1a224b3adc19975f9da758ec8fd68ba6bba4',
    },
    'rows': 1730,
}


def convert_row(row):
    options = ast.literal_eval(row['options']) if isinstance(row['options'], str) else row['options']
    if not isinstance(options, list) or not 2 <= len(options) <= 26:
        raise ValueError(f'Invalid options: {row["id"]}')
    labels = string.ascii_uppercase[:len(options)]
    if row['answer'] not in tuple(labels):
        raise ValueError(f'Invalid reference option: {row["id"]}')
    slots = [i for i in range(1, 8) if row.get(f'image_{i}') is not None]
    if not slots or slots != list(range(1, len(slots) + 1)):
        raise ValueError(f'Missing/nonconsecutive image slots: {row["id"]}')
    stem = hashlib.sha256(str(row['id']).encode()).hexdigest()
    images, paths = [], []
    for slot in slots:
        data = row[f'image_{slot}']['bytes']
        if not data:
            raise ValueError(f'Image bytes absent: {row["id"]}, image {slot}')
        images.append(base64.b64encode(data).decode('ascii'))
        paths.append(f'{stem}_{slot}.png')
    return {'index': row['id'], 'question': row['question'], 'answer': row['answer'],
            'category': row['subject'], **dict(zip(labels, options)),
            'image': json.dumps(images), 'image_path': json.dumps(paths)}


def prepare(root=None):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    from .prepare_data import file_sha256
    root = Path(root or os.environ.get('LMUData', Path.home() / 'LMUData'))
    root.mkdir(parents=True, exist_ok=True)
    target = root / 'MMMU_Pro_4c.tsv'
    manifest = root / 'MMMU_Pro_4c.source.json'
    with (root / '.mmmupro-download.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists():
            saved = json.loads(manifest.read_text()) if manifest.exists() else {}
            if saved.get('source') != SOURCE or saved.get('sha256') != file_sha256(target):
                raise ValueError('MMMU-Pro cache has unknown/different provenance; use a clean LMUData directory')
            return target
        paths = []
        for filename, sha256 in SOURCE['files'].items():
            path = hf_hub_download(SOURCE['repo_id'], filename, repo_type='dataset', revision=SOURCE['revision'])
            if file_sha256(path) != sha256:
                raise ValueError(f'MMMU-Pro source checksum mismatch: {filename}')
            paths.append(path)
        temporary = target.with_suffix('.tsv.tmp')
        ids = set()
        with temporary.open('w', newline='') as handle:
            fields = ['index', 'question', 'answer', 'category', *string.ascii_uppercase, 'image', 'image_path']
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t')
            writer.writeheader()
            for path in paths:
                for batch in pq.ParquetFile(path).iter_batches(batch_size=16):
                    for row in batch.to_pylist():
                        if row['id'] in ids:
                            raise ValueError('Duplicate MMMU-Pro source ID')
                        ids.add(row['id'])
                        writer.writerow(convert_row(row))
        if len(ids) != SOURCE['rows']:
            raise ValueError(f'MMMU-Pro row count differs: {len(ids)}')
        metadata = {'source': SOURCE, 'sha256': file_sha256(temporary)}
        manifest_tmp = manifest.with_suffix('.json.tmp')
        manifest_tmp.write_text(json.dumps(metadata, indent=2) + '\n')
        manifest_tmp.replace(manifest)
        temporary.replace(target)
    return target
