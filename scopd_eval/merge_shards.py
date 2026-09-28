"""Merge a complete disjoint inference shard set before benchmark post-processing."""
import argparse
import json
from pathlib import Path


def merge(root, shards, output):
    all_ids, rows, reference = set(), [], None
    contracts = ('model', 'model_revision', 'dataset', 'retention', 'max_new_tokens',
                 'min_pixels', 'max_pixels', 'greedy', 'protocol', 'shards')
    for shard in range(shards):
        directory = root / f'shard_{shard:03d}'
        manifest = json.loads((directory / 'input_manifest.json').read_text())
        if manifest['shard'] != shard or manifest['shards'] != shards or manifest['limit']:
            raise ValueError('Shard identity/count differs, or input was a smoke subset')
        if reference is None:
            reference = manifest
        if any(manifest[key] != reference[key] for key in contracts):
            raise ValueError('Shards use different models/configurations')
        ids = set(manifest['sample_ids'])
        if ids & all_ids:
            raise ValueError('Overlapping shards')
        part = [json.loads(line) for line in (directory / 'predictions.jsonl').read_text().splitlines()]
        if len(part) != len(ids) or {r['sample_id'] for r in part} != ids:
            raise ValueError('Incomplete or duplicated shard predictions')
        if any(row['dataset'] != manifest['dataset'] for row in part):
            raise ValueError('Wrong dataset in shard outputs')
        all_ids.update(ids)
        rows.extend(part)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    return len(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, required=True)
    p.add_argument('--shards', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.shards < 1:
        p.error('shards must be positive')
    print(f'Merged {merge(a.input_dir, a.shards, a.output)} outputs')


if __name__ == '__main__':
    main()
