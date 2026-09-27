"""Check the ordered training manifest without distributing images or model targets."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--image-root', type=Path, required=True)
    p.add_argument('--order', type=Path, default=Path(__file__).resolve().parents[1] / 'manifests/train20k_order.csv')
    p.add_argument('--limit', type=int, default=0)
    a = p.parse_args()
    with a.order.open() as handle:
        expected = list(csv.DictReader(handle))
    if a.limit:
        expected = expected[:a.limit]
    with a.dataset.open() as handle:
        for position, target in enumerate(expected):
            line = next(handle, None)
            if line is None:
                raise ValueError('Training data ends before the expected manifest')
            row = json.loads(line)
            for key in ('sample_id', 'source_id', 'image'):
                if str(row[key]) != target[key]:
                    raise ValueError(f'Order/identity differs at row {position}: {key}')
            actual = hashlib.sha256(str(row['question']).encode()).hexdigest()
            if actual != target['question_sha256']:
                raise ValueError(f'Question preprocessing differs at row {position}')
            if not (a.image_root / row['image']).is_file():
                raise FileNotFoundError(f'Missing image for row {position}')
    print(json.dumps({'verified_prefix_rows': len(expected), 'images_exist': True,
                      'dataset_sha256': hashlib.sha256(a.dataset.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
