"""Single-GPU inference with disjoint shards, exact resume contracts and raw outputs."""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import time

from .datasets import DATASETS, load, make_record
from .postprocess import atomic_json
from .protocol import digest
from .runtime import activate


def visionzip_settings(retention):
    if retention == 1:
        return {'enable_visionzip': False, 'visionzip_ratio': 0.0}
    if not .05 < retention < 1:
        raise ValueError('Native VisionZip needs retention > 5%; no silent dominant-only workaround')
    # The frozen implementation allocates 5% contextual tokens in addition to
    # (1 - backend_ratio) dominant tokens. This is NOT a retention argument.
    return {'enable_visionzip': True, 'visionzip_ratio': round(1.05 - retention, 8)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, help='Local pinned MERGED checkpoint directory')
    p.add_argument('--model-revision', required=True, help='Commit or content hash of the merged checkpoint')
    p.add_argument('--dataset', choices=DATASETS, required=True)
    p.add_argument('--retention', type=float, default=.1)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--max-new-tokens', type=int, default=2048)
    p.add_argument('--min-image-tokens', type=int, default=1280)
    p.add_argument('--max-image-tokens', type=int, default=4096)
    a = p.parse_args()
    if not 0 <= a.shard < a.shards or a.limit < 0 or a.max_new_tokens < 1:
        p.error('Invalid shard, limit or length')
    if not 1 <= a.min_image_tokens <= a.max_image_tokens:
        p.error('Invalid image-token bounds')
    model_path = Path(a.model).resolve()
    if not (model_path / 'config.json').exists() or (model_path / 'adapter_config.json').exists():
        p.error('Supply a local pinned base or merged checkpoint, not an adapter')
    budget = visionzip_settings(a.retention)
    activate()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('An allocated GPU is required for inference')
    from .runtime_fixes import install_wrapper
    from .visionzip_memory import install
    install_wrapper(video_timing=False)
    install()
    from vlmeval.vlm.qwen2_vl.model import Qwen2VLChat
    dataset = load(a.dataset)
    selected = dataset.data.iloc[a.shard::a.shards]
    if a.limit:
        selected = selected.head(a.limit)
    ids = selected['index'].astype(str).tolist()
    contract = {'protocol': 'reasoning-image-inference-v1', 'model': str(model_path),
                'model_revision': a.model_revision, 'dataset': DATASETS[a.dataset],
                'retention': a.retention, **budget, 'sample_ids': ids,
                'shard': a.shard, 'shards': a.shards, 'limit': a.limit,
                'max_new_tokens': a.max_new_tokens, 'min_pixels': a.min_image_tokens * 784,
                'max_pixels': a.max_image_tokens * 784, 'greedy': True,
                'metadata_sha256': digest(selected.drop(columns=['image'], errors='ignore').astype(str).to_dict('records'))}
    work = a.output_dir / f'shard_{a.shard:03d}'
    work.mkdir(parents=True, exist_ok=True)
    with (work / '.infer.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = work / 'input_manifest.json'
        if manifest.exists() and json.loads(manifest.read_text()) != contract:
            raise ValueError('Inference input/configuration changed; use a new output directory')
        atomic_json(manifest, contract)
        path = work / 'predictions.jsonl'
        done = {}
        if path.exists():
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if row['sample_id'] in done:
                    raise ValueError('Duplicate saved prediction')
                done[row['sample_id']] = row
        if set(done) - set(ids):
            raise ValueError('Unexpected saved prediction IDs')
        if set(done) == set(ids):
            print('Inference already complete')
            return
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(42)
        model = Qwen2VLChat(model_path=str(model_path), temperature=0., enable_thinking=True,
                            max_new_tokens=a.max_new_tokens, min_pixels=a.min_image_tokens * 784,
                            max_pixels=a.max_image_tokens * 784, **budget)
        model.set_dump_image(dataset.dump_image)
        with path.open('a') as handle:
            for _, row in selected.iterrows():
                if str(row['index']) in done:
                    continue
                prompt = (model.build_prompt(row, dataset=dataset.dataset_name)
                          if model.use_custom_prompt(dataset.dataset_name) else dataset.build_prompt(row))
                images = [m['value'] for m in prompt if m['type'] == 'image']
                if not images:
                    raise ValueError('Image benchmark prompt has no images')
                started = time.monotonic()
                response = model.generate(copy.deepcopy(prompt), dataset=dataset.dataset_name)
                record = make_record(dataset.dataset_name, row, response, images)
                record.update(elapsed_seconds=time.monotonic() - started,
                              input_text=[m['value'] for m in prompt if m['type'] == 'text'])
                if 'Failed to obtain answer' in record['raw_prediction']:
                    raise RuntimeError(record['raw_prediction'])
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                handle.flush()
                os.fsync(handle.fileno())
                print(f'{a.dataset} {row["index"]} complete', flush=True)
        atomic_json(work / 'inference_summary.json', {'status': 'completed', **contract,
                    'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30})


if __name__ == '__main__':
    main()
