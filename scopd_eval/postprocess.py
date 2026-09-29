"""Shared, resumable, mismatch-only Qwen27B post-processing for saved outputs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

from .protocol import DEFAULT_MODEL, PROTOCOL, exact_match, image_content, make_request, validate
from .scoring import summarize


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def query(base_url, payload, timeout=180):
    headers = {'Content-Type': 'application/json'}
    if os.environ.get('JUDGE_API_KEY'):
        headers['Authorization'] = 'Bearer ' + os.environ['JUDGE_API_KEY']
    request = urllib.request.Request(base_url.rstrip('/') + '/chat/completions',
                                     data=json.dumps(payload).encode(), headers=headers)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.load(response)
            choice = body['choices'][0]
            text = choice['message']['content'].strip()
            if choice.get('finish_reason') == 'length' or text not in ('CORRECT', 'INCORRECT'):
                raise ValueError(f'Invalid judge verdict: {text!r}')
            return text == 'CORRECT', body
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def run(input_path, output_dir, *, image_root, base_url, model, revision, concurrency=4, dry_run=False):
    if revision == 'unspecified' or not revision.strip():
        raise ValueError('Pin the judge checkpoint revision (or local content hash)')
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raw = input_path.read_bytes()
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
        if not records or len({(r['dataset'], str(r['sample_id'])) for r in records}) != len(records):
            raise ValueError('Empty input or duplicate sample IDs')
        contract = {'protocol': PROTOCOL, 'input_sha256': hashlib.sha256(raw).hexdigest(),
                    'model': model, 'judge_revision': revision, 'rows': len(records)}
        manifest = output_dir / 'manifest.json'
        if manifest.exists() and json.loads(manifest.read_text()) != contract:
            raise ValueError('Existing outputs have a different input/model/protocol; use a new directory')
        atomic_json(manifest, contract)
        cache_path = output_dir / 'judge_cache.jsonl'
        cache = {}
        if cache_path.exists():
            for line in cache_path.read_text().splitlines():
                cached = json.loads(line)
                if cached['fingerprint'] in cache:
                    raise ValueError('Duplicate judge-cache entry')
                cache[cached['fingerprint']] = cached
        results, pending = [], []
        for row in records:
            validate(row)
            matched = exact_match(row)
            result = {**row, 'exact_match': matched, 'correct': matched,
                      'source': 'deterministic_match' if matched else 'qwen27_pending', 'protocol': PROTOCOL}
            results.append(result)
            if matched:
                for image in row['images']:
                    image_content(str(image), Path(image_root))
                continue
            payload, key, identity = make_request(row, image_root, model, revision)
            result['judge_fingerprint'] = key
            if key in cache:
                result.update(correct=cache[key]['correct'], source='qwen27_visual_gt')
            else:
                pending.append((result, payload, key, identity))
        preflight = {**contract, 'exact_matches': sum(r['exact_match'] for r in results),
                     'fallback_total': sum(not r['exact_match'] for r in results),
                     'pending_requests': len(pending), 'status': 'prepared'}
        atomic_json(output_dir / 'preflight.json', preflight)
        if dry_run:
            return preflight
        errors = []
        with ThreadPoolExecutor(max_workers=concurrency) as pool, cache_path.open('a') as handle:
            futures = {pool.submit(query, base_url, payload): (result, key, identity)
                       for result, payload, key, identity in pending}
            for future in as_completed(futures):
                result, key, identity = futures[future]
                try:
                    correct, body = future.result()
                    cached = {'fingerprint': key, 'correct': correct, 'identity': identity, 'response': body}
                    handle.write(json.dumps(cached, ensure_ascii=False) + '\n')
                    handle.flush()
                    os.fsync(handle.fileno())
                    result.update(correct=correct, source='qwen27_visual_gt')
                except Exception as error:
                    errors.append({'sample_id': result['sample_id'], 'error': str(error)})
        if errors:
            atomic_json(output_dir / 'summary.json', {**preflight, 'status': 'incomplete', 'errors': errors})
            raise RuntimeError(f'{len(errors)} judge requests failed; successful entries cached, no score published')
        temp = output_dir / 'judged.jsonl.tmp'
        with temp.open('w') as handle:
            for result in results:
                handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
        temp.replace(output_dir / 'judged.jsonl')
        report = {**preflight, 'status': 'completed', 'pending_requests': 0,
                  'scores': summarize(results), 'official_protocol_equivalent': False}
        atomic_json(output_dir / 'summary.json', report)
        return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--image-root', type=Path, default=Path('.'))
    p.add_argument('--base-url', default='http://127.0.0.1:8000/v1')
    p.add_argument('--model', default=DEFAULT_MODEL)
    p.add_argument('--judge-revision', required=True)
    p.add_argument('--concurrency', type=int, default=4)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if a.concurrency < 1:
        p.error('concurrency must be positive')
    print(json.dumps(run(a.input, a.output_dir, image_root=a.image_root, base_url=a.base_url,
                         model=a.model, revision=a.judge_revision, concurrency=a.concurrency,
                         dry_run=a.dry_run), indent=2))


if __name__ == '__main__':
    main()
