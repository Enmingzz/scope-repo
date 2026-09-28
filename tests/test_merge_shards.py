import json

import pytest

from scopd_eval.merge_shards import merge


def write_shard(root, shard, sample_id, *, model='model', limit=0, rows=None):
    directory = root / f'shard_{shard:03d}'
    directory.mkdir(parents=True)
    manifest = {
        'model': model, 'model_revision': 'pinned', 'dataset': 'MMStar',
        'retention': .1, 'max_new_tokens': 2048, 'min_pixels': 1003520,
        'max_pixels': 3211264, 'greedy': True,
        'protocol': 'reasoning-image-inference-v1',
        'shards': 2, 'shard': shard, 'limit': limit,
        'sample_ids': [sample_id],
    }
    (directory / 'input_manifest.json').write_text(json.dumps(manifest))
    if rows is None:
        rows = [{'sample_id': sample_id, 'dataset': 'MMStar', 'raw_prediction': 'A'}]
    (directory / 'predictions.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in rows))


def test_merge_complete_disjoint_shards(tmp_path):
    write_shard(tmp_path, 0, 'a')
    write_shard(tmp_path, 1, 'b')
    output = tmp_path / 'all.jsonl'
    assert merge(tmp_path, 2, output) == 2
    assert [json.loads(line)['sample_id'] for line in output.read_text().splitlines()] == ['a', 'b']
    with pytest.raises(FileExistsError):
        merge(tmp_path, 2, output)


@pytest.mark.parametrize('failure', ['missing', 'incomplete', 'duplicate', 'overlap',
                                     'model', 'smoke', 'dataset'])
def test_reject_bad_shards_without_publishing_output(tmp_path, failure):
    write_shard(tmp_path, 0, 'a')
    kwargs = {}
    if failure == 'model':
        kwargs['model'] = 'another-model'
    elif failure == 'smoke':
        kwargs['limit'] = 1
    elif failure == 'incomplete':
        kwargs['rows'] = []
    elif failure == 'duplicate':
        kwargs['rows'] = [{'sample_id': 'b', 'dataset': 'MMStar'}] * 2
    elif failure == 'dataset':
        kwargs['rows'] = [{'sample_id': 'b', 'dataset': 'MME'}]
    if failure != 'missing':
        write_shard(tmp_path, 1, 'a' if failure == 'overlap' else 'b', **kwargs)
    output = tmp_path / 'all.jsonl'
    with pytest.raises((ValueError, FileNotFoundError)):
        merge(tmp_path, 2, output)
    assert not output.exists()
