import base64
import json
from pathlib import Path

import pytest

from scopd_eval import postprocess
from scopd_eval.datasets import make_record
from scopd_eval.infer import visionzip_settings
from scopd_eval.protocol import DEFAULT_MODEL, candidate_answer, exact_match, make_request
from scopd_eval.scoring import summarize

IMAGE = 'data:image/png;base64,' + base64.b64encode(b'test-image-bytes').decode()


def record(answer='B', candidate='<think>A is tempting</think><answer>B</answer>', **extra):
    return {'dataset': 'MMStar', 'sample_id': '1', 'question': 'Which object?',
            'options': {'A': 'red cube', 'B': 'blue sphere'}, 'reference': answer,
            'images': [IMAGE], 'raw_prediction': candidate, **extra}


@pytest.mark.parametrize('candidate,match', [
    ('<think>A</think><answer>B</answer>', True),
    ('<answer>A</answer>', False), ('blue sphere', True),
    ('B or A', False), ('I think A, not B', False),
    ('<think>the answer is B', False), ('<answer>B', False),
    ('<answer>A</answer><answer>B</answer>', False),
    ('<answer>B</answer> Actually A', False),
])
def test_match_is_strict(candidate, match):
    assert exact_match(record(candidate=candidate)) is match


def test_numeric_equivalence_without_substring_matching():
    assert exact_match(record(answer='0.5', candidate='1/2', options={}))
    assert not exact_match(record(answer='0.5', candidate='0.5 or 0.6', options={}))
    assert not exact_match(record(answer='5', candidate='15', options={}))


def test_judge_sees_question_gt_candidate_and_all_images():
    row = record(candidate='the sphere', images=[IMAGE, IMAGE])
    payload, key, identity = make_request(row, '.', revision='revision1')
    assert payload['model'] == DEFAULT_MODEL
    content = payload['messages'][1]['content']
    context = json.loads(content[0]['text'])
    assert context['reference_answer'] == 'B' and context['candidate_final_answer'] == 'the sphere'
    assert context['question'] == row['question'] and context['options'] == row['options']
    assert [x['type'] for x in content] == ['text', 'image_url', 'image_url']
    assert 'A is tempting' not in content[0]['text']
    assert len(key) == 64 and len(identity['image_sha256']) == 2
    assert payload['chat_template_kwargs']['enable_thinking'] is False


def test_cache_changes_with_gt_image_revision_and_candidate():
    row = record()
    key = make_request(row, '.', revision='a')[1]
    for changed in [dict(row, reference='A'), dict(row, raw_prediction='A'),
                    dict(row, images=['data:image/png;base64,' + base64.b64encode(b'other').decode()])]:
        assert make_request(changed, '.', revision='a')[1] != key
    assert make_request(row, '.', revision='b')[1] != key


def test_missing_image_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError):
        make_request(record(images=['missing.png']), tmp_path, revision='a')
    with pytest.raises(ValueError):
        make_request(record(images=[]), tmp_path, revision='a')


def test_only_mismatches_call_judge_and_resume(tmp_path, monkeypatch):
    rows = [record(), record(candidate='the sphere', sample_id='2'),
            record(candidate='A', sample_id='3')]
    path = tmp_path / 'predictions.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    calls = []
    def query(url, payload):
        text = json.loads(payload['messages'][1]['content'][0]['text'])['candidate_final_answer']
        calls.append(text)
        return text == 'the sphere', {'mock': True}
    monkeypatch.setattr(postprocess, 'query', query)
    kwargs = dict(image_root=tmp_path, base_url='http://localhost', model=DEFAULT_MODEL, revision='pin')
    result = postprocess.run(path, tmp_path / 'out', **kwargs)
    assert sorted(calls) == ['A', 'the sphere']
    assert result['fallback_total'] == 2
    assert result['scores']['MMStar']['accuracy_percent'] == pytest.approx(200 / 3)
    postprocess.run(path, tmp_path / 'out', **kwargs)
    assert len(calls) == 2
    with pytest.raises(ValueError):
        postprocess.run(path, tmp_path / 'out', **dict(kwargs, revision='other'))


def test_failure_not_converted_to_incorrect_score(tmp_path, monkeypatch):
    path = tmp_path / 'predictions.jsonl'
    path.write_text(json.dumps(record(candidate='unknown')) + '\n')
    def query(*args):
        raise TimeoutError('test failure')
    monkeypatch.setattr(postprocess, 'query', query)
    with pytest.raises(RuntimeError):
        postprocess.run(path, tmp_path / 'out', image_root=tmp_path, base_url='http://localhost',
                        model=DEFAULT_MODEL, revision='pin')
    summary = json.loads((tmp_path / 'out/summary.json').read_text())
    assert summary['status'] == 'incomplete' and 'scores' not in summary


def test_mme_pair_and_mmvp_pair_scoring():
    rows = [dict(record(sample_id=str(i), dataset='MME'), correct=i != 2, exact_match=i != 2,
                 metadata={'category': 'count', 'image_path': str(i // 2)}) for i in range(4)]
    assert summarize(rows)['MME']['mme_score'] == 125
    pairs = [dict(r, dataset='MMVP', sample_id=str(i + 1)) for i, r in enumerate(rows)]
    assert summarize(pairs)['MMVP']['pair_accuracy_percent'] == 50


@pytest.mark.parametrize('retention,backend', [(.1, .95), (.2, .85), (.3, .75), (1., 0.)])
def test_retention_is_not_backend_ratio(retention, backend):
    assert visionzip_settings(retention)['visionzip_ratio'] == backend


def test_hallusion_boolean_reference():
    row = make_record('HallusionBench', {'index': 'x', 'question': 'Is it blue?', 'answer': 1}, 'Yes', ['x.png'])
    assert row['reference'] == 'Yes' and exact_match(row)


def test_cvbench_keeps_source_macro_then_2d_3d_macro():
    rows = []
    for i, (source, split, correct) in enumerate([('a', '2D', True), ('b', '2D', False),
                                               ('b', '2D', False), ('b', '2D', False),
                                               ('x', '3D', True)]):
        rows.append(dict(record(dataset='CVBench', sample_id=str(i)), correct=correct, exact_match=correct,
                         metadata={'source': source, 'split': split}))
    report = summarize(rows)['CVBench']
    assert report['2d_source_macro_percent'] == 50
    assert report['3d_accuracy_percent'] == 100
    assert report['cvbench_macro_percent'] == 75


def test_mmereal_duplicate_option_labels_are_not_repaired():
    source = {'index': 1, 'question': 'Which?', 'answer': 'E',
              'A': 'first', 'B': 'second', 'C': 'third', 'D': 'fourth', 'E': 'fifth',
              'multi-choice options': '(A) first\n(B) second\n(C) third\n(D) fourth\n(D) fifth'}
    row = make_record('MME-RealWorld-Lite', source, 'fifth', [IMAGE])
    assert row['options']['E'] == 'E' and not exact_match(row)
    payload, _, _ = make_request(row, '.', revision='pin')
    assert '(D) fifth' in payload['messages'][1]['content'][0]['text']
