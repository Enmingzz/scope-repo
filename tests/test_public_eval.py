import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scopd_eval import prepare_data
from scopd_eval.postprocess import run


ROOT = Path(__file__).resolve().parents[1]


def test_inference_metadata_digest_handles_missing_values():
    import pandas as pd
    from scopd_eval.infer import metadata_digest
    from scopd_eval.protocol import digest
    frame = pd.DataFrame({'index': [1], 'question': ['Question'], 'optional': [float('nan')],
                          'image': ['not part of metadata']})
    assert metadata_digest(frame) == digest([{'index': '1', 'question': 'Question', 'optional': None}])
    complete = frame.drop(columns='optional')
    assert metadata_digest(complete) == digest(complete.drop(columns='image').astype(str).to_dict('records'))


@pytest.mark.parametrize('source', prepare_data.SOURCES.values(),
                         ids=lambda source: source['filename'])
def test_mirror_preserves_original_dataset_checksum(source):
    expected = {}
    for path in (ROOT / 'third_party/VLMEvalKit/vlmeval/dataset').glob('image_*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == 'DATASET_MD5' for t in node.targets):
                try:
                    expected.update(ast.literal_eval(node.value))
                except (ValueError, TypeError):
                    continue
    name = source.get('local_filename', source['filename'])
    assert source['md5'] == expected[Path(name).stem]
    assert len(source['sha256']) == 64 and len(source['revision']) == 40


@pytest.mark.parametrize('folder', ['merged', 'scopd-plus', 'Qwen2-VL-misleading-name'])
def test_model_dispatch_reads_config_not_directory(folder, monkeypatch):
    # Execute the actual constructor dispatch block without loading 7B weights.
    import transformers
    source = ast.parse((ROOT / 'third_party/VLMEvalKit/vlmeval/vlm/qwen2_vl/model.py').read_text())
    constructor = next(n for c in source.body if isinstance(c, ast.ClassDef) and c.name == 'Qwen2VLChat'
                       for n in c.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    start = next(i for i, n in enumerate(constructor.body)
                 if isinstance(n, ast.ImportFrom) and any(a.name == 'AutoConfig' for a in n.names))
    block = ast.Module(body=constructor.body[start:start + 3], type_ignores=[])
    config = transformers.Qwen2_5_VLConfig(text_config={'model_type': 'qwen2_5_vl_text'})
    assert config.model_type == 'qwen2_5_vl_text'
    monkeypatch.setattr(transformers.AutoConfig, 'from_pretrained', lambda path: config)
    monkeypatch.setattr(transformers.AutoProcessor, 'from_pretrained', lambda path: object())
    values = {'model_path': '/models/' + folder, 'self': SimpleNamespace()}
    exec(compile(block, 'constructor_dispatch', 'exec'), values)
    assert values['MODEL_CLS'] is transformers.Qwen2_5_VLForConditionalGeneration


def test_mmstar_verified_download_and_resume(tmp_path, monkeypatch):
    import huggingface_hub
    source = tmp_path / 'public.tsv'
    source.write_text('index\tquestion\n0\tQuestion\n')
    metadata = prepare_data.SOURCES['mmstar']
    monkeypatch.setitem(metadata, 'sha256', prepare_data.file_sha256(source))
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        return str(source)
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', download)
    target = prepare_data.prepare_dataset('mmstar', tmp_path / 'cache')
    assert prepare_data.prepare_dataset('mmstar', tmp_path / 'cache') == target
    assert len(calls) == 1 and calls[0]['revision'] == metadata['revision']
    target.write_text('corrupted')
    with pytest.raises(ValueError, match='checksum'):
        prepare_data.prepare_dataset('mmstar', tmp_path / 'cache')


def test_exact_match_preflight_checks_image(tmp_path):
    path = tmp_path / 'predictions.jsonl'
    path.write_text(json.dumps({'sample_id': '0', 'dataset': 'MMStar', 'question': 'Which?',
                               'reference': 'A', 'images': ['missing.png'],
                               'raw_prediction': '<answer>A</answer>'}) + '\n')
    with pytest.raises(FileNotFoundError):
        run(path, tmp_path / 'out', image_root=tmp_path, base_url='http://localhost',
            model='test', revision='pinned', dry_run=True)


def test_mmmupro_conversion_preserves_question_options_and_image_order():
    import base64
    from scopd_eval.prepare_mmmupro import convert_row
    row = {'id': 'example_1', 'question': 'Compare <image 1> and <image 2>.',
           'options': "['one', 'two', 'three', 'four']", 'answer': 'C',
           'subject': 'Geometry', 'explanation': 'must not be exported',
           'image_1': {'bytes': b'first'}, 'image_2': {'bytes': b'second'}}
    result = convert_row(row)
    assert result['question'] == row['question'] and result['index'] == row['id']
    assert [result[key] for key in 'ABCD'] == ['one', 'two', 'three', 'four']
    assert result['answer'] == 'C' and 'explanation' not in result
    assert [base64.b64decode(x) for x in json.loads(result['image'])] == [b'first', b'second']
    row.update(options=['one', 'two', 'three', 'four', 'five'], answer='E')
    assert convert_row(row)['E'] == 'five'
    row['image_2'], row['image_3'] = None, {'bytes': b'third'}
    with pytest.raises(ValueError, match='image slots'):
        convert_row(row)


@pytest.mark.parametrize('path', [None, 'nested/0.png'])
def test_image_localization_bounds_workers(tmp_path, monkeypatch, path):
    import os
    import pandas as pd
    source = ast.parse((ROOT / 'third_party/VLMEvalKit/vlmeval/smp/file.py').read_text())
    node = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'localize_df')
    calls = []
    class Pool:
        def __init__(self, count):
            calls.append(count)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def map(self, function, values):
            return [function(value) for value in values]
    namespace = {'os': os, 'osp': os.path, 'mp': SimpleNamespace(Pool=Pool),
                 'LMUDataRoot': lambda: str(tmp_path),
                 'decode_img_omni': lambda item: [os.path.join(item[0], item[2])]}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'localize', 'exec'), namespace)
    monkeypatch.setenv('VLMEVAL_LOCALIZE_WORKERS', '2')
    monkeypatch.setattr(os, 'sched_getaffinity', lambda pid: {0, 1, 2, 3})
    row = {'index': '0', 'image': 'a' * 100}
    if path:
        row['image_path'] = path
    result = namespace['localize_df'](pd.DataFrame([row]), 'test', nproc=16)
    assert calls == [2] and 'image' not in result
    assert result['image_path'][0] == str(tmp_path / 'images/test' / (path or '0.jpg'))
