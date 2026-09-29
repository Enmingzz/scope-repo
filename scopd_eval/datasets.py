"""The image benchmarks used in the main tables, with explicit subset names."""
import ast
import math
import re
import string

DATASETS = {
    'mme': 'MME', 'mmstar': 'MMStar', 'mathvista': 'MathVista_MINI',
    'mathverse': 'MathVerse_MINI_Vision_Only', 'mmmupro': 'MMMU_Pro_4c',
    'hallusion': 'HallusionBench', 'cvbench': 'CVBench', 'logicvista': 'LogicVista',
    'blink': 'BLINK', 'visonly': 'VisOnlyQA-VLMEvalKit', 'hr4k': 'HRBench4K',
    'mmvp': 'MMVP', 'mmereal_lite': 'MME-RealWorld-Lite', 'realworldqa': 'RealWorldQA',
    'pope': 'POPE', 'mathvision': 'MathVision_MINI',
}


def present(value):
    return value is not None and not (isinstance(value, float) and math.isnan(value)) and str(value).strip() != ''


def options_for(row):
    options = {key: str(row[key]) for key in string.ascii_uppercase if key in row and present(row[key])}
    if options:
        return options
    if present(row.get('choices')):
        choices = row['choices']
        if isinstance(choices, str):
            choices = ast.literal_eval(choices)
        if isinstance(choices, list):
            return {string.ascii_uppercase[i]: str(v) for i, v in enumerate(choices)}
    return {}


def make_record(dataset, row, response, images):
    reference = str(row['answer'])
    options = options_for(row)
    original_options = str(row.get('multi-choice options', ''))
    if dataset.startswith('MME-RealWorld'):
        labels = re.findall(r'(?m)^\s*\(?([A-E])[).]', original_options)
        if len(labels) != len(set(labels)):
            # Preserve the original duplicate labels; do not silently reinterpret
            # the positional E column as a repaired option in the judge prompt.
            options = {key: key for key in 'ABCDE'}
    if dataset == 'MathVista_MINI' and present(row.get('answer_option')):
        reference = str(row['answer_option'])
    if dataset == 'HallusionBench':
        reference = {'0': 'No', '1': 'Yes', '0.0': 'No', '1.0': 'Yes'}.get(reference, reference)
    if isinstance(response, dict):
        extra = response.get('extra_records', {})
        raw = extra.get('raw_prediction')
        if raw is None:
            raise ValueError('Raw model output was not preserved')
        prediction = response['prediction']
        tokens = extra.get('generated_token_len')
    else:
        raw, prediction, tokens = str(response), str(response), None
    metadata = {}
    for field in ('category', 'l2-category', 'task', 'answer_type', 'precision', 'split', 'type', 'source'):
        if present(row.get(field)):
            metadata[field] = str(row[field])
    if dataset == 'MME':
        metadata['image_path'] = str(row.get('image_path', images[0]))
    return {'sample_id': str(row['index']), 'dataset': dataset, 'question': str(row['question']),
            'reference': reference, 'options': options, 'images': [str(p) for p in images],
            'raw_prediction': raw, 'prediction': prediction, 'generated_tokens': tokens,
            'metadata': metadata, 'original_options_text': original_options}


def load(key):
    from .prepare_data import SOURCES, prepare_dataset
    if key in SOURCES or key == 'mmmupro':
        prepare_dataset(key)
    from vlmeval.dataset import build_dataset, ConcatDataset
    if key == 'cvbench':
        ConcatDataset.DATASET_SETS['CVBench'] = ['CV-Bench-2D', 'CV-Bench-3D']
    dataset = build_dataset(DATASETS[key])
    if dataset is None:
        raise ValueError(f'Unknown pinned dataset: {DATASETS[key]}')
    if not dataset.data['index'].astype(str).is_unique:
        raise ValueError('Duplicate benchmark IDs')
    return dataset
