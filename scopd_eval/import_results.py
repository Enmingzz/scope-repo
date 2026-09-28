"""Convert existing VLMEvalKit outputs; never regenerate already available responses."""
import argparse
import json
from pathlib import Path

from .datasets import DATASETS, load, make_record
from .runtime import activate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=DATASETS, required=True)
    p.add_argument('--result-file', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    activate()
    import pandas as pd
    dataset = load(a.dataset)
    expected = dataset.data.set_index(dataset.data['index'].astype(str))
    if a.result_file.suffix == '.xlsx':
        frame = pd.read_excel(a.result_file, keep_default_na=False)
    elif a.result_file.suffix == '.jsonl':
        frame = pd.read_json(a.result_file, lines=True, dtype={'index': str})
    else:
        frame = pd.read_csv(a.result_file, keep_default_na=False)
    if not frame['index'].astype(str).is_unique or not set(frame['index'].astype(str)) <= set(expected.index):
        raise ValueError('Duplicate or unknown result IDs')
    rows = []
    for _, result in frame.iterrows():
        source = expected.loc[str(result['index'])]
        for field in ('question', 'answer'):
            if field in result and str(result[field]) != str(source[field]):
                raise ValueError(f'{field} differs for {result["index"]}')
        if not str(result.get('raw_prediction', '')).strip():
            raise ValueError('Existing output lacks raw_prediction; do not silently judge only a lossy parser output')
        images = dataset.dump_image(source)
        if isinstance(images, str):
            images = [images]
        response = {'prediction': result['prediction'], 'extra_records': {
                    'raw_prediction': result['raw_prediction'], 'generated_token_len': None}}
        rows.append(make_record(dataset.dataset_name, source, response, images))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    print(f'Imported {len(rows)} saved outputs; no generation and no judging performed')


if __name__ == '__main__':
    main()
