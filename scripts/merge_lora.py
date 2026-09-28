"""Checked LoRA merge using the same patched training model implementation."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-model', required=True)
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        raise FileExistsError('Refusing to overwrite a merged model')
    from scopd.visionzip_aokvqa.qwen_wrapper import import_qwen25_modules
    cls, _, processor_cls = import_qwen25_modules()
    import torch
    from peft import PeftModel
    model = cls.from_pretrained(a.base_model, torch_dtype=torch.bfloat16,
                                attn_implementation='flash_attention_2', device_map='auto')
    model = PeftModel.from_pretrained(model, str(a.adapter)).merge_and_unload(safe_merge=True)
    model.save_pretrained(a.output, safe_serialization=True)
    processor_cls.from_pretrained(a.base_model).save_pretrained(a.output)
    (a.output / 'merge_manifest.json').write_text(json.dumps({
        'base_model': a.base_model, 'adapter': str(a.adapter), 'safe_merge': True,
        'dtype': 'bfloat16'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
