"""Checked LoRA merge using the same patched training model implementation."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-model')
    p.add_argument('--adapter', type=Path)
    p.add_argument('--download-manifest', type=Path,
                   help='Manifest from python -m scopd_eval.download_model')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    a = p.parse_args()
    provenance = {}
    if a.download_manifest:
        if a.base_model or a.adapter:
            p.error('Use --download-manifest OR --base-model and --adapter')
        provenance = json.loads(a.download_manifest.read_text())
        a.base_model, a.adapter = provenance['base_path'], Path(provenance['adapter_path'])
    if not a.base_model or not a.adapter:
        p.error('Supply --download-manifest OR both --base-model and --adapter')
    if a.output.exists() and any(a.output.iterdir()):
        raise FileExistsError('Refusing to overwrite a merged model')
    from scopd.visionzip_aokvqa.qwen_wrapper import import_qwen25_modules
    cls, _, processor_cls = import_qwen25_modules()
    import torch
    from peft import PeftModel
    if a.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA merge requires an allocated GPU; use --device cpu with sufficient host RAM')
    model = cls.from_pretrained(a.base_model, torch_dtype=torch.bfloat16,
                                attn_implementation='eager', device_map=a.device)
    model = PeftModel.from_pretrained(model, str(a.adapter)).merge_and_unload(safe_merge=True)
    model.save_pretrained(a.output, safe_serialization=True)
    processor_cls.from_pretrained(a.base_model).save_pretrained(a.output)
    (a.output / 'merge_manifest.json').write_text(json.dumps({
        **provenance, 'base_model_path': a.base_model, 'adapter': str(a.adapter), 'safe_merge': True,
        'dtype': 'bfloat16'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
