# Runtime Reproduction

Use Python 3.11. The tested training/evaluation environment imported:

| Component | Version |
|---|---|
| torch | 2.9.1 (CUDA12.6 build) |
| torchvision | 0.24.1 |
| Transformers | 4.57.0 plus the included source overlays |
| tokenizers | 0.22.2 |
| huggingface-hub | 0.34.3 |
| peft | 0.18.1 |
| accelerate | 1.13.0 |
| flash-attn | 2.8.3, built against the selected PyTorch |
| qwen-vl-utils | 0.0.14 |
| datasets | 4.8.5 |
| safetensors | 0.7.0 |
| pandas | 3.0.0 |
| pyarrow | 23.0.1 |
| Pillow imported runtime | 9.5.0.post2 (Pillow-SIMD) |

Cluster-local build suffixes are intentionally not written as universally
installable package versions. `requirements.txt` is a source-version list, not
a promise that every wheel exists for every platform. Install compatible CUDA,
torch/torchvision and FlashAttention first. The original Pillow metadata and
imported implementation differed; verify `PIL.__version__`, not only pip freeze.
Using a different Pillow build is not asserted to be pixel-identical.

`runtime/prepare.py` reconstructs separate train/eval Transformers trees from
the SHA256-pinned 4.57.0 wheel plus six source overlays per tree. This version was
yanked on PyPI; an explicit exact-version download is intentional for reproducing
the existing runtime. It is not a recommendation for unrelated projects.

The evaluator sources and VisionZip helper are vendored without their Git
histories, datasets, demos or weights. Upstream copyright/license notices remain.
`runtime/export_manifest.json` records source/export hashes, including portable
path substitutions. All trainable math and native pruning functions are reused.
The portable evaluator also preserves multiple image inputs instead of taking
the legacy video-message branch, retains verified HTTPS defaults, and uses
literal parsing rather than evaluating MME-RealWorld option strings as code.

The judge must use a separate environment: tested vLLM0.25.1 with
torch2.11.0+cu130 and Qwen3.6-27B-FP8. Do not put the training Transformers
override on the judge's PYTHONPATH. The provided server binds to localhost.
The image-aware judge is a new protocol and needs its own end-to-end throughput
and memory test; do not reuse timings from the old text-only judge.

The server defaults to image cap1003520 pixels and context8192. Multiple images
or unusually long questions may require a larger context or lower concurrency.
Do not silently drop images or truncate questions to fit. Resource errors must
be repaired and resumed, not scored as model mistakes.

## Third-Party Attribution

- Transformers: Hugging Face, Apache-2.0; see `runtime/TRANSFORMERS_LICENSE` and
  the retained source-file notices.
- VLMEvalKit: OpenBMB/OpenCompass and source contributors; see
  `third_party/VLMEvalKit/LICENSE` and retained file notices. The runtime snapshot
  is based on revision `51682a6baab948d3dbb4b867a3eab178504ac3f5` with local patches.
- VisionZip: retained implementation and upstream license under `third_party/VisionZip/`.
- EPIC helper: upstream attribution remains in `scopd/visionzip_aokvqa/epic_official.py`.

The repository is a private source transfer, not an anonymous reviewer site.
Do not remove third-party licensing attribution for double-blind anonymity.
