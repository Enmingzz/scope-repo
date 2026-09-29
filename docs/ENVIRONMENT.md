# Runtime Reproduction

## Public Installation

From a Python 3.11 shell with a CUDA toolkit (`nvcc`), run
`bash scripts/setup_eval.sh`, then `source .venv-eval/bin/activate` and
`source runtime/activate.sh eval`. Set `VENV_DIR` to put the environment on
scratch; the script prints the activation path. It installs torch/torchvision
first, both requirement lists with explicit constraints, FlashAttention build
prerequisites, and the verified runtime overlays. On an L40S it compiles the
compatible sm80 FlashAttention kernels only. `MAX_JOBS` defaults to 1 to limit
host memory; a native build can take several hours on a low-CPU allocation.
This needs a CUDA toolkit,
not just the NVIDIA driver. CUDA 12.6 matches the PyTorch wheel; other toolkit
versions may produce warnings or need local adjustment.

The public recipe pins **Pillow 9.5.0**, available as standard CPython wheels.
It does not install a cluster-specific Pillow-SIMD package or claim pixel-exact
equivalence to that historical build. All methods being compared should use the
same public recipe. Record `PIL.__version__` and its imported file alongside the
checkpoint, runtime and judge revisions for numerical reproduction.

Use `bash scripts/setup_judge.sh` for the isolated `.venv-judge` environment
(`JUDGE_VENV_DIR` overrides its location). Never activate the patched runtime
in that environment. Full copyable download, merge, generation and scoring
commands are in [First Evaluation](QUICKSTART.md#first-evaluation-one-gpu).

## Generation Environment

Use Python 3.11. The public setup script pins the following environment:

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
| Pillow | 9.5.0 |

Cluster-local build suffixes are intentionally not written as universally
installable package versions. `requirements.txt` is a source-version list, not
a promise that every wheel exists for every platform. Install compatible CUDA,
torch/torchvision and FlashAttention first. The original private environment
imported Pillow-SIMD 9.5.0.post2, with different package metadata. Verify
`PIL.__version__`, not only pip freeze. Using the public Pillow wheel is not
asserted to reproduce that build's pixels exactly.

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
torch2.11.0+cu130, Transformers5.17.0, Pillow12.3.0 and Qwen3.6-27B-FP8. These
are independent of the generation environment's Transformers/Pillow versions.
Do not put the training Transformers
override on the judge's PYTHONPATH. The provided server binds to localhost.
The image-aware judge is a distinct protocol; do not reuse timings from a
text-only judge or treat a routing smoke test as a quality benchmark.

The server defaults to image cap1003520 pixels, context8192 and at most four images
per request. Set `JUDGE_MAX_IMAGES` explicitly for tasks needing more. Multiple images
or unusually long questions may require a larger context or lower concurrency.
Do not silently drop images or truncate questions to fit. Resource errors must
be repaired and resumed, not scored as model mistakes.

Large TSV preparation uses at most two image-decoding workers by default and
respects CPU affinity. `VLMEVAL_LOCALIZE_WORKERS` can lower or raise this cap.
This changes only preparation parallelism, not image contents or scoring.

## Third-Party Attribution

- Transformers: Hugging Face, Apache-2.0; see `runtime/TRANSFORMERS_LICENSE` and
  the retained source-file notices.
- VLMEvalKit: OpenBMB/OpenCompass and source contributors; see
  `third_party/VLMEvalKit/LICENSE` and retained file notices. The runtime snapshot
  is based on revision `51682a6baab948d3dbb4b867a3eab178504ac3f5` with local patches.
- VisionZip: retained implementation and upstream license under `third_party/VisionZip/`.
- EPIC helper: upstream attribution remains in `scopd/visionzip_aokvqa/epic_official.py`.

The public repository is not an anonymous reviewer site.
Do not remove third-party licensing attribution for double-blind anonymity.
