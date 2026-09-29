# Public Evaluation Validation

Date: 2026-09-28.

## Scope

The public evaluation recipe was tested from an anonymous fresh clone,
empty dataset/model caches, and separate new generation and judge environments.
No private model, image cache, benchmark outputs, or pre-existing research
Python environment is used. The host provides Python 3.11, a CUDA toolkit and
an allocated NVIDIA L40S (48 GB). This is not a fresh operating-system test.

This document distinguishes dependency/data checks from actual model inference.
The four-example smoke is not a benchmark-performance result.

## Pinned Inputs

| Input | Revision |
|---|---|
| `enmingzhangzz/SCOPD`, `models/scopd-plus` | `a47010e4c68c4c7c883b4e957e85d9eb7b9747e9` |
| `Qwen/Qwen2.5-VL-7B-Instruct` | `cc594898137f460bfe9f0759e9844b3ce807cfb5` |
| `mm-eval/VLMEvalKit` public TSV mirror | `772639df87331ba19da76ea9cc5fa77836a15a18` |
| `Qwen/Qwen3.6-27B-FP8` visual judge | `e89b16ebf1988b3d6befa7de50abc2d76f26eb09` |

Model download helpers resolve an input revision to an immutable commit and
save it in a local manifest. The SCOPD+ adapter SHA256 was checked against its
public provenance. Its actual LoRA merge completed in a generically named
`merged` directory, with the architecture selected from model configuration.

MMStar SHA256:
`5b581d485fcfbb0f4c8fc0440fc4ecea17fb4ba1750c6446fd20833070552e95`.
The mirror's MD5 matches the bundled evaluator's original MMStar MD5. Other
mirrored sources and their checksums are in
[`prepare_data.py`](../scopd_eval/prepare_data.py). TLS checks remain enabled.

## Completed Checks

- Public base, adapter and judge downloads, without an authentication token.
- Safe BF16 LoRA merge on one GPU.
- Fresh Qwen27B environment installation and actual image-aware judge startup.
- Two non-exact-match judge controls: an equivalent answer accepted and a
  different answer rejected, using the same public image and question.
- Judge-cache reuse without duplicated requests for those controls.
- Public dataset checksum, loading and image-decoding checks.
- CPU regression tests and dependency consistency checks.

Final GPU regression result: **123 passed, 2 skipped**. Both skips require the
training dataset, which is not part of this public evaluation smoke. The
FlashAttention GPU tests passed after building the unmodified public 2.8.3 source.

All 16 image dataset loaders completed. Their first sample's images were
decoded and verified, and the model prompt and judge input were constructed
without dropping images. This is not all-sample inference:

| Dataset | Rows |
|---|---:|
| MME | 2374 |
| MMStar | 1500 |
| MathVista-MINI | 1000 |
| MathVerse-MINI-Vision-Only | 788 |
| MMMU-Pro standard (4 options) | 1730 |
| HallusionBench | 951 |
| CVBench | 2638 |
| LogicVista | 447 |
| BLINK | 1901 |
| VisOnlyQA | 1150 |
| HRBench4K | 800 |
| MMVP | 300 |
| MME-RealWorld-Lite | 1919 |
| RealWorldQA | 765 |
| POPE | 5127 |
| MathVision-MINI | 304 |

MMMU-Pro is converted from official public Parquet files with source/output
checksums. CVBench combines the original 1288-row 2D and 1350-row 3D files.
Neither conversion silently drops options or images. Image localization uses
bounded CPU workers and resolves relative image paths.

## Real End-to-End Smoke

The public adapter was merged into the pinned base model. The first four
MMStar examples were then generated greedily at each budget using the documented
2048 response cap and 1280-to-4096 image-token bounds:

| Visual retention | Generated rows | Real judge calls | Peak allocated GPU memory |
|---|---:|---:|---:|
| 10% | 4 | 4 | 17.27 GiB |
| 100% (no pruning) | 4 | 4 | 15.76 GiB |

All eight responses completed without an inference error or OOM. All eight
required the nonmatch judge path; both final reports have `status: completed`.
Restarting inference left each prediction file byte-identical with four unique
IDs. Repeating scoring left the judge cache byte-identical with no duplicate
requests. The two additional synthetic judge controls tested an equivalent
answer and a distinct wrong answer. These tiny subsets are not performance
estimates and must not be used as reported benchmark scores.

Generation and the FP8 27B judge ran sequentially on the same GPU. During judge
startup, model weights occupied 28.51 GiB; this is not the total server footprint.

The real smoke exposed a merged-model dispatch error not covered by the initial
mock: Transformers 4.57 forwards some composite configuration attributes to its
text sub-config. Dispatch now reads the configuration class's model type and
has a regression test using the actual nested Qwen configuration. Weight values,
native pruning, generation settings and training math were not changed.

Additional one-example generation checks passed for CVBench and BLINK (the
latter with all three images). Their judge-request preflights also passed;
these two extra responses were not scored. The CVBench run exposed Pandas 3
preserving missing values during string conversion. Inference metadata now
serializes those values as JSON null, with a regression test; existing complete
metadata hashes and completed MMStar inference remain resumable.

Environment bootstrap was assembled and repaired in the new environments,
then the setup script was rerun successfully. The first FlashAttention build
hit an audit timeout and resumed from compiled objects. A later optional
wheel rebuild with upgraded build tools was stopped when it began recompiling
all kernels; it did not replace the successfully tested installation. This is
not a claim of an unattended install on every supported operating system.

## Limitations

- A four-example smoke cannot establish full benchmark scores or judge quality.
- No optimizer update or full training run is part of this evaluation audit.
  Reconstructing the ordered training data from its manifest is separate work.
- The public recipe uses Pillow 9.5.0, not a cluster-specific Pillow-SIMD build;
  historical pixel- or score-identical reproduction is not asserted.
- Source-building FlashAttention can take substantially longer than running
  the smoke itself. A CUDA toolkit is required in addition to a driver.
- The judge uses its own CUDA 13.0 environment and requires a compatible driver.
  Generation and judging run sequentially on the single GPU.
- Datasets without a pinned mirror still use the bundled upstream loaders;
  access rules and download availability must be checked separately.
