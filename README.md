<div align="center">

# SCOPD / SCOPD+

**Post-training vision-language models for visual-token pruning.**

[Models on Hugging Face](https://huggingface.co/enmingzhangzz/SCOPD) | [Quick Start](docs/QUICKSTART.md) | [Training](docs/TRAINING.md) | [Evaluation](docs/EVALUATION.md)

</div>

## News

- **2026-09-28:** Added a public one-GPU evaluation Quick Start, verified with
  fresh environments, public downloads, real generation and Qwen27B judging.
  See the [validation scope and limitations](docs/VALIDATION.md).
- **2026-09-28:** Released **13 LoRA checkpoints** for Qwen2.5-VL-7B and
  Qwen3-VL-4B, together with an evaluation bundle, on
  [Hugging Face](https://huggingface.co/enmingzhangzz/SCOPD).
- **2026-09-28:** Released the core Qwen2.5-VL-7B training code, pinned runtime
  sources, and a unified image-aware answer-judging protocol.

## Overview

Visual-token pruning reduces the visual context available to a vision-language
model. SCOPD and SCOPD+ adapt the language decoder to this compressed input using
on-policy distillation from a full-visual-context teacher.

| Method | Response-token supervision |
|---|---|
| **SCOPD** | Forward KL from the full-context EMA teacher on all generated response tokens. |
| **SCOPD+** | The same objective on the top 10% of response tokens ranked by student budget sensitivity. |

SCOPD+ measures sensitivity using Jensen-Shannon divergence between the
student's predictions at **10% and 11% visual-token retention**, conditioned on
the same student-generated response. Token selection is detached; only the
selected teacher-student KL losses receive gradients. See the
[training contract](docs/TRAINING.md) for the exact objective and reductions.

## Models

All released weights are hosted in **[enmingzhangzz/SCOPD](https://huggingface.co/enmingzhangzz/SCOPD)**.
The collection contains **LoRA adapters only**, not merged base models.
Download the matching base model at the revision recorded in each adapter's
`provenance.json`; merge locally for evaluation using the release instructions.

### Main Checkpoints

| Base model | SCOPD | SCOPD+ | Evaluation |
|---|---|---|---|
| Qwen2.5-VL-7B-Instruct | [LoRA](https://huggingface.co/enmingzhangzz/SCOPD/tree/main/models/scopd) | [LoRA](https://huggingface.co/enmingzhangzz/SCOPD/tree/main/models/scopd-plus) | [Quick Start](docs/QUICKSTART.md#evaluation) |
| Qwen3-VL-4B-Instruct | [LoRA](https://huggingface.co/enmingzhangzz/SCOPD/tree/main/models/qwen3-vl-4b-scopd) | [LoRA](https://huggingface.co/enmingzhangzz/SCOPD/tree/main/models/qwen3-vl-4b-scopd-plus) | [Qwen3 evaluation bundle](https://huggingface.co/enmingzhangzz/SCOPD#run-a-benchmark) |

Qwen3 uses `<analysis>...</analysis><answer>...</answer>` and its own native
VisionZip wrapper. **The training entry point in this GitHub repository is
Qwen2.5-specific**; use the Qwen3 runtime provided with the model release for
Qwen3 evaluation.

<details>
<summary><strong>Additional Qwen2.5-VL-7B checkpoints</strong></summary>

| Variant | Weights |
|---|---|
| SCOPD+ Top20 | [LoRA](https://huggingface.co/enmingzhangzz/SCOPD/tree/main/models/scopd-plus-top20) |
| SCOPD+ Top40 | [LoRA](https://huggingface.co/enmingzhangzz/SCOPD/tree/main/models/scopd-plus-top40) |
| SCOPD+ Top60 | [LoRA](https://huggingface.co/enmingzhangzz/SCOPD/tree/main/models/scopd-plus-top60) |
| SCOPD+ Reverse KL | [LoRA](https://huggingface.co/enmingzhangzz/SCOPD/tree/main/models/scopd-plus-reverse-kl) |
| SCOPD+ JSD | [LoRA](https://huggingface.co/enmingzhangzz/SCOPD/tree/main/models/scopd-plus-jsd) |
| SCOPD Off-Policy | [LoRA](https://huggingface.co/enmingzhangzz/SCOPD/tree/main/models/scopd-offpolicy) |
| SFT | [LoRA](https://huggingface.co/enmingzhangzz/SCOPD/tree/main/models/sft) |
| EPIC | [LoRA](https://huggingface.co/enmingzhangzz/SCOPD/tree/main/models/epic) |
| GRPO | [LoRA](https://huggingface.co/enmingzhangzz/SCOPD/tree/main/models/grpo) |

</details>

**Top10/20/40/60 refer to response-token selection, not visual-token retention.**
Checkpoint-specific objectives, base revisions, and weight hashes are listed in
the release [model index](https://huggingface.co/enmingzhangzz/SCOPD/blob/main/models.json).
The additional checkpoints are provided for evaluation; this repository exposes
the two core training methods rather than a trainer for every listed variant.

## Getting Started

| Task | Guide |
|---|---|
| Install, download, merge and evaluate on one GPU | [First Evaluation](docs/QUICKSTART.md#first-evaluation-one-gpu) |
| Install the pinned training/evaluation runtime | [Environment](docs/ENVIRONMENT.md) |
| Train SCOPD or SCOPD+ and resume a run | [Quick Start](docs/QUICKSTART.md#training) |
| Inspect the loss, token selection, EMA, and data contract | [Training](docs/TRAINING.md) |
| Run benchmarks, merge shards, and judge answers | [Quick Start](docs/QUICKSTART.md#evaluation) |
| Inspect the judge prompt and scoring policy | [Evaluation](docs/EVALUATION.md) |

The core implementation is [train_scopd.py](train_scopd.py). Current training
defaults use LLM-only LoRA (rank 16, alpha 32, dropout 0), a constant learning
rate of `2e-5`, effective batch size 32, and 10% student visual retention.
The response cap is **1024 new tokens**; both image pixel bounds are
`1280 * 28 * 28`. These are separate budgets.

The ordered 20K training ID manifest is included under [manifests/](manifests/).
Images and training targets are not redistributed. Check the local dataset
before training and preserve its ordering; the default run consumes 10240
examples, corresponding to 320 optimizer updates.

## Evaluation

The image runner supports MME, MMStar, MathVista, MathVerse, MMMU-Pro,
HallusionBench, CVBench, LogicVista, BLINK, VisOnlyQA, HRBench4K, MMVP,
MME-RealWorld-Lite, RealWorldQA, POPE, and MathVision-MINI. Exact subsets and
commands are documented in the [Quick Start](docs/QUICKSTART.md#evaluation).
Use retention `1.0` for no pruning, or `0.10`, `0.20`, and `0.30` for pruned
inference. Do not confuse retention with the backend's pruning parameter.

Every compared method uses the same answer-judging protocol, including MMStar:
accept deterministic matches directly, then send **all nonmatches** to
Qwen3.6-27B-FP8 with the question, options, reference answer, candidate answer,
and original images. This is a versioned evaluation protocol, not a claim of
equivalence to every benchmark's official judge. Do not mix scores obtained
with different protocols.

## Validation

Tests cover the loss, token selection, runtime hooks, checkpoint resume,
inference sharding, and judge request routing. See
[Tests and Validation](docs/QUICKSTART.md#tests-and-validation).
The public Qwen2.5 evaluation path passed a one-L40S end-to-end smoke from fresh
environments and public downloads: SCOPD+ generation at 10% and 100% retention,
real image-aware Qwen27B scoring, and exact resume/cache checks. GPU regression
tests passed (123 passed, 2 training-data-dependent skips). All 16 image dataset
loaders and first-sample prompt/record checks passed. These checks are not full
benchmark scores or maximum-input memory guarantees. See the detailed
[validation record](docs/VALIDATION.md).

## Licenses and Acknowledgements

The released adapters are available under the license stated in their
[model card](https://huggingface.co/enmingzhangzz/SCOPD#license).
We build on Qwen, Transformers, PEFT, VisionZip, and VLMEvalKit. Third-party
source notices and licenses are retained; see the
[attributions](docs/ENVIRONMENT.md#third-party-attribution).
