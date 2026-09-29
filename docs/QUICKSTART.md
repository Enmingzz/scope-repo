# Quick Start

[Back to the project overview](../README.md)

Run the commands below from the repository root. This guide covers the
Qwen2.5-VL-7B training and evaluation code included in this repository.
Public adapters and the Qwen3-specific evaluation bundle are available in the
[Hugging Face release](https://huggingface.co/enmingzhangzz/SCOPD).

Core Qwen2.5-VL-7B training, merged-model image evaluation, and a single
versioned Qwen27B visual post-processing protocol. Experimental outputs,
model weights, datasets, personal paths, job histories and credentials are
not included.

## Methods

| CLI name | Method |
|---|---|
| `scopd` | SCOPD: dense forward KL on every generated response token |
| `scopd+` | SCOPD+: forward KL on the top 10% budget-JSD response tokens |

## First Evaluation (One GPU)

Start with Python 3.11, an allocated CUDA GPU, and a CUDA toolkit with `nvcc`
on `PATH` (CUDA 12.6 is recommended for the pinned PyTorch wheel). No training
data, private environment, or Hugging Face login is needed for this example.
The single-GPU path targets a 48 GB GPU (such as L40S) with at least 32 GB of
host RAM. The separate judge uses CUDA 13.0 wheels and needs a compatible NVIDIA
driver (580-series or newer); the generation environment uses CUDA 12.6 wheels.
Allow disk space for the base (about 16 GB), local merged model (about 16 GB),
and, separately, the FP8 judge (about 30 GB) plus both Python environments.
FlashAttention may compile from source; the first installation can take much
longer than subsequent evaluation runs. Compilation is not model inference.

```bash
git clone https://github.com/Enmingzz/scopd-repo.git
cd scopd-repo
export WORK_ROOT="${SCRATCH:-$PWD/.cache}/scopd"
export HF_HOME="$WORK_ROOT/hf"
export TMPDIR="$WORK_ROOT/tmp"
export XDG_CACHE_HOME="$WORK_ROOT/cache"
export LMUData="$WORK_ROOT/benchmarks"
mkdir -p "$HF_HOME" "$TMPDIR" "$XDG_CACHE_HOME" "$LMUData"
MAX_JOBS=2 NVCC_THREADS=2 bash scripts/setup_eval.sh
source .venv-eval/bin/activate
source runtime/activate.sh eval

export RELEASE_DIR="$WORK_ROOT/scopd-plus"
python -m scopd_eval.download_model --variant scopd-plus --output "$RELEASE_DIR"
export MERGED_MODEL="$RELEASE_DIR/merged"
python scripts/merge_lora.py --download-manifest "$RELEASE_DIR/download_manifest.json" \
  --output "$MERGED_MODEL" --device cuda
export MODEL_REVISION=$(python -c 'import json,os; print(json.load(open(os.environ["RELEASE_DIR"]+"/download_manifest.json"))["release_revision"])')
export EVAL_ROOT="$WORK_ROOT/eval/scopd-plus"
python -m scopd_eval.prepare_data --dataset mmstar
python -m scopd_eval.infer --model "$MERGED_MODEL" --model-revision "$MODEL_REVISION" \
  --dataset mmstar --retention 0.10 --limit 4 --output-dir "$EVAL_ROOT/mmstar-smoke"
```

The four-example run tests generation, not benchmark performance. Remove
`--limit 4` and choose a new output directory for the complete benchmark.
Use `--retention 1` for no pruning. The default response cap is 2048 and image
bounds are 1280 to 4096 nominal tokens. Model architecture is read from
`config.json`; a merged directory may have any name.

Downloads are pinned in `download_manifest.json` and resume at the same revision.
The adapter hash is checked against public provenance. Merging refuses to
overwrite an existing directory: when resuming an already merged model, skip
the merge command. Inference resumes completed examples with identical arguments.
MMStar and several other datasets are fetched from a pinned public Hugging Face
mirror and SHA256-checked. Each mirrored TSV matches the MD5 recorded in the
bundled VLMEvalKit sources; question/option text is not reformatted. The source
revision and checksums are recorded in the inference manifest. HTTPS verification
is never disabled. `python -m scopd_eval.prepare_data --help` lists mirrored
datasets; other benchmarks use their bundled sources and may require separate
dataset access/preparation.

MMMU-Pro-4c is materialized from the pinned official `standard (4 options)` test
split (1730 questions). The converter preserves official IDs, questions, option
ordering and image ordering, and records source and output checksums. Despite
the upstream split name, individual rows contain between 2 and 9 options; none
are truncated to four. It does
not use an undocumented local TSV. With `--dataset mmmupro`, this preparation is
automatic; it can also be run via `python -m scopd_eval.prepare_data --dataset mmmupro`.
Converted inputs are a versioned protocol; do not mix them with an older TSV
without checking its contents. Multi-image tasks can require a larger judge
image/context limit than the MMStar example below.

### Score the Saved Responses

Generation and judging run **sequentially on the same GPU**, or on separate
GPUs. Do not keep the 7B generation model loaded when starting the 27B judge.
Download the judge from the evaluation environment:

```bash
python -m scopd_eval.download_judge --output "$WORK_ROOT/judge.json"
export JUDGE_MODEL_PATH=$(python -c 'import json,os; print(json.load(open(os.environ["WORK_ROOT"]+"/judge.json"))["path"])')
export JUDGE_MODEL_REVISION=$(python -c 'import json,os; print(json.load(open(os.environ["WORK_ROOT"]+"/judge.json"))["revision"])')
bash scripts/setup_judge.sh
```

In a second shell in the same GPU allocation, from the repository root:

```bash
export WORK_ROOT="${SCRATCH:-$PWD/.cache}/scopd"
export HF_HOME="$WORK_ROOT/hf"
export TMPDIR="$WORK_ROOT/tmp"
export XDG_CACHE_HOME="$WORK_ROOT/cache"
mkdir -p "$HF_HOME" "$TMPDIR" "$XDG_CACHE_HOME"
source .venv-judge/bin/activate
unset PYTHONPATH PYTHONHOME
export JUDGE_MODEL_PATH=$(python -c 'import json,os; print(json.load(open(os.environ["WORK_ROOT"]+"/judge.json"))["path"])')
export JUDGE_MODEL_REVISION=$(python -c 'import json,os; print(json.load(open(os.environ["WORK_ROOT"]+"/judge.json"))["revision"])')
bash scripts/serve_judge.sh
```

Wait for the server's application-startup message, then in the first shell:

```bash
python -m scopd_eval.postprocess \
  --input "$EVAL_ROOT/mmstar-smoke/shard_000/predictions.jsonl" \
  --output-dir "$EVAL_ROOT/mmstar-smoke/judged" \
  --judge-revision "$JUDGE_MODEL_REVISION"
```

The final report is `judged/summary.json`; individual verdicts are in
`judged/judged.jsonl`. It must say `status: completed`. Failed judge requests
are cached/resumable and never silently counted as wrong. A four-example smoke
score is not a full MMStar score. Stop the judge with Ctrl-C before other GPU work.
The server allows up to four images per request by default; set
`JUDGE_MAX_IMAGES` and `JUDGE_CONTEXT` explicitly for larger multi-image tasks.
Images are never silently omitted.

## Training

The shared core is [train_scopd.py](../train_scopd.py). Necessary trainer,
EMA, optimizer, LoRA and pruning helpers are included under `scopd/`.

These are the current code defaults, not a claim that every released adapter
used identical settings. Consult each checkpoint's model card and provenance
for its base revision and original configuration.

| Setting | Default |
|---|---|
| Student | Native VisionZip, retention 10% |
| SCOPD+ probe | Native VisionZip, retention 11%, same response prefix |
| Teacher | Full image, EMA LoRA, no ground-truth access |
| Loss | Forward `KL(teacher || student)` in FP32 |
| Trainable modules | LLM-only LoRA, rank 16, alpha 32, dropout 0 |
| Learning rate | Constant AdamW, 2e-5 |
| Effective batch | 32 examples; microbatch 1 per GPU |
| Response length | At most **1024 new tokens** |
| Training image budget | **1280** nominal visual tokens |
| Processor pixel limits | `min_pixels = max_pixels = 1280 * 28 * 28 = 1003520` |
| Run length | 10240 global examples = 320 optimizer updates |
| Checkpoint interval | 1024 examples; step-0 and full final state included |
| Output format | `<think>...</think><answer>...</answer>` |

The actual resized token count can differ because of native grid rounding.
Use `--image-token-budget` and `--max-new-tokens` to set these budgets explicitly.
Resume requires both values to match the saved configuration.

Use the environment installed above, then reconstruct/activate training sources:

```bash
python runtime/prepare.py
source runtime/activate.sh train
python -m pip install --no-deps -e .

torchrun --standalone --nproc_per_node=4 train_scopd.py \
  --method scopd+ --model "$BASE_MODEL" \
  --dataset "$TRAIN_JSONL" --image-root "$IMAGE_ROOT" \
  --output-dir "$RUN_ROOT/scopd_plus"
```

Use `--method scopd` for the dense control. One GPU uses the same command with
`python` instead of `torchrun`; accumulation preserves effective batch 32.
Use `--max-samples 20480` to consume a prepared ordered file of at least 20480
entries without wrapping. A 20000-row file is not enough for 20480 examples.
Use `--max-samples 20000` to consume exactly 20000 rows (625 optimizer updates).

The ordered 20000-example ID manifest is included in `manifests/`; images,
questions and training targets are not redistributed. Check a local copy before
training, including the first 10240 examples used by the default configuration:

```bash
python scripts/check_training_data.py --dataset "$TRAIN_JSONL" \
  --image-root "$IMAGE_ROOT" --limit 10240
```

Resume using identical arguments plus `--resume-latest`. LoRA, AdamW, EMA,
data position and per-rank RNG must all be present. The final output is a LoRA
checkpoint, not a merged model.

```bash
python scripts/merge_lora.py --base-model "$BASE_MODEL" \
  --adapter "$RUN_ROOT/scopd_plus/final" --output "$MERGED_MODEL"
```

See [training details](TRAINING.md) and [environment notes](ENVIRONMENT.md).

## Evaluation

Use a local pinned base/merged checkpoint. Retention is specified directly;
the entry point converts it to the backend's different parameter convention.
Do not pass `0.10` directly to the backend's `visionzip_ratio`.

```bash
source runtime/activate.sh eval
python -m scopd_eval.infer --model "$MERGED_MODEL" --model-revision "$MODEL_REVISION" \
  --dataset mmstar --retention 0.10 --output-dir "$EVAL_ROOT/mmstar"
```

Use `--retention 1` for no pruning. Inference remains greedy with the existing
reasoning prompt, cap 2048, and evaluation image bounds 1280 to 4096 tokens.
**The training image-budget change does not change evaluation defaults.**

Supported image tasks: MME, MMStar, MathVista-MINI, MathVerse-MINI-Vision-Only,
MMMU-Pro-4c, HallusionBench, CVBench, LogicVista, BLINK, VisOnlyQA,
HRBench4K, MMVP, MME-RealWorld-Lite, RealWorldQA, POPE, MathVision-MINI.
Use `python -m scopd_eval.infer --help` for their short keys. Independent
`--shard N --shards K` jobs may each use one GPU; do not score an incomplete
set of shards as a full-benchmark result.

```bash
python -m scopd_eval.merge_shards --input-dir "$EVAL_ROOT/mmstar" \
  --shards 4 --output "$EVAL_ROOT/mmstar/all_predictions.jsonl"
```

### One Post-Processing Protocol

**MMStar uses the same Qwen3.6-27B-FP8 visual judge as every other exported task.**
The legacy MMStar 7B fallback is disabled. Processing is:

1. Extract the candidate's final answer and check deterministic equivalence.
2. If it matches the reference, accept without a judge call.
3. Otherwise give Qwen27B the candidate final answer, reference answer, original
   question, options and original image(s).
4. Judge correctness strictly; never solve the question to repair the candidate.
5. Retain raw output, deterministic result, judge verdict and provenance separately.

The gate is **all nonmatches**, including parsed wrong answers, not only parse
failures. All methods use the same gate, prompt, judge checkpoint and images.
Scores obtained with different judge models, inputs or protocols are not directly
comparable. Reprocess every compared method under this protocol.
See [exact prompt and scoring policy](EVALUATION.md).

Start the judge in its separate environment on an allocated GPU:

```bash
export JUDGE_MODEL_PATH=/path/to/pinned/Qwen3.6-27B-FP8
export JUDGE_MODEL_REVISION=PINNED_CHECKPOINT_REVISION
bash scripts/serve_judge.sh
```

From another shell:

```bash
python -m scopd_eval.postprocess \
  --input "$EVAL_ROOT/mmstar/all_predictions.jsonl" \
  --output-dir "$EVAL_ROOT/mmstar/judged" \
  --judge-revision "$JUDGE_MODEL_REVISION"
```

For a single unsharded run, use `shard_000/predictions.jsonl` directly. For
multiple shards, use the merged file rather than scoring only the first shard.

`--dry-run` checks data and image availability, including exact-match records,
without querying the judge.
For existing VLMEvalKit files, use `scopd_eval.import_results`; there is no need
to regenerate responses. A model/parser/judge failure is not silently turned
into an incorrect answer. Resume uses request fingerprints; changing the model,
reference, image, candidate, protocol or input file invalidates reuse.

## Tests and Validation

```bash
source runtime/activate.sh train
python -m pytest -q
```

The bundled source overlays are needed even when `transformers.__version__`
already says 4.57.0. Experiment outputs and model weights are not included.
Tests cover loss and gradient equivalence, token selection, runtime hooks,
resume contracts, inference sharding and judge request routing. Tests requiring
an unavailable dataset or GPU are skipped explicitly. Unit judge-routing tests
use a mock server; separately, the public recipe passed real SCOPD+ generation
and Qwen27B image-aware scoring on one L40S, with resume/cache checks and no
private data or model caches. See [the validation record](VALIDATION.md) for
exact scope, pinned revisions and limitations. This does not establish full
benchmark scores or memory safety for every maximum-size input.

The vendored toolkit may print an optional `.env` warning and a PyAV notice.
Neither is needed for this local image-only evaluation path. No remote API key
is required for the provided local judge.

A private repository under an identifiable account is **not an anonymous
reviewer link**. Export a source archive without `.git` and check paper/PDF
metadata separately before a double-blind submission.
