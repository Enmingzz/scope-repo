# SCOPE / SCOPD+ Reproduction

Core Qwen2.5-VL-7B training, merged-model image evaluation, and a single
versioned Qwen27B visual post-processing protocol. Experimental outputs,
model weights, datasets, personal paths, job histories and credentials are
not included.

## Names

| CLI name | Method | Historical name |
|---|---|---|
| `scope` | Dense forward KL on every generated response token | OPSD |
| `scopd+` | Forward KL on the top 10% budget-JSD response tokens | SCOPE |

These are the requested new names, not changes to the objectives. In particular,
`--method scope` now means the dense baseline.

## Training

The shared core is [train_opsd_scope.py](train_opsd_scope.py). Necessary trainer,
EMA, optimizer, LoRA and pruning helpers are included under `opsd/`.

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
These are the new entry-point defaults, not a relabeling of historical runs.
Older image settings require `--image-token-budget 1080`; older response caps
must be supplied explicitly on resume. Changing either fails the resume contract.

Activate a compatible environment and reconstruct the patched sources:

```bash
python runtime/prepare.py
source runtime/activate.sh train
python -m pip install --no-deps -e .

torchrun --standalone --nproc_per_node=4 train_opsd_scope.py \
  --method scopd+ --model "$BASE_MODEL" \
  --dataset "$TRAIN_JSONL" --image-root "$IMAGE_ROOT" \
  --output-dir "$RUN_ROOT/scopd_plus"
```

Use `--method scope` for the dense control. One GPU uses the same command with
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

See [training details](docs/TRAINING.md) and [environment notes](docs/ENVIRONMENT.md).

## Evaluation

Use a local pinned base/merged checkpoint. Retention is specified directly;
the entry point converts it to the backend's different parameter convention.
Do not pass `0.10` directly to the backend's `visionzip_ratio`.

```bash
source runtime/activate.sh eval
python -m scope_eval.infer --model "$MERGED_MODEL" --model-revision "$MODEL_REVISION" \
  --dataset mmstar --retention 0.10 --output-dir "$EVAL_ROOT/mmstar"
```

Use `--retention 1` for no pruning. Inference remains greedy with the existing
reasoning prompt, cap 2048, and evaluation image bounds 1280 to 4096 tokens.
**The training image-budget change does not change evaluation defaults.**

Supported image tasks: MME, MMStar, MathVista-MINI, MathVerse-MINI-Vision-Only,
MMMU-Pro-4c, HallusionBench, CVBench, LogicVista, BLINK, VisOnlyQA,
HRBench4K, MMVP, MME-RealWorld-Lite, RealWorldQA, POPE, MathVision-MINI.
Use `python -m scope_eval.infer --help` for their short keys. Independent
`--shard N --shards K` jobs may each use one GPU; do not score an incomplete
set of shards as a full-benchmark result.

```bash
python -m scope_eval.merge_shards --input-dir "$EVAL_ROOT/mmstar" \
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
This is a **new protocol**: historical scores from text-only, GT-blind or 7B
judges must not be silently reused as if they came from this protocol.
See [exact prompt and scoring policy](docs/EVALUATION.md).

Start the judge in its separate environment on an allocated GPU:

```bash
export JUDGE_MODEL_PATH=/path/to/pinned/Qwen3.6-27B-FP8
export JUDGE_MODEL_REVISION=PINNED_CHECKPOINT_REVISION
bash scripts/serve_judge.sh
```

From another shell:

```bash
python -m scope_eval.postprocess \
  --input "$EVAL_ROOT/mmstar/all_predictions.jsonl" \
  --output-dir "$EVAL_ROOT/mmstar/judged" \
  --judge-revision "$JUDGE_MODEL_REVISION"
```

For a single unsharded run, use `shard_000/predictions.jsonl` directly. For
multiple shards, use the merged file rather than scoring only the first shard.

`--dry-run` checks data and image availability without querying the judge.
For existing VLMEvalKit files, use `scope_eval.import_results`; there is no need
to regenerate responses. A model/parser/judge failure is not silently turned
into an incorrect answer. Resume uses request fingerprints; changing the model,
reference, image, candidate, protocol or input file invalidates reuse.

## Tests and Scope

```bash
source runtime/activate.sh train
python -m pytest -q
```

The bundled source overlays are needed even when `transformers.__version__`
already says 4.57.0. The full historical experiment tree is intentionally absent.
This release does not submit jobs, run new full training, or overwrite old scores.
The export was checked with 95 passing tests and three skips for unbundled
historical data/reference code. Judge request routing is covered with a mock
server; the real 27B image-aware judge has not been validated end to end here.
Large GPU runs and a clean-machine environment installation still need local
validation; passing unit tests is not a memory guarantee for maximum-length runs.

A private repository under an identifiable account is **not an anonymous
reviewer link**. Export a source archive without `.git` and check paper/PDF
metadata separately before a double-blind submission.
