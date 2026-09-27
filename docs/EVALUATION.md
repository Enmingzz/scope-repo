# Unified Visual Judge Protocol

Protocol ID: `qwen27-visual-gt-mismatch-v1`.
Judge: **Qwen/Qwen3.6-27B-FP8**, the multimodal checkpoint used by the later local
judge runtime, not the text-only Qwen3-27B. The caller must record a pinned
checkpoint revision or local content digest. Use the same checkpoint for every
method, retention ratio and benchmark.

The actual system prompt is the `SYSTEM_PROMPT` constant in
`scope_eval/protocol.py`. It is hashed into every fallback request. The request
contains a JSON object with `question`, `options`, `reference_answer`,
`candidate_final_answer`, and `candidate_extraction`, followed by one image
content block per original image. Paths are resolved locally and image bytes are
sent as data URLs. Missing images are errors, never a silent text-only fallback.

Generation is deterministic: temperature0, seed42, thinking disabled, maximum
16 judge output tokens, constrained output `CORRECT`/`INCORRECT`. These settings
and the judge revision are distinct from model inference settings.

## Gating and Candidate Extraction

- Use a unique closed `<answer>` span when present. Conflicting answer spans or
  trailing text are not accepted by an exact-match shortcut.
- Otherwise remove closed reasoning blocks and inspect explicit final-answer
  declarations or the remaining direct answer.
- Unfinished reasoning is not used to infer an answer. An unclosed answer span
  is sent to the strict judge rather than directly accepted.
- A unique option label, exact option text, or exact rational numeric equivalence
  is a deterministic match. No substring search for the GT inside reasoning.
- **Every nonmatch** goes to the same judge, including confidently parsed wrong
  answers. A mismatch is not the same as a parser failure.
- The judge may recognize semantic equivalence, but may not repair the candidate
  using the image/question/GT. This evaluates final-answer correctness; it is not
  the stricter manual reasoning-evidence audit used in pass@k studies.

## Results and Aggregation

`predictions.jsonl` preserves generated text. `judged.jsonl` keeps deterministic
matches and fallback verdicts in separate fields. `judge_cache.jsonl` preserves
model replies, prompt/image fingerprints and revision metadata. Summary JSON
includes the protocol ID and makes no claim of equivalence to an official
benchmark's judge protocol.

Most tasks use sample answer accuracy. MME also uses per-category accuracy plus
paired-image accuracy. MMVP reports paired accuracy separately from question
accuracy. HallusionBench reports aAcc, fAcc and qAcc. POPE is **accuracy**, not F1.
CVBench first averages 2D accuracy across its data sources, then averages that
score with 3D accuracy; it is not pooled accuracy over all 2638 questions.
Per-category accuracies are retained so specialized macro scores can be inspected.
This does not replace graded MM-Vet or LLaVA-Bench rubrics with binary accuracy;
those tasks are intentionally outside the image runner's supported registry.

Complete all shards before presenting a full-dataset score, especially for paired
or grouped metrics. Do not compare old 7B/text-only scores with the new visual
judge scores as if the evaluation protocol were identical. Reprocess all compared
methods, using existing raw outputs when available.

GPU inference and judge serving are separate processes/environments. All exact
matches skip the judge. Errors produce an incomplete run, never a published
partial accuracy. Completed judge requests are reusable only if input and request
fingerprints match; a running directory is protected against concurrent writers.

## Import Existing Results

```bash
source runtime/activate.sh eval
python -m scope_eval.import_results --dataset mmstar \
  --result-file "$OLD_RESULT_XLSX" --output "$NEW_INPUT_JSONL"
python -m scope_eval.postprocess --input "$NEW_INPUT_JSONL" \
  --output-dir "$NEW_POSTPROCESS_DIR" --judge-revision "$JUDGE_MODEL_REVISION"
```

The importer aligns sample IDs, questions and references against benchmark
metadata and resolves original images. It requires raw responses and never
modifies the original result file. Inference prompts never contain reference
answers; reference answers are added only after generation for scoring.
