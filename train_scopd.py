#!/usr/bin/env python3
"""Core SCOPD and SCOPD+ training, with one shared reproduction entry point.

SCOPD:  mean_t KL(q_full_EMA || p_b) on a student-generated response.
SCOPD+: the same loss, averaged over the top ceil(rho * T) response positions
        ranked by B_t = JSD(p_b, p_{b+delta}). Selection is stop-gradient.

Defaults: Qwen2.5-VL-7B, native VisionZip b=.10, delta=.01 (10% -> 11%),
rho=.10, LLM-only LoRA r16/alpha32/dropout0, greedy rollout cap1024, BF16,
FP32 divergences, EMA decay .9999, constant AdamW LR2e-5, effective batch32.
Image budget: min_pixels=max_pixels=1280*28*28=1003520. Native resize rounding
can make the actual visual-token count differ from this nominal budget.
Training rollouts generate at most 1024 response tokens, excluding the prompt
and image tokens.
--max-samples 10240 means 320 optimizer updates, NOT 10240 updates.

This file contains the algorithm, not a fork of the model/trainer. Run inside
the scopd checkout: it reuses visionzip_aokvqa for the patched Qwen/VisionZip
backend, data/prompt processing, DDP, accumulation, EMA updates, and complete
resume checkpoints. No experiment-directory imports are used. Stock PyPI
Transformers does not implement this VisionZip backend.

Usage (same arguments and data order for both methods):
  python train_scopd.py --method scopd --dataset DATA.jsonl \
      --image-root IMAGES --output-dir OUT/scopd
  torchrun --standalone --nproc_per_node=4 train_scopd.py --method scopd+ \
      --dataset DATA.jsonl --image-root IMAGES --output-dir OUT/scopd_plus
Resume with the original arguments plus --resume-latest. See
docs/TRAINING.md for the runtime and data requirements.
Resume requires a matching method and configuration contract.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def forward_kl(teacher: torch.Tensor, student: torch.Tensor,
               chunk_size: int = 32) -> torch.Tensor:
    """Mean KL(teacher || student), in nats, temperature 1, full vocabulary.

    Only the student receives gradients. Chunking matches the existing trainer's
    reduction without storing a differentiable per-vocabulary loss tensor.
    Inputs are already sliced to the response positions to supervise.
    """
    if teacher.shape != student.shape or student.ndim != 2 or not len(student):
        raise ValueError("Expected aligned, nonempty [response_tokens, vocabulary] logits")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    loss = student.new_zeros((), dtype=torch.float32)
    for start in range(0, len(student), chunk_size):
        q = teacher[start:start + chunk_size].detach().float().softmax(-1)
        log_p = student[start:start + chunk_size].float().log_softmax(-1)
        loss = loss + F.kl_div(log_p, q, reduction="sum")
    return loss / len(student)


@torch.no_grad()
def budget_jsd(student: torch.Tensor, probe: torch.Tensor,
               chunk_size: int = 32) -> torch.Tensor:
    """Detached B_t = .5 KL(p_b || m) + .5 KL(p_plus || m), m=(p_b+p_plus)/2.

    This is Jensen-Shannon divergence, not the symmetric average of the two
    directional KLs. All probability calculations use FP32 and all vocab IDs.
    """
    if student.shape != probe.shape or student.ndim != 2 or not len(student):
        raise ValueError("Expected aligned, nonempty response logits")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    chunks = []
    half = student.new_tensor(0.5, dtype=torch.float32)
    for start in range(0, len(student), chunk_size):
        log_p = student[start:start + chunk_size].float().log_softmax(-1)
        log_plus = probe[start:start + chunk_size].float().log_softmax(-1)
        log_m = torch.logsumexp(torch.stack((log_p + torch.log1p(-half),
                                            log_plus + half.log())), dim=0)
        kl_p = F.kl_div(log_m, log_p, log_target=True, reduction="none")
        kl_plus = F.kl_div(log_m, log_plus, log_target=True, reduction="none")
        chunks.append((half * kl_plus + (1 - half) * kl_p).sum(-1).clamp_min(0))
    values = torch.cat(chunks)
    if not torch.isfinite(values).all():
        raise FloatingPointError("Nonfinite budget JSD")
    return values


@torch.no_grad()
def select_response_tokens(scores: torch.Tensor, valid: torch.Tensor,
                           fraction: float) -> torch.Tensor:
    """Per-trajectory top fraction; stable ties favor the earlier token."""
    if scores.ndim != 1 or scores.shape != valid.shape or not 0 < fraction <= 1:
        raise ValueError("Invalid response scores, mask, or selection fraction")
    positions = valid.to(device=scores.device, dtype=torch.bool).nonzero().flatten()
    if not positions.numel() or not torch.isfinite(scores[positions]).all():
        raise ValueError("No valid response tokens, or nonfinite selection scores")
    count = max(1, math.ceil(fraction * positions.numel()))
    order = torch.argsort(scores[positions], descending=True, stable=True)
    return positions[order[:count]].sort().values


def training_step(model: Any, processor: Any, sample: Any, cfg: dict,
                  retention_ratio: float, *, teacher_model=None, ema_shadow=None,
                  teacher_adapter_name="", teacher_uses_ground_truth=False,
                  rollout_seed=None, progress_step=None, total_steps=None,
                  fixed_rollout_token_ids=None, fixed_rollout_text=None,
                  fixed_rollout_metadata=None, capture_rollout=False):
    """One on-policy sample; the shared trainer averages sample losses equally.

    The EMA teacher is scored before constructing the student's autograd graph
    so temporary parameter swapping cannot mutate saved student activations.
    Native b and b+delta selection/merging are independent, not nested masks.
    """
    from scopd.visionzip_aokvqa import train as runtime

    if (teacher_model is not None or teacher_adapter_name or teacher_uses_ground_truth
            or fixed_rollout_token_ids is not None):
        raise ValueError("This entry point supports on-policy, no-GT, EMA-LoRA SCOPD/SCOPD+ only")
    if retention_ratio != cfg["pruning"]["train_retention_ratios"][0]:
        raise ValueError("Unexpected student retention ratio")
    device = runtime.primary_device(model)
    prompt = runtime.encode_prompt(processor, sample,
                                   image_root=cfg["dataset"]["image_root"], device=device)
    with torch.no_grad(), runtime.torch_seed_scope(rollout_seed, device), \
            runtime.temporary_cached_rollout(model):
        ids, text, generation = runtime.generate_pruned(
            model, processor, prompt, retention_ratio,
            max_new_tokens=cfg["generation"]["max_new_tokens"],
            do_sample=False, temperature=0.0, top_p=0.9, top_k=None,
            manual_decode=False, stop_on_parse=False,
            sample_id=sample.sample_id, question=sample.question)
    if ids.ndim != 2 or ids.shape[0] != 1 or not ids.numel() or (ids < 0).any():
        raise ValueError("Expected one nonempty unpadded generated response")
    count = ids.numel()
    prompt_len = prompt["input_ids"].shape[1]
    sequence = runtime.sequence_inputs_from_prompt(prompt, ids)
    if not torch.equal(sequence["input_ids"][:, prompt_len:], ids):
        raise AssertionError("Response prefix changed")

    # Use the wrapper's explicit full-token branch, never a pruned teacher.
    teacher_weights = runtime.swapped_ema_parameters(model, ema_shadow) if ema_shadow is not None else nullcontext()
    with torch.no_grad(), teacher_weights, \
            runtime.temporary_eval(model):
        teacher_out, teacher_context = runtime.forward_pruned(
            model, sequence, 1.0, prompt_len=prompt_len)
        q = runtime.extract_generated_logits(teacher_out.logits, prompt_len, count).detach().clone()
        del teacher_out
    teacher_meta = teacher_context["metadata"]
    if teacher_meta["num_kept_visual_tokens"] != teacher_meta["num_full_visual_tokens"]:
        raise AssertionError("Teacher did not receive the full image")

    student_out, student_context = runtime.forward_pruned(
        model, sequence, retention_ratio, prompt_len=prompt_len,
        sample_id=sample.sample_id, question=sample.question)
    meta = student_context["metadata"]
    p = runtime.extract_generated_logits(student_out.logits, meta["student_prompt_len"], count)
    if (generation["num_kept_visual_tokens"] != meta["num_kept_visual_tokens"]
            or meta["num_full_visual_tokens"] != teacher_meta["num_full_visual_tokens"]):
        raise AssertionError("Rollout/scoring visual budgets differ")
    if p.shape != q.shape or p.shape[0] != count or not p.requires_grad or q.requires_grad:
        raise AssertionError("Response alignment or student/teacher gradient ownership is wrong")
    # Generation has no padding. The causal slice excludes all prompt/image
    # tokens, includes EOS if generated, and includes truncated responses.
    valid = torch.ones(count, dtype=torch.bool, device=p.device)
    selected = valid.nonzero().flatten()
    signal = None
    probe_meta = None
    selection = cfg["scopd"]["budget_token_selection"]
    if selection["enabled"]:
        plus = round(retention_ratio + selection["delta"], 8)
        raw_model = runtime.unwrap_model(model)
        modes = [module.training for module in raw_model.modules()]
        with torch.no_grad(), runtime.temporary_eval(raw_model):
            probe_out, probe_context = runtime.forward_pruned(
                raw_model, sequence, plus, prompt_len=prompt_len,
                sample_id=sample.sample_id, question=sample.question)
            probe_meta = probe_context["metadata"]
            probe = runtime.extract_generated_logits(
                probe_out.logits, probe_meta["student_prompt_len"], count)
            signal = budget_jsd(p.detach(), probe, selection["kl_chunk_size"])
            del probe, probe_out
        if modes != [module.training for module in raw_model.modules()]:
            raise AssertionError("Probe did not restore model modes")
        if (probe_meta["num_full_visual_tokens"] != meta["num_full_visual_tokens"]
                or probe_meta["num_kept_visual_tokens"] <= meta["num_kept_visual_tokens"]):
            raise AssertionError("Probe must retain more visual tokens from the same image")
        selected = select_response_tokens(signal, valid, selection["top_fraction"])

    # Normalize by selected count, NOT the original length or selection ratio.
    # No KL floor, projection/F weighting, auxiliary JSD loss, or trajectory weighting.
    if signal is None:
        loss = forward_kl(q, p, selection["kl_chunk_size"])
        unselected_loss = loss.detach()
    else:
        loss = forward_kl(q[selected], p[selected], selection["kl_chunk_size"])
        with torch.no_grad():
            unselected_loss = forward_kl(q, p, selection["kl_chunk_size"])
    metrics = {
        "loss_type": "scopd_plus_top_budget_jsd_forward_kl" if signal is not None else "scopd_nogt_forward_kl",
        "kl_loss": float(loss.detach()), "unweighted_kl_loss": float(unselected_loss),
        "generated_tokens": count, "selected_tokens": selected.numel(),
        "sampled_b": retention_ratio,
        "sampled_b_plus": round(retention_ratio + selection["delta"], 8) if signal is not None else None,
        "teacher_context": "student_prompt_no_ground_truth", "teacher_ground_truth_access": False,
        "teacher_source": "ema_lora_shadow" if ema_shadow is not None else "ema_uninitialized_current",
        "scopd_teacher_strategy": "ema", "rollout_use_cache": True,
        "teacher_visual_tokens": teacher_meta["num_kept_visual_tokens"],
        "student_visual_tokens": meta["num_kept_visual_tokens"],
        "probe_visual_tokens": probe_meta["num_kept_visual_tokens"] if probe_meta else None,
        "budget_jsd_mean": float(signal.mean()) if signal is not None else None,
        "budget_jsd_max": float(signal.max()) if signal is not None else None,
    }
    record = {
        "step": progress_step, "sample_id": sample.sample_id, "question": sample.question,
        "prompt": sample.prompt, "student_text": text, "rollout_seed": rollout_seed,
        "response_token_ids": ids.detach().reshape(-1).cpu().tolist(),
        "selected_response_positions": selected.cpu().tolist(),
        "budget_jsd": signal.cpu().tolist() if signal is not None else None,
        **metrics,
    }
    if cfg.get("output_dir"):
        rank = int(os.environ.get("RANK", "0"))
        runtime.write_jsonl(Path(cfg["output_dir"]) / f"rank{rank}_core_rollouts.jsonl", record)
    metrics[runtime.STUDENT_TEXT_LOG_KEY] = {k: v for k, v in record.items() if k != "step"}
    if capture_rollout:
        cpu_ids = ids.detach().cpu().clone()
        metrics[runtime.ROLLOUT_CACHE_KEY] = {
            "token_ids": cpu_ids, "token_ids_sha256": hashlib.sha256(cpu_ids.numpy().tobytes()).hexdigest(),
            "text": text, "generation_metadata": runtime.numeric_metadata(generation),
        }
    return loss, metrics


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=("scopd", "scopd+"), required=True,
                   help="scopd: all response tokens; scopd+: top budget-JSD response tokens")
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct", help="Local pinned base-model snapshot or HF ID")
    p.add_argument("--dataset", type=Path, required=True, help="Ordered training JSONL, not validation data")
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-samples", type=int, default=10240, help="Global example count, not optimizer updates")
    p.add_argument("--effective-batch-size", type=int, default=32, help="Across GPUs; microbatch is fixed at 1")
    p.add_argument("--max-new-tokens", type=int, default=1024,
                   help="Maximum generated response tokens per training sample (default: 1024; excludes prompt/image tokens)")
    p.add_argument("--image-token-budget", type=int, default=1280,
                   help="Nominal image budget; both min_pixels and max_pixels equal this value * 28 * 28")
    p.add_argument("--retention", type=float, default=0.10, help="Actual retained fraction, not backend visionzip_ratio")
    p.add_argument("--delta", type=float, default=0.01, help="Absolute extra retention; .01 means 10%% -> 11%%")
    p.add_argument("--top-fraction", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=1024, help="LoRA snapshots, in examples")
    p.add_argument("--resume-every", type=int, default=1024, help="Full checkpoints, in examples")
    p.add_argument("--stop-at-samples", type=int, help="Stop at an optimizer boundary; keep original training target")
    p.add_argument("--resume-latest", action="store_true")
    p.add_argument("--print-config", action="store_true", help="Print resolved defaults without loading data/model")
    return p


def build_config(args: argparse.Namespace, world_size: int) -> dict:
    batch = args.effective_batch_size
    stop = args.stop_at_samples if args.stop_at_samples is not None else args.max_samples
    if world_size < 1 or batch < 1 or batch % world_size:
        raise ValueError("Effective batch size must be divisible by WORLD_SIZE")
    if (not 0 < args.retention < 1 or not 0 < args.delta < 1 - args.retention
            or not 0 < args.top_fraction <= 1 or args.max_new_tokens < 1
            or args.image_token_budget < 1):
        raise ValueError("Invalid budget, intervention, selection fraction, or generation length")
    for name, value in (("max_samples", args.max_samples), ("stop_at_samples", stop),
                        ("eval_every", args.eval_every), ("resume_every", args.resume_every)):
        if value <= 0 or value % batch:
            raise ValueError(f"{name} must be positive and divisible by effective batch size {batch}")
    if stop > args.max_samples:
        raise ValueError("Stop point exceeds training target")
    return {
        "base_model": args.model,
        "experiment": {"name": args.method, "parameter_scope": "language_decoder_only",
                       "entrypoint": "train_scopd.py", "algorithm_version": 3},
        "dataset": {"name": str(args.dataset.resolve()), "image_root": str(args.image_root.resolve()),
                    "shuffle": False, "use_splits": ["train"], "limit": 0,
                    "min_pixels": args.image_token_budget * 28 * 28,
                    "max_pixels": args.image_token_budget * 28 * 28},
        "prompt": {"enable_thinking": True},
        "training": {"method": "scopd_nogt", "max_steps": args.max_samples, "start_step": 0,
                     "seed": args.seed, "bf16": True, "attn_implementation": "flash_attention_2",
                     "device_map": {"": 0}, "use_lora": True,
                     "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.0,
                     "lora_layers_to_transform": list(range(28)), "lora_layers_pattern": "layers",
                     "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                     "expected_trainable_tensors": 392, "expected_trainable_parameters": 40370176,
                     "learning_rate": 2e-5, "weight_decay": 0.0, "save_every": 0,
                     "gradient_accumulation_steps": batch // world_size, "micro_batch_size": 1,
                     "max_sample_retries": 0},
        "generation": {"max_new_tokens": args.max_new_tokens, "max_unparseable_new_tokens": 0,
                       "stop_on_parse": False, "temperature": 0.0, "top_p": 0.9, "top_k": 0,
                       "manual_pruned_generate": False, "require_kv_cache": True},
        "pruning": {"method": "visionzip", "retention_ratio_schedule": "paired_deterministic_uniform",
                    "train_retention_ratios": [args.retention], "allow_embedding_fallback": False},
        "paired_sampling": {"enabled": True, "allow_custom_retention_ratios": True,
                            "namespace": "scopd_pair_v1",
                            "ratio_seed": args.seed, "rollout_seed": args.seed},
        "scopd": {"teacher_strategy": "ema", "use_ema_teacher": True, "ema_decay": 0.9999,
                 "ema_lazy_init": True, "teacher_ground_truth_access": False, "temperature": 1.0,
                 "native_budget_weighting": {"enabled": False},
                 "budget_token_selection": {"enabled": args.method == "scopd+", "top_fraction": args.top_fraction,
                                            "delta": args.delta, "kl_chunk_size": 32}},
        "checkpointing": {"enabled": True, "eval_snapshot_every": args.eval_every,
                          "resumable_every": args.resume_every, "save_step_zero": True,
                          "save_final_full": True, "log_sample_assignments": True,
                          "resume_from": "", "stop_at_step": stop},
        "output_dir": str(args.output_dir.resolve()),
    }


def checked_adapter_loader(original, output_dir: str | None = None):
    """Preserve FP32 adapter bits when PEFT construction passes through BF16."""
    def load(model, **kwargs):
        result = original(model, **kwargs)
        if kwargs.get("adapter_path"):
            from peft import get_peft_model_state_dict, set_peft_model_state_dict
            from safetensors.torch import load_file
            saved = load_file(str(Path(kwargs["adapter_path"]) / "adapter_model.safetensors"))
            live = get_peft_model_state_dict(result)
            if saved.keys() != live.keys() or any(saved[k].dtype != live[k].dtype for k in saved):
                raise RuntimeError("Resume adapter keys/dtypes differ from checkpoint")
            set_peft_model_state_dict(result, saved, adapter_name="default")
            live = get_peft_model_state_dict(result)
            if not all(torch.equal(saved[k], live[k].detach().cpu()) for k in saved):
                raise RuntimeError("Resume did not restore adapter tensors exactly")
        if output_dir:
            record_runtime(output_dir, kwargs.get("adapter_path", ""))
        return result
    return load


def record_runtime(output_dir: str, resumed_adapter: str) -> None:
    """Record actual imported source hashes, not just package metadata versions."""
    names = ("torch", "transformers", "peft", "tokenizers", "PIL", "flash_attn",
             "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl",
             "transformers.modeling_flash_attention_utils",
             "scopd.visionzip_aokvqa.qwen_wrapper", "scopd.visionzip_aokvqa.train")
    modules = {}
    for name in names:
        module = sys.modules.get(name)
        path = Path(module.__file__) if module is not None and getattr(module, "__file__", None) else None
        modules[name] = {
            "version": getattr(module, "__version__", None),
            "path": str(path) if path else None,
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path and path.is_file() else None,
        }
    payload = {"python": sys.version, "modules": modules, "cuda": torch.version.cuda,
               "entrypoint_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "resumed_adapter": resumed_adapter, "resume_adapter_exactly_verified": bool(resumed_adapter)}
    rank = int(os.environ.get("RANK", "0"))
    (Path(output_dir) / f"rank{rank}_runtime.json").write_text(json.dumps(payload, indent=2) + "\n")


@contextmanager
def install_core(runtime, output_dir: str | None = None):
    """Process-local hooks only; existing trainer files/runs are not changed."""
    old_step, old_lora = runtime.scopd_nogt_step, runtime.apply_lora
    runtime.scopd_nogt_step = training_step
    runtime.apply_lora = checked_adapter_loader(old_lora, output_dir)
    try:
        yield
    finally:
        runtime.scopd_nogt_step, runtime.apply_lora = old_step, old_lora


def prepare_run(cfg: dict, resume_latest: bool) -> bool:
    """Fail closed on changed data, incomplete resume state, or reused outputs."""
    from scopd.visionzip_aokvqa import train as runtime
    from scopd.visionzip_aokvqa.data_integrity import sha256_file

    dataset = Path(cfg["dataset"]["name"])
    if not dataset.is_file() or not Path(cfg["dataset"]["image_root"]).is_dir():
        raise FileNotFoundError("Training JSONL and image root must exist")
    cfg["dataset"]["sha256"] = sha256_file(dataset)
    with dataset.open() as handle:
        rows = sum(1 for line in handle if line.strip())
    if rows < cfg["training"]["max_steps"]:
        raise ValueError("Not enough training rows; this entry point does not silently repeat data")
    out = Path(cfg["output_dir"])
    checkpoint = None
    if resume_latest:
        candidates = sorted((out / "resume_checkpoints").glob("step_*/COMPLETE"))
        if not candidates:
            raise FileNotFoundError("--resume-latest requires a COMPLETE full checkpoint")
        checkpoint = candidates[-1].parent
        cfg["checkpointing"]["resume_from"] = str(checkpoint)
        _, state = runtime.prepare_resume_config(cfg, out)
        for name in ("adapter_model.safetensors", "adapter_config.json", "optimizer.pt", "ema_shadow.pt"):
            if not (checkpoint / name).is_file():
                raise FileNotFoundError(f"Missing resume state: {checkpoint / name}")
        world = int(os.environ.get("WORLD_SIZE", "1"))
        if state["world_size"] != world:
            raise ValueError("Resume must keep the original WORLD_SIZE")
        for rank in range(world):
            if not (checkpoint / "rank_rng_states" / f"rank_{rank:02d}.pt").is_file():
                raise FileNotFoundError(f"Missing RNG state for rank {rank}")
        if state["global_step"] >= cfg["checkpointing"]["stop_at_step"]:
            print(f"Requested target already complete: {checkpoint}", flush=True)
            return False
        rank = int(os.environ.get("RANK", "0"))
        runtime.trim_jsonl_to_step(out / f"rank{rank}_core_rollouts.jsonl", state["global_step"] - 1)
    elif out.exists() and any(out.iterdir()):
        raise FileExistsError("Output is nonempty; use a new directory or --resume-latest")
    out.mkdir(parents=True, exist_ok=True)
    return True


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    cfg = build_config(args, int(os.environ.get("WORLD_SIZE", "1")))
    if args.print_config:
        print(json.dumps(cfg, indent=2))
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("Training needs an allocated CUDA GPU; use --print-config on a login node")
    from scopd.visionzip_aokvqa import train as runtime
    distributed, _, _, _ = runtime.setup_distributed()
    try:
        if not prepare_run(cfg, args.resume_latest):
            return 0
        # Every rank checks the empty output before any rank writes config/logs.
        runtime.distributed_barrier(distributed)
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        random.seed(args.seed + int(os.environ.get("RANK", "0")))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        with install_core(runtime, cfg["output_dir"]):
            runtime.train(cfg)
    finally:
        runtime.cleanup_distributed(distributed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
