#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from contextlib import contextmanager, nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DISALLOWED_QWEN25_BOOTSTRAP = "/__disabled_legacy_runtime__"
sys.path = [path for path in sys.path if not path or not path.startswith(DISALLOWED_QWEN25_BOOTSTRAP)]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from scopd.visionzip_aokvqa.aokvqa import FormattedAOKVQASample, load_aokvqa_dataset
from scopd.visionzip_aokvqa.data_integrity import verify_decontaminated_training_data
from scopd.visionzip_aokvqa.epic_official import (
    UPSTREAM_COMMIT as EPIC_UPSTREAM_COMMIT,
    UPSTREAM_REPOSITORY as EPIC_UPSTREAM_REPOSITORY,
    UPSTREAM_TRAINER_SHA256 as EPIC_UPSTREAM_TRAINER_SHA256,
    enable_visual_checkpoint_input_grads,
    extract_official_epic_response_logits,
    sample_official_epic_curriculum,
)
from scopd.visionzip_aokvqa.losses import (
    compute_budget_gradient_alignment,
    compute_budget_gradient_geometry,
    compute_budget_gradient_projection_geometry,
    compute_per_token_bhattacharyya_affinity,
    compute_teacher_gradient_budget_consensus,
    compute_teacher_mass_on_student_support,
    compute_budget_contrastive_per_token_kl,
    compute_forward_kl,
    compute_generalized_jsd,
    compute_per_token_generalized_jsd,
    compute_per_token_kl,
    compute_sequence_logprob,
    compute_token_ce,
    grpo_policy_loss,
    keep_mask_above_kl_floor,
    keep_mask_after_topk_exclusion,
    resolve_token_outlier_top_k,
)
from scopd.visionzip_aokvqa.native_budget_weighting import (
    budget_jsd_conditioned_abs_projection_grouped_weights,
    budget_gradient_aligned_bridge_gate,
    budget_gradient_consensus_weights,
    budget_counterfactual_teachability_weights,
    budget_tangent_residual_weights,
    budget_residual_hardness_weights,
    budget_contrastive_gate,
    budget_consistent_rank_weights,
    counterfactual_budget_bridge,
    counterfactual_cancellation_strength,
    counterfactual_gradient_residual_gate,
    counterfactual_rescue_amplification_weights,
    counterfactual_teachability_mixture_weights,
    counterfactual_teachability_modulation_weights,
    conditional_rescue_residual_weights,
    deterministic_random_token_drop_partition,
    deterministic_random_token_keep_partition,
    forward_kl_top_fraction_partition,
    generated_token_valid_mask,
    grouped_kl_mass_weights,
    hellinger_curriculum_weights,
    kl_abs_projection_soft_or_topk_partition,
    max_kl_fraction_inverse_jsd_weights,
    max_kl_fraction_softmax_inverse_jsd_group_balanced_weights,
    max_kl_fraction_softmax_inverse_jsd_weights,
    native_budget_robustness_weights,
    normalize_candidate_loss_mass,
    projection_fraction_bottom_drop_partition,
    projection_fraction_grouped_weights,
    projection_fraction_top_drop_partition,
    projection_fraction_token_partition,
    projection_mass_grouped_weights,
    symmetric_teacher_gap_stability_weights,
    teacher_gap_persistence_weights,
)
from scopd.visionzip_aokvqa.paired_sampling import (
    paired_retention_ratio,
    paired_rollout_seed,
    torch_seed_scope,
)
from scopd.visionzip_aokvqa.phase_ratio_scaling import (
    resolve_phase_ratio_scale,
    validate_phase_ratio_scaling_config,
)
from scopd.visionzip_aokvqa.trajectory_weighting import (
    AdaptiveBudgetFrontierState,
    ProgressAdaptiveFrontierState,
    RobustnessGatedCurriculumState,
    SensitivityFrontierState,
    competence_frontier_probability_weights,
    direct_inverse_sensitivity_probability_weights,
    effective_batch_local_objective,
    global_f_curriculum_trajectory_weights,
    globally_calibrated_trajectory_weights,
    hard_trajectory_partition_weights,
    inverse_sensitivity_probability_weights,
    ratio_group_angle_probability_weights,
    ratio_group_angle_sample_probability_weights,
    ratio_group_fraction_probability_weights,
    ratio_group_projection_probability_weights,
    residualized_budget_sensitivity,
    robustness_gated_curriculum_weights,
    sensitivity_frontier_weights,
    softmax_inverse_sensitivity_probability_weights,
    softmax_trajectory_signal_probability_weights,
    teacher_gap_mass_robustness,
    trajectory_priority_downweights,
    trajectory_rank_downweights,
    trajectory_sigmoid_downweights,
    uniform_trajectory_probability_weights,
)
from scopd.visionzip_aokvqa.prompting import build_scopd_teacher_prompt, normalize_prompt_mode, parse_final_answer
from scopd.visionzip_aokvqa.qwen_wrapper import (
    apply_lora,
    encode_prompt,
    encode_prompt_and_response,
    encode_prompt_text,
    extract_generated_logits,
    forward_pruned,
    generate_pruned,
    load_qwen_model_and_processor,
    model_input_subset,
    normalize_pruning_method,
    primary_device,
    teacher_adapter_disabled,
)


OUTPUT_ROOT = Path("outputs/visionzip_aokvqa_reasoning")
METHODS = (
    "sft",
    "grpo",
    "epic",
    "epic_official",
    "scopd",
    "scopd_fixed_teacher",
    "scopd_nogt",
    "scopd_gt_prompt",
    "offpolicy",
)
SCOPD_DYNAMIC_TEACHER_ALIASES = {"", "dynamic", "dynamic_shared_current", "shared_current", "latest"}
SCOPD_FIXED_TEACHER_ALIASES = {"fixed_base", "fixed_teacher", "legacy_fixed_base", "base"}
SCOPD_EMA_TEACHER_ALIASES = {"ema", "ema_teacher", "ema_shared", "ema_reference"}
SCOPD_EXTERNAL_TEACHER_ALIASES = {"external", "external_adapter", "teacher_adapter", "sft_teacher"}
DEFAULT_TEACHER_ADAPTER_NAME = "teacher"
STUDENT_TEXT_LOG_KEY = "_student_text_log"
ROLLOUT_CACHE_KEY = "_effective_batch_rollout_cache"
EFFECTIVE_BATCH_PROBABILITY_MODES = {
    "global_calibrated_counterfactual_teachability_batch",
    "global_f_intermediate_curriculum_batch",
    "jsd_over_current_kl_batch",
    "jsd_over_current_kl_direct_inverse_batch",
    "jsd_over_current_kl_softmax_batch",
    "jsd_over_step0_kl_batch",
    "ratio_group_counterfactual_teachability_batch",
    "trajectory_counterfactual_teachability_softmax_batch",
    "trajectory_projection_fraction_top20_batch",
    "trajectory_projection_fraction_bottom80_batch",
    "progress_adaptive_robust_frontier_batch",
}
COUNTERFACTUAL_TEACHABILITY_MODES = {
    "global_calibrated_counterfactual_teachability_batch",
    "global_f_intermediate_curriculum_batch",
    "ratio_group_counterfactual_teachability_batch",
    "trajectory_counterfactual_teachability_softmax_batch",
    "trajectory_projection_fraction_top20_batch",
    "trajectory_projection_fraction_bottom80_batch",
    "progress_adaptive_robust_frontier_batch",
    "adaptive_budget_frontier_sampler_batch",
}
RAW_TRAJECTORY_F_MODES = {
    "global_calibrated_counterfactual_teachability_batch",
    "global_f_intermediate_curriculum_batch",
    "trajectory_projection_fraction_top20_batch",
    "trajectory_projection_fraction_bottom80_batch",
}
DIRECT_GLOBAL_F_MODES = {
    "global_calibrated_counterfactual_teachability_batch",
    "global_f_intermediate_curriculum_batch",
}
TOKEN_PROJECTION_PARTITION_MODES = {
    "token_projection_fraction_top20",
    "token_projection_fraction_bottom80",
}
TOKEN_RANDOM_DROP_MODE = "token_random_drop20"
TOKEN_RANDOM_KEEP_MODE = "token_random_keep20"
TOKEN_FORWARD_KL_TOP_MODE = "token_forward_kl_top20"
TOKEN_PROJECTION_MASS_GROUPED_MODE = "token_projection_mass_grouped"
TOKEN_PROJECTION_FRACTION_GROUPED_MODE = "token_projection_fraction_grouped"
TOKEN_BUDGET_JSD_ABS_F_GROUPED_MODE = "token_budget_jsd_abs_f_grouped"
TOKEN_PROJECTION_BOTTOM_DROP_MODE = "token_projection_fraction_drop_bottom"
TOKEN_PROJECTION_TOP_DROP_MODE = "token_projection_fraction_drop_top"
TOKEN_HELLINGER_CURRICULUM_MODE = "token_hellinger_curriculum"
TOKEN_TIP_KL_ABS_PROJECTION_MODE = "token_tip_kl_abs_projection_topk"


def load_yaml(path: str | Path) -> dict[str, Any]:
    if not path:
        return {}
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def get_nested(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def set_nested(cfg: dict[str, Any], dotted: str, value: Any) -> None:
    cur = cfg
    parts = dotted.split(".")
    for key in parts[:-1]:
        cur = cur.setdefault(key, {})
    cur[parts[-1]] = value


def verify_decontaminated_dataset(cfg: dict[str, Any]) -> dict[str, Any] | None:
    manifest_value = str(get_nested(cfg, "dataset.decontamination_manifest", "") or "").strip()
    if not manifest_value:
        return None
    return verify_decontaminated_training_data(
        train_path=str(get_nested(cfg, "dataset.name")),
        manifest_path=manifest_value,
        expected_rows=int(get_nested(cfg, "dataset.expected_rows", 10_000)),
    )


def grpo_group_advantages(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rewards = rewards.float().flatten()
    if rewards.numel() == 0:
        raise ValueError("GRPO rewards must be non-empty.")
    return (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(float(eps))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--method", choices=METHODS, default=None)
    p.add_argument("--output_dir", default="")
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--start_step", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--allow_embedding_fallback", action="store_true")
    p.add_argument("--adapter_path", default="")
    p.add_argument("--selected_ids_path", default="")
    p.add_argument("--gradient_accumulation_steps", type=int, default=None)
    p.add_argument("--prompt_mode", default=None)
    p.add_argument("--enable_thinking", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=None)
    p.add_argument("--attn_implementation", default=None)
    p.add_argument("--resume_from_checkpoint", default="")
    return p


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_yaml(args.config)
    if args.method:
        set_nested(cfg, "training.method", args.method)
    method = get_nested(cfg, "training.method", "sft")
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    cfg.setdefault("output_dir", str(OUTPUT_ROOT / "checkpoints" / method))
    if args.max_steps is not None:
        set_nested(cfg, "training.max_steps", args.max_steps)
    if args.start_step is not None:
        set_nested(cfg, "training.start_step", args.start_step)
    if args.limit is not None:
        set_nested(cfg, "dataset.limit", args.limit)
    if args.smoke:
        set_nested(cfg, "training.max_steps", 1)
        set_nested(cfg, "dataset.limit", 16)
        cfg["smoke"] = True
    if args.allow_embedding_fallback:
        set_nested(cfg, "pruning.allow_embedding_fallback", True)
    if args.adapter_path:
        set_nested(cfg, "training.adapter_path", args.adapter_path)
    if args.selected_ids_path:
        set_nested(cfg, "dataset.selected_ids_path", args.selected_ids_path)
    if args.gradient_accumulation_steps is not None:
        set_nested(cfg, "training.gradient_accumulation_steps", args.gradient_accumulation_steps)
    if args.prompt_mode:
        set_nested(cfg, "prompt.mode", args.prompt_mode)
    if args.enable_thinking:
        set_nested(cfg, "prompt.enable_thinking", True)
    if args.max_new_tokens is not None:
        set_nested(cfg, "generation.max_new_tokens", int(args.max_new_tokens))
    if args.attn_implementation:
        set_nested(cfg, "training.attn_implementation", args.attn_implementation)
    if args.resume_from_checkpoint:
        set_nested(cfg, "checkpointing.resume_from", args.resume_from_checkpoint)
    return cfg


def setup_distributed() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP training requires CUDA.")
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            timeout_minutes = int(os.environ.get("SCOPD_DDP_TIMEOUT_MINUTES", "10"))
            dist.init_process_group(backend="nccl", timeout=timedelta(minutes=timeout_minutes))
    return distributed, rank, local_rank, world_size


def cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return int(rank) == 0


def unwrap_model(model: Any) -> Any:
    return getattr(model, "module", model)


def active_lora_adapter_name(model: Any) -> str:
    peft_model = unwrap_model(model)
    active = getattr(peft_model, "active_adapter", None)
    if isinstance(active, str) and active:
        return active
    active_adapters = getattr(peft_model, "active_adapters", None)
    if callable(active_adapters):
        adapters = active_adapters()
        if isinstance(adapters, (list, tuple)) and adapters:
            return str(adapters[0])
        if isinstance(adapters, str) and adapters:
            return adapters
    peft_config = getattr(peft_model, "peft_config", {})
    if isinstance(peft_config, dict) and "default" in peft_config:
        return "default"
    return ""


def set_lora_adapter_requires_grad(model: Any, adapter_name: str, requires_grad: bool) -> int:
    marker = f".{adapter_name}."
    count = 0
    for name, param in unwrap_model(model).named_parameters():
        if marker in name:
            param.requires_grad_(requires_grad)
            count += 1
    return count


def load_shared_teacher_lora_adapter(model: Any, adapter_path: str | Path, adapter_name: str) -> None:
    peft_model = unwrap_model(model)
    if not hasattr(peft_model, "load_adapter") or not hasattr(peft_model, "set_adapter"):
        raise RuntimeError("A shared LoRA teacher requires a PEFT model with load_adapter() and set_adapter().")
    previous_adapter = active_lora_adapter_name(peft_model)
    peft_model.load_adapter(str(adapter_path), adapter_name=adapter_name, is_trainable=False)
    if previous_adapter:
        peft_model.set_adapter(previous_adapter)
    frozen_count = set_lora_adapter_requires_grad(peft_model, adapter_name, False)
    if frozen_count == 0:
        raise RuntimeError(f"Loaded teacher adapter {adapter_name!r}, but found no matching LoRA parameters.")


@contextmanager
def active_lora_adapter(model: Any, adapter_name: str):
    peft_model = unwrap_model(model)
    if not adapter_name:
        raise ValueError("adapter_name must be non-empty.")
    if not hasattr(peft_model, "set_adapter"):
        raise RuntimeError("Model does not support set_adapter(); cannot switch to shared teacher LoRA adapter.")
    previous_adapter = active_lora_adapter_name(peft_model)
    peft_model.set_adapter(adapter_name)
    try:
        yield model
    finally:
        if previous_adapter:
            peft_model.set_adapter(previous_adapter)


def apply_selected_ids(dataset: list[FormattedAOKVQASample], ids_path: str | Path | None) -> list[FormattedAOKVQASample]:
    if not ids_path:
        return dataset
    path = Path(ids_path)
    ids = json.loads(path.read_text(encoding="utf-8"))
    wanted = [str(x) for x in ids]
    by_id = {sample.sample_id: sample for sample in dataset}
    missing = [sample_id for sample_id in wanted if sample_id not in by_id]
    if missing:
        raise KeyError(f"Selected training ids missing from dataset: {missing[:10]}")
    return [by_id[sample_id] for sample_id in wanted]


def _truthy_config_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def prompt_mode_from_config(cfg: dict[str, Any]) -> str:
    return normalize_prompt_mode(
        get_nested(cfg, "prompt.mode", None),
        enable_thinking=_truthy_config_value(get_nested(cfg, "prompt.enable_thinking", False)),
    )


def image_pixel_bounds_from_config(cfg: dict[str, Any]) -> tuple[int | None, int | None]:
    min_pixels = get_nested(cfg, "dataset.min_pixels", get_nested(cfg, "training.min_pixels", None))
    max_pixels = get_nested(cfg, "dataset.max_pixels", get_nested(cfg, "training.max_pixels", None))
    return (
        int(min_pixels) if min_pixels is not None else None,
        int(max_pixels) if max_pixels is not None else None,
    )


def configure_pruning_backend(cfg: dict[str, Any]) -> str:
    method = normalize_pruning_method(str(get_nested(cfg, "pruning.method", "visionzip") or "visionzip"))
    os.environ["SCOPD_PRUNING_METHOD"] = method
    if method == "random":
        os.environ["SCOPD_RANDOM_PRUNER_SEED"] = str(
            int(get_nested(cfg, "pruning.random_seed", get_nested(cfg, "training.seed", 42)))
        )
    if method == "fastv":
        os.environ["SCOPD_FASTV_TOKENS_ANCHOR"] = str(get_nested(cfg, "pruning.fastv_tokens_anchor", "all") or "all")
        os.environ["SCOPD_FASTV_TOKENS_PRUNE_LAYERS"] = str(get_nested(cfg, "pruning.fastv_tokens_prune_layers", "4") or "4")
    return method


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_student_text_log(
    sample: FormattedAOKVQASample,
    retention_ratio: float,
    generated_text: str,
    generated_tokens: int,
    teacher_source: str = "",
    teacher_strategy: str = "",
    rollout_decoder: str = "",
    rollout_use_cache: bool | None = None,
) -> dict[str, Any]:
    parsed = parse_final_answer(generated_text)
    return {
        "sample_id": sample.sample_id,
        "question": sample.question,
        "options": sample.options,
        "correct_letter": sample.correct_letter,
        "parsed_answer": parsed,
        "parseable": parsed is not None,
        "student_correct": parsed == sample.correct_letter,
        "retention_ratio": float(retention_ratio),
        "generated_tokens": int(generated_tokens),
        "teacher_source": teacher_source,
        "scopd_teacher_strategy": teacher_strategy,
        "rollout_decoder": rollout_decoder,
        "rollout_use_cache": rollout_use_cache,
        "student_text": generated_text,
    }


def is_retryable_training_error(exc: Exception) -> bool:
    message = str(exc)
    retryable_markers = (
        "Image features and image tokens do not match",
        "Image features and image placeholders do not match",
    )
    return any(marker in message for marker in retryable_markers)


def save_resolved_config(cfg: dict[str, Any], output_dir: Path) -> None:
    import yaml

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config_resolved.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def sample_retention_ratio(
    cfg: dict[str, Any],
    rng: random.Random,
    progress_step: int | None = None,
    total_steps: int | None = None,
    sample_id: str | None = None,
) -> float:
    ratios = [float(x) for x in get_nested(cfg, "pruning.train_retention_ratios", [0.1, 0.2, 0.3, 0.4])]
    schedule = str(get_nested(cfg, "pruning.retention_ratio_schedule", "random") or "random").strip().lower()
    if schedule in {"progressive", "curriculum"}:
        if progress_step is None or total_steps is None or total_steps <= 0:
            raise ValueError("progressive retention ratio scheduling requires progress_step and total_steps.")
        ordered = sorted(ratios, reverse=True)
        phase_end_steps_raw = get_nested(cfg, "pruning.progressive_phase_end_steps", None)
        if phase_end_steps_raw is not None:
            phase_end_steps = [int(value) for value in phase_end_steps_raw]
            if len(phase_end_steps) != len(ordered):
                raise ValueError(
                    "pruning.progressive_phase_end_steps must match the number of retention ratios; "
                    f"got {len(phase_end_steps)} boundaries for {len(ordered)} ratios."
                )
            if any(end <= 0 for end in phase_end_steps) or any(
                left >= right for left, right in zip(phase_end_steps, phase_end_steps[1:])
            ):
                raise ValueError("pruning.progressive_phase_end_steps must be positive and strictly increasing.")
            if phase_end_steps[-1] != int(total_steps):
                raise ValueError(
                    "The final progressive phase boundary must equal training.max_steps; "
                    f"got {phase_end_steps[-1]} vs {total_steps}."
                )
            for stage, end_step in enumerate(phase_end_steps):
                if int(progress_step) < end_step:
                    return float(ordered[stage])
            return float(ordered[-1])
        progress = min(max(float(progress_step) / float(total_steps), 0.0), 0.999999)
        stage = min(len(ordered) - 1, int(progress * len(ordered)))
        return float(ordered[stage])
    if schedule == "paired_deterministic_uniform":
        if progress_step is None or sample_id is None:
            raise ValueError(
                "paired_deterministic_uniform requires progress_step/global_index and sample_id."
            )
        weights_raw = get_nested(cfg, "pruning.train_retention_ratio_weights", None)
        if weights_raw is not None:
            raise ValueError("paired_deterministic_uniform does not accept ratio weights.")
        return paired_retention_ratio(
            ratios,
            seed=int(get_nested(cfg, "paired_sampling.ratio_seed", get_nested(cfg, "training.seed", 42))),
            global_index=int(progress_step),
            sample_id=str(sample_id),
            namespace=str(get_nested(cfg, "paired_sampling.namespace", "scopd_pair_v1")),
        )
    if schedule not in {"random", "weighted_random", "uniform_random"}:
        raise ValueError(f"Unsupported pruning.retention_ratio_schedule={schedule!r}.")
    weights_raw = get_nested(cfg, "pruning.train_retention_ratio_weights", None)
    if weights_raw is None:
        return float(rng.choice(ratios))
    weights = [float(x) for x in weights_raw]
    if len(weights) != len(ratios):
        raise ValueError(
            "pruning.train_retention_ratio_weights must have the same length as pruning.train_retention_ratios; "
            f"got {len(weights)} weights for {len(ratios)} ratios."
        )
    if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
        raise ValueError("pruning.train_retention_ratio_weights must be non-negative and sum to a positive value.")
    return float(rng.choices(ratios, weights=weights, k=1)[0])


def resolve_scopd_teacher_strategy(
    cfg: dict[str, Any],
    teacher_model: Any | None,
    teacher_adapter_name: str = "",
) -> str:
    raw = str(get_nested(cfg, "scopd.teacher_strategy", "") or "").strip().lower()
    use_ema = bool(get_nested(cfg, "scopd.use_ema_teacher", False))
    if teacher_model is not None and (use_ema or raw in SCOPD_EMA_TEACHER_ALIASES):
        return "ema"
    if teacher_model is not None:
        return "external"
    if teacher_adapter_name:
        if use_ema or raw in SCOPD_EMA_TEACHER_ALIASES:
            return "ema"
        if not raw or raw in SCOPD_EXTERNAL_TEACHER_ALIASES:
            return "external_adapter"
    fixed_teacher = bool(get_nested(cfg, "scopd.fixed_teacher", False))
    if fixed_teacher and not raw:
        return "fixed_base"
    if use_ema:
        return "ema"
    if raw in SCOPD_EMA_TEACHER_ALIASES:
        return "ema"
    if raw in SCOPD_DYNAMIC_TEACHER_ALIASES:
        return "dynamic_shared_current"
    if raw in SCOPD_FIXED_TEACHER_ALIASES:
        return "fixed_base"
    if raw in SCOPD_EXTERNAL_TEACHER_ALIASES:
        raise ValueError("scopd.teacher_strategy='external' requires scopd.teacher_adapter_path.")
    raise ValueError(
        "Unsupported scopd.teacher_strategy="
        f"{raw!r}. Use dynamic_shared_current for the online shared path, ema for the official EMA reference path, "
        "external with scopd.teacher_adapter_path for a shared teacher LoRA, or fixed_base for the legacy ablation."
    )


def trainable_parameter_names(model: Any) -> list[str]:
    return [name for name, param in unwrap_model(model).named_parameters() if param.requires_grad]


def copy_named_parameters(source_model: Any, target_model: Any, names: list[str]) -> None:
    source_params = dict(unwrap_model(source_model).named_parameters())
    target_params = dict(unwrap_model(target_model).named_parameters())
    missing = [name for name in names if name not in target_params]
    if missing:
        raise KeyError(f"EMA teacher is missing student trainable parameters: {missing[:10]}")
    with torch.no_grad():
        for name in names:
            target_params[name].data.copy_(source_params[name].detach().data)


def _configured(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def resolve_ema_update_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    raw_decay = get_nested(cfg, "scopd.ema_decay", None)
    raw_alpha = get_nested(cfg, "scopd.ema_alpha", None)
    has_decay = _configured(raw_decay)
    has_alpha = _configured(raw_alpha)
    if has_decay and has_alpha:
        raise ValueError("Specify only one of scopd.ema_decay (official) or scopd.ema_alpha (legacy ablation).")
    if has_decay:
        decay = float(raw_decay)
        if decay <= 0.0 or decay > 1.0:
            raise ValueError(f"scopd.ema_decay must be in (0, 1], got {decay}.")
        return {
            "mode": "official_decay_freeze" if decay == 1.0 else "official_decay",
            "decay": decay,
            "alpha": 1.0 - decay,
            "lazy_init": bool(get_nested(cfg, "scopd.ema_lazy_init", True)),
        }
    if has_alpha:
        alpha = float(raw_alpha)
        if alpha <= 0.0 or alpha > 1.0:
            raise ValueError(f"scopd.ema_alpha must be in (0, 1], got {alpha}.")
        return {
            "mode": "legacy_alpha",
            "decay": 1.0 - alpha,
            "alpha": alpha,
            "lazy_init": bool(get_nested(cfg, "scopd.ema_lazy_init", False)),
        }
    decay = float(get_nested(cfg, "scopd.ema_decay_default", 0.9999))
    if decay <= 0.0 or decay >= 1.0:
        raise ValueError(f"scopd.ema_decay_default must be in (0, 1), got {decay}.")
    return {
        "mode": "official_decay_default",
        "decay": decay,
        "alpha": 1.0 - decay,
        "lazy_init": bool(get_nested(cfg, "scopd.ema_lazy_init", True)),
    }


def update_ema_teacher(source_model: Any, target_model: Any, names: list[str], decay: float) -> None:
    source_params = dict(unwrap_model(source_model).named_parameters())
    target_params = dict(unwrap_model(target_model).named_parameters())
    decay = float(decay)
    if decay < 0.0 or decay > 1.0:
        raise ValueError(f"EMA decay must be in [0, 1], got {decay}.")
    if decay == 1.0:
        return
    with torch.no_grad():
        for name in names:
            target_params[name].data.mul_(decay).add_(source_params[name].detach().data, alpha=1.0 - decay)


def _remap_adapter_parameter_name(name: str, source_adapter: str, target_adapter: str) -> str:
    marker = f".{source_adapter}."
    if marker not in name:
        raise ValueError(f"Parameter {name!r} does not belong to adapter {source_adapter!r}.")
    return name.replace(marker, f".{target_adapter}.", 1)


def update_ema_adapter(
    model: Any,
    names: list[str],
    source_adapter: str,
    target_adapter: str,
    decay: float,
) -> None:
    params = dict(unwrap_model(model).named_parameters())
    decay = float(decay)
    if decay < 0.0 or decay > 1.0:
        raise ValueError(f"EMA decay must be in [0, 1], got {decay}.")
    if decay == 1.0:
        return
    missing: list[str] = []
    with torch.no_grad():
        for source_name in names:
            target_name = _remap_adapter_parameter_name(source_name, source_adapter, target_adapter)
            if target_name not in params:
                missing.append(target_name)
                continue
            params[target_name].data.mul_(decay).add_(params[source_name].detach().data, alpha=1.0 - decay)
    if missing:
        raise KeyError(f"EMA teacher adapter is missing parameters: {missing[:10]}")


def create_ema_shadow(model: Any, names: list[str]) -> dict[str, torch.Tensor]:
    params = dict(unwrap_model(model).named_parameters())
    with torch.no_grad():
        return {name: params[name].detach().clone() for name in names}


def load_ema_shadow(path: str | Path, model: Any, names: list[str]) -> dict[str, torch.Tensor] | None:
    if not path:
        return None
    ema_path = Path(path) / "ema_shadow.pt"
    if not ema_path.exists():
        return None
    params = dict(unwrap_model(model).named_parameters())
    payload = torch.load(ema_path, map_location="cpu")
    missing = [name for name in names if name not in payload]
    if missing:
        raise KeyError(f"EMA shadow checkpoint is missing parameters: {missing[:10]}")
    return {
        name: payload[name].to(device=params[name].device, dtype=params[name].dtype).clone()
        for name in names
    }


def update_ema_shadow(model: Any, shadow: dict[str, torch.Tensor], names: list[str], decay: float) -> None:
    params = dict(unwrap_model(model).named_parameters())
    decay = float(decay)
    if decay < 0.0 or decay > 1.0:
        raise ValueError(f"EMA decay must be in [0, 1], got {decay}.")
    if decay == 1.0:
        return
    with torch.no_grad():
        for name in names:
            shadow[name].mul_(decay).add_(params[name].detach().data, alpha=1.0 - decay)


@contextmanager
def swapped_ema_parameters(model: Any, shadow: dict[str, torch.Tensor]):
    params = dict(unwrap_model(model).named_parameters())
    originals = {name: params[name].detach().clone() for name in shadow}
    try:
        with torch.no_grad():
            for name, value in shadow.items():
                params[name].data.copy_(value.to(device=params[name].device, dtype=params[name].dtype))
        yield
    finally:
        with torch.no_grad():
            for name, value in originals.items():
                params[name].data.copy_(value)


@contextmanager
def temporary_eval(model: Any):
    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        yield
    finally:
        if was_training:
            model.train()


@contextmanager
def temporary_cached_rollout(model: Any):
    """Match inference-time model state while generating cached SCOPD rollouts."""

    target = unwrap_model(model)
    model_config = getattr(target, "config", None)
    generation_config = getattr(target, "generation_config", None)
    original_model_use_cache = getattr(model_config, "use_cache", None)
    original_generation_use_cache = getattr(generation_config, "use_cache", None)
    with temporary_eval(model):
        if model_config is not None:
            model_config.use_cache = True
        if generation_config is not None:
            generation_config.use_cache = True
        try:
            if model_config is not None and model_config.use_cache is not True:
                raise RuntimeError("Failed to enable model KV cache for cached SCOPD rollout.")
            if generation_config is not None and generation_config.use_cache is not True:
                raise RuntimeError("Failed to enable generation KV cache for cached SCOPD rollout.")
            yield
        finally:
            if model_config is not None:
                model_config.use_cache = original_model_use_cache
            if generation_config is not None:
                generation_config.use_cache = original_generation_use_cache


def save_ema_shadow(path: Path, shadow: dict[str, torch.Tensor] | None) -> None:
    if shadow is None:
        return
    torch.save({name: value.detach().cpu() for name, value in shadow.items()}, path / "ema_shadow.pt")


def sequence_inputs_from_prompt(prompt_inputs: dict[str, torch.Tensor], generated_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    generated_ids = generated_ids.reshape(1, -1).to(device=prompt_inputs["input_ids"].device)
    out = dict(prompt_inputs)
    out["input_ids"] = torch.cat([prompt_inputs["input_ids"], generated_ids], dim=1)
    ones = torch.ones_like(generated_ids, dtype=prompt_inputs["attention_mask"].dtype)
    out["attention_mask"] = torch.cat([prompt_inputs["attention_mask"], ones], dim=1)
    if "mm_token_type_ids" in prompt_inputs and prompt_inputs["mm_token_type_ids"] is not None:
        zeros = torch.zeros_like(generated_ids, dtype=prompt_inputs["mm_token_type_ids"].dtype)
        out["mm_token_type_ids"] = torch.cat([prompt_inputs["mm_token_type_ids"], zeros], dim=1)
    return out


def generation_top_k(cfg: dict[str, Any]) -> int | None:
    top_k = int(get_nested(cfg, "generation.top_k", 0) or 0)
    return top_k if top_k > 0 else None


def generation_do_sample(cfg: dict[str, Any]) -> bool:
    return float(get_nested(cfg, "generation.temperature", 0.7) or 0.0) > 0.0


def generation_max_unparseable_tokens(cfg: dict[str, Any]) -> int | None:
    value = int(get_nested(cfg, "generation.max_unparseable_new_tokens", 256) or 0)
    return value if value > 0 else None


def generation_stop_on_parse(cfg: dict[str, Any]) -> bool:
    return _truthy_config_value(get_nested(cfg, "generation.stop_on_parse", True))


def generate_full_teacher(model: Any, processor: Any, prompt_inputs: dict[str, torch.Tensor], cfg: dict[str, Any]) -> tuple[torch.Tensor, str]:
    eos_token_id = getattr(processor.tokenizer, "eos_token_id", None)
    pad_token_id = getattr(processor.tokenizer, "pad_token_id", None) or eos_token_id
    generate_model = unwrap_model(model)
    with torch.no_grad():
        do_sample = generation_do_sample(cfg)
        kwargs = {
            **model_input_subset(prompt_inputs),
            "max_new_tokens": int(get_nested(cfg, "generation.max_new_tokens", 128)),
            "do_sample": do_sample,
            "use_cache": True,
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
        }
        if do_sample:
            kwargs["temperature"] = float(get_nested(cfg, "generation.temperature", 0.7))
            kwargs["top_p"] = float(get_nested(cfg, "generation.top_p", 0.9))
            top_k = generation_top_k(cfg)
            if top_k is not None:
                kwargs["top_k"] = top_k
        output_ids = generate_model.generate(**kwargs)
    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    gen = output_ids[:, prompt_len:]
    text = processor.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    return gen, text


def corrected_epic_target(text: str, correct_letter: str) -> str:
    parsed = parse_final_answer(text)
    if parsed == correct_letter:
        return text.strip()
    lines = [line for line in str(text).strip().splitlines() if line.strip()]
    kept = []
    for line in lines:
        if "final" in line.lower() and "answer" in line.lower():
            continue
        kept.append(line)
    reasoning = "\n".join(kept).strip()
    if "Reasoning:" not in reasoning:
        reasoning = f"Reasoning: {reasoning or 'The correct option is supported by the image and question.'}"
    return f"{reasoning}\nFinal answer: {correct_letter}"


def build_epic_target(text: str, correct_letter: str, cfg: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    parsed = parse_final_answer(text)
    teacher_correct = parsed == correct_letter
    correct_final_answer = bool(get_nested(cfg, "epic.correct_final_answer", False))
    filter_teacher_correct = bool(get_nested(cfg, "epic.filter_teacher_correct", True))
    if teacher_correct:
        return text.strip(), {
            "teacher_parseable": parsed is not None,
            "teacher_correct": True,
            "epic_target_policy": "teacher_correct",
        }
    if filter_teacher_correct and not correct_final_answer:
        return None, {
            "teacher_parseable": parsed is not None,
            "teacher_correct": False,
            "epic_target_policy": "filtered_teacher_incorrect",
        }
    if correct_final_answer:
        return corrected_epic_target(text, correct_letter), {
            "teacher_parseable": parsed is not None,
            "teacher_correct": False,
            "epic_target_policy": "corrected_final_answer",
        }
    return None, {
        "teacher_parseable": parsed is not None,
        "teacher_correct": False,
        "epic_target_policy": "dropped_teacher_incorrect",
    }


def epic_teacher_retention_ratio(cfg: dict[str, Any], student_retention_ratio: float) -> float | str:
    """Map the student retention ratio to EPIC's easier teacher view.

    Official EPIC/TCD runs the same model twice: a more compressed student
    forward and an easier teacher forward, then distills teacher token
    distributions into the student.  Their LLaVA code expresses compression as
    a reduction ratio, where teacher_reduction = student_reduction - gap.  In
    this Qwen2.5-VL/VisionZip adaptation retention = 1 - reduction, so the
    equivalent teacher retention is student_retention + gap.
    """

    policy = str(get_nested(cfg, "epic.teacher_retention_policy", "easier_by_gap"))
    if policy == "full":
        return "full"
    if policy != "easier_by_gap":
        raise ValueError(f"Unsupported EPIC teacher_retention_policy={policy!r}.")
    gap = float(get_nested(cfg, "epic.teacher_retention_gap", 0.3))
    return min(1.0, max(float(student_retention_ratio), float(student_retention_ratio) + gap))


def sft_like_step(
    model: Any,
    processor: Any,
    sample: FormattedAOKVQASample,
    cfg: dict[str, Any],
    retention_ratio: float,
    target_response: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    device = primary_device(model)
    _, full_inputs, answer_ids = encode_prompt_and_response(
        processor,
        sample,
        target_response,
        image_root=get_nested(cfg, "dataset.image_root", ""),
        device=device,
    )
    prompt_len = int(full_inputs["input_ids"].shape[1] - answer_ids.numel())
    outputs, pruned = forward_pruned(
        model,
        full_inputs,
        retention_ratio,
        prompt_len=prompt_len,
        allow_embedding_fallback=bool(get_nested(cfg, "pruning.allow_embedding_fallback", False)),
    )
    student_prompt_len = int(pruned["metadata"]["student_prompt_len"])
    logits = extract_generated_logits(outputs.logits, student_prompt_len, int(answer_ids.numel()))
    loss = compute_token_ce(logits, answer_ids)
    return loss, {
        "loss_type": "ce",
        "generated_tokens": int(answer_ids.numel()),
        **numeric_metadata(pruned["metadata"]),
    }


def epic_tcd_step(
    model: Any,
    processor: Any,
    sample: FormattedAOKVQASample,
    cfg: dict[str, Any],
    retention_ratio: float,
    teacher_retention_override: float | str | None = None,
    official_logit_alignment: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """EPIC-style token consistency distillation adapted to Qwen2.5-VL/VisionZip.

    This follows the official EPIC repo's TCD training structure rather than
    our earlier response-level generated-target baseline:
      1. run student with compressed visual tokens and supervised labels;
      2. run the same model under no_grad with an easier/full visual-token view;
      3. align logits by answer-token index;
      4. optimize alpha * forward-KL + (1 - alpha) * SFT CE.

    The teacher pass intentionally keeps the current LoRA adapter enabled,
    matching EPIC's "teacher and student share weights" self-consistency setup.
    """

    device = primary_device(model)
    _, full_inputs, answer_ids = encode_prompt_and_response(
        processor,
        sample,
        sample.target,
        image_root=get_nested(cfg, "dataset.image_root", ""),
        device=device,
    )
    prompt_len = int(full_inputs["input_ids"].shape[1] - answer_ids.numel())
    student_outputs, student_pruned = forward_pruned(
        model,
        full_inputs,
        retention_ratio,
        prompt_len=prompt_len,
        allow_embedding_fallback=bool(get_nested(cfg, "pruning.allow_embedding_fallback", False)),
    )
    student_prompt_len = int(student_pruned["metadata"]["student_prompt_len"])
    student_logits = extract_generated_logits(student_outputs.logits, student_prompt_len, int(answer_ids.numel()))
    student_loss_sft = compute_token_ce(student_logits, answer_ids)
    student_distillation_logits = (
        extract_official_epic_response_logits(
            student_outputs.logits,
            response_start=student_prompt_len,
            response_count=int(answer_ids.numel()),
        )
        if official_logit_alignment
        else student_logits
    )

    teacher_ratio = (
        epic_teacher_retention_ratio(cfg, retention_ratio)
        if teacher_retention_override is None
        else teacher_retention_override
    )
    with torch.no_grad():
        if teacher_ratio == "full" or float(teacher_ratio) >= 1.0:
            teacher_outputs = model(**model_input_subset(full_inputs), use_cache=False)
            teacher_prompt_len = prompt_len
            teacher_meta: dict[str, Any] = {
                "teacher_num_full_visual_tokens": int(student_pruned["metadata"].get("num_full_visual_tokens", 0)),
                "teacher_num_kept_visual_tokens": int(student_pruned["metadata"].get("num_full_visual_tokens", 0)),
            }
        else:
            teacher_outputs, teacher_pruned = forward_pruned(
                model,
                full_inputs,
                float(teacher_ratio),
                prompt_len=prompt_len,
                allow_embedding_fallback=bool(get_nested(cfg, "pruning.allow_embedding_fallback", False)),
            )
            teacher_prompt_len = int(teacher_pruned["metadata"]["student_prompt_len"])
            teacher_meta = {
                "teacher_num_full_visual_tokens": int(teacher_pruned["metadata"].get("num_full_visual_tokens", 0)),
                "teacher_num_kept_visual_tokens": int(teacher_pruned["metadata"].get("num_kept_visual_tokens", 0)),
            }
    teacher_logits = (
        extract_official_epic_response_logits(
            teacher_outputs.logits,
            response_start=teacher_prompt_len,
            response_count=int(answer_ids.numel()),
        )
        if official_logit_alignment
        else extract_generated_logits(teacher_outputs.logits, teacher_prompt_len, int(answer_ids.numel()))
    ).detach()
    distillation_loss = compute_forward_kl(
        teacher_logits,
        student_distillation_logits,
        temperature=float(get_nested(cfg, "epic.temperature", 1.0)),
    )
    alpha = float(get_nested(cfg, "epic.alpha", 0.5))
    loss = alpha * distillation_loss + (1.0 - alpha) * student_loss_sft
    return loss, {
        "loss_type": "epic_tcd",
        "generated_tokens": int(answer_ids.numel()),
        "student_retention_ratio": float(retention_ratio),
        "teacher_retention_ratio": "full" if teacher_ratio == "full" else float(teacher_ratio),
        "distillation_loss": float(distillation_loss.detach().cpu()),
        "student_loss_sft": float(student_loss_sft.detach().cpu()),
        "epic_alpha": alpha,
        "epic_temperature": float(get_nested(cfg, "epic.temperature", 1.0)),
        "epic_logit_alignment": (
            "official_unshifted_response_label_positions"
            if official_logit_alignment
            else "legacy_causal_target_positions"
        ),
        "epic_distilled_positions": int(student_distillation_logits.shape[0]),
        "epic_reference": "ZichenWen1/EPIC TCD adapted to Qwen2.5-VL/VisionZip",
        **teacher_meta,
        **numeric_metadata(student_pruned["metadata"]),
    }


def scopd_nogt_step(
    model: Any,
    processor: Any,
    sample: FormattedAOKVQASample,
    cfg: dict[str, Any],
    retention_ratio: float,
    teacher_model: Any | None = None,
    ema_shadow: dict[str, torch.Tensor] | None = None,
    teacher_adapter_name: str = "",
    teacher_uses_ground_truth: bool = False,
    rollout_seed: int | None = None,
    progress_step: int | None = None,
    total_steps: int | None = None,
    fixed_rollout_token_ids: torch.Tensor | None = None,
    fixed_rollout_text: str | None = None,
    fixed_rollout_metadata: dict[str, Any] | None = None,
    capture_rollout: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    device = primary_device(model)
    prompt_inputs = encode_prompt(processor, sample, image_root=get_nested(cfg, "dataset.image_root", ""), device=device)
    manual_rollout = bool(get_nested(cfg, "generation.manual_pruned_generate", True))
    require_kv_cache = bool(get_nested(cfg, "generation.require_kv_cache", False))
    if require_kv_cache and manual_rollout:
        raise ValueError("generation.require_kv_cache=true is incompatible with manual_pruned_generate=true.")
    if fixed_rollout_token_ids is not None:
        if fixed_rollout_text is None:
            raise ValueError("Fixed rollout token IDs require the matching generated text.")
        gen_ids = fixed_rollout_token_ids.detach().to(device=device, dtype=torch.long).reshape(-1)
        gen_text = str(fixed_rollout_text)
        generation_meta = dict(fixed_rollout_metadata or {})
        generation_meta["rollout_decoder"] = "effective_batch_fixed_prefix_replay"
        rollout_decoder = "effective_batch_fixed_prefix_replay"
    else:
        rollout_decoder = "manual_no_cache" if manual_rollout else "hf_generate_kv_cache"
        rollout_context = nullcontext() if manual_rollout else temporary_cached_rollout(model)
        with torch_seed_scope(rollout_seed, device), rollout_context:
            gen_ids, gen_text, generation_meta = generate_pruned(
                model,
                processor,
                prompt_inputs,
                retention_ratio,
                max_new_tokens=int(get_nested(cfg, "generation.max_new_tokens", 128)),
                do_sample=generation_do_sample(cfg),
                temperature=float(get_nested(cfg, "generation.temperature", 0.7)),
                top_p=float(get_nested(cfg, "generation.top_p", 0.9)),
                top_k=generation_top_k(cfg),
                allow_embedding_fallback=bool(get_nested(cfg, "pruning.allow_embedding_fallback", False)),
                manual_decode=manual_rollout,
                max_unparseable_tokens=generation_max_unparseable_tokens(cfg),
                stop_on_parse=generation_stop_on_parse(cfg),
                sample_id=sample.sample_id,
                question=sample.question,
            )
    rollout_decoder = str(generation_meta.get("rollout_decoder", rollout_decoder))
    if gen_ids.numel() == 0:
        raise RuntimeError("SCOPD student generated zero tokens.")
    student_seq_inputs = sequence_inputs_from_prompt(prompt_inputs, gen_ids)
    student_prompt_len = int(prompt_inputs["input_ids"].shape[1])
    if teacher_uses_ground_truth:
        teacher_prompt = build_scopd_teacher_prompt(
            sample.question,
            sample.options,
            sample.target,
            prompt_mode=prompt_mode_from_config(cfg),
        )
        teacher_prompt_inputs = encode_prompt_text(
            processor,
            sample,
            teacher_prompt,
            image_root=get_nested(cfg, "dataset.image_root", ""),
            device=device,
        )
        teacher_seq_inputs = sequence_inputs_from_prompt(teacher_prompt_inputs, gen_ids)
        teacher_prompt_len = int(teacher_prompt_inputs["input_ids"].shape[1])
        teacher_context = "ground_truth_reference_solution"
        source_context = "privileged_gt_prompt"
    else:
        teacher_seq_inputs = student_seq_inputs
        teacher_prompt_len = student_prompt_len
        teacher_context = "student_prompt_no_ground_truth"
        source_context = "no_gt"

    raw_teacher_strategy = str(get_nested(cfg, "scopd.teacher_strategy", "") or "").strip()
    explicit_teacher_strategy = (
        teacher_model is not None
        or bool(teacher_adapter_name)
        or ema_shadow is not None
        or raw_teacher_strategy
        or bool(get_nested(cfg, "scopd.use_ema_teacher", False))
    )
    teacher_strategy = (
        resolve_scopd_teacher_strategy(cfg, teacher_model, teacher_adapter_name) if explicit_teacher_strategy else "fixed_base"
    )
    if teacher_strategy == "external":
        with torch.no_grad():
            teacher_outputs = teacher_model(**model_input_subset(teacher_seq_inputs), use_cache=False)
        teacher_source = f"external_{source_context}_full_token"
    elif teacher_strategy == "external_adapter":
        with torch.no_grad(), active_lora_adapter(model, teacher_adapter_name), temporary_eval(model):
            teacher_outputs = model(**model_input_subset(teacher_seq_inputs), use_cache=False)
        teacher_source = f"external_adapter_{source_context}_full_token"
    elif teacher_strategy == "ema":
        if teacher_model is not None:
            with torch.no_grad():
                teacher_outputs = teacher_model(**model_input_subset(teacher_seq_inputs), use_cache=False)
            teacher_source = f"ema_{source_context}_full_token"
        elif teacher_adapter_name:
            with torch.no_grad(), active_lora_adapter(model, teacher_adapter_name), temporary_eval(model):
                teacher_outputs = model(**model_input_subset(teacher_seq_inputs), use_cache=False)
            teacher_source = f"ema_adapter_{source_context}_full_token"
        elif ema_shadow is not None:
            with torch.no_grad(), swapped_ema_parameters(model, ema_shadow), temporary_eval(model):
                teacher_outputs = model(**model_input_subset(teacher_seq_inputs), use_cache=False)
            teacher_source = f"ema_lora_shadow_{source_context}_full_token"
        else:
            with torch.no_grad(), temporary_eval(model):
                teacher_outputs = model(**model_input_subset(teacher_seq_inputs), use_cache=False)
            teacher_source = f"ema_uninitialized_current_{source_context}_full_token"
    elif teacher_strategy == "dynamic_shared_current":
        with torch.no_grad(), temporary_eval(model):
            teacher_outputs = model(**model_input_subset(teacher_seq_inputs), use_cache=False)
        teacher_source = f"dynamic_shared_current_{source_context}_full_token"
    elif teacher_strategy == "fixed_base":
        with torch.no_grad(), teacher_adapter_disabled(model):
            teacher_outputs = model(**model_input_subset(teacher_seq_inputs), use_cache=False)
        teacher_source = "base_privileged_gt_prompt_full_token" if teacher_uses_ground_truth else "base_full_token"
    else:
        raise AssertionError(teacher_strategy)

    teacher_logits = extract_generated_logits(
        teacher_outputs.logits,
        teacher_prompt_len,
        int(gen_ids.numel()),
    ).detach().clone()
    del teacher_outputs

    native_weighting_enabled = bool(get_nested(cfg, "scopd.native_budget_weighting.enabled", False))
    weighting_mode = str(
        get_nested(cfg, "scopd.native_budget_weighting.mode", "inverse_student_gap")
    ).strip().lower()
    native_requires_b_plus = native_weighting_enabled and weighting_mode not in {
        TOKEN_HELLINGER_CURRICULUM_MODE,
        TOKEN_RANDOM_KEEP_MODE,
        TOKEN_FORWARD_KL_TOP_MODE,
    }
    b_plus_ratio: float | None = None
    b_plus_logits: torch.Tensor | None = None
    b_plus_metadata: dict[str, Any] = {}
    if native_requires_b_plus:
        budget_delta_mode = str(
            get_nested(cfg, "scopd.native_budget_weighting.budget_delta_mode", "absolute")
        ).strip().lower()
        if budget_delta_mode == "absolute":
            b_plus_ratio = float(retention_ratio) + float(
                get_nested(cfg, "scopd.native_budget_weighting.budget_delta", 0.05)
            )
        elif budget_delta_mode == "relative":
            relative_fraction = float(
                get_nested(cfg, "scopd.native_budget_weighting.budget_delta_fraction", 0.25)
            )
            b_plus_ratio = float(retention_ratio) * (1.0 + relative_fraction)
        else:
            raise ValueError(f"Unsupported native budget delta mode: {budget_delta_mode!r}.")
        if b_plus_ratio >= 1.0:
            raise ValueError(f"Native budget probe must remain pruned; got b_plus={b_plus_ratio}.")

    student_outputs, pruned = forward_pruned(
        model,
        student_seq_inputs,
        retention_ratio,
        prompt_len=student_prompt_len,
        allow_embedding_fallback=bool(get_nested(cfg, "pruning.allow_embedding_fallback", False)),
        sample_id=sample.sample_id,
        question=sample.question,
    )
    if normalize_pruning_method() == "random":
        rollout_mask_hash = str(generation_meta.get("random_mask_hash", ""))
        scoring_mask_hash = str(pruned["metadata"].get("random_mask_hash", ""))
        if not rollout_mask_hash or rollout_mask_hash != scoring_mask_hash:
            raise RuntimeError(
                "RandomPruner SCOPD must reuse one mask for rollout and student scoring: "
                f"rollout={rollout_mask_hash!r}, scoring={scoring_mask_hash!r}."
            )
    student_logits = extract_generated_logits(student_outputs.logits, int(pruned["metadata"]["student_prompt_len"]), int(gen_ids.numel()))
    scopd_temperature = float(get_nested(cfg, "scopd.temperature", 1.0))
    trajectory_scalar_kl: torch.Tensor | None = None
    if native_weighting_enabled and weighting_mode in {
        "trajectory_probe",
        "symmetric_teacher_gap_stability",
    }:
        trajectory_scalar_kl = compute_forward_kl(
            teacher_logits,
            student_logits,
            temperature=scopd_temperature,
        )
    # Preserve the original teacher -> differentiable student forward order.
    # The auxiliary branch is no-grad and runs only after the SCOPD graph exists.
    if native_requires_b_plus:
        if b_plus_ratio is None:
            raise AssertionError("Native budget weighting requires b_plus ratio.")
        with torch.no_grad():
            b_plus_outputs, b_plus_pruned = forward_pruned(
                unwrap_model(model),
                student_seq_inputs,
                b_plus_ratio,
                prompt_len=student_prompt_len,
                allow_embedding_fallback=bool(
                    get_nested(cfg, "pruning.allow_embedding_fallback", False)
                ),
                sample_id=sample.sample_id,
                question=sample.question,
            )
            b_plus_metadata = b_plus_pruned["metadata"]
            b_plus_logits = extract_generated_logits(
                b_plus_outputs.logits,
                int(b_plus_metadata["student_prompt_len"]),
                int(gen_ids.numel()),
            ).detach().clone()
        random_b_subset_b_plus: bool | None = None
        if normalize_pruning_method() == "random":
            b_indices = pruned.get("random_keep_indices")
            b_plus_indices = b_plus_pruned.get("random_keep_indices")
            if b_indices is None or b_plus_indices is None:
                raise RuntimeError("RandomPruner native-budget probe did not expose retained indices.")
            b_set = {int(index) for index in b_indices.tolist()}
            b_plus_set = {int(index) for index in b_plus_indices.tolist()}
            random_b_subset_b_plus = b_set.issubset(b_plus_set)
            if not random_b_subset_b_plus:
                raise RuntimeError(
                    "RandomPruner native-budget masks must be nested for the same sample: "
                    f"b={retention_ratio}, b_plus={b_plus_ratio}."
                )
        del b_plus_outputs
    else:
        random_b_subset_b_plus = None
    weighting_metrics: dict[str, Any] = {
        "native_budget_weighting_enabled": native_weighting_enabled,
        "native_budget_weighting_mode": weighting_mode if native_weighting_enabled else None,
        "native_budget_delta_mode": (
            str(get_nested(cfg, "scopd.native_budget_weighting.budget_delta_mode", "absolute"))
            if native_requires_b_plus
            else None
        ),
        "sampled_b": float(retention_ratio),
        "sampled_b_plus": b_plus_ratio,
        "native_b_num_full_visual_tokens": int(
            pruned["metadata"].get("num_full_visual_tokens", 0)
        ),
        "native_b_num_kept_visual_tokens": int(
            pruned["metadata"].get("num_kept_visual_tokens", 0)
        ),
        "native_b_plus_num_full_visual_tokens": (
            int(b_plus_metadata.get("num_full_visual_tokens", 0))
            if native_requires_b_plus
            else None
        ),
        "native_b_plus_num_kept_visual_tokens": (
            int(b_plus_metadata.get("num_kept_visual_tokens", 0))
            if native_requires_b_plus
            else None
        ),
        "native_b_plus_random_mask_hash": (
            b_plus_metadata.get("random_mask_hash") if native_requires_b_plus else None
        ),
        "native_random_b_subset_b_plus": random_b_subset_b_plus,
    }
    if native_weighting_enabled:
        if native_requires_b_plus and b_plus_logits is None:
            raise AssertionError("Native budget weighting requires b_plus logits.")
        per_token_scopd = compute_per_token_kl(
            teacher_logits,
            student_logits.detach() if weighting_mode == "trajectory_probe" else student_logits,
            temperature=scopd_temperature,
            chunk_size=int(get_nested(cfg, "scopd.native_budget_weighting.kl_chunk_size", 32)),
        )
        per_token_bridge: torch.Tensor | None = None
        token_projection_partition = None
        token_random_drop_partition = None
        token_random_keep_partition = None
        token_forward_kl_top_partition = None
        token_tip_kl_abs_projection_partition = None
        token_projection_mass_group = None
        if weighting_mode in {
            "counterfactual_budget_bridge",
            "budget_gradient_aligned_bridge",
            "counterfactual_gradient_residual",
        }:
            per_token_bridge = compute_per_token_kl(
                b_plus_logits,
                student_logits,
                temperature=scopd_temperature,
                chunk_size=int(get_nested(cfg, "scopd.native_budget_weighting.kl_chunk_size", 32)),
            )
        with torch.no_grad():
            if weighting_mode in {TOKEN_RANDOM_KEEP_MODE, TOKEN_FORWARD_KL_TOP_MODE}:
                sensitivity = torch.zeros_like(per_token_scopd.detach().float())
            elif weighting_mode == TOKEN_HELLINGER_CURRICULUM_MODE:
                affinity = compute_per_token_bhattacharyya_affinity(
                    teacher_logits,
                    student_logits.detach(),
                    temperature=scopd_temperature,
                    chunk_size=int(
                        get_nested(cfg, "scopd.native_budget_weighting.kl_chunk_size", 32)
                    ),
                )
                sensitivity = 1.0 - affinity
            elif weighting_mode in {
                "max_kl_fraction_inverse_jsd",
                "max_kl_fraction_softmax_inverse_jsd",
                "max_kl_fraction_softmax_inverse_jsd_group_balanced",
                *TOKEN_PROJECTION_PARTITION_MODES,
                TOKEN_RANDOM_DROP_MODE,
                TOKEN_PROJECTION_MASS_GROUPED_MODE,
                TOKEN_PROJECTION_FRACTION_GROUPED_MODE,
                TOKEN_BUDGET_JSD_ABS_F_GROUPED_MODE,
                TOKEN_PROJECTION_BOTTOM_DROP_MODE,
                TOKEN_PROJECTION_TOP_DROP_MODE,
                TOKEN_TIP_KL_ABS_PROJECTION_MODE,
            }:
                sensitivity = compute_per_token_generalized_jsd(
                    b_plus_logits,
                    student_logits.detach(),
                    beta=0.5,
                    temperature=float(
                        get_nested(
                            cfg,
                            "scopd.native_budget_weighting.sensitivity_temperature",
                            1.0,
                        )
                    ),
                    top_k=None,
                    token_clip=None,
                    clip_mode="token",
                    chunk_size=int(
                        get_nested(cfg, "scopd.native_budget_weighting.kl_chunk_size", 32)
                    ),
                )
            else:
                sensitivity = compute_per_token_kl(
                    b_plus_logits,
                    student_logits.detach(),
                    temperature=float(
                        get_nested(
                            cfg,
                            "scopd.native_budget_weighting.sensitivity_temperature",
                            1.0,
                        )
                    ),
                    chunk_size=int(
                        get_nested(cfg, "scopd.native_budget_weighting.kl_chunk_size", 32)
                    ),
                )
            valid_mask = generated_token_valid_mask(gen_ids)
            eps = float(get_nested(cfg, "scopd.native_budget_weighting.eps", 1e-8))
            if weighting_mode == TOKEN_RANDOM_KEEP_MODE:
                top_fraction = float(
                    get_nested(cfg, "scopd.native_budget_weighting.top_fraction", 0.2)
                )
                min_teacher_kl = float(
                    get_nested(cfg, "scopd.native_budget_weighting.min_teacher_kl", 0.0)
                )
                token_random_keep_partition = deterministic_random_token_keep_partition(
                    per_token_scopd,
                    valid_mask,
                    sample_key=f"{sample.sample_id}:{rollout_seed}",
                    seed=int(
                        get_nested(cfg, "scopd.native_budget_weighting.random_keep_seed", 42)
                    ),
                    keep_fraction=top_fraction,
                    min_kl=min_teacher_kl,
                )
                eligible = token_random_keep_partition.eligible_mask
                valid = token_random_keep_partition.selected_mask
                excluded = token_random_keep_partition.excluded_mask
                token_weight = torch.zeros_like(per_token_scopd.detach().float())
                token_weight[valid] = 1.0
                eligible_count = int(eligible.sum().cpu())
                valid_count = int(valid_mask.sum().cpu())
                selected_count = int(valid.sum().cpu())
                excluded_count = int(excluded.sum().cpu())
                below_kl_floor_count = int((valid_mask & ~eligible).sum().cpu())
                mode_metrics = {
                    "native_token_partition": "random_keep20",
                    "native_token_partition_top_fraction": top_fraction,
                    "native_token_partition_min_teacher_kl": min_teacher_kl,
                    "native_token_partition_random_seed": int(
                        get_nested(cfg, "scopd.native_budget_weighting.random_keep_seed", 42)
                    ),
                    "native_token_partition_valid_tokens": valid_count,
                    "native_token_partition_eligible_tokens": eligible_count,
                    "native_token_partition_below_kl_floor_tokens": below_kl_floor_count,
                    "native_token_partition_below_kl_floor_selected_tokens": 0,
                    "native_token_partition_excluded_low_kl_tokens": below_kl_floor_count,
                    "native_token_partition_top_tokens": selected_count,
                    "native_token_partition_random_selected_tokens": selected_count,
                    "native_token_partition_random_excluded_tokens": excluded_count,
                    "native_token_partition_selected_tokens": selected_count,
                    "native_token_partition_selected_fraction_of_eligible": (
                        float(selected_count / eligible_count) if eligible_count else 0.0
                    ),
                    "native_token_partition_selected_fraction_of_valid": (
                        float(selected_count / valid_count) if valid_count else 0.0
                    ),
                    "native_token_partition_degenerate_eligible": eligible_count < 2,
                    "native_token_partition_empty_selected": selected_count == 0,
                    "native_selection_definition": (
                        "deterministic_sha256_random_keep_over_valid_response_tokens"
                    ),
                    "native_trajectory_weight_mean": 1.0 if selected_count else 0.0,
                    "native_loss_mass_scale": 1.0,
                }
                loss_type = "scopd_nogt_token_random_keep20_forward_kl"
            elif weighting_mode == TOKEN_FORWARD_KL_TOP_MODE:
                top_fraction = float(
                    get_nested(cfg, "scopd.native_budget_weighting.top_fraction", 0.2)
                )
                min_teacher_kl = float(
                    get_nested(cfg, "scopd.native_budget_weighting.min_teacher_kl", 0.0)
                )
                token_forward_kl_top_partition = forward_kl_top_fraction_partition(
                    per_token_scopd,
                    valid_mask,
                    top_fraction=top_fraction,
                    min_kl=min_teacher_kl,
                )
                eligible = token_forward_kl_top_partition.eligible_mask
                valid = token_forward_kl_top_partition.selected_mask
                excluded = token_forward_kl_top_partition.excluded_mask
                token_weight = valid.to(dtype=per_token_scopd.dtype).detach()
                eligible_count = int(eligible.sum().cpu())
                valid_count = int(valid_mask.sum().cpu())
                selected_count = int(valid.sum().cpu())
                excluded_count = int(excluded.sum().cpu())
                below_kl_floor_count = int((valid_mask & ~eligible).sum().cpu())
                selected_kl = per_token_scopd.detach().float()[valid]
                excluded_kl = per_token_scopd.detach().float()[excluded]
                mode_metrics = {
                    "native_token_partition": "forward_kl_top20",
                    "native_token_partition_top_fraction": top_fraction,
                    "native_token_partition_min_teacher_kl": min_teacher_kl,
                    "native_token_partition_valid_tokens": valid_count,
                    "native_token_partition_eligible_tokens": eligible_count,
                    "native_token_partition_below_kl_floor_tokens": below_kl_floor_count,
                    "native_token_partition_selected_tokens": selected_count,
                    "native_token_partition_excluded_tokens": excluded_count,
                    "native_token_partition_selected_fraction_of_eligible": (
                        float(selected_count / eligible_count) if eligible_count else 0.0
                    ),
                    "native_token_partition_selected_fraction_of_valid": (
                        float(selected_count / valid_count) if valid_count else 0.0
                    ),
                    "native_token_partition_selected_kl_min": (
                        float(selected_kl.min().cpu()) if selected_count else None
                    ),
                    "native_token_partition_selected_kl_mean": (
                        float(selected_kl.mean().cpu()) if selected_count else None
                    ),
                    "native_token_partition_excluded_kl_max": (
                        float(excluded_kl.max().cpu()) if excluded_count else None
                    ),
                    "native_token_partition_empty_selected": selected_count == 0,
                    "native_selection_definition": (
                        "stable_descending_top_fraction_of_detached_forward_kl_per_response"
                    ),
                    "native_trajectory_weight_mean": 1.0 if selected_count else 0.0,
                    "native_loss_mass_scale": 1.0,
                }
                loss_type = "scopd_nogt_token_forward_kl_top20"
            elif weighting_mode == TOKEN_HELLINGER_CURRICULUM_MODE:
                curriculum_scale = float(
                    get_nested(cfg, "scopd.native_budget_weighting.curriculum_scale", 8.0)
                )
                weights = hellinger_curriculum_weights(
                    affinity,
                    valid_mask,
                    scale=curriculum_scale,
                )
                valid = weights.valid_mask
                token_weight = weights.weight
                mode_metrics = {
                    "native_affinity_definition": "sum_v_sqrt_q_full_times_p_student",
                    "native_difficulty_definition": "one_minus_bhattacharyya_affinity",
                    "native_curriculum_definition": "scale_times_d_times_one_minus_d",
                    "native_curriculum_scale": curriculum_scale,
                    "native_curriculum_normalization": "none",
                    "native_affinity_min": float(weights.affinity[valid].min().cpu()),
                    "native_affinity_mean": float(weights.affinity[valid].mean().cpu()),
                    "native_affinity_max": float(weights.affinity[valid].max().cpu()),
                    "native_difficulty_min": float(weights.difficulty[valid].min().cpu()),
                    "native_difficulty_mean": float(weights.difficulty[valid].mean().cpu()),
                    "native_difficulty_max": float(weights.difficulty[valid].max().cpu()),
                    "native_trajectory_weight_mean": float(token_weight[valid].mean().cpu()),
                    "native_loss_mass_scale": 1.0,
                }
                loss_type = "scopd_nogt_token_hellinger_curriculum_forward_kl"
            elif weighting_mode == "inverse_student_gap":
                weights = native_budget_robustness_weights(
                    sensitivity,
                    valid_mask,
                    eps=eps,
                )
                valid = weights.valid_mask
                token_weight = weights.weight
                mode_metrics = {
                    "native_tau": float(weights.tau.cpu()),
                    "native_robustness_mean": float(weights.robustness[valid].mean().cpu()),
                }
                loss_type = "scopd_nogt_native_budget_weighted_forward_kl"
            elif weighting_mode in {
                "max_kl_fraction_inverse_jsd",
                "max_kl_fraction_softmax_inverse_jsd",
                "max_kl_fraction_softmax_inverse_jsd_group_balanced",
            }:
                max_kl_fraction = float(
                    get_nested(
                        cfg,
                        "scopd.native_budget_weighting.max_kl_fraction",
                        0.10,
                    )
                )
                if weighting_mode == "max_kl_fraction_inverse_jsd":
                    weights = max_kl_fraction_inverse_jsd_weights(
                        per_token_scopd,
                        sensitivity,
                        valid_mask,
                        max_kl_fraction=max_kl_fraction,
                        eps=eps,
                    )
                    weight_transform = "direct_inverse"
                    softmax_temperature = None
                    high_group_coefficient = None
                    balanced_unweighted_kl = None
                elif weighting_mode == "max_kl_fraction_softmax_inverse_jsd":
                    softmax_temperature = float(
                        get_nested(
                            cfg,
                            "scopd.native_budget_weighting.softmax_temperature",
                            0.05,
                        )
                    )
                    weights = max_kl_fraction_softmax_inverse_jsd_weights(
                        per_token_scopd,
                        sensitivity,
                        valid_mask,
                        max_kl_fraction=max_kl_fraction,
                        temperature=softmax_temperature,
                    )
                    weight_transform = "softmax_inverse"
                    high_group_coefficient = None
                    balanced_unweighted_kl = None
                else:
                    softmax_temperature = float(
                        get_nested(
                            cfg,
                            "scopd.native_budget_weighting.softmax_temperature",
                            0.05,
                        )
                    )
                    high_group_coefficient = float(
                        get_nested(
                            cfg,
                            "scopd.native_budget_weighting.high_group_coefficient",
                            0.10,
                        )
                    )
                    weights = max_kl_fraction_softmax_inverse_jsd_group_balanced_weights(
                        per_token_scopd,
                        sensitivity,
                        valid_mask,
                        max_kl_fraction=max_kl_fraction,
                        temperature=softmax_temperature,
                        high_group_coefficient=high_group_coefficient,
                    )
                    balanced_unweighted_kl = (
                        weights.balanced_unweighted_weight[weights.valid_mask]
                        * per_token_scopd[weights.valid_mask]
                    ).mean()
                    weight_transform = "group_balanced_softmax_inverse"
                valid = weights.valid_mask
                high = weights.high_group_mask
                token_weight = weights.weight
                high_jsd = weights.sensitivity[high]
                high_weight = weights.weight[high]
                mode_metrics = {
                    "native_sensitivity_definition": "per_token_symmetric_jsd_student_b_plus_student_b",
                    "native_max_kl_fraction": max_kl_fraction,
                    "native_group_weight_transform": weight_transform,
                    "native_group_softmax_temperature": softmax_temperature,
                    "native_high_group_coefficient": high_group_coefficient,
                    "native_low_group_coefficient": (
                        1.0 - high_group_coefficient
                        if high_group_coefficient is not None
                        else None
                    ),
                    "native_group_balanced_unweighted_loss": (
                        float(balanced_unweighted_kl.detach().cpu())
                        if balanced_unweighted_kl is not None
                        else None
                    ),
                    "native_group_batch_normalization": False,
                    "native_high_group_kl_threshold": float(weights.threshold.cpu()),
                    "native_high_group_tokens": int(high.sum().cpu()),
                    "native_high_group_fraction": float(high.float().sum().cpu() / valid.sum().cpu()),
                    "native_high_group_jsd_min": (
                        float(high_jsd.min().cpu()) if high.any() else None
                    ),
                    "native_high_group_jsd_mean": (
                        float(high_jsd.mean().cpu()) if high.any() else None
                    ),
                    "native_high_group_jsd_max": (
                        float(high_jsd.max().cpu()) if high.any() else None
                    ),
                    "native_high_group_weight_mean": (
                        float(high_weight.mean().cpu()) if high.any() else None
                    ),
                    "native_high_group_weight_max": (
                        float(high_weight.max().cpu()) if high.any() else None
                    ),
                    "native_trajectory_weight_mean": float(token_weight[valid].mean().cpu()),
                    "native_loss_mass_scale": 1.0,
                }
                loss_type = (
                    "scopd_nogt_max10_group_inverse_jsd_d25_forward_kl"
                    if weighting_mode == "max_kl_fraction_inverse_jsd"
                    else "scopd_nogt_max10_group_softmax_inverse_jsd_d25_forward_kl"
                    if weighting_mode == "max_kl_fraction_softmax_inverse_jsd"
                    else "scopd_nogt_max10_group_lambda_softmax_inverse_jsd_d25_forward_kl"
                )
            elif weighting_mode == TOKEN_TIP_KL_ABS_PROJECTION_MODE:
                teacher_js_tokens = compute_per_token_generalized_jsd(
                    teacher_logits,
                    student_logits.detach(),
                    beta=0.5,
                    temperature=float(
                        get_nested(
                            cfg,
                            "scopd.native_budget_weighting.sensitivity_temperature",
                            1.0,
                        )
                    ),
                    top_k=None,
                    token_clip=None,
                    clip_mode="token",
                    chunk_size=int(
                        get_nested(cfg, "scopd.native_budget_weighting.kl_chunk_size", 32)
                    ),
                ).detach().float()
                teacher_plus_js_tokens = compute_per_token_generalized_jsd(
                    teacher_logits,
                    b_plus_logits,
                    beta=0.5,
                    temperature=float(
                        get_nested(
                            cfg,
                            "scopd.native_budget_weighting.sensitivity_temperature",
                            1.0,
                        )
                    ),
                    top_k=None,
                    token_clip=None,
                    clip_mode="token",
                    chunk_size=int(
                        get_nested(cfg, "scopd.native_budget_weighting.kl_chunk_size", 32)
                    ),
                ).detach().float()
                top_fraction = float(
                    get_nested(cfg, "scopd.native_budget_weighting.top_fraction", 0.20)
                )
                projection_clip_quantile = float(
                    get_nested(
                        cfg,
                        "scopd.native_budget_weighting.projection_clip_quantile",
                        0.98,
                    )
                )
                token_tip_kl_abs_projection_partition = (
                    kl_abs_projection_soft_or_topk_partition(
                        teacher_js_tokens,
                        sensitivity,
                        teacher_plus_js_tokens,
                        per_token_scopd,
                        valid_mask,
                        top_fraction=top_fraction,
                        projection_clip_quantile=projection_clip_quantile,
                        eps=eps,
                    )
                )
                weights = token_tip_kl_abs_projection_partition
                valid = weights.selected_mask
                token_weight = valid.to(dtype=per_token_scopd.dtype).detach()
                scores = weights.soft_or_score[weights.valid_mask]
                selected_scores = weights.soft_or_score[valid]
                unselected = weights.valid_mask & ~valid
                unselected_scores = weights.soft_or_score[unselected]
                projection_values = weights.absolute_projection_mass[weights.valid_mask]
                mode_metrics = {
                    "native_tip_selection": "deterministic_response_topk",
                    "native_tip_selection_detached": True,
                    "native_tip_token_retention_fraction": top_fraction,
                    "native_tip_actual_selected_fraction": float(
                        valid.sum().cpu() / weights.valid_mask.sum().cpu()
                    ),
                    "native_tip_score_formula": "s=kl_hat+absP_hat-kl_hat*absP_hat",
                    "native_tip_projection_definition": "abs((A+B-C)/2)",
                    "native_tip_projection_clip_quantile": projection_clip_quantile,
                    "native_tip_projection_clip_value": float(
                        weights.projection_clip_value.cpu()
                    ),
                    "native_tip_normalization_scope": "valid_tokens_within_each_response",
                    "native_tip_kl_direction": "forward",
                    "native_tip_valid_token_count": int(weights.valid_mask.sum().cpu()),
                    "native_tip_selected_token_count": int(valid.sum().cpu()),
                    "native_tip_kl_degenerate": weights.kl_degenerate,
                    "native_tip_projection_degenerate": weights.projection_degenerate,
                    "native_tip_uniform_fallback": weights.uniform_fallback,
                    "native_tip_normalized_kl_mean": float(
                        weights.normalized_kl[weights.valid_mask].mean().cpu()
                    ),
                    "native_tip_normalized_abs_projection_mean": float(
                        weights.normalized_absolute_projection[weights.valid_mask]
                        .mean()
                        .cpu()
                    ),
                    "native_tip_abs_projection_mean": float(projection_values.mean().cpu()),
                    "native_tip_abs_projection_max": float(projection_values.max().cpu()),
                    "native_tip_soft_or_score_min": float(scores.min().cpu()),
                    "native_tip_soft_or_score_mean": float(scores.mean().cpu()),
                    "native_tip_soft_or_score_max": float(scores.max().cpu()),
                    "native_tip_selected_score_mean": float(selected_scores.mean().cpu()),
                    "native_tip_unselected_score_mean": (
                        float(unselected_scores.mean().cpu()) if unselected.any() else None
                    ),
                    "native_tip_score_threshold": float(selected_scores.min().cpu()),
                    "native_tip_objective": "mean_selected_forward_KL(P_teacher||P_student)",
                    "native_trajectory_weight_mean": float(token_weight[weights.valid_mask].mean().cpu()),
                    "native_loss_mass_scale": 1.0,
                }
                loss_type = "scopd_nogt_tip_kl_abs_projection_topk_forward_kl"
            elif weighting_mode in {*TOKEN_PROJECTION_PARTITION_MODES, TOKEN_RANDOM_DROP_MODE}:
                teacher_js_tokens = compute_per_token_generalized_jsd(
                    teacher_logits,
                    student_logits.detach(),
                    beta=0.5,
                    temperature=float(
                        get_nested(
                            cfg,
                            "scopd.native_budget_weighting.sensitivity_temperature",
                            1.0,
                        )
                    ),
                    top_k=None,
                    token_clip=None,
                    clip_mode="token",
                    chunk_size=int(
                        get_nested(cfg, "scopd.native_budget_weighting.kl_chunk_size", 32)
                    ),
                ).detach().float()
                teacher_plus_js_tokens = compute_per_token_generalized_jsd(
                    teacher_logits,
                    b_plus_logits,
                    beta=0.5,
                    temperature=float(
                        get_nested(
                            cfg,
                            "scopd.native_budget_weighting.sensitivity_temperature",
                            1.0,
                        )
                    ),
                    top_k=None,
                    token_clip=None,
                    clip_mode="token",
                    chunk_size=int(
                        get_nested(cfg, "scopd.native_budget_weighting.kl_chunk_size", 32)
                    ),
                ).detach().float()
                top_fraction = float(
                    get_nested(cfg, "scopd.native_budget_weighting.top_fraction", 0.2)
                )
                min_teacher_kl = float(
                    get_nested(cfg, "scopd.native_budget_weighting.min_teacher_kl", 1e-5)
                )
                if weighting_mode == TOKEN_RANDOM_DROP_MODE:
                    token_random_drop_partition = deterministic_random_token_drop_partition(
                        per_token_scopd,
                        valid_mask,
                        sample_key=f"{sample.sample_id}:{rollout_seed}",
                        seed=int(
                            get_nested(cfg, "scopd.native_budget_weighting.random_drop_seed", 42)
                        ),
                        drop_fraction=top_fraction,
                        min_kl=min_teacher_kl,
                    )
                    eligible = token_random_drop_partition.eligible_mask
                    top = token_random_drop_partition.dropped_mask
                    valid = token_random_drop_partition.selected_mask
                    projection_mass = 0.5 * (
                        teacher_js_tokens + sensitivity - teacher_plus_js_tokens
                    )
                    projection_fraction = projection_mass / (teacher_js_tokens + eps)
                    partition_name = "random_drop20"
                else:
                    token_projection_partition = projection_fraction_token_partition(
                        teacher_js_tokens,
                        sensitivity,
                        teacher_plus_js_tokens,
                        per_token_scopd,
                        valid_mask,
                        top_fraction=top_fraction,
                        min_kl=min_teacher_kl,
                        select=(
                            "top"
                            if weighting_mode == "token_projection_fraction_top20"
                            else "bottom"
                        ),
                        eps=eps,
                    )
                    eligible = token_projection_partition.eligible_mask
                    top = token_projection_partition.top_mask
                    valid = token_projection_partition.selected_mask
                    projection_fraction = token_projection_partition.projection_fraction
                    projection_mass = token_projection_partition.projection_mass
                    partition_name = (
                        "top20" if weighting_mode.endswith("top20") else "bottom80"
                    )
                token_weight = torch.zeros_like(per_token_scopd.detach().float())
                token_weight[valid] = 1.0
                eligible_count = int(eligible.sum().cpu())
                selected_count = int(valid.sum().cpu())
                selected_eligible_count = int((valid & eligible).sum().cpu())
                below_kl_floor = valid_mask & ~eligible
                below_kl_floor_count = int(below_kl_floor.sum().cpu())
                selected_below_kl_floor_count = int(
                    (valid & below_kl_floor).sum().cpu()
                )
                eligible_projection_fraction = projection_fraction[eligible]
                mode_metrics = {
                    "native_token_partition": partition_name,
                    "native_token_partition_top_fraction": top_fraction,
                    "native_token_partition_min_teacher_kl": min_teacher_kl,
                    "native_token_partition_random_seed": (
                        int(get_nested(cfg, "scopd.native_budget_weighting.random_drop_seed", 42))
                        if weighting_mode == TOKEN_RANDOM_DROP_MODE
                        else None
                    ),
                    "native_token_partition_valid_tokens": int(valid_mask.sum().cpu()),
                    "native_token_partition_eligible_tokens": eligible_count,
                    "native_token_partition_below_kl_floor_tokens": below_kl_floor_count,
                    "native_token_partition_below_kl_floor_selected_tokens": (
                        selected_below_kl_floor_count
                    ),
                    "native_token_partition_excluded_low_kl_tokens": (
                        below_kl_floor_count - selected_below_kl_floor_count
                    ),
                    "native_token_partition_top_tokens": int(top.sum().cpu()),
                    "native_token_partition_random_dropped_tokens": (
                        int(top.sum().cpu())
                        if weighting_mode == TOKEN_RANDOM_DROP_MODE
                        else None
                    ),
                    "native_token_partition_selected_tokens": selected_count,
                    "native_token_partition_selected_fraction_of_eligible": (
                        float(selected_eligible_count / eligible_count)
                        if eligible_count
                        else 0.0
                    ),
                    "native_token_partition_selected_fraction_of_valid": (
                        float(selected_count / int(valid_mask.sum().cpu()))
                        if valid_mask.any()
                        else 0.0
                    ),
                    "native_token_partition_degenerate_eligible": eligible_count < 2,
                    "native_token_partition_empty_selected": selected_count == 0,
                    "native_token_projection_fraction_min": (
                        float(eligible_projection_fraction.min().cpu())
                        if eligible_count
                        else None
                    ),
                    "native_token_projection_fraction_mean": (
                        float(eligible_projection_fraction.mean().cpu())
                        if eligible_count
                        else None
                    ),
                    "native_token_projection_fraction_max": (
                        float(eligible_projection_fraction.max().cpu())
                        if eligible_count
                        else None
                    ),
                    "native_trajectory_budget_projection_mass": float(
                        projection_mass[valid_mask].sum().cpu()
                    ),
                    "native_trajectory_teacher_js_mass": float(
                        teacher_js_tokens[valid_mask].sum().cpu()
                    ),
                    "native_trajectory_raw_projection_fraction": float(
                        projection_mass[valid_mask]
                        .sum()
                        .div(teacher_js_tokens[valid_mask].sum().clamp_min(eps))
                        .cpu()
                    ),
                    "native_loss_mass_scale": 1.0,
                }
                loss_type = f"scopd_nogt_{weighting_mode}_forward_kl"
            elif weighting_mode == TOKEN_PROJECTION_MASS_GROUPED_MODE:
                teacher_js_tokens = compute_per_token_generalized_jsd(
                    teacher_logits,
                    student_logits.detach(),
                    beta=0.5,
                    temperature=float(
                        get_nested(
                            cfg,
                            "scopd.native_budget_weighting.sensitivity_temperature",
                            1.0,
                        )
                    ),
                    top_k=None,
                    token_clip=None,
                    clip_mode="token",
                    chunk_size=int(
                        get_nested(cfg, "scopd.native_budget_weighting.kl_chunk_size", 32)
                    ),
                ).detach().float()
                teacher_plus_js_tokens = compute_per_token_generalized_jsd(
                    teacher_logits,
                    b_plus_logits,
                    beta=0.5,
                    temperature=float(
                        get_nested(
                            cfg,
                            "scopd.native_budget_weighting.sensitivity_temperature",
                            1.0,
                        )
                    ),
                    top_k=None,
                    token_clip=None,
                    clip_mode="token",
                    chunk_size=int(
                        get_nested(cfg, "scopd.native_budget_weighting.kl_chunk_size", 32)
                    ),
                ).detach().float()
                top_fraction = float(
                    get_nested(cfg, "scopd.native_budget_weighting.top_fraction", 0.10)
                )
                high_group_lambda = float(
                    get_nested(cfg, "scopd.native_budget_weighting.high_group_lambda", 0.30)
                )
                preserve_loss_mass = bool(
                    get_nested(cfg, "scopd.native_budget_weighting.preserve_loss_mass", False)
                )
                token_projection_mass_group = projection_mass_grouped_weights(
                    teacher_js_tokens,
                    sensitivity,
                    teacher_plus_js_tokens,
                    per_token_scopd,
                    valid_mask,
                    top_fraction=top_fraction,
                    high_group_lambda=high_group_lambda,
                    preserve_loss_mass=preserve_loss_mass,
                    eps=eps,
                )
                weights = token_projection_mass_group
                valid = weights.valid_mask
                token_weight = weights.weight
                high = weights.high_mask
                low = weights.low_mask
                positive_total = weights.positive_projection_mass[valid].sum()
                high_positive = weights.positive_projection_mass[high].sum()
                signed_total = weights.projection_mass[valid].sum()
                high_signed = weights.projection_mass[high].sum()
                raw_grouped_kl = (
                    weights.raw_weight[valid] * per_token_scopd[valid]
                ).mean()
                mode_metrics = {
                    "native_projection_metric": "relu((A+B-C)/2)",
                    "native_group_objective": "lambda_high_mean_plus_one_minus_lambda_low_mean",
                    "native_top_fraction": top_fraction,
                    "native_high_group_lambda": high_group_lambda,
                    "native_low_group_lambda": 1.0 - high_group_lambda,
                    "native_preserve_loss_mass": preserve_loss_mass,
                    "native_high_group_tokens": int(high.sum().cpu()),
                    "native_low_group_tokens": int(low.sum().cpu()),
                    "native_high_group_fraction": float(high.sum().cpu() / valid.sum().cpu()),
                    "native_projection_positive_token_fraction": float(
                        (weights.projection_mass[valid] > 0).float().mean().cpu()
                    ),
                    "native_high_group_positive_projection_mass_share": (
                        float((high_positive / positive_total).cpu())
                        if float(positive_total) > eps
                        else 0.0
                    ),
                    "native_high_group_signed_projection_mass_share": (
                        float((high_signed / signed_total).cpu())
                        if abs(float(signed_total)) > eps
                        else None
                    ),
                    "native_projection_mass_min": float(
                        weights.projection_mass[valid].min().cpu()
                    ),
                    "native_projection_mass_mean": float(
                        weights.projection_mass[valid].mean().cpu()
                    ),
                    "native_projection_mass_max": float(
                        weights.projection_mass[valid].max().cpu()
                    ),
                    "native_high_group_scopd_kl_mean": float(
                        per_token_scopd[high].detach().mean().cpu()
                    ),
                    "native_low_group_scopd_kl_mean": float(
                        per_token_scopd[low].detach().mean().cpu()
                    ),
                    "native_raw_grouped_kl": float(raw_grouped_kl.detach().cpu()),
                    "native_raw_token_weight_min": float(
                        weights.raw_weight[valid].min().cpu()
                    ),
                    "native_raw_token_weight_mean": float(
                        weights.raw_weight[valid].mean().cpu()
                    ),
                    "native_raw_token_weight_max": float(
                        weights.raw_weight[valid].max().cpu()
                    ),
                    "native_loss_mass_scale": float(weights.loss_mass_scale.cpu()),
                    "native_projection_group_degenerate": weights.degenerate,
                }
                loss_type = "scopd_nogt_token_projection_mass_grouped_forward_kl"
            elif weighting_mode == TOKEN_PROJECTION_FRACTION_GROUPED_MODE:
                teacher_js_tokens = compute_per_token_generalized_jsd(
                    teacher_logits,
                    student_logits.detach(),
                    beta=0.5,
                    temperature=float(
                        get_nested(
                            cfg,
                            "scopd.native_budget_weighting.sensitivity_temperature",
                            1.0,
                        )
                    ),
                    top_k=None,
                    token_clip=None,
                    clip_mode="token",
                    chunk_size=int(
                        get_nested(cfg, "scopd.native_budget_weighting.kl_chunk_size", 32)
                    ),
                ).detach().float()
                teacher_plus_js_tokens = compute_per_token_generalized_jsd(
                    teacher_logits,
                    b_plus_logits,
                    beta=0.5,
                    temperature=float(
                        get_nested(
                            cfg,
                            "scopd.native_budget_weighting.sensitivity_temperature",
                            1.0,
                        )
                    ),
                    top_k=None,
                    token_clip=None,
                    clip_mode="token",
                    chunk_size=int(
                        get_nested(cfg, "scopd.native_budget_weighting.kl_chunk_size", 32)
                    ),
                ).detach().float()
                group_fraction = float(
                    get_nested(cfg, "scopd.native_budget_weighting.top_fraction", 0.10)
                )
                selected_group_lambda = float(
                    get_nested(
                        cfg,
                        "scopd.native_budget_weighting.selected_group_lambda",
                        0.30,
                    )
                )
                selection = str(
                    get_nested(cfg, "scopd.native_budget_weighting.selection", "top")
                ).strip().lower()
                min_teacher_kl = float(
                    get_nested(cfg, "scopd.native_budget_weighting.min_teacher_kl", 1e-5)
                )
                preserve_loss_mass = bool(
                    get_nested(cfg, "scopd.native_budget_weighting.preserve_loss_mass", False)
                )
                weights = projection_fraction_grouped_weights(
                    teacher_js_tokens,
                    sensitivity,
                    teacher_plus_js_tokens,
                    per_token_scopd,
                    valid_mask,
                    group_fraction=group_fraction,
                    selected_group_lambda=selected_group_lambda,
                    selection=selection,
                    min_kl=min_teacher_kl,
                    preserve_loss_mass=preserve_loss_mass,
                    eps=eps,
                )
                valid = weights.valid_mask
                token_weight = weights.weight
                selected = weights.selected_mask
                complement = weights.complement_mask
                raw_grouped_kl = (
                    weights.raw_weight[valid] * per_token_scopd[valid]
                ).mean()
                selected_teacher_kl = per_token_scopd.detach().float()[selected]
                selected_teacher_kl_ge_1e5 = selected_teacher_kl >= 1e-5
                mode_metrics = {
                    "native_projection_metric": "((A+B-C)/2)/A",
                    "native_group_objective": (
                        "lambda_selected_mean_plus_one_minus_lambda_complement_mean"
                    ),
                    "native_f_group_selection": selection,
                    "native_top_fraction": group_fraction,
                    "native_selected_group_lambda": selected_group_lambda,
                    "native_complement_group_lambda": 1.0 - selected_group_lambda,
                    "native_min_teacher_kl": min_teacher_kl,
                    "native_preserve_loss_mass": preserve_loss_mass,
                    "native_eligible_tokens": int(weights.eligible_mask.sum().cpu()),
                    "native_selected_group_tokens": int(selected.sum().cpu()),
                    "native_selected_group_teacher_kl_ge_1e5_tokens": int(
                        selected_teacher_kl_ge_1e5.sum().cpu()
                    ),
                    "native_selected_group_teacher_kl_lt_1e5_tokens": int(
                        (~selected_teacher_kl_ge_1e5).sum().cpu()
                    ),
                    "native_complement_group_tokens": int(complement.sum().cpu()),
                    "native_selected_group_fraction": float(
                        selected.sum().cpu() / valid.sum().cpu()
                    ),
                    "native_projection_fraction_min": float(
                        weights.projection_fraction[valid].min().cpu()
                    ),
                    "native_projection_fraction_mean": float(
                        weights.projection_fraction[valid].mean().cpu()
                    ),
                    "native_projection_fraction_max": float(
                        weights.projection_fraction[valid].max().cpu()
                    ),
                    "native_selected_group_scopd_kl_mean": (
                        float(per_token_scopd[selected].detach().mean().cpu())
                        if selected.any()
                        else None
                    ),
                    "native_complement_group_scopd_kl_mean": (
                        float(per_token_scopd[complement].detach().mean().cpu())
                        if complement.any()
                        else None
                    ),
                    "native_raw_grouped_kl": float(raw_grouped_kl.detach().cpu()),
                    "native_raw_token_weight_min": float(
                        weights.raw_weight[valid].min().cpu()
                    ),
                    "native_raw_token_weight_mean": float(
                        weights.raw_weight[valid].mean().cpu()
                    ),
                    "native_raw_token_weight_max": float(
                        weights.raw_weight[valid].max().cpu()
                    ),
                    "native_loss_mass_scale": float(weights.loss_mass_scale.cpu()),
                    "native_projection_group_degenerate": weights.degenerate,
                }
                loss_type = "scopd_nogt_token_projection_fraction_grouped_forward_kl"
            elif weighting_mode == TOKEN_BUDGET_JSD_ABS_F_GROUPED_MODE:
                teacher_js_tokens = compute_per_token_generalized_jsd(
                    teacher_logits,
                    student_logits.detach(),
                    beta=0.5,
                    temperature=float(
                        get_nested(
                            cfg,
                            "scopd.native_budget_weighting.sensitivity_temperature",
                            1.0,
                        )
                    ),
                    top_k=None,
                    token_clip=None,
                    clip_mode="token",
                    chunk_size=int(
                        get_nested(cfg, "scopd.native_budget_weighting.kl_chunk_size", 32)
                    ),
                ).detach().float()
                teacher_plus_js_tokens = compute_per_token_generalized_jsd(
                    teacher_logits,
                    b_plus_logits,
                    beta=0.5,
                    temperature=float(
                        get_nested(
                            cfg,
                            "scopd.native_budget_weighting.sensitivity_temperature",
                            1.0,
                        )
                    ),
                    top_k=None,
                    token_clip=None,
                    clip_mode="token",
                    chunk_size=int(
                        get_nested(cfg, "scopd.native_budget_weighting.kl_chunk_size", 32)
                    ),
                ).detach().float()
                budget_top_fraction = float(
                    get_nested(
                        cfg,
                        "scopd.native_budget_weighting.budget_top_fraction",
                        0.50,
                    )
                )
                within_budget_fraction = float(
                    get_nested(
                        cfg,
                        "scopd.native_budget_weighting.within_budget_fraction",
                        0.50,
                    )
                )
                selected_group_lambda = float(
                    get_nested(
                        cfg,
                        "scopd.native_budget_weighting.selected_group_lambda",
                        0.50,
                    )
                )
                selection = str(
                    get_nested(cfg, "scopd.native_budget_weighting.selection", "top")
                ).strip().lower()
                preserve_loss_mass = bool(
                    get_nested(cfg, "scopd.native_budget_weighting.preserve_loss_mass", False)
                )
                weights = budget_jsd_conditioned_abs_projection_grouped_weights(
                    teacher_js_tokens,
                    sensitivity,
                    teacher_plus_js_tokens,
                    per_token_scopd,
                    valid_mask,
                    budget_top_fraction=budget_top_fraction,
                    within_budget_fraction=within_budget_fraction,
                    selected_group_lambda=selected_group_lambda,
                    selection=selection,
                    preserve_loss_mass=preserve_loss_mass,
                    eps=eps,
                )
                valid = weights.valid_mask
                token_weight = weights.weight
                budget_sensitive = weights.budget_sensitive_mask
                budget_insensitive = valid & ~budget_sensitive
                selected = weights.selected_mask
                budget_unselected = budget_sensitive & ~selected
                complement = weights.complement_mask
                raw_grouped_kl = (
                    weights.raw_weight[valid] * per_token_scopd[valid]
                ).mean()
                mode_metrics = {
                    "native_budget_prefilter_metric": "B=JSD(p_b,p_b_plus)",
                    "native_projection_metric": "abs(((A+B-C)/2)/A)",
                    "native_group_objective": (
                        "lambda_selected_mean_plus_one_minus_lambda_complement_mean"
                    ),
                    "native_abs_f_group_selection": selection,
                    "native_budget_top_fraction": budget_top_fraction,
                    "native_within_budget_fraction": within_budget_fraction,
                    "native_selected_group_lambda": selected_group_lambda,
                    "native_complement_group_lambda": 1.0 - selected_group_lambda,
                    "native_preserve_loss_mass": preserve_loss_mass,
                    "native_budget_sensitive_tokens": int(budget_sensitive.sum().cpu()),
                    "native_budget_insensitive_tokens": int(budget_insensitive.sum().cpu()),
                    "native_selected_group_tokens": int(selected.sum().cpu()),
                    "native_budget_unselected_tokens": int(budget_unselected.sum().cpu()),
                    "native_complement_group_tokens": int(complement.sum().cpu()),
                    "native_selected_group_fraction": float(
                        selected.sum().cpu() / valid.sum().cpu()
                    ),
                    "native_budget_sensitive_b_jsd_min": float(
                        weights.budget_js[budget_sensitive].min().cpu()
                    ),
                    "native_budget_sensitive_b_jsd_mean": float(
                        weights.budget_js[budget_sensitive].mean().cpu()
                    ),
                    "native_budget_insensitive_b_jsd_max": (
                        float(weights.budget_js[budget_insensitive].max().cpu())
                        if budget_insensitive.any()
                        else None
                    ),
                    "native_selected_abs_f_min": float(
                        weights.abs_projection_fraction[selected].min().cpu()
                    ),
                    "native_selected_abs_f_mean": float(
                        weights.abs_projection_fraction[selected].mean().cpu()
                    ),
                    "native_selected_abs_f_max": float(
                        weights.abs_projection_fraction[selected].max().cpu()
                    ),
                    "native_budget_unselected_abs_f_min": (
                        float(weights.abs_projection_fraction[budget_unselected].min().cpu())
                        if budget_unselected.any()
                        else None
                    ),
                    "native_budget_unselected_abs_f_mean": (
                        float(weights.abs_projection_fraction[budget_unselected].mean().cpu())
                        if budget_unselected.any()
                        else None
                    ),
                    "native_budget_unselected_abs_f_max": (
                        float(weights.abs_projection_fraction[budget_unselected].max().cpu())
                        if budget_unselected.any()
                        else None
                    ),
                    "native_selected_group_scopd_kl_mean": float(
                        per_token_scopd[selected].detach().mean().cpu()
                    ),
                    "native_complement_group_scopd_kl_mean": (
                        float(per_token_scopd[complement].detach().mean().cpu())
                        if complement.any()
                        else None
                    ),
                    "native_raw_grouped_kl": float(raw_grouped_kl.detach().cpu()),
                    "native_raw_token_weight_min": float(
                        weights.raw_weight[valid].min().cpu()
                    ),
                    "native_raw_token_weight_mean": float(
                        weights.raw_weight[valid].mean().cpu()
                    ),
                    "native_raw_token_weight_max": float(
                        weights.raw_weight[valid].max().cpu()
                    ),
                    "native_loss_mass_scale": float(weights.loss_mass_scale.cpu()),
                    "native_projection_group_degenerate": weights.degenerate,
                }
                loss_type = "scopd_nogt_token_budget_jsd_abs_f_grouped_forward_kl"
            elif weighting_mode in {
                TOKEN_PROJECTION_BOTTOM_DROP_MODE,
                TOKEN_PROJECTION_TOP_DROP_MODE,
            }:
                teacher_js_tokens = compute_per_token_generalized_jsd(
                    teacher_logits,
                    student_logits.detach(),
                    beta=0.5,
                    temperature=float(
                        get_nested(
                            cfg,
                            "scopd.native_budget_weighting.sensitivity_temperature",
                            1.0,
                        )
                    ),
                    top_k=None,
                    token_clip=None,
                    clip_mode="token",
                    chunk_size=int(
                        get_nested(cfg, "scopd.native_budget_weighting.kl_chunk_size", 32)
                    ),
                ).detach().float()
                teacher_plus_js_tokens = compute_per_token_generalized_jsd(
                    teacher_logits,
                    b_plus_logits,
                    beta=0.5,
                    temperature=float(
                        get_nested(
                            cfg,
                            "scopd.native_budget_weighting.sensitivity_temperature",
                            1.0,
                        )
                    ),
                    top_k=None,
                    token_clip=None,
                    clip_mode="token",
                    chunk_size=int(
                        get_nested(cfg, "scopd.native_budget_weighting.kl_chunk_size", 32)
                    ),
                ).detach().float()
                drop_fraction = float(
                    get_nested(cfg, "scopd.native_budget_weighting.drop_fraction", 0.10)
                )
                drop_selection = (
                    "bottom"
                    if weighting_mode == TOKEN_PROJECTION_BOTTOM_DROP_MODE
                    else "top"
                )
                drop_partition_fn = (
                    projection_fraction_bottom_drop_partition
                    if drop_selection == "bottom"
                    else projection_fraction_top_drop_partition
                )
                f_drop = drop_partition_fn(
                    teacher_js_tokens,
                    sensitivity,
                    teacher_plus_js_tokens,
                    valid_mask,
                    drop_fraction=drop_fraction,
                    eps=eps,
                )
                valid = f_drop.kept_mask
                dropped = f_drop.dropped_mask
                token_weight = torch.zeros_like(per_token_scopd.detach().float())
                token_weight[valid] = 1.0
                dropped_kl = per_token_scopd.detach().float()[dropped]
                kept_kl = per_token_scopd.detach().float()[valid]
                full_kl_mass = per_token_scopd.detach().float()[valid_mask].sum()
                dropped_kl_mass = dropped_kl.sum()
                mode_metrics = {
                    "native_projection_metric": "((A+B-C)/2)/(A+eps)",
                    "native_token_partition": f"drop_{drop_selection}_F_keep_complement",
                    "native_token_partition_drop_fraction": drop_fraction,
                    "native_token_partition_valid_tokens": int(valid_mask.sum().cpu()),
                    "native_token_partition_dropped_tokens": int(dropped.sum().cpu()),
                    "native_token_partition_kept_tokens": int(valid.sum().cpu()),
                    "native_token_partition_actual_dropped_fraction": float(
                        dropped.float().sum().cpu() / valid_mask.sum().cpu()
                    ),
                    "native_token_partition_uses_kl_floor": False,
                    "native_token_partition_dropped_f_min": float(
                        f_drop.projection_fraction[dropped].min().cpu()
                    ) if dropped.any() else None,
                    "native_token_partition_dropped_f_mean": float(
                        f_drop.projection_fraction[dropped].mean().cpu()
                    ) if dropped.any() else None,
                    "native_token_partition_dropped_f_max": float(
                        f_drop.projection_fraction[dropped].max().cpu()
                    ) if dropped.any() else None,
                    "native_token_partition_kept_f_mean": float(
                        f_drop.projection_fraction[valid].mean().cpu()
                    ),
                    "native_token_partition_kept_f_min": float(
                        f_drop.projection_fraction[valid].min().cpu()
                    ),
                    "native_token_partition_kept_f_max": float(
                        f_drop.projection_fraction[valid].max().cpu()
                    ),
                    "native_token_partition_dropped_scopd_kl_mean": (
                        float(dropped_kl.mean().cpu()) if dropped.any() else None
                    ),
                    "native_token_partition_kept_scopd_kl_mean": float(kept_kl.mean().cpu()),
                    "native_token_partition_dropped_scopd_kl_mass_fraction": float(
                        (dropped_kl_mass / full_kl_mass.clamp_min(1e-8)).cpu()
                    ),
                    "native_loss_mass_scale": 1.0,
                }
                loss_type = (
                    f"scopd_nogt_token_projection_fraction_drop_{drop_selection}_forward_kl"
                )
            elif weighting_mode == "trajectory_probe":
                student_budget_jsd = compute_generalized_jsd(
                    b_plus_logits,
                    student_logits.detach(),
                    beta=0.5,
                    temperature=float(
                        get_nested(cfg, "scopd.native_budget_weighting.sensitivity_temperature", 1.0)
                    ),
                    top_k=None,
                    token_clip=None,
                    clip_mode="token",
                    chunk_size=int(
                        get_nested(cfg, "scopd.native_budget_weighting.kl_chunk_size", 32)
                    ),
                ).detach().float()
                trajectory_mode = str(
                    get_nested(cfg, "scopd.trajectory_weighting.mode", "")
                ).strip().lower()
                teachability_metrics: dict[str, float] = {}
                if trajectory_mode in COUNTERFACTUAL_TEACHABILITY_MODES:
                    teacher_student_jsd_b = compute_generalized_jsd(
                        teacher_logits,
                        student_logits.detach(),
                        beta=0.5,
                        temperature=float(
                            get_nested(
                                cfg,
                                "scopd.native_budget_weighting.sensitivity_temperature",
                                1.0,
                            )
                        ),
                        top_k=None,
                        token_clip=None,
                        clip_mode="token",
                        chunk_size=int(
                            get_nested(cfg, "scopd.native_budget_weighting.kl_chunk_size", 32)
                        ),
                    ).detach().float()
                    teacher_student_jsd_b_plus = compute_generalized_jsd(
                        teacher_logits,
                        b_plus_logits,
                        beta=0.5,
                        temperature=float(
                            get_nested(
                                cfg,
                                "scopd.native_budget_weighting.sensitivity_temperature",
                                1.0,
                            )
                        ),
                        top_k=None,
                        token_clip=None,
                        clip_mode="token",
                        chunk_size=int(
                            get_nested(cfg, "scopd.native_budget_weighting.kl_chunk_size", 32)
                        ),
                    ).detach().float()
                    budget_projection_mass = 0.5 * (
                        teacher_student_jsd_b
                        + student_budget_jsd
                        - teacher_student_jsd_b_plus
                    )
                    budget_explained_fraction = (
                        budget_projection_mass
                        / (teacher_student_jsd_b + eps)
                    ).clamp(0.0, 1.0)
                    teachability_metrics = {
                        "native_teacher_student_jsd_b_mean": float(
                            teacher_student_jsd_b.cpu()
                        ),
                        "native_teacher_student_jsd_b_plus_mean": float(
                            teacher_student_jsd_b_plus.cpu()
                        ),
                        "native_trajectory_budget_explained_fraction": float(
                            budget_explained_fraction.cpu()
                        ),
                        "native_trajectory_budget_projection_mass": float(
                            budget_projection_mass.cpu()
                        ),
                        "native_trajectory_teacher_js_mass": float(
                            teacher_student_jsd_b.cpu()
                        ),
                    }
                per_token_b_plus_teacher_gap = compute_per_token_kl(
                    teacher_logits,
                    b_plus_logits,
                    temperature=scopd_temperature,
                    chunk_size=int(get_nested(cfg, "scopd.native_budget_weighting.kl_chunk_size", 32)),
                ).detach().float()
                gap_b = per_token_scopd.detach().float()
                valid = valid_mask
                token_weight = torch.zeros_like(gap_b)
                token_weight[valid] = 1.0
                positive_closure = (gap_b - per_token_b_plus_teacher_gap).clamp_min(0.0)
                gap_mass = gap_b[valid].sum()
                closure_mass = positive_closure[valid].sum() / gap_mass.clamp_min(eps)
                nonnegative_gap = gap_b[valid].clamp_min(0.0)
                nonnegative_sensitivity = sensitivity[valid].detach().float().clamp_min(0.0)
                local_relative_sensitivity = nonnegative_sensitivity / (
                    nonnegative_gap + nonnegative_sensitivity + eps
                )
                trajectory_teacher_gap_change = (
                    gap_b[valid].mean() - per_token_b_plus_teacher_gap[valid].mean()
                )
                trajectory_teacher_gap_sensitivity = trajectory_teacher_gap_change.abs()
                trajectory_mass_robustness = teacher_gap_mass_robustness(
                    gap_b,
                    sensitivity,
                    valid_mask,
                    eps=eps,
                )
                mode_metrics = {
                    "native_teacher_gap_b_mean": float(gap_b[valid].mean().cpu()),
                    "native_student_budget_jsd_mean": float(student_budget_jsd.cpu()),
                    "native_teacher_gap_b_plus_mean": float(
                        per_token_b_plus_teacher_gap[valid].mean().cpu()
                    ),
                    "native_trajectory_teacher_gap_change": float(
                        trajectory_teacher_gap_change.cpu()
                    ),
                    "native_trajectory_teacher_gap_sensitivity": float(
                        trajectory_teacher_gap_sensitivity.cpu()
                    ),
                    "native_trajectory_budget_closure_mass": float(closure_mass.cpu()),
                    "native_trajectory_positive_closure_mean": float(
                        positive_closure[valid].mean().cpu()
                    ),
                    "native_trajectory_mass_robustness": float(
                        trajectory_mass_robustness.clamp(0.0, 1.0).cpu()
                    ),
                    "native_local_relative_sensitivity_mean": float(
                        local_relative_sensitivity.mean().cpu()
                    ),
                    "native_trajectory_scalar_objective": "compute_forward_kl_exact",
                    "native_loss_mass_scale": 1.0,
                    **teachability_metrics,
                }
                loss_type = "scopd_nogt_native_budget_trajectory_probe_forward_kl"
            elif weighting_mode == "teacher_gap_persistence":
                per_token_b_plus_teacher_gap = compute_per_token_kl(
                    teacher_logits,
                    b_plus_logits,
                    temperature=scopd_temperature,
                    chunk_size=int(get_nested(cfg, "scopd.native_budget_weighting.kl_chunk_size", 32)),
                )
                weights = teacher_gap_persistence_weights(
                    per_token_scopd,
                    per_token_b_plus_teacher_gap,
                    valid_mask,
                    alpha=float(get_nested(cfg, "scopd.native_budget_weighting.alpha", 0.5)),
                    eps=eps,
                )
                valid = weights.valid_mask
                token_weight = weights.weight
                gap_b = weights.teacher_gap_b[valid]
                gap_b_plus = weights.teacher_gap_b_plus[valid]
                mode_metrics = {
                    "native_teacher_gap_b_mean": float(gap_b.mean().cpu()),
                    "native_teacher_gap_b_median": float(torch.quantile(gap_b, 0.5).cpu()),
                    "native_teacher_gap_b_plus_mean": float(gap_b_plus.mean().cpu()),
                    "native_teacher_gap_b_plus_median": float(torch.quantile(gap_b_plus, 0.5).cpu()),
                    "native_teacher_gap_tau": float(weights.tau_teacher_gap.cpu()),
                    "native_budget_rescue_mean": float(weights.rescue_fraction[valid].mean().cpu()),
                    "native_teacher_gap_persistence_mean": float(weights.persistence[valid].mean().cpu()),
                    "native_teacher_gap_confidence_mean": float(weights.confidence[valid].mean().cpu()),
                    "native_teacher_gap_priority_mean": float(weights.priority[valid].mean().cpu()),
                    "native_teacher_gap_alpha": float(
                        get_nested(cfg, "scopd.native_budget_weighting.alpha", 0.5)
                    ),
                    "native_loss_mass_scale": float(weights.loss_mass_scale.cpu()),
                }
                loss_type = "scopd_nogt_native_budget_teacher_gap_persistence_weighted_forward_kl"
            elif weighting_mode == "symmetric_teacher_gap_stability":
                per_token_b_plus_teacher_gap = compute_per_token_kl(
                    teacher_logits,
                    b_plus_logits,
                    temperature=scopd_temperature,
                    chunk_size=int(get_nested(cfg, "scopd.native_budget_weighting.kl_chunk_size", 32)),
                )
                weights = symmetric_teacher_gap_stability_weights(
                    per_token_scopd,
                    per_token_b_plus_teacher_gap,
                    valid_mask,
                    alpha=float(get_nested(cfg, "scopd.native_budget_weighting.alpha", 0.25)),
                    eps=eps,
                )
                valid = weights.valid_mask
                token_weight = weights.weight
                mode_metrics = {
                    "native_teacher_gap_b_mean": float(weights.teacher_gap_b[valid].mean().cpu()),
                    "native_teacher_gap_b_plus_mean": float(
                        weights.teacher_gap_b_plus[valid].mean().cpu()
                    ),
                    "native_signed_budget_change_mean": float(
                        weights.signed_budget_change[valid].mean().cpu()
                    ),
                    "native_normalized_budget_change_mean": float(
                        weights.normalized_budget_change[valid].mean().cpu()
                    ),
                    "native_normalized_budget_change_median": float(
                        torch.quantile(weights.normalized_budget_change[valid], 0.5).cpu()
                    ),
                    "native_normalized_budget_change_max": float(
                        weights.normalized_budget_change[valid].max().cpu()
                    ),
                    "native_budget_robustness_mean": float(weights.robustness[valid].mean().cpu()),
                    "native_teacher_gap_stability_alpha": float(
                        get_nested(cfg, "scopd.native_budget_weighting.alpha", 0.25)
                    ),
                    "native_loss_mass_scale": float(weights.loss_mass_scale.cpu()),
                    "native_token_scalar_objective": "compute_forward_kl_plus_zero_value_gradient_redistribution",
                }
                loss_type = "scopd_nogt_symmetric_teacher_gap_stability_forward_kl"
            elif weighting_mode == "counterfactual_rescue_amplification":
                per_token_b_plus_teacher_gap = compute_per_token_kl(
                    teacher_logits,
                    b_plus_logits,
                    temperature=scopd_temperature,
                    chunk_size=int(get_nested(cfg, "scopd.native_budget_weighting.kl_chunk_size", 32)),
                )
                weights = counterfactual_rescue_amplification_weights(
                    per_token_scopd,
                    per_token_b_plus_teacher_gap,
                    valid_mask,
                    alpha=float(get_nested(cfg, "scopd.native_budget_weighting.alpha", 0.5)),
                    eps=eps,
                )
                valid = weights.valid_mask
                token_weight = weights.weight
                mode_metrics = {
                    "native_teacher_gap_b_mean": float(weights.teacher_gap_b[valid].mean().cpu()),
                    "native_teacher_gap_b_plus_mean": float(
                        weights.teacher_gap_b_plus[valid].mean().cpu()
                    ),
                    "native_counterfactual_rescue_mean": float(
                        weights.rescue_fraction[valid].mean().cpu()
                    ),
                    "native_counterfactual_rescue_alpha": float(
                        get_nested(cfg, "scopd.native_budget_weighting.alpha", 0.5)
                    ),
                    "native_loss_mass_scale": float(weights.loss_mass_scale.cpu()),
                }
                loss_type = (
                    "scopd_nogt_native_budget_counterfactual_rescue_amplification_forward_kl"
                )
            elif weighting_mode in {
                "native_budget_rescue_grouped",
                "counterfactual_teachability_grouped",
                "teacher_gap_grouped_control",
            }:
                per_token_b_plus_teacher_gap = compute_per_token_kl(
                    teacher_logits,
                    b_plus_logits,
                    temperature=scopd_temperature,
                    chunk_size=int(
                        get_nested(cfg, "scopd.native_budget_weighting.kl_chunk_size", 32)
                    ),
                )
                if weighting_mode == "native_budget_rescue_grouped":
                    ranking_signal = (
                        per_token_scopd.detach().float()
                        - per_token_b_plus_teacher_gap.detach().float()
                    ).clamp_min(0.0)
                    rescue_fraction = (
                        ranking_signal
                        / per_token_scopd.detach().float().clamp_min(eps)
                    ).clamp(0.0, 1.0)
                elif weighting_mode == "counterfactual_teachability_grouped":
                    modulation = counterfactual_teachability_modulation_weights(
                        per_token_scopd,
                        per_token_b_plus_teacher_gap,
                        valid_mask,
                        alpha=0.0,
                        rescue_modulation=float(
                            get_nested(
                                cfg,
                                "scopd.native_budget_weighting.rescue_modulation",
                                0.1,
                            )
                        ),
                        eps=eps,
                    )
                    ranking_signal = modulation.priority
                    rescue_fraction = modulation.rescue_fraction
                else:
                    ranking_signal = per_token_scopd.detach().float()
                    rescue_fraction = torch.zeros_like(ranking_signal)
                weights = grouped_kl_mass_weights(
                    per_token_scopd,
                    ranking_signal,
                    valid_mask,
                    top_fraction=float(
                        get_nested(cfg, "scopd.native_budget_weighting.top_fraction", 0.2)
                    ),
                    high_group_mass=float(
                        get_nested(cfg, "scopd.native_budget_weighting.high_group_mass", 0.5)
                    ),
                    eps=eps,
                )
                valid = weights.valid_mask
                token_weight = weights.weight
                mode_metrics = {
                    "native_teacher_gap_b_mean": float(per_token_scopd[valid].mean().cpu()),
                    "native_teacher_gap_b_plus_mean": float(
                        per_token_b_plus_teacher_gap[valid].mean().cpu()
                    ),
                    "native_group_ranking_signal_mean": float(
                        weights.ranking_signal[valid].mean().cpu()
                    ),
                    "native_counterfactual_rescue_mean": float(
                        rescue_fraction[valid].mean().cpu()
                    ),
                    "native_group_high_fraction": float(
                        weights.high_group_mask[valid].float().mean().cpu()
                    ),
                    "native_group_top_fraction": float(
                        get_nested(cfg, "scopd.native_budget_weighting.top_fraction", 0.2)
                    ),
                    "native_group_high_mass": float(
                        get_nested(cfg, "scopd.native_budget_weighting.high_group_mass", 0.5)
                    ),
                    "native_loss_mass_scale": float(weights.loss_mass_scale.cpu()),
                }
                loss_type = f"scopd_nogt_{weighting_mode}_forward_kl"
            elif weighting_mode == "counterfactual_teachability_mixture":
                per_token_b_plus_teacher_gap = compute_per_token_kl(
                    teacher_logits,
                    b_plus_logits,
                    temperature=scopd_temperature,
                    chunk_size=int(get_nested(cfg, "scopd.native_budget_weighting.kl_chunk_size", 32)),
                )
                weights = counterfactual_teachability_mixture_weights(
                    per_token_scopd,
                    per_token_b_plus_teacher_gap,
                    valid_mask,
                    alpha=float(get_nested(cfg, "scopd.native_budget_weighting.alpha", 0.5)),
                    rescue_mix=float(
                        get_nested(cfg, "scopd.native_budget_weighting.rescue_mix", 0.1)
                    ),
                    eps=eps,
                )
                valid = weights.valid_mask
                token_weight = weights.weight
                mode_metrics = {
                    "native_teacher_gap_b_mean": float(weights.teacher_gap_b[valid].mean().cpu()),
                    "native_teacher_gap_b_plus_mean": float(
                        weights.teacher_gap_b_plus[valid].mean().cpu()
                    ),
                    "native_teacher_gap_rank_mean": float(
                        weights.teacher_gap_rank[valid].mean().cpu()
                    ),
                    "native_counterfactual_rescue_mean": float(
                        weights.rescue_fraction[valid].mean().cpu()
                    ),
                    "native_counterfactual_teachability_priority_mean": float(
                        weights.priority[valid].mean().cpu()
                    ),
                    "native_counterfactual_teachability_alpha": float(
                        get_nested(cfg, "scopd.native_budget_weighting.alpha", 0.5)
                    ),
                    "native_counterfactual_rescue_mix": float(
                        get_nested(cfg, "scopd.native_budget_weighting.rescue_mix", 0.1)
                    ),
                    "native_loss_mass_scale": float(weights.loss_mass_scale.cpu()),
                }
                loss_type = (
                    "scopd_nogt_native_budget_counterfactual_teachability_mixture_forward_kl"
                )
            elif weighting_mode == "counterfactual_teachability_modulation":
                per_token_b_plus_teacher_gap = compute_per_token_kl(
                    teacher_logits,
                    b_plus_logits,
                    temperature=scopd_temperature,
                    chunk_size=int(get_nested(cfg, "scopd.native_budget_weighting.kl_chunk_size", 32)),
                )
                weights = counterfactual_teachability_modulation_weights(
                    per_token_scopd,
                    per_token_b_plus_teacher_gap,
                    valid_mask,
                    alpha=float(get_nested(cfg, "scopd.native_budget_weighting.alpha", 0.5)),
                    rescue_modulation=float(
                        get_nested(
                            cfg,
                            "scopd.native_budget_weighting.rescue_modulation",
                            0.1,
                        )
                    ),
                    eps=eps,
                )
                valid = weights.valid_mask
                token_weight = weights.weight
                mode_metrics = {
                    "native_teacher_gap_b_mean": float(weights.teacher_gap_b[valid].mean().cpu()),
                    "native_teacher_gap_b_plus_mean": float(
                        weights.teacher_gap_b_plus[valid].mean().cpu()
                    ),
                    "native_teacher_gap_rank_mean": float(
                        weights.teacher_gap_rank[valid].mean().cpu()
                    ),
                    "native_counterfactual_rescue_mean": float(
                        weights.rescue_fraction[valid].mean().cpu()
                    ),
                    "native_counterfactual_teachability_priority_mean": float(
                        weights.priority[valid].mean().cpu()
                    ),
                    "native_counterfactual_teachability_alpha": float(
                        get_nested(cfg, "scopd.native_budget_weighting.alpha", 0.5)
                    ),
                    "native_counterfactual_rescue_modulation": float(
                        get_nested(
                            cfg,
                            "scopd.native_budget_weighting.rescue_modulation",
                            0.1,
                        )
                    ),
                    "native_loss_mass_scale": float(weights.loss_mass_scale.cpu()),
                }
                loss_type = (
                    "scopd_nogt_native_budget_counterfactual_teachability_modulation_forward_kl"
                )
            elif weighting_mode == "conditional_rescue_residual":
                per_token_b_plus_teacher_gap = compute_per_token_kl(
                    teacher_logits,
                    b_plus_logits,
                    temperature=scopd_temperature,
                    chunk_size=int(get_nested(cfg, "scopd.native_budget_weighting.kl_chunk_size", 32)),
                )
                weights = conditional_rescue_residual_weights(
                    per_token_scopd,
                    per_token_b_plus_teacher_gap,
                    valid_mask,
                    alpha=float(get_nested(cfg, "scopd.native_budget_weighting.alpha", 0.1)),
                    difficulty_bins=int(
                        get_nested(cfg, "scopd.native_budget_weighting.difficulty_bins", 5)
                    ),
                    eps=eps,
                )
                valid = weights.valid_mask
                token_weight = weights.weight
                mode_metrics = {
                    "native_teacher_gap_b_mean": float(weights.teacher_gap_b[valid].mean().cpu()),
                    "native_teacher_gap_b_plus_mean": float(
                        weights.teacher_gap_b_plus[valid].mean().cpu()
                    ),
                    "native_teacher_gap_rank_mean": float(
                        weights.teacher_gap_rank[valid].mean().cpu()
                    ),
                    "native_counterfactual_rescue_mean": float(
                        weights.rescue_fraction[valid].mean().cpu()
                    ),
                    "native_expected_rescue_mean": float(
                        weights.expected_rescue[valid].mean().cpu()
                    ),
                    "native_rescue_residual_mean": float(
                        weights.rescue_residual[valid].mean().cpu()
                    ),
                    "native_rescue_residual_abs_mean": float(
                        weights.rescue_residual[valid].abs().mean().cpu()
                    ),
                    "native_conditional_rescue_alpha": float(
                        get_nested(cfg, "scopd.native_budget_weighting.alpha", 0.1)
                    ),
                    "native_difficulty_bins": int(
                        get_nested(cfg, "scopd.native_budget_weighting.difficulty_bins", 5)
                    ),
                    "native_loss_mass_scale": float(weights.loss_mass_scale.cpu()),
                }
                loss_type = (
                    "scopd_nogt_native_budget_conditional_rescue_residual_forward_kl"
                )
            elif weighting_mode == "budget_consistent_rank":
                per_token_b_plus_teacher_gap = compute_per_token_kl(
                    teacher_logits,
                    b_plus_logits,
                    temperature=scopd_temperature,
                    chunk_size=int(get_nested(cfg, "scopd.native_budget_weighting.kl_chunk_size", 32)),
                )
                weights = budget_consistent_rank_weights(
                    per_token_scopd,
                    per_token_b_plus_teacher_gap,
                    valid_mask,
                    alpha=float(get_nested(cfg, "scopd.native_budget_weighting.alpha", 1.0)),
                    eps=eps,
                )
                valid = weights.valid_mask
                token_weight = weights.weight
                mode_metrics = {
                    "native_teacher_gap_b_mean": float(weights.teacher_gap_b[valid].mean().cpu()),
                    "native_teacher_gap_b_plus_mean": float(weights.teacher_gap_b_plus[valid].mean().cpu()),
                    "native_persistent_gap_mean": float(weights.persistent_gap[valid].mean().cpu()),
                    "native_persistent_rank_mean": float(weights.persistent_rank[valid].mean().cpu()),
                    "native_teacher_gap_alpha": float(
                        get_nested(cfg, "scopd.native_budget_weighting.alpha", 1.0)
                    ),
                    "native_loss_mass_scale": float(weights.loss_mass_scale.cpu()),
                }
                loss_type = "scopd_nogt_native_budget_consistent_rank_weighted_forward_kl"
            elif weighting_mode == "budget_residual_hardness":
                per_token_b_plus_teacher_gap = compute_per_token_kl(
                    teacher_logits,
                    b_plus_logits,
                    temperature=scopd_temperature,
                    chunk_size=int(get_nested(cfg, "scopd.native_budget_weighting.kl_chunk_size", 32)),
                )
                weights = budget_residual_hardness_weights(
                    per_token_scopd,
                    per_token_b_plus_teacher_gap,
                    valid_mask,
                    alpha=float(get_nested(cfg, "scopd.native_budget_weighting.alpha", 1.0)),
                    persistence_mix=float(
                        get_nested(cfg, "scopd.native_budget_weighting.persistence_mix", 0.1)
                    ),
                    eps=eps,
                )
                valid = weights.valid_mask
                token_weight = weights.weight
                mode_metrics = {
                    "native_teacher_gap_b_mean": float(weights.teacher_gap_b[valid].mean().cpu()),
                    "native_teacher_gap_b_plus_mean": float(weights.teacher_gap_b_plus[valid].mean().cpu()),
                    "native_persistent_gap_mean": float(weights.persistent_gap[valid].mean().cpu()),
                    "native_teacher_gap_rank_mean": float(weights.teacher_gap_rank[valid].mean().cpu()),
                    "native_persistent_rank_mean": float(weights.persistent_rank[valid].mean().cpu()),
                    "native_budget_residual_priority_mean": float(weights.priority[valid].mean().cpu()),
                    "native_teacher_gap_alpha": float(
                        get_nested(cfg, "scopd.native_budget_weighting.alpha", 1.0)
                    ),
                    "native_persistence_mix": float(
                        get_nested(cfg, "scopd.native_budget_weighting.persistence_mix", 0.1)
                    ),
                    "native_loss_mass_scale": float(weights.loss_mass_scale.cpu()),
                }
                loss_type = "scopd_nogt_native_budget_residual_hardness_weighted_forward_kl"
            elif weighting_mode == "budget_gradient_consensus":
                gradient_consensus, gradient_norm_consistency = (
                    compute_teacher_gradient_budget_consensus(
                        teacher_logits,
                        b_plus_logits,
                        student_logits,
                        temperature=scopd_temperature,
                        chunk_size=int(
                            get_nested(cfg, "scopd.native_budget_weighting.kl_chunk_size", 32)
                        ),
                        eps=eps,
                    )
                )
                weights = budget_gradient_consensus_weights(
                    per_token_scopd,
                    gradient_consensus,
                    gradient_norm_consistency,
                    valid_mask,
                    alpha=float(get_nested(cfg, "scopd.native_budget_weighting.alpha", 0.5)),
                    eps=eps,
                )
                valid = weights.valid_mask
                token_weight = weights.weight
                mode_metrics = {
                    "native_teacher_gap_b_mean": float(weights.teacher_gap_b[valid].mean().cpu()),
                    "native_teacher_gap_rank_mean": float(weights.teacher_gap_rank[valid].mean().cpu()),
                    "native_gradient_consensus_mean": float(
                        weights.gradient_consensus[valid].mean().cpu()
                    ),
                    "native_gradient_consensus_positive_fraction": float(
                        (weights.gradient_consensus[valid] > 0.0).float().mean().cpu()
                    ),
                    "native_gradient_norm_consistency_mean": float(
                        weights.gradient_norm_consistency[valid].mean().cpu()
                    ),
                    "native_invariant_priority_mean": float(
                        weights.invariant_priority[valid].mean().cpu()
                    ),
                    "native_teacher_gap_alpha": float(
                        get_nested(cfg, "scopd.native_budget_weighting.alpha", 0.5)
                    ),
                    "native_loss_mass_scale": float(weights.loss_mass_scale.cpu()),
                }
                loss_type = "scopd_nogt_native_budget_gradient_consensus_weighted_forward_kl"
            elif weighting_mode == "counterfactual_budget_bridge":
                if per_token_bridge is None:
                    raise AssertionError("Counterfactual budget bridge requires differentiable bridge KL.")
                per_token_b_plus_teacher_gap = compute_per_token_kl(
                    teacher_logits,
                    b_plus_logits,
                    temperature=scopd_temperature,
                    chunk_size=int(get_nested(cfg, "scopd.native_budget_weighting.kl_chunk_size", 32)),
                )
                weights = counterfactual_budget_bridge(
                    per_token_scopd,
                    per_token_b_plus_teacher_gap,
                    per_token_bridge,
                    valid_mask,
                    max_bridge_fraction=float(
                        get_nested(cfg, "scopd.native_budget_weighting.max_bridge_fraction", 0.5)
                    ),
                    eps=eps,
                )
                valid = weights.valid_mask
                token_weight = weights.full_teacher_weight + weights.bridge_teacher_weight
                mode_metrics = {
                    "native_teacher_gap_b_mean": float(weights.teacher_gap_b[valid].mean().cpu()),
                    "native_teacher_gap_b_plus_mean": float(weights.teacher_gap_b_plus[valid].mean().cpu()),
                    "native_bridge_gap_mean": float(weights.bridge_gap[valid].mean().cpu()),
                    "native_budget_rescue_mean": float(weights.rescue_fraction[valid].mean().cpu()),
                    "native_teacher_gap_confidence_mean": float(weights.confidence[valid].mean().cpu()),
                    "native_bridge_fraction_mean": float(weights.bridge_fraction[valid].mean().cpu()),
                    "native_bridge_fraction_max": float(weights.bridge_fraction[valid].max().cpu()),
                    "native_full_teacher_weight_mean": float(weights.full_teacher_weight[valid].mean().cpu()),
                    "native_bridge_teacher_weight_mean": float(weights.bridge_teacher_weight[valid].mean().cpu()),
                    "native_max_bridge_fraction": float(
                        get_nested(cfg, "scopd.native_budget_weighting.max_bridge_fraction", 0.5)
                    ),
                    "native_loss_mass_scale": float(weights.loss_mass_scale.cpu()),
                }
                loss_type = "scopd_nogt_counterfactual_budget_bridge_forward_kl"
            elif weighting_mode == "budget_gradient_aligned_bridge":
                if per_token_bridge is None:
                    raise AssertionError("Gradient-aligned bridge requires differentiable bridge KL.")
                per_token_b_plus_teacher_gap = compute_per_token_kl(
                    teacher_logits,
                    b_plus_logits,
                    temperature=scopd_temperature,
                    chunk_size=int(get_nested(cfg, "scopd.native_budget_weighting.kl_chunk_size", 32)),
                )
                gradient_alignment = compute_budget_gradient_alignment(
                    teacher_logits,
                    b_plus_logits,
                    student_logits,
                    temperature=scopd_temperature,
                    chunk_size=int(get_nested(cfg, "scopd.native_budget_weighting.kl_chunk_size", 32)),
                    eps=eps,
                )
                weights = budget_gradient_aligned_bridge_gate(
                    per_token_scopd,
                    per_token_b_plus_teacher_gap,
                    gradient_alignment,
                    valid_mask,
                    max_bridge_fraction=float(
                        get_nested(cfg, "scopd.native_budget_weighting.max_bridge_fraction", 0.5)
                    ),
                    eps=eps,
                )
                valid = weights.valid_mask
                with torch.enable_grad():
                    per_token_aligned_candidate = (
                        per_token_scopd + weights.bridge_fraction * per_token_bridge
                    )
                mass_normalization = normalize_candidate_loss_mass(
                    per_token_scopd,
                    per_token_aligned_candidate,
                    torch.ones_like(per_token_scopd),
                    valid,
                    eps=eps,
                )
                token_weight = mass_normalization.weight
                aligned = weights.gradient_alignment[valid]
                mode_metrics = {
                    "native_teacher_gap_b_mean": float(weights.teacher_gap_b[valid].mean().cpu()),
                    "native_teacher_gap_b_plus_mean": float(weights.teacher_gap_b_plus[valid].mean().cpu()),
                    "native_budget_rescue_mean": float(weights.rescue_fraction[valid].mean().cpu()),
                    "native_teacher_gap_confidence_mean": float(weights.confidence[valid].mean().cpu()),
                    "native_gradient_alignment_mean": float(aligned.mean().cpu()),
                    "native_gradient_alignment_positive_fraction": float((aligned > 0.0).float().mean().cpu()),
                    "native_bridge_fraction_mean": float(weights.bridge_fraction[valid].mean().cpu()),
                    "native_bridge_fraction_max": float(weights.bridge_fraction[valid].max().cpu()),
                    "native_max_bridge_fraction": float(
                        get_nested(cfg, "scopd.native_budget_weighting.max_bridge_fraction", 0.5)
                    ),
                    "native_loss_mass_scale": float(mass_normalization.loss_mass_scale.cpu()),
                }
                loss_type = "scopd_nogt_budget_gradient_aligned_bridge_forward_kl"
            elif weighting_mode == "counterfactual_gradient_residual":
                if per_token_bridge is None:
                    raise AssertionError(
                        "Counterfactual gradient residualization requires differentiable bridge KL."
                    )
                per_token_b_plus_teacher_gap = compute_per_token_kl(
                    teacher_logits,
                    b_plus_logits,
                    temperature=scopd_temperature,
                    chunk_size=int(get_nested(cfg, "scopd.native_budget_weighting.kl_chunk_size", 32)),
                )
                (
                    gradient_alignment,
                    projection_coefficient,
                    teacher_gradient_norm_sq,
                    budget_gradient_norm_sq,
                    gradient_dot_product,
                ) = compute_budget_gradient_projection_geometry(
                    teacher_logits,
                    b_plus_logits,
                    student_logits,
                    temperature=scopd_temperature,
                    chunk_size=int(get_nested(cfg, "scopd.native_budget_weighting.kl_chunk_size", 32)),
                    eps=eps,
                )
                cancellation_schedule = str(
                    get_nested(
                        cfg,
                        "scopd.native_budget_weighting.cancellation_schedule",
                        "constant",
                    )
                )
                base_cancellation_strength = float(
                    get_nested(cfg, "scopd.native_budget_weighting.cancellation_strength", 0.5)
                )
                effective_cancellation_strength = counterfactual_cancellation_strength(
                    base_cancellation_strength,
                    schedule=cancellation_schedule,
                    progress_step=progress_step,
                    total_steps=total_steps,
                    decay_fraction=float(
                        get_nested(
                            cfg,
                            "scopd.native_budget_weighting.cancellation_decay_fraction",
                            0.5,
                        )
                    ),
                )
                weights = counterfactual_gradient_residual_gate(
                    per_token_scopd,
                    per_token_b_plus_teacher_gap,
                    gradient_alignment,
                    projection_coefficient,
                    teacher_gradient_norm_sq,
                    budget_gradient_norm_sq,
                    gradient_dot_product,
                    valid_mask,
                    cancellation_strength=effective_cancellation_strength,
                    max_projection_coefficient=float(
                        get_nested(
                            cfg,
                            "scopd.native_budget_weighting.max_projection_coefficient",
                            1.0,
                        )
                    ),
                    eps=eps,
                )
                valid = weights.valid_mask
                token_weight = torch.ones_like(per_token_scopd).detach()
                with torch.enable_grad():
                    per_token_gradient_residual = (
                        per_token_scopd
                        - weights.cancellation_coefficient * per_token_bridge
                    )
                    raw_residual_mean = per_token_gradient_residual[valid].mean()
                    unweighted_forward_mean = per_token_scopd[valid].mean()
                # Preserve the exact vanilla SCOPD scalar while retaining the
                # counterfactual-residual gradient. The correction is detached.
                gradient_residual_scalar_correction = (
                    unweighted_forward_mean.detach() - raw_residual_mean.detach()
                )
                rescued = weights.budget_rescue_indicator[valid]
                coefficients = weights.cancellation_coefficient[valid]
                mode_metrics = {
                    "native_teacher_gap_b_mean": float(weights.teacher_gap_b[valid].mean().cpu()),
                    "native_teacher_gap_b_plus_mean": float(
                        weights.teacher_gap_b_plus[valid].mean().cpu()
                    ),
                    "native_gradient_alignment_mean": float(
                        weights.gradient_alignment[valid].mean().cpu()
                    ),
                    "native_gradient_alignment_positive_fraction": float(
                        (weights.gradient_alignment[valid] > 0.0).float().mean().cpu()
                    ),
                    "native_budget_rescue_indicator_mean": float(rescued.mean().cpu()),
                    "native_raw_projection_coefficient_mean": float(
                        weights.raw_projection_coefficient[valid].mean().cpu()
                    ),
                    "native_clipped_projection_coefficient_mean": float(
                        weights.clipped_projection_coefficient[valid].mean().cpu()
                    ),
                    "native_cancellation_coefficient_mean": float(coefficients.mean().cpu()),
                    "native_cancellation_coefficient_max": float(coefficients.max().cpu()),
                    "native_residual_gradient_norm_ratio_mean": float(
                        weights.residual_gradient_norm_ratio[valid].mean().cpu()
                    ),
                    "native_cancellation_strength": effective_cancellation_strength,
                    "native_base_cancellation_strength": base_cancellation_strength,
                    "native_cancellation_schedule": cancellation_schedule,
                    "native_cancellation_decay_fraction": float(
                        get_nested(
                            cfg,
                            "scopd.native_budget_weighting.cancellation_decay_fraction",
                            0.5,
                        )
                    ),
                    "native_cancellation_progress_step": progress_step,
                    "native_cancellation_total_steps": total_steps,
                    "native_max_projection_coefficient": float(
                        get_nested(
                            cfg,
                            "scopd.native_budget_weighting.max_projection_coefficient",
                            1.0,
                        )
                    ),
                    "native_gradient_residual_raw_loss": float(raw_residual_mean.detach().cpu()),
                    "native_gradient_residual_scalar_correction": float(
                        gradient_residual_scalar_correction.cpu()
                    ),
                }
                loss_type = "scopd_nogt_counterfactual_gradient_residual_forward_kl"
            elif weighting_mode == "budget_tangent_residual":
                per_token_b_plus_teacher_gap = compute_per_token_kl(
                    teacher_logits,
                    b_plus_logits,
                    temperature=scopd_temperature,
                    chunk_size=int(get_nested(cfg, "scopd.native_budget_weighting.kl_chunk_size", 32)),
                )
                gradient_alignment, explained_fraction, residual_fraction = (
                    compute_budget_gradient_geometry(
                        teacher_logits,
                        b_plus_logits,
                        student_logits,
                        temperature=scopd_temperature,
                        chunk_size=int(get_nested(cfg, "scopd.native_budget_weighting.kl_chunk_size", 32)),
                        eps=eps,
                    )
                )
                weights = budget_tangent_residual_weights(
                    per_token_scopd,
                    gradient_alignment,
                    explained_fraction,
                    valid_mask,
                    alpha=float(get_nested(cfg, "scopd.native_budget_weighting.alpha", 1.0)),
                    eps=eps,
                )
                valid = weights.valid_mask
                token_weight = weights.weight
                mode_metrics = {
                    "native_teacher_gap_b_mean": float(weights.teacher_gap_b[valid].mean().cpu()),
                    "native_teacher_gap_b_plus_mean": float(per_token_b_plus_teacher_gap[valid].mean().cpu()),
                    "native_gradient_alignment_mean": float(weights.gradient_alignment[valid].mean().cpu()),
                    "native_gradient_alignment_positive_fraction": float(
                        (weights.gradient_alignment[valid] > 0.0).float().mean().cpu()
                    ),
                    "native_budget_explained_fraction_mean": float(
                        weights.budget_explained_fraction[valid].mean().cpu()
                    ),
                    "native_budget_residual_fraction_mean": float(
                        weights.budget_residual_fraction[valid].mean().cpu()
                    ),
                    "native_budget_tangent_priority_mean": float(weights.priority[valid].mean().cpu()),
                    "native_teacher_gap_alpha": float(
                        get_nested(cfg, "scopd.native_budget_weighting.alpha", 1.0)
                    ),
                    "native_loss_mass_scale": float(weights.loss_mass_scale.cpu()),
                }
                loss_type = "scopd_nogt_budget_tangent_residual_weighted_forward_kl"
            elif weighting_mode == "budget_counterfactual_teachability":
                per_token_b_plus_teacher_gap = compute_per_token_kl(
                    teacher_logits,
                    b_plus_logits,
                    temperature=scopd_temperature,
                    chunk_size=int(get_nested(cfg, "scopd.native_budget_weighting.kl_chunk_size", 32)),
                )
                gradient_alignment, explained_fraction, residual_fraction = (
                    compute_budget_gradient_geometry(
                        teacher_logits,
                        b_plus_logits,
                        student_logits,
                        temperature=scopd_temperature,
                        chunk_size=int(get_nested(cfg, "scopd.native_budget_weighting.kl_chunk_size", 32)),
                        eps=eps,
                    )
                )
                support_top_k = int(
                    get_nested(cfg, "scopd.native_budget_weighting.support_top_k", 32)
                )
                teacher_support_coverage = compute_teacher_mass_on_student_support(
                    teacher_logits,
                    student_logits,
                    top_k=support_top_k,
                    temperature=scopd_temperature,
                    chunk_size=int(get_nested(cfg, "scopd.native_budget_weighting.kl_chunk_size", 32)),
                )
                weights = budget_counterfactual_teachability_weights(
                    per_token_scopd,
                    gradient_alignment,
                    explained_fraction,
                    teacher_support_coverage,
                    valid_mask,
                    alpha=float(get_nested(cfg, "scopd.native_budget_weighting.alpha", 1.0)),
                    eps=eps,
                )
                valid = weights.valid_mask
                token_weight = weights.weight
                mode_metrics = {
                    "native_teacher_gap_b_mean": float(weights.teacher_gap_b[valid].mean().cpu()),
                    "native_teacher_gap_b_plus_mean": float(per_token_b_plus_teacher_gap[valid].mean().cpu()),
                    "native_gradient_alignment_mean": float(weights.gradient_alignment[valid].mean().cpu()),
                    "native_gradient_alignment_positive_fraction": float(
                        (weights.gradient_alignment[valid] > 0.0).float().mean().cpu()
                    ),
                    "native_budget_explained_fraction_mean": float(
                        weights.budget_explained_fraction[valid].mean().cpu()
                    ),
                    "native_budget_residual_fraction_mean": float(
                        weights.budget_residual_fraction[valid].mean().cpu()
                    ),
                    "native_teacher_support_coverage_mean": float(
                        weights.teacher_support_coverage[valid].mean().cpu()
                    ),
                    "native_support_coverage_rank_mean": float(
                        weights.support_coverage_rank[valid].mean().cpu()
                    ),
                    "native_budget_counterfactual_teachability_priority_mean": float(
                        weights.priority[valid].mean().cpu()
                    ),
                    "native_teacher_gap_alpha": float(
                        get_nested(cfg, "scopd.native_budget_weighting.alpha", 1.0)
                    ),
                    "native_support_top_k": support_top_k,
                    "native_loss_mass_scale": float(weights.loss_mass_scale.cpu()),
                }
                loss_type = "scopd_nogt_budget_counterfactual_teachability_weighted_forward_kl"
            elif weighting_mode == "budget_contrastive_target":
                per_token_b_plus_teacher_gap = compute_per_token_kl(
                    teacher_logits,
                    b_plus_logits,
                    temperature=scopd_temperature,
                    chunk_size=int(get_nested(cfg, "scopd.native_budget_weighting.kl_chunk_size", 32)),
                )
                weights = budget_contrastive_gate(
                    per_token_scopd,
                    per_token_b_plus_teacher_gap,
                    valid_mask,
                    beta_max=float(get_nested(cfg, "scopd.native_budget_weighting.beta_max", 0.5)),
                    eps=eps,
                )
                valid = weights.valid_mask
                with torch.enable_grad():
                    per_token_contrastive, per_token_target_shift = compute_budget_contrastive_per_token_kl(
                        teacher_logits,
                        b_plus_logits,
                        student_logits,
                        weights.shaping_strength,
                        advantage_clip=float(
                            get_nested(cfg, "scopd.native_budget_weighting.advantage_clip", 2.0)
                        ),
                        temperature=scopd_temperature,
                        chunk_size=int(get_nested(cfg, "scopd.native_budget_weighting.kl_chunk_size", 32)),
                        return_target_shift=True,
                    )
                mass_normalization = normalize_candidate_loss_mass(
                    per_token_scopd,
                    per_token_contrastive,
                    torch.ones_like(per_token_scopd),
                    valid,
                    eps=eps,
                )
                loss_mass_scale = mass_normalization.loss_mass_scale
                token_weight = mass_normalization.weight
                mode_metrics = {
                    "native_teacher_gap_b_mean": float(weights.teacher_gap_b[valid].mean().cpu()),
                    "native_teacher_gap_b_plus_mean": float(weights.teacher_gap_b_plus[valid].mean().cpu()),
                    "native_budget_rescue_mean": float(weights.rescue_fraction[valid].mean().cpu()),
                    "native_teacher_gap_confidence_mean": float(weights.confidence[valid].mean().cpu()),
                    "native_contrastive_strength_mean": float(weights.shaping_strength[valid].mean().cpu()),
                    "native_contrastive_strength_max": float(weights.shaping_strength[valid].max().cpu()),
                    "native_target_shift_mean": float(per_token_target_shift[valid].mean().cpu()),
                    "native_target_shift_max": float(per_token_target_shift[valid].max().cpu()),
                    "native_beta_max": float(get_nested(cfg, "scopd.native_budget_weighting.beta_max", 0.5)),
                    "native_advantage_clip": float(
                        get_nested(cfg, "scopd.native_budget_weighting.advantage_clip", 2.0)
                    ),
                    "native_loss_mass_scale": float(loss_mass_scale.cpu()),
                }
                loss_type = "scopd_nogt_budget_contrastive_target_forward_kl"
            elif weighting_mode == "dual_budget_decomposition":
                per_token_b_plus_teacher_gap = compute_per_token_kl(
                    teacher_logits,
                    b_plus_logits,
                    temperature=scopd_temperature,
                    chunk_size=int(get_nested(cfg, "scopd.native_budget_weighting.kl_chunk_size", 32)),
                )
                hardness = budget_residual_hardness_weights(
                    per_token_scopd,
                    per_token_b_plus_teacher_gap,
                    valid_mask,
                    alpha=float(get_nested(cfg, "scopd.native_budget_weighting.alpha", 1.0)),
                    persistence_mix=float(
                        get_nested(cfg, "scopd.native_budget_weighting.persistence_mix", 0.1)
                    ),
                    eps=eps,
                )
                gate = budget_contrastive_gate(
                    per_token_scopd,
                    per_token_b_plus_teacher_gap,
                    valid_mask,
                    beta_max=float(get_nested(cfg, "scopd.native_budget_weighting.beta_max", 0.5)),
                    eps=eps,
                )
                valid = hardness.valid_mask
                with torch.enable_grad():
                    per_token_contrastive, per_token_target_shift = compute_budget_contrastive_per_token_kl(
                        teacher_logits,
                        b_plus_logits,
                        student_logits,
                        gate.shaping_strength,
                        advantage_clip=float(
                            get_nested(cfg, "scopd.native_budget_weighting.advantage_clip", 2.0)
                        ),
                        temperature=scopd_temperature,
                        chunk_size=int(get_nested(cfg, "scopd.native_budget_weighting.kl_chunk_size", 32)),
                        return_target_shift=True,
                    )
                mass_normalization = normalize_candidate_loss_mass(
                    per_token_scopd,
                    per_token_contrastive,
                    hardness.raw_weight,
                    valid,
                    eps=eps,
                )
                loss_mass_scale = mass_normalization.loss_mass_scale
                token_weight = mass_normalization.weight
                mode_metrics = {
                    "native_teacher_gap_b_mean": float(hardness.teacher_gap_b[valid].mean().cpu()),
                    "native_teacher_gap_b_plus_mean": float(hardness.teacher_gap_b_plus[valid].mean().cpu()),
                    "native_persistent_gap_mean": float(hardness.persistent_gap[valid].mean().cpu()),
                    "native_budget_residual_priority_mean": float(hardness.priority[valid].mean().cpu()),
                    "native_budget_rescue_mean": float(gate.rescue_fraction[valid].mean().cpu()),
                    "native_teacher_gap_confidence_mean": float(gate.confidence[valid].mean().cpu()),
                    "native_contrastive_strength_mean": float(gate.shaping_strength[valid].mean().cpu()),
                    "native_contrastive_strength_max": float(gate.shaping_strength[valid].max().cpu()),
                    "native_target_shift_mean": float(per_token_target_shift[valid].mean().cpu()),
                    "native_target_shift_max": float(per_token_target_shift[valid].max().cpu()),
                    "native_teacher_gap_alpha": float(
                        get_nested(cfg, "scopd.native_budget_weighting.alpha", 1.0)
                    ),
                    "native_persistence_mix": float(
                        get_nested(cfg, "scopd.native_budget_weighting.persistence_mix", 0.1)
                    ),
                    "native_beta_max": float(get_nested(cfg, "scopd.native_budget_weighting.beta_max", 0.5)),
                    "native_advantage_clip": float(
                        get_nested(cfg, "scopd.native_budget_weighting.advantage_clip", 2.0)
                    ),
                    "native_loss_mass_scale": float(loss_mass_scale.cpu()),
                }
                loss_type = "scopd_nogt_dual_budget_decomposition_forward_kl"
            else:
                raise ValueError(f"Unknown native budget weighting mode: {weighting_mode!r}.")
        if weighting_mode in {"trajectory_probe", "symmetric_teacher_gap_stability"}:
            if trajectory_scalar_kl is None:
                raise AssertionError("Exact-scalar weighting requires the original scalar SCOPD KL.")
            unweighted_kl = trajectory_scalar_kl
            if weighting_mode == "trajectory_probe":
                kl = trajectory_scalar_kl
        elif weighting_mode in {
            *TOKEN_PROJECTION_PARTITION_MODES,
            TOKEN_RANDOM_DROP_MODE,
            TOKEN_RANDOM_KEEP_MODE,
            TOKEN_FORWARD_KL_TOP_MODE,
            TOKEN_PROJECTION_BOTTOM_DROP_MODE,
            TOKEN_PROJECTION_TOP_DROP_MODE,
            TOKEN_TIP_KL_ABS_PROJECTION_MODE,
        }:
            if weighting_mode in {
                TOKEN_PROJECTION_BOTTOM_DROP_MODE,
                TOKEN_PROJECTION_TOP_DROP_MODE,
            }:
                unweighted_kl = per_token_scopd[valid_mask].mean()
            elif weighting_mode == TOKEN_RANDOM_DROP_MODE:
                if token_random_drop_partition is None:
                    raise AssertionError("Random token drop partition was not computed.")
                unweighted_kl = per_token_scopd[token_random_drop_partition.valid_mask].mean()
            elif weighting_mode == TOKEN_RANDOM_KEEP_MODE:
                if token_random_keep_partition is None:
                    raise AssertionError("Random token keep partition was not computed.")
                unweighted_kl = per_token_scopd[token_random_keep_partition.valid_mask].mean()
            elif weighting_mode == TOKEN_FORWARD_KL_TOP_MODE:
                if token_forward_kl_top_partition is None:
                    raise AssertionError("Forward-KL top partition was not computed.")
                unweighted_kl = per_token_scopd[
                    token_forward_kl_top_partition.valid_mask
                ].mean()
            elif weighting_mode == TOKEN_TIP_KL_ABS_PROJECTION_MODE:
                if token_tip_kl_abs_projection_partition is None:
                    raise AssertionError("KL/absolute-P SoftOR partition was not computed.")
                unweighted_kl = per_token_scopd[
                    token_tip_kl_abs_projection_partition.valid_mask
                ].mean()
            else:
                if token_projection_partition is None:
                    raise AssertionError("Token projection partition was not computed.")
                unweighted_kl = per_token_scopd[token_projection_partition.valid_mask].mean()
        else:
            unweighted_kl = per_token_scopd[valid].mean()
        if weighting_mode == "trajectory_probe":
            pass
        elif weighting_mode == "symmetric_teacher_gap_stability":
            weighted_token_kl = (token_weight[valid] * per_token_scopd[valid]).mean()
            unweighted_token_kl = per_token_scopd[valid].mean()
            kl = trajectory_scalar_kl + weighted_token_kl - unweighted_token_kl
        elif weighting_mode in {
            *TOKEN_PROJECTION_PARTITION_MODES,
            TOKEN_RANDOM_DROP_MODE,
            TOKEN_RANDOM_KEEP_MODE,
            TOKEN_FORWARD_KL_TOP_MODE,
            TOKEN_PROJECTION_BOTTOM_DROP_MODE,
            TOKEN_PROJECTION_TOP_DROP_MODE,
            TOKEN_TIP_KL_ABS_PROJECTION_MODE,
        }:
            # Keep a graph-connected zero for a legitimately empty partition.
            # This preserves strict top/complement semantics without a vanilla-loss fallback.
            kl = (
                per_token_scopd[valid].mean()
                if valid.any()
                else per_token_scopd.sum() * 0.0
            )
        elif weighting_mode == "counterfactual_budget_bridge":
            if per_token_bridge is None:
                raise AssertionError("Counterfactual budget bridge KL was not computed.")
            kl = (
                weights.full_teacher_weight[valid] * per_token_scopd[valid]
                + weights.bridge_teacher_weight[valid] * per_token_bridge[valid]
            ).mean()
        elif weighting_mode == "budget_gradient_aligned_bridge":
            kl = (token_weight[valid] * per_token_aligned_candidate[valid]).mean()
        elif weighting_mode == "counterfactual_gradient_residual":
            kl = (
                per_token_gradient_residual[valid].mean()
                + gradient_residual_scalar_correction
            )
        elif weighting_mode in {"budget_contrastive_target", "dual_budget_decomposition"}:
            kl = (token_weight[valid] * per_token_contrastive[valid]).mean()
        elif weighting_mode == "max_kl_fraction_softmax_inverse_jsd_group_balanced":
            high = weights.high_group_mask
            low = valid & ~high
            if high.any() and low.any():
                high_group_kl = (
                    weights.within_high_weight[high] * per_token_scopd[high]
                ).mean()
                low_group_kl = per_token_scopd[low].mean()
                coefficient = float(weights.high_group_coefficient)
                kl = coefficient * high_group_kl + (1.0 - coefficient) * low_group_kl
                equivalent_token_weight_kl = (
                    token_weight[valid] * per_token_scopd[valid]
                ).mean()
                mode_metrics.update(
                    {
                        "native_group_objective": "lambda_high_mean_plus_one_minus_lambda_low_mean",
                        "native_high_group_weighted_kl": float(high_group_kl.detach().cpu()),
                        "native_low_group_unweighted_kl": float(low_group_kl.detach().cpu()),
                        "native_direct_lambda_reconstruction_error": float(
                            (kl.detach() - equivalent_token_weight_kl.detach()).abs().cpu()
                        ),
                    }
                )
            else:
                kl = per_token_scopd[valid].mean()
                mode_metrics.update(
                    {
                        "native_group_objective": "vanilla_fallback_for_degenerate_group",
                        "native_high_group_weighted_kl": None,
                        "native_low_group_unweighted_kl": None,
                        "native_direct_lambda_reconstruction_error": 0.0,
                    }
                )
        else:
            kl = (token_weight[valid] * per_token_scopd[valid]).mean()
        sensitivity_valid = sensitivity.detach().float()[valid]
        weight_valid = token_weight[valid]
        kl_mass_ratio = kl.detach().float() / unweighted_kl.detach().float().clamp_min(1e-8)
        has_weighted_tokens = bool(valid.any())
        weighting_metrics.update(
            {
                "loss_type": loss_type,
                "unweighted_kl_loss": float(unweighted_kl.detach().cpu()),
                "weighted_kl_loss": float(kl.detach().cpu()),
                "native_weighted_to_unweighted_kl_ratio": float(kl_mass_ratio.cpu()),
                "native_b_num_full_visual_tokens": int(pruned["metadata"].get("num_full_visual_tokens", 0)),
                "native_b_num_kept_visual_tokens": int(pruned["metadata"].get("num_kept_visual_tokens", 0)),
                "native_b_plus_num_full_visual_tokens": (
                    int(b_plus_metadata.get("num_full_visual_tokens", 0))
                    if native_requires_b_plus
                    else None
                ),
                "native_b_plus_num_kept_visual_tokens": (
                    int(b_plus_metadata.get("num_kept_visual_tokens", 0))
                    if native_requires_b_plus
                    else None
                ),
                "native_sensitivity_min": (
                    float(sensitivity_valid.min().cpu()) if has_weighted_tokens else None
                ),
                "native_sensitivity_mean": (
                    float(sensitivity_valid.mean().cpu()) if has_weighted_tokens else None
                ),
                "native_sensitivity_median": (
                    float(torch.quantile(sensitivity_valid, 0.5).cpu())
                    if has_weighted_tokens
                    else None
                ),
                "native_sensitivity_max": (
                    float(sensitivity_valid.max().cpu()) if has_weighted_tokens else None
                ),
                "native_token_weight_min": (
                    float(weight_valid.min().cpu()) if has_weighted_tokens else None
                ),
                "native_token_weight_mean": (
                    float(weight_valid.mean().cpu()) if has_weighted_tokens else None
                ),
                "native_token_weight_max": (
                    float(weight_valid.max().cpu()) if has_weighted_tokens else None
                ),
                "native_valid_generated_tokens": int(valid.sum().cpu()),
                "native_weight_detached": not token_weight.requires_grad,
                "native_probe_grad_enabled": False,
                **mode_metrics,
            }
        )
        del b_plus_logits
    else:
        kl_floor_filter_enabled = bool(
            get_nested(cfg, "scopd.token_kl_floor_filter.enabled", False)
        )
        outlier_exclusion_enabled = bool(
            get_nested(cfg, "scopd.token_outlier_exclusion.enabled", False)
        )
        if kl_floor_filter_enabled:
            min_kl = float(get_nested(cfg, "scopd.token_kl_floor_filter.min_kl", 1e-5))
            valid = generated_token_valid_mask(gen_ids)
            forward_per_token_kl = compute_per_token_kl(
                teacher_logits,
                student_logits,
                temperature=scopd_temperature,
                chunk_size=int(
                    get_nested(cfg, "scopd.token_kl_floor_filter.kl_chunk_size", 32)
                ),
            )
            keep_mask, removed_indices = keep_mask_above_kl_floor(
                forward_per_token_kl.detach(),
                valid,
                min_kl,
            )
            unfiltered_forward_kl = forward_per_token_kl[valid].mean()
            kl = (
                forward_per_token_kl[keep_mask].mean()
                if keep_mask.any()
                else forward_per_token_kl.sum() * 0.0
            )
            detached_forward_kl = forward_per_token_kl.detach().float()
            removed_forward = detached_forward_kl[removed_indices]
            valid_forward_sum = detached_forward_kl[valid].sum()
            removed_forward_sum = removed_forward.sum()
            removed_mass_fraction = removed_forward_sum / valid_forward_sum.clamp_min(1e-8)
            valid_count = int(valid.sum().cpu())
            kept_count = int(keep_mask.sum().cpu())
            removed_count = int(removed_indices.numel())
            weighting_metrics.update(
                {
                    "loss_type": "scopd_nogt_forward_kl_floor_filtered",
                    "unweighted_kl_loss": float(unfiltered_forward_kl.detach().cpu()),
                    "weighted_kl_loss": float(kl.detach().cpu()),
                    "token_kl_floor_direction": "KL(teacher || student)",
                    "token_kl_floor_threshold": min_kl,
                    "token_kl_floor_comparison": "keep_kl_greater_than_or_equal_threshold",
                    "token_kl_floor_valid_tokens": valid_count,
                    "token_kl_floor_kept_tokens": kept_count,
                    "token_kl_floor_removed_tokens": removed_count,
                    "token_kl_floor_kept_fraction": kept_count / valid_count,
                    "token_kl_floor_removed_fraction": removed_count / valid_count,
                    "token_kl_floor_removed_kl_mass_fraction": float(
                        removed_mass_fraction.cpu()
                    ),
                    "token_kl_floor_remaining_mean_normalized": True,
                    "token_kl_floor_selection_detached": not keep_mask.requires_grad,
                    "token_kl_floor_empty_selection": kept_count == 0,
                    "token_kl_floor_filtered_to_unfiltered_loss_ratio": float(
                        (
                            kl.detach().float()
                            / unfiltered_forward_kl.detach().float().clamp_min(1e-8)
                        ).cpu()
                    ),
                }
            )
        elif outlier_exclusion_enabled:
            requested_top_k = resolve_token_outlier_top_k(
                int(get_nested(cfg, "scopd.token_outlier_exclusion.top_k", 0) or 0),
                get_nested(cfg, "scopd.token_outlier_exclusion.top_k_by_ratio", None),
                retention_ratio,
            )
            valid = generated_token_valid_mask(gen_ids)
            forward_per_token_kl = compute_per_token_kl(
                teacher_logits,
                student_logits,
                temperature=scopd_temperature,
                chunk_size=int(
                    get_nested(cfg, "scopd.token_outlier_exclusion.kl_chunk_size", 32)
                ),
            )
            keep_mask, removed_indices = keep_mask_after_topk_exclusion(
                forward_per_token_kl.detach(),
                valid,
                requested_top_k,
            )
            unfiltered_forward_kl = forward_per_token_kl[valid].mean()
            kl = forward_per_token_kl[keep_mask].mean()
            removed_forward = forward_per_token_kl.detach().float()[removed_indices]
            flat_generated_ids = gen_ids.detach().reshape(-1)
            valid_forward_sum = forward_per_token_kl.detach().float()[valid].sum()
            removed_forward_sum = removed_forward.sum()
            removed_mass_fraction = removed_forward_sum / valid_forward_sum.clamp_min(1e-8)
            if removed_indices.numel() > 0:
                removed_forward_min = float(removed_forward.min().cpu())
                removed_forward_mean = float(removed_forward.mean().cpu())
                removed_forward_max = float(removed_forward.max().cpu())
            else:
                removed_forward_min = None
                removed_forward_mean = None
                removed_forward_max = None
            weighting_metrics.update(
                {
                    "loss_type": (
                        "scopd_nogt_forward_kl_topk_excluded"
                        if not teacher_uses_ground_truth
                        else "scopd_gt_prompt_forward_kl_topk_excluded"
                    ),
                    "unweighted_kl_loss": float(unfiltered_forward_kl.detach().cpu()),
                    "weighted_kl_loss": float(kl.detach().cpu()),
                    "token_outlier_ranking_kl_direction": "KL(teacher || student)",
                    "token_outlier_training_kl_direction": "KL(teacher || student)",
                    "token_outlier_requested_top_k": requested_top_k,
                    "token_outlier_effective_top_k": int(removed_indices.numel()),
                    "token_outlier_valid_tokens": int(valid.sum().cpu()),
                    "token_outlier_kept_tokens": int(keep_mask.sum().cpu()),
                    "token_outlier_remaining_mean_normalized": True,
                    "token_outlier_removed_forward_kl_min": removed_forward_min,
                    "token_outlier_removed_forward_kl_mean": removed_forward_mean,
                    "token_outlier_removed_forward_kl_max": removed_forward_max,
                    "token_outlier_removed_forward_kl_mass_fraction": float(
                        removed_mass_fraction.cpu()
                    ),
                    "token_outlier_removed_positions": [
                        int(value) for value in removed_indices.cpu().tolist()
                    ],
                    "token_outlier_removed_token_ids": [
                        int(flat_generated_ids[int(value)].cpu().item())
                        for value in removed_indices.cpu().tolist()
                    ],
                    "token_outlier_removed_forward_kl_values": [
                        float(value) for value in removed_forward.cpu().tolist()
                    ],
                }
            )
        else:
            kl = compute_forward_kl(teacher_logits, student_logits, temperature=scopd_temperature)
            weighting_metrics.update(
                {
                    "loss_type": "scopd_gt_prompt_forward_kl" if teacher_uses_ground_truth else "scopd_nogt_forward_kl",
                    "unweighted_kl_loss": float(kl.detach().cpu()),
                    "weighted_kl_loss": None,
                }
            )
    parsed = parse_final_answer(gen_text)
    output_metrics = {
        "loss_type": weighting_metrics.pop("loss_type"),
        "generated_tokens": int(gen_ids.numel()),
        "kl_loss": float(kl.detach().cpu()),
        "parseable": parsed is not None,
        "student_correct": parsed == sample.correct_letter,
        "teacher_source": teacher_source,
        "scopd_teacher_strategy": teacher_strategy,
        "teacher_context": teacher_context,
        "teacher_ground_truth_access": teacher_uses_ground_truth,
        "teacher_prompt_tokens": teacher_prompt_len,
        "student_prompt_tokens": student_prompt_len,
        "rollout_decoder": rollout_decoder,
        "rollout_use_cache": not manual_rollout,
        "rollout_model_eval": not manual_rollout,
        "rollout_num_full_visual_tokens": int(generation_meta.get("num_full_visual_tokens", 0)),
        "rollout_num_kept_visual_tokens": int(generation_meta.get("num_kept_visual_tokens", 0)),
        "rollout_random_mask_hash": generation_meta.get("random_mask_hash"),
        "scoring_random_mask_hash": pruned["metadata"].get("random_mask_hash"),
        "cached_vs_no_cache_greedy_equal": generation_meta.get("cached_vs_no_cache_greedy_equal"),
        "random_mask_reused_for_rollout_and_scoring": (
            generation_meta.get("random_mask_hash") == pruned["metadata"].get("random_mask_hash")
            if normalize_pruning_method() == "random"
            else None
        ),
        STUDENT_TEXT_LOG_KEY: build_student_text_log(
            sample,
            retention_ratio,
            gen_text,
            int(gen_ids.numel()),
            teacher_source=teacher_source,
            teacher_strategy=teacher_strategy,
            rollout_decoder=rollout_decoder,
            rollout_use_cache=not manual_rollout,
        ),
        "scopd_reference": (
            "privileged_gt_prompt_ema_teacher_ablation"
            if teacher_uses_ground_truth and teacher_strategy == "ema"
            else "privileged_gt_prompt_teacher_ablation"
            if teacher_uses_ground_truth
            else "no_gt_ema_teacher_ablation"
            if teacher_strategy == "ema"
            else "no_gt_dynamic_shared_current_teacher_ablation"
            if teacher_strategy == "dynamic_shared_current"
            else "legacy_no_gt_ablation"
            if teacher_strategy == "fixed_base"
            else "no_gt_external_adapter_teacher_ablation"
            if teacher_strategy == "external_adapter"
            else "no_gt_external_teacher_ablation"
        ),
        **weighting_metrics,
        **numeric_metadata(pruned["metadata"]),
    }
    if capture_rollout:
        token_ids_cpu = gen_ids.detach().to(device="cpu", dtype=torch.long).clone()
        output_metrics[ROLLOUT_CACHE_KEY] = {
            "token_ids": token_ids_cpu,
            "token_ids_sha256": hashlib.sha256(token_ids_cpu.numpy().tobytes()).hexdigest(),
            "text": gen_text,
            "generation_metadata": numeric_metadata(generation_meta),
        }
    return kl, output_metrics


def scopd_step(
    model: Any,
    processor: Any,
    sample: FormattedAOKVQASample,
    cfg: dict[str, Any],
    retention_ratio: float,
    teacher_model: Any | None = None,
    ema_shadow: dict[str, torch.Tensor] | None = None,
    teacher_adapter_name: str = "",
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Official-style SCOPD adapted to A-OKVQA/Qwen2.5-VL/VisionZip.

    Student rolls out from the normal visual-question prompt.  Teacher scores
    the same generated suffix from a privileged prompt containing the reference
    A-OKVQA solution, matching the official SCOPD student/teacher context split.
    """

    device = primary_device(model)
    image_root = get_nested(cfg, "dataset.image_root", "")
    student_prompt_inputs = encode_prompt(processor, sample, image_root=image_root, device=device)
    gen_ids, gen_text, _ = generate_pruned(
        model,
        processor,
        student_prompt_inputs,
        retention_ratio,
        max_new_tokens=int(get_nested(cfg, "generation.max_new_tokens", 128)),
        do_sample=generation_do_sample(cfg),
        temperature=float(get_nested(cfg, "generation.temperature", 0.7)),
        top_p=float(get_nested(cfg, "generation.top_p", 0.9)),
        top_k=generation_top_k(cfg),
        allow_embedding_fallback=bool(get_nested(cfg, "pruning.allow_embedding_fallback", False)),
        manual_decode=bool(get_nested(cfg, "generation.manual_pruned_generate", True)),
        max_unparseable_tokens=generation_max_unparseable_tokens(cfg),
        stop_on_parse=generation_stop_on_parse(cfg),
    )
    if gen_ids.numel() == 0:
        raise RuntimeError("SCOPD student generated zero tokens.")

    student_seq_inputs = sequence_inputs_from_prompt(student_prompt_inputs, gen_ids)
    student_prompt_len = int(student_prompt_inputs["input_ids"].shape[1])

    teacher_prompt = build_scopd_teacher_prompt(
        sample.question,
        sample.options,
        sample.target,
        prompt_mode=prompt_mode_from_config(cfg),
    )
    teacher_prompt_inputs = encode_prompt_text(processor, sample, teacher_prompt, image_root=image_root, device=device)
    teacher_seq_inputs = sequence_inputs_from_prompt(teacher_prompt_inputs, gen_ids)
    teacher_prompt_len = int(teacher_prompt_inputs["input_ids"].shape[1])

    teacher_strategy = resolve_scopd_teacher_strategy(cfg, teacher_model, teacher_adapter_name)
    if teacher_strategy == "external":
        with torch.no_grad():
            teacher_outputs = teacher_model(**model_input_subset(teacher_seq_inputs), use_cache=False)
        teacher_source = "external_privileged_full_token"
    elif teacher_strategy == "external_adapter":
        with torch.no_grad(), active_lora_adapter(model, teacher_adapter_name), temporary_eval(model):
            teacher_outputs = model(**model_input_subset(teacher_seq_inputs), use_cache=False)
        teacher_source = "external_adapter_privileged_full_token"
    elif teacher_strategy == "ema":
        if teacher_model is not None:
            with torch.no_grad():
                teacher_outputs = teacher_model(**model_input_subset(teacher_seq_inputs), use_cache=False)
            teacher_source = "ema_privileged_full_token"
        elif teacher_adapter_name:
            with torch.no_grad(), active_lora_adapter(model, teacher_adapter_name), temporary_eval(model):
                teacher_outputs = model(**model_input_subset(teacher_seq_inputs), use_cache=False)
            teacher_source = "ema_adapter_privileged_full_token"
        elif ema_shadow is not None:
            with torch.no_grad(), swapped_ema_parameters(model, ema_shadow), temporary_eval(model):
                teacher_outputs = model(**model_input_subset(teacher_seq_inputs), use_cache=False)
            teacher_source = "ema_lora_shadow_privileged_full_token"
        else:
            with torch.no_grad(), temporary_eval(model):
                teacher_outputs = model(**model_input_subset(teacher_seq_inputs), use_cache=False)
            teacher_source = "ema_uninitialized_current_privileged_full_token"
    elif teacher_strategy == "dynamic_shared_current":
        with torch.no_grad():
            teacher_outputs = model(**model_input_subset(teacher_seq_inputs), use_cache=False)
        teacher_source = "dynamic_shared_current_privileged_full_token"
    elif teacher_strategy == "fixed_base":
        with torch.no_grad(), teacher_adapter_disabled(model):
            teacher_outputs = model(**model_input_subset(teacher_seq_inputs), use_cache=False)
        teacher_source = "fixed_base_privileged_full_token"
    else:
        raise AssertionError(teacher_strategy)

    token_count = int(gen_ids.numel())
    teacher_logits = extract_generated_logits(teacher_outputs.logits, teacher_prompt_len, token_count).detach()
    del teacher_outputs

    student_outputs, pruned = forward_pruned(
        model,
        student_seq_inputs,
        retention_ratio,
        prompt_len=student_prompt_len,
        allow_embedding_fallback=bool(get_nested(cfg, "pruning.allow_embedding_fallback", False)),
    )
    student_logits = extract_generated_logits(student_outputs.logits, int(pruned["metadata"]["student_prompt_len"]), token_count)

    gt_weight = float(get_nested(cfg, "scopd.ground_truth_ce_weight", 0.0) or 0.0)
    if gt_weight != 0.0:
        raise ValueError(
            "scopd.ground_truth_ce_weight is not part of the official-aligned SCOPD path. "
            "Use training.method=scopd_nogt for the legacy no-GT ablation or create a separate explicit CE ablation."
        )
    beta = float(get_nested(cfg, "scopd.beta", 0.0))
    temperature = float(get_nested(cfg, "scopd.temperature", 1.0))
    top_k_raw = int(get_nested(cfg, "scopd.top_k_loss", 0) or 0)
    top_k = top_k_raw if top_k_raw > 0 else None
    token_clip_raw = float(get_nested(cfg, "scopd.jsd_token_clip", 0.05) or 0.0)
    token_clip = token_clip_raw if token_clip_raw > 0.0 else None
    distillation_loss = compute_generalized_jsd(
        teacher_logits,
        student_logits,
        beta=beta,
        temperature=temperature,
        top_k=top_k,
        token_clip=token_clip,
        clip_mode=str(get_nested(cfg, "scopd.jsd_clip_mode", "token")),
    )
    parsed = parse_final_answer(gen_text)
    return distillation_loss, {
        "loss_type": "official_scopd_generalized_jsd",
        "generated_tokens": token_count,
        "distillation_loss": float(distillation_loss.detach().cpu()),
        "kl_loss": float(distillation_loss.detach().cpu()) if beta == 0.0 else None,
        "parseable": parsed is not None,
        "student_correct": parsed == sample.correct_letter,
        "teacher_source": teacher_source,
        "scopd_teacher_strategy": teacher_strategy,
        "teacher_context": "ground_truth_reference_solution",
        "teacher_prompt_tokens": teacher_prompt_len,
        "student_prompt_tokens": student_prompt_len,
        "scopd_beta": beta,
        "scopd_temperature": temperature,
        "scopd_top_k_loss": top_k_raw,
        "scopd_jsd_token_clip": token_clip_raw,
        "scopd_jsd_clip_mode": str(get_nested(cfg, "scopd.jsd_clip_mode", "token")),
        "scopd_reference": (
            "siyan-zhao/SCOPD EMA reference teacher adapted to Qwen2.5-VL/VisionZip"
            if teacher_strategy == "ema"
            else
            "siyan-zhao/SCOPD latest dynamic shared-current teacher adapted to Qwen2.5-VL/VisionZip"
            if teacher_strategy == "dynamic_shared_current"
            else "legacy_fixed_base_teacher_ablation"
            if teacher_strategy == "fixed_base"
            else "external_teacher_ablation"
        ),
        **numeric_metadata(pruned["metadata"]),
    }


def offpolicy_step(
    model: Any,
    processor: Any,
    sample: FormattedAOKVQASample,
    cfg: dict[str, Any],
    retention_ratio: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    device = primary_device(model)
    prompt_inputs = encode_prompt(processor, sample, image_root=get_nested(cfg, "dataset.image_root", ""), device=device)
    with teacher_adapter_disabled(model):
        gen_ids, gen_text = generate_full_teacher(model, processor, prompt_inputs, cfg)
    if gen_ids.numel() == 0:
        raise RuntimeError("Off-policy teacher generated zero tokens.")

    seq_inputs = sequence_inputs_from_prompt(prompt_inputs, gen_ids)
    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    with torch.no_grad(), teacher_adapter_disabled(model):
        teacher_outputs = model(**model_input_subset(seq_inputs), use_cache=False)
    student_outputs, pruned = forward_pruned(
        model,
        seq_inputs,
        retention_ratio,
        prompt_len=prompt_len,
        allow_embedding_fallback=bool(get_nested(cfg, "pruning.allow_embedding_fallback", False)),
    )
    token_count = int(gen_ids.numel())
    teacher_logits = extract_generated_logits(teacher_outputs.logits, prompt_len, token_count).detach()
    student_logits = extract_generated_logits(student_outputs.logits, int(pruned["metadata"]["student_prompt_len"]), token_count)
    temperature = float(get_nested(cfg, "offpolicy.temperature", 1.0))
    kl = compute_forward_kl(teacher_logits, student_logits, temperature=temperature)
    parsed = parse_final_answer(gen_text)
    return kl, {
        "loss_type": "offpolicy_kl",
        "generated_tokens": token_count,
        "kl_loss": float(kl.detach().cpu()),
        "teacher_parseable": parsed is not None,
        "teacher_correct": parsed == sample.correct_letter,
        "teacher_retention_ratio": "full",
        "offpolicy_temperature": temperature,
        **numeric_metadata(pruned["metadata"]),
    }


def grpo_step(
    model: Any,
    processor: Any,
    sample: FormattedAOKVQASample,
    cfg: dict[str, Any],
    retention_ratio: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    device = primary_device(model)
    prompt_inputs = encode_prompt(processor, sample, image_root=get_nested(cfg, "dataset.image_root", ""), device=device)
    group_size = int(get_nested(cfg, "grpo.group_size", 4))
    logprobs = []
    rewards = []
    parseable_count = 0
    generated_lengths = []
    for _ in range(group_size):
        gen_ids, text, _ = generate_pruned(
            model,
            processor,
            prompt_inputs,
            retention_ratio,
            max_new_tokens=int(get_nested(cfg, "generation.max_new_tokens", 128)),
            do_sample=generation_do_sample(cfg),
            temperature=float(get_nested(cfg, "generation.temperature", 0.7)),
            top_p=float(get_nested(cfg, "generation.top_p", 0.9)),
            top_k=generation_top_k(cfg),
            allow_embedding_fallback=bool(get_nested(cfg, "pruning.allow_embedding_fallback", False)),
            manual_decode=bool(get_nested(cfg, "generation.manual_pruned_generate", True)),
            max_unparseable_tokens=generation_max_unparseable_tokens(cfg),
            stop_on_parse=generation_stop_on_parse(cfg),
        )
        if gen_ids.numel() == 0:
            continue
        parsed = parse_final_answer(text)
        parseable = parsed is not None
        parseable_count += int(parseable)
        reward = (1.0 if parsed == sample.correct_letter else 0.0) + (0.1 if parseable else 0.0)
        seq_inputs = sequence_inputs_from_prompt(prompt_inputs, gen_ids)
        outputs, pruned = forward_pruned(
            model,
            seq_inputs,
            retention_ratio,
            prompt_len=int(prompt_inputs["input_ids"].shape[1]),
            allow_embedding_fallback=bool(get_nested(cfg, "pruning.allow_embedding_fallback", False)),
        )
        logits = extract_generated_logits(outputs.logits, int(pruned["metadata"]["student_prompt_len"]), int(gen_ids.numel()))
        logprobs.append(compute_sequence_logprob(logits, gen_ids))
        rewards.append(reward)
        generated_lengths.append(int(gen_ids.numel()))
    if not logprobs:
        raise RuntimeError("GRPO generated no trainable completions.")
    logprobs_t = torch.stack(logprobs)
    rewards_t = torch.tensor(rewards, device=logprobs_t.device, dtype=torch.float32)
    advantages = grpo_group_advantages(rewards_t)
    return grpo_policy_loss(logprobs_t, advantages), {
        "loss_type": "grpo",
        "reward_mean": float(rewards_t.mean().detach().cpu()),
        "reward_std": float(rewards_t.std(unbiased=False).detach().cpu()),
        "parseable_rate": parseable_count / max(1, len(rewards)),
        "generated_tokens": sum(generated_lengths) / max(1, len(generated_lengths)),
    }


def numeric_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in meta.items():
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                out[key] = float(value.detach().cpu().item())
        elif isinstance(value, (int, float, str, bool)) or value is None:
            out[key] = value
    return out


def aggregate_microbatch_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if not metrics:
        return {}
    out: dict[str, Any] = {}
    for key in sorted({key for item in metrics for key in item}):
        values = [item[key] for item in metrics if key in item]
        if values and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            out[key] = sum(float(value) for value in values) / len(values)
        elif values and all(value == values[0] for value in values):
            out[key] = values[0]
        else:
            out[key] = values
    return out


def trajectory_probability_mode(cfg: dict[str, Any]) -> str | None:
    if not bool(get_nested(cfg, "scopd.trajectory_weighting.enabled", False)):
        return None
    mode = str(get_nested(cfg, "scopd.trajectory_weighting.mode", "")).strip().lower()
    return mode if mode in EFFECTIVE_BATCH_PROBABILITY_MODES else None


def ratio_group_weight_transform(cfg: dict[str, Any]) -> tuple[str, float | None]:
    transform = str(
        get_nested(cfg, "scopd.trajectory_weighting.group_transform", "linear")
    ).strip().lower()
    if transform not in {"linear", "softmax"}:
        raise ValueError(
            f"Unknown ratio-group transform {transform!r}; expected 'linear' or 'softmax'."
        )
    if transform == "linear":
        return transform, None
    temperature = float(
        get_nested(cfg, "scopd.trajectory_weighting.temperature", float("nan"))
    )
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(
            "Ratio-group softmax projection weighting requires a finite positive temperature."
        )
    return transform, temperature


def ratio_group_statistic(cfg: dict[str, Any]) -> str:
    statistic = str(
        get_nested(
            cfg,
            "scopd.trajectory_weighting.group_statistic",
            "teacher_directed_projection_mass_over_teacher_js_mass",
        )
    ).strip().lower()
    supported = {
        "teacher_directed_projection_mass_over_teacher_js_mass",
        "teacher_directed_projection_cosine",
        "teacher_directed_projection_cosine_sample_normalized",
        "teacher_directed_projection_fraction_exact",
    }
    if statistic not in supported:
        raise ValueError(
            f"Unknown ratio-group statistic {statistic!r}; expected one of {sorted(supported)}."
        )
    return statistic


def global_trajectory_calibration(cfg: dict[str, Any]) -> dict[str, float]:
    calibration = get_nested(cfg, "scopd.trajectory_weighting.calibration", None)
    if not isinstance(calibration, dict):
        raise ValueError(
            "global_calibrated_counterfactual_teachability_batch requires a frozen "
            "trajectory_weighting.calibration mapping."
        )
    values = {
        "q05": float(calibration.get("q05", float("nan"))),
        "q95": float(calibration.get("q95", float("nan"))),
        "normalized_mean": float(
            calibration.get("normalized_mean", float("nan"))
        ),
        "coefficient": float(
            get_nested(cfg, "scopd.trajectory_weighting.coefficient", 1.0)
        ),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError(f"Global trajectory calibration must be finite: {values}.")
    if values["q95"] <= values["q05"]:
        raise ValueError("Global trajectory calibration requires q95 > q05.")
    if not 0.0 <= values["normalized_mean"] <= 1.0:
        raise ValueError("Global trajectory normalized_mean must be in [0, 1].")
    if values["coefficient"] < 0.0:
        raise ValueError("Global trajectory coefficient must be nonnegative.")
    if 1.0 - values["coefficient"] * values["normalized_mean"] <= 0.0:
        raise ValueError(
            "Global trajectory coefficient makes the minimum possible weight nonpositive."
        )
    return values


def trajectory_sensitivity_signal(
    metrics: dict[str, Any],
    cfg: dict[str, Any],
    mode: str,
) -> float:
    eps = float(get_nested(cfg, "scopd.trajectory_weighting.eps", 1e-8))
    jsd = max(float(metrics["native_student_budget_jsd_mean"]), 0.0)
    if mode in {
        "jsd_over_current_kl_batch",
        "jsd_over_current_kl_direct_inverse_batch",
        "jsd_over_current_kl_softmax_batch",
    }:
        denominator = max(float(metrics["native_teacher_gap_b_mean"]), eps)
    elif mode == "jsd_over_step0_kl_batch":
        calibration = get_nested(
            cfg,
            "scopd.trajectory_weighting.step0_teacher_kl_by_ratio",
            None,
        )
        if not isinstance(calibration, dict):
            raise ValueError(
                "jsd_over_step0_kl_batch requires step0_teacher_kl_by_ratio calibration."
            )
        key = f"{float(metrics['sampled_b']):.2f}"
        if key not in calibration:
            raise KeyError(f"Missing step-0 teacher KL calibration for ratio {key}.")
        denominator = float(calibration[key])
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise ValueError(
                f"Invalid step-0 teacher KL calibration for ratio {key}: {denominator}."
            )
    elif mode in {
        "progress_adaptive_robust_frontier_batch",
        "adaptive_budget_frontier_sampler_batch",
    }:
        explained = float(metrics["native_trajectory_budget_explained_fraction"])
        teacher_gap = float(metrics["native_teacher_gap_b_mean"])
        if not math.isfinite(explained) or not 0.0 <= explained <= 1.0:
            raise FloatingPointError(
                f"Invalid budget-explained fraction: {explained}."
            )
        if not math.isfinite(teacher_gap) or teacher_gap < 0.0:
            raise FloatingPointError(f"Invalid teacher gap: {teacher_gap}.")
        if mode == "adaptive_budget_frontier_sampler_batch":
            sampler_metric = str(
                get_nested(
                    cfg,
                    "scopd.trajectory_weighting.sampler_metric",
                    "robust_need",
                )
            ).strip().lower()
            if sampler_metric == "teacher_kl":
                return teacher_gap
            if sampler_metric != "robust_need":
                raise ValueError(
                    f"Unsupported adaptive sampler metric: {sampler_metric!r}."
                )
        return teacher_gap * (1.0 - explained)
    elif mode in RAW_TRAJECTORY_F_MODES:
        projection = float(metrics["native_trajectory_budget_projection_mass"])
        teacher_js = float(metrics["native_trajectory_teacher_js_mass"])
        signal = projection / max(teacher_js, eps)
        if not math.isfinite(signal):
            raise FloatingPointError(
                f"Invalid raw budget-explained fraction: {signal}."
            )
        return signal
    elif mode in COUNTERFACTUAL_TEACHABILITY_MODES:
        signal = float(metrics["native_trajectory_budget_explained_fraction"])
        if not math.isfinite(signal) or not 0.0 <= signal <= 1.0:
            raise FloatingPointError(
                f"Invalid counterfactual-teachability signal: {signal}."
            )
        return signal
    else:
        raise ValueError(f"Unsupported probability weighting mode: {mode!r}.")
    signal = jsd / denominator
    if not math.isfinite(signal) or signal < 0.0:
        raise FloatingPointError(f"Invalid trajectory sensitivity signal: {signal}.")
    return signal


def prepare_effective_batch_probability_window(
    model: Any,
    processor: Any,
    dataset: Any,
    cfg: dict[str, Any],
    rng: random.Random,
    *,
    teacher_model: Any | None,
    ema_shadow: dict[str, torch.Tensor] | None,
    teacher_adapter_name: str,
    start_step: int,
    local_step_start: int,
    accumulation_steps: int,
    max_steps: int,
    distributed: bool,
    rank: int,
    world_size: int,
    curriculum_state: (
        AdaptiveBudgetFrontierState
        |
        ProgressAdaptiveFrontierState
        | RobustnessGatedCurriculumState
        | SensitivityFrontierState
        | None
    ) = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Probe one complete effective batch, then assign 32-way weights.

    Rollout token IDs are cached on CPU and replayed by the differentiable pass,
    so the probe and train passes use exactly the same student-generated prefix.
    """

    mode = trajectory_probability_mode(cfg)
    if mode is None:
        raise ValueError("Effective-batch probability probing requires a probability mode.")
    local_records: list[dict[str, Any]] = []
    probe_model = unwrap_model(model)
    global_step_unit = world_size if distributed else 1
    for offset in range(int(accumulation_steps)):
        local_step = int(local_step_start) + offset
        global_index = int(start_step) + local_step * global_step_unit + (rank if distributed else 0)
        sample = dataset[global_index % len(dataset)]
        if mode == "adaptive_budget_frontier_sampler_batch":
            if not isinstance(curriculum_state, AdaptiveBudgetFrontierState):
                raise AssertionError(
                    "Adaptive budget sampling requires synchronized sampler state."
                )
            ratio = curriculum_state.select_ratio(global_index, sample.sample_id)
        else:
            ratio = sample_retention_ratio(
                cfg,
                rng,
                progress_step=global_index,
                total_steps=max_steps,
                sample_id=sample.sample_id,
            )
        rollout_seed = paired_rollout_seed(
            seed=int(
                get_nested(
                    cfg,
                    "paired_sampling.rollout_seed",
                    get_nested(cfg, "training.seed", 42),
                )
            ),
            global_index=global_index,
            sample_id=sample.sample_id,
            namespace=str(get_nested(cfg, "paired_sampling.namespace", "scopd_pair_v1")),
        )
        with torch.no_grad():
            probe_loss, probe_metrics = scopd_nogt_step(
                probe_model,
                processor,
                sample,
                cfg,
                ratio,
                teacher_model=teacher_model,
                ema_shadow=ema_shadow,
                teacher_adapter_name=teacher_adapter_name,
                teacher_uses_ground_truth=False,
                rollout_seed=rollout_seed,
                progress_step=global_index,
                total_steps=max_steps,
                capture_rollout=True,
            )
        rollout_cache = probe_metrics.pop(ROLLOUT_CACHE_KEY, None)
        if not isinstance(rollout_cache, dict):
            raise AssertionError("Effective-batch probe did not return a rollout cache.")
        probe_metrics.pop(STUDENT_TEXT_LOG_KEY, None)
        signal = trajectory_sensitivity_signal(probe_metrics, cfg, mode)
        local_records.append(
            {
                "global_index": global_index,
                "sample_id": sample.sample_id,
                "ratio": float(ratio),
                "rollout_seed": int(rollout_seed),
                "probe_loss": float(probe_loss.detach().float().cpu()),
                "signal": signal,
                "budget_explained_fraction": float(
                    probe_metrics.get(
                        "native_trajectory_budget_explained_fraction", 0.0
                    )
                ),
                "projection_mass": float(
                    probe_metrics.get("native_trajectory_budget_projection_mass", 0.0)
                ),
                "teacher_js_mass": float(
                    probe_metrics.get("native_trajectory_teacher_js_mass", 0.0)
                ),
                "jsd": float(probe_metrics["native_student_budget_jsd_mean"]),
                "current_teacher_kl": float(probe_metrics["native_teacher_gap_b_mean"]),
                "b_plus_teacher_kl": float(
                    probe_metrics["native_teacher_gap_b_plus_mean"]
                ),
                "sampled_b_plus": float(probe_metrics["sampled_b_plus"]),
                "b_visual_tokens": int(probe_metrics["native_b_num_kept_visual_tokens"]),
                "b_plus_visual_tokens": int(
                    probe_metrics["native_b_plus_num_kept_visual_tokens"]
                ),
                "b_random_mask_hash": probe_metrics.get("random_mask_hash"),
                "b_plus_random_mask_hash": probe_metrics.get(
                    "native_b_plus_random_mask_hash"
                ),
                "random_b_subset_b_plus": probe_metrics.get(
                    "native_random_b_subset_b_plus"
                ),
                "rollout": rollout_cache,
            }
        )
        del probe_loss, probe_metrics

    device = primary_device(model)
    local_signals = torch.tensor(
        [record["signal"] for record in local_records], dtype=torch.float32, device=device
    )
    local_losses = torch.tensor(
        [record["probe_loss"] for record in local_records], dtype=torch.float32, device=device
    )
    use_distributed = distributed and dist.is_initialized() and world_size > 1
    if use_distributed:
        gathered_signals = [torch.empty_like(local_signals) for _ in range(world_size)]
        gathered_losses = [torch.empty_like(local_losses) for _ in range(world_size)]
        dist.all_gather(gathered_signals, local_signals)
        dist.all_gather(gathered_losses, local_losses)
        global_signals = torch.cat(gathered_signals)
        global_losses = torch.cat(gathered_losses)
        gathered_records: list[list[dict[str, Any]] | None] = [None] * world_size
        public_records = [
            {key: value for key, value in record.items() if key != "rollout"}
            for record in local_records
        ]
        dist.all_gather_object(gathered_records, public_records)
        global_records = [
            record
            for rank_records in gathered_records
            if rank_records is not None
            for record in rank_records
        ]
    else:
        global_signals = local_signals
        global_losses = local_losses
        global_records = [
            {key: value for key, value in record.items() if key != "rollout"}
            for record in local_records
        ]

    eps = float(get_nested(cfg, "scopd.trajectory_weighting.eps", 1e-8))
    weight_transform = "median_inverse"
    weight_temperature: float | None = None
    frontier_state_metrics: dict[str, Any] = {}
    global_ratios: torch.Tensor | None = None
    global_projection_mass: torch.Tensor | None = None
    global_teacher_js_mass: torch.Tensor | None = None
    global_budget_js_mass: torch.Tensor | None = None
    if mode in COUNTERFACTUAL_TEACHABILITY_MODES:
        global_ratios = torch.tensor(
            [float(record["ratio"]) for record in global_records],
            dtype=torch.float32,
            device=global_signals.device,
        )
        global_projection_mass = torch.tensor(
            [float(record["projection_mass"]) for record in global_records],
            dtype=torch.float32,
            device=global_signals.device,
        )
        global_teacher_js_mass = torch.tensor(
            [float(record["teacher_js_mass"]) for record in global_records],
            dtype=torch.float32,
            device=global_signals.device,
        )
        global_budget_js_mass = torch.tensor(
            [float(record["jsd"]) for record in global_records],
            dtype=torch.float32,
            device=global_signals.device,
        )
    if mode == "global_calibrated_counterfactual_teachability_batch":
        calibration = global_trajectory_calibration(cfg)
        weights = globally_calibrated_trajectory_weights(
            global_signals,
            q05=calibration["q05"],
            q95=calibration["q95"],
            calibration_mean=calibration["normalized_mean"],
            coefficient=calibration["coefficient"],
            eps=eps,
        )
        group_signal = global_signals
        batch_tau = None
        weight_transform = "fixed_global_robust_affine"
        frontier_state_metrics = {
            "trajectory_global_calibration_q05": calibration["q05"],
            "trajectory_global_calibration_q95": calibration["q95"],
            "trajectory_global_calibration_normalized_mean": calibration[
                "normalized_mean"
            ],
            "trajectory_global_calibration_coefficient": calibration[
                "coefficient"
            ],
            "trajectory_global_normalized_signal_min": float(
                weights.normalized_signal.min().cpu()
            ),
            "trajectory_global_normalized_signal_mean": float(
                weights.normalized_signal.mean().cpu()
            ),
            "trajectory_global_normalized_signal_max": float(
                weights.normalized_signal.max().cpu()
            ),
            "trajectory_global_objective_weight_min": float(
                weights.objective_weight.min().cpu()
            ),
            "trajectory_global_objective_weight_mean": float(
                weights.objective_weight.mean().cpu()
            ),
            "trajectory_global_objective_weight_max": float(
                weights.objective_weight.max().cpu()
            ),
            "trajectory_global_batch_renormalized": False,
        }
    elif mode == "global_f_intermediate_curriculum_batch":
        gamma = float(get_nested(cfg, "scopd.trajectory_weighting.gamma", 4.0))
        weights = global_f_curriculum_trajectory_weights(
            global_signals,
            gamma=gamma,
        )
        group_signal = global_signals
        batch_tau = None
        weight_transform = "fixed_global_gamma_f_one_minus_f"
        frontier_state_metrics = {
            "trajectory_global_curriculum_gamma": gamma,
            "trajectory_global_clipped_signal_min": float(
                weights.clipped_signal.min().cpu()
            ),
            "trajectory_global_clipped_signal_mean": float(
                weights.clipped_signal.mean().cpu()
            ),
            "trajectory_global_clipped_signal_max": float(
                weights.clipped_signal.max().cpu()
            ),
            "trajectory_global_objective_weight_min": float(
                weights.objective_weight.min().cpu()
            ),
            "trajectory_global_objective_weight_mean": float(
                weights.objective_weight.mean().cpu()
            ),
            "trajectory_global_objective_weight_max": float(
                weights.objective_weight.max().cpu()
            ),
            "trajectory_global_batch_renormalized": False,
        }
    elif mode in {
        "trajectory_projection_fraction_top20_batch",
        "trajectory_projection_fraction_bottom80_batch",
    }:
        global_indices = [int(record["global_index"]) for record in global_records]
        batch_ordinal = min(global_indices) // len(global_indices)
        selection = (
            "top"
            if mode == "trajectory_projection_fraction_top20_batch"
            else "bottom"
        )
        weights = hard_trajectory_partition_weights(
            global_signals,
            top_fraction=float(
                get_nested(cfg, "scopd.trajectory_weighting.top_fraction", 0.2)
            ),
            batch_ordinal=batch_ordinal,
            select=selection,
        )
        group_signal = global_signals
        batch_tau = None
        weight_transform = f"hard_projection_fraction_{selection}_partition"
        frontier_state_metrics = {
            "trajectory_partition_selection": selection,
            "trajectory_partition_top_fraction": float(
                get_nested(cfg, "scopd.trajectory_weighting.top_fraction", 0.2)
            ),
            "trajectory_partition_batch_ordinal": batch_ordinal,
            "trajectory_partition_top_count": int(weights.top_count),
            "trajectory_partition_selected_count": int(weights.selected_mask.sum().cpu()),
            "trajectory_partition_selected_fraction": float(
                weights.selected_mask.float().mean().cpu()
            ),
            "trajectory_global_batch_renormalized": True,
        }
    elif mode == "progress_adaptive_robust_frontier_batch":
        if not isinstance(curriculum_state, ProgressAdaptiveFrontierState):
            raise AssertionError(
                "Progress-adaptive robust frontier requires its synchronized state."
            )
        batch_tau = curriculum_state.tau
        weights = competence_frontier_probability_weights(
            global_losses,
            global_signals,
            tau=batch_tau,
            eps=eps,
        )
        group_signal = global_signals
        weight_transform = "progress_adaptive_robust_competence_frontier"
        state_update = curriculum_state.update(
            float(global_signals.mean().cpu()),
            int(global_signals.numel()),
        )
        frontier_state_metrics = {
            "trajectory_frontier_initial_tau": float(curriculum_state.initial_tau),
            "trajectory_frontier_initial_robust_need_mean": float(
                curriculum_state.initial_robust_need_mean
            ),
            "trajectory_frontier_tau": float(batch_tau),
            "trajectory_frontier_tau_after_update": float(state_update["tau_after"]),
            "trajectory_frontier_progress_after_update": float(
                state_update["progress_after"]
            ),
            "trajectory_frontier_ema_robust_need": float(
                state_update["ema_robust_need_mean"]
            ),
            "trajectory_frontier_ema_decay": float(state_update["ema_decay"]),
            "trajectory_frontier_updates": int(state_update["updates"]),
            "trajectory_frontier_trajectories_seen": int(
                state_update["trajectories_seen"]
            ),
            "trajectory_frontier_loss_mass_scale": float(
                weights.loss_mass_scale.cpu()
            ),
        }
    elif mode == "adaptive_budget_frontier_sampler_batch":
        if not isinstance(curriculum_state, AdaptiveBudgetFrontierState):
            raise AssertionError(
                "Adaptive budget sampling requires synchronized sampler state."
            )
        probabilities_before = curriculum_state.probabilities()
        tau_before = curriculum_state.tau
        weights = uniform_trajectory_probability_weights(
            int(global_signals.numel()), device=global_signals.device
        )
        group_signal = global_signals
        batch_tau = tau_before
        weight_transform = "adaptive_budget_frontier_sampling_with_vanilla_scopd_loss"
        state_update = curriculum_state.update(
            [float(record["ratio"]) for record in global_records],
            [float(record["signal"]) for record in global_records],
        )
        frontier_state_metrics = {
            "budget_sampler_metric": str(
                get_nested(
                    cfg,
                    "scopd.trajectory_weighting.sampler_metric",
                    "robust_need",
                )
            ).strip().lower(),
            "budget_sampler_ready_before": bool(state_update["ready_before"]),
            "budget_sampler_ready_after": bool(state_update["ready_after"]),
            "budget_sampler_tau": tau_before,
            "budget_sampler_tau_after_update": state_update["tau_after"],
            "budget_sampler_initial_tau": state_update["initial_tau"],
            "budget_sampler_initial_metric_mean": state_update[
                "initial_metric_mean"
            ],
            "budget_sampler_probabilities_before": {
                f"r{int(round(ratio * 100)):03d}": float(probability)
                for ratio, probability in zip(
                    curriculum_state.retention_ratios, probabilities_before
                )
            },
            "budget_sampler_probabilities_after": {
                f"r{int(round(ratio * 100)):03d}": float(probability)
                for ratio, probability in zip(
                    curriculum_state.retention_ratios,
                    state_update["probabilities_after"],
                )
            },
            "budget_sampler_initial_group_metric": state_update[
                "initial_group_metric"
            ],
            "budget_sampler_ema_group_metric": state_update["ema_group_metric"],
            "budget_sampler_calibration_counts": state_update[
                "calibration_counts"
            ],
            "budget_sampler_calibration_complete_at_trajectory": state_update[
                "calibration_complete_at_trajectory"
            ],
            "budget_sampler_updates": int(state_update["updates"]),
            "budget_sampler_trajectories_seen": int(
                state_update["trajectories_seen"]
            ),
            "budget_sampler_loss_is_vanilla_mean": True,
            "budget_sampler_state_update_scope": "synchronized_microbatch",
        }
    elif mode == "ratio_group_counterfactual_teachability_batch":
        group_transform, group_temperature = ratio_group_weight_transform(cfg)
        group_statistic = ratio_group_statistic(cfg)
        if group_statistic == "teacher_directed_projection_cosine":
            if group_transform != "softmax" or group_temperature is None:
                raise ValueError("Ratio-group angle weighting requires softmax.")
            weights = ratio_group_angle_probability_weights(
                global_projection_mass,
                global_teacher_js_mass,
                global_budget_js_mass,
                global_ratios,
                temperature=group_temperature,
                eps=eps,
            )
            weight_transform = "ratio_group_softmax_angle"
        elif group_statistic == "teacher_directed_projection_cosine_sample_normalized":
            if group_transform != "softmax" or group_temperature is None:
                raise ValueError(
                    "Sample-normalized ratio-group angle weighting requires softmax."
                )
            weights = ratio_group_angle_sample_probability_weights(
                global_projection_mass,
                global_teacher_js_mass,
                global_budget_js_mass,
                global_ratios,
                temperature=group_temperature,
                eps=eps,
            )
            weight_transform = "ratio_group_softmax_angle_sample_normalized"
        elif group_statistic == "teacher_directed_projection_fraction_exact":
            if group_transform != "softmax" or group_temperature is None:
                raise ValueError("Exact ratio-group projection weighting requires softmax.")
            weights = ratio_group_projection_probability_weights(
                global_projection_mass,
                global_teacher_js_mass,
                global_ratios,
                temperature=group_temperature,
                eps=eps,
            )
            weight_transform = "ratio_group_softmax_projection_exact"
        else:
            weights = ratio_group_fraction_probability_weights(
                global_projection_mass,
                global_teacher_js_mass,
                global_ratios,
                transform=group_transform,
                temperature=group_temperature,
                eps=eps,
            )
            weight_transform = f"ratio_group_{group_transform}_projection"
        batch_tau: float | None = None
        group_signal = weights.group_signal
        weight_temperature = group_temperature
    elif mode == "trajectory_counterfactual_teachability_softmax_batch":
        weight_temperature = float(
            get_nested(cfg, "scopd.trajectory_weighting.temperature")
        )
        weights = softmax_trajectory_signal_probability_weights(
            global_signals,
            temperature=weight_temperature,
        )
        batch_tau = None
        group_signal = global_signals
        weight_transform = "trajectory_softmax_projection"
    elif mode == "jsd_over_current_kl_direct_inverse_batch":
        weights = direct_inverse_sensitivity_probability_weights(global_signals, eps=eps)
        batch_tau = None
        group_signal = global_signals
        weight_transform = "direct_inverse"
    elif mode == "jsd_over_current_kl_softmax_batch":
        weights = softmax_inverse_sensitivity_probability_weights(
            global_signals,
            temperature=float(
                get_nested(cfg, "scopd.trajectory_weighting.temperature")
            ),
        )
        batch_tau = None
        group_signal = global_signals
        weight_transform = "softmax_negative_sensitivity"
        weight_temperature = float(
            get_nested(cfg, "scopd.trajectory_weighting.temperature")
        )
    else:
        weights = inverse_sensitivity_probability_weights(global_signals, eps=eps)
        batch_tau = float(weights.tau.cpu())
        group_signal = global_signals
    expected_size = int(accumulation_steps) * (world_size if use_distributed else 1)
    if int(weights.probability_weight.numel()) != expected_size:
        raise AssertionError(
            f"Effective-batch weight count mismatch: {weights.probability_weight.numel()} vs {expected_size}."
        )
    start = rank * int(accumulation_steps) if use_distributed else 0
    local_weights = weights.probability_weight[start : start + int(accumulation_steps)]
    local_group_signal = group_signal[start : start + int(accumulation_steps)]
    for record, probability_weight, current_group_signal in zip(
        local_records, local_weights, local_group_signal
    ):
        record["probability_weight"] = float(probability_weight.cpu())
        record["ratio_group_signal"] = float(current_group_signal.cpu())
        record["weight_signal"] = float(current_group_signal.cpu())
    for record, probability_weight, current_group_signal in zip(
        global_records, weights.probability_weight, group_signal
    ):
        record["probability_weight"] = float(probability_weight.cpu())
        record["ratio_group_signal"] = float(current_group_signal.cpu())
        record["weight_signal"] = float(current_group_signal.cpu())

    weighted_kl = (weights.probability_weight * global_losses).sum()
    unweighted_kl = global_losses.mean()
    effective_multiplier = weights.probability_weight * expected_size
    effective_sample_size = weights.probability_weight.sum().square().div(
        weights.probability_weight.square().sum().clamp_min(eps)
    )
    ratio_group_stats: dict[str, dict[str, float | int]] = {}
    if mode in COUNTERFACTUAL_TEACHABILITY_MODES:
        if (
            global_ratios is None
            or global_projection_mass is None
            or global_teacher_js_mass is None
            or global_budget_js_mass is None
        ):
            raise AssertionError("Counterfactual teachability tensors are unavailable.")
        for ratio in torch.unique(global_ratios, sorted=True):
            members = torch.isclose(global_ratios, ratio, rtol=0.0, atol=1e-6)
            group_probability = weights.probability_weight[members]
            group_losses = global_losses[members]
            weighted_contribution = (group_probability * group_losses).sum()
            ratio_group_stats[f"r{int(round(float(ratio.cpu()) * 100)):03d}"] = {
                "retention_ratio": float(ratio.cpu()),
                "count": int(members.sum().cpu()),
                "projection_mass_sum": float(global_projection_mass[members].sum().cpu()),
                "teacher_js_mass_sum": float(global_teacher_js_mass[members].sum().cpu()),
                "budget_js_mass_sum": float(global_budget_js_mass[members].sum().cpu()),
                "projection_fraction": float(
                    global_projection_mass[members]
                    .sum()
                    .div(global_teacher_js_mass[members].sum().clamp_min(eps))
                    .clamp(0.0, 1.0)
                    .cpu()
                ),
                "projection_cosine": float(
                    global_projection_mass[members]
                    .sum()
                    .div(
                        (
                            global_teacher_js_mass[members].sum()
                            * global_budget_js_mass[members].sum()
                        )
                        .clamp_min(0.0)
                        .sqrt()
                        .clamp_min(eps)
                    )
                    .clamp(-1.0, 1.0)
                    .cpu()
                ),
                "trajectory_signal_min": float(global_signals[members].min().cpu()),
                "trajectory_signal_mean": float(global_signals[members].mean().cpu()),
                "trajectory_signal_max": float(global_signals[members].max().cpu()),
                "probability_mass": float(group_probability.sum().cpu()),
                "mean_probability_weight": float(group_probability.mean().cpu()),
                "mean_multiplier": float(effective_multiplier[members].mean().cpu()),
                "unweighted_kl_mean": float(group_losses.mean().cpu()),
                "unweighted_kl_contribution_fraction": float(
                    group_losses.sum().div(global_losses.sum().clamp_min(eps)).cpu()
                ),
                "weighted_kl_contribution": float(weighted_contribution.cpu()),
                "weighted_kl_contribution_fraction": float(
                    weighted_contribution.div(weighted_kl.clamp_min(eps)).cpu()
                ),
            }
    summary = {
        "trajectory_weighting_mode": mode,
        "trajectory_normalization": (
            "fixed_global_calibration_no_batch_renormalization"
            if mode == "global_calibrated_counterfactual_teachability_batch"
            else "fixed_global_f_curriculum_no_batch_renormalization"
            if mode == "global_f_intermediate_curriculum_batch"
            else "effective_batch_kl_mass_preserving"
            if mode == "progress_adaptive_robust_frontier_batch"
            else (
                "vanilla_mean_after_adaptive_ratio_sampling"
                if mode == "adaptive_budget_frontier_sampler_batch"
                else "effective_batch_probability_sum_one"
            )
        ),
        "trajectory_effective_batch_size": expected_size,
        "trajectory_probability_weight_sum": float(weights.probability_weight.sum().cpu()),
        "trajectory_probability_weight_min": float(weights.probability_weight.min().cpu()),
        "trajectory_probability_weight_max": float(weights.probability_weight.max().cpu()),
        "trajectory_effective_multiplier_min": float(
            (weights.probability_weight * expected_size).min().cpu()
        ),
        "trajectory_effective_multiplier_max": float(
            (weights.probability_weight * expected_size).max().cpu()
        ),
        "trajectory_batch_tau": batch_tau,
        "trajectory_weight_transform": weight_transform,
        "trajectory_weight_temperature": weight_temperature,
        "trajectory_ratio_group_signal_min": float(group_signal.min().cpu()),
        "trajectory_ratio_group_signal_mean": float(group_signal.mean().cpu()),
        "trajectory_ratio_group_signal_max": float(group_signal.max().cpu()),
        "trajectory_weight_signal_min": float(group_signal.min().cpu()),
        "trajectory_weight_signal_mean": float(group_signal.mean().cpu()),
        "trajectory_weight_signal_max": float(group_signal.max().cpu()),
        "trajectory_signal_min": float(global_signals.min().cpu()),
        "trajectory_signal_mean": float(global_signals.mean().cpu()),
        "trajectory_signal_max": float(global_signals.max().cpu()),
        "trajectory_global_unweighted_kl": float(unweighted_kl.cpu()),
        "trajectory_global_weighted_kl": float(weighted_kl.cpu()),
        "trajectory_global_scalar_error": float((weighted_kl - unweighted_kl).cpu()),
        "trajectory_loss_scale_ratio": float(
            (weighted_kl / unweighted_kl.clamp_min(eps)).cpu()
        ),
        "trajectory_effective_sample_size": float(effective_sample_size.cpu()),
        "trajectory_effective_sample_size_fraction": float(
            effective_sample_size.div(expected_size).cpu()
        ),
        "trajectory_ratio_groups": ratio_group_stats,
        "trajectory_weight_detached": not weights.probability_weight.requires_grad,
        "global_records": global_records,
        **frontier_state_metrics,
    }
    return local_records, summary


def initialize_trajectory_curriculum_state(
    cfg: dict[str, Any],
) -> (
    AdaptiveBudgetFrontierState
    |
    ProgressAdaptiveFrontierState
    | RobustnessGatedCurriculumState
    | SensitivityFrontierState
    | None
):
    if not bool(get_nested(cfg, "scopd.trajectory_weighting.enabled", False)):
        return None
    mode = str(get_nested(cfg, "scopd.trajectory_weighting.mode", "")).strip().lower()
    if mode == "adaptive_budget_frontier_sampler_batch":
        ratios = tuple(
            float(value)
            for value in get_nested(
                cfg, "pruning.train_retention_ratios", [0.1, 0.2, 0.3, 0.4]
            )
        )
        return AdaptiveBudgetFrontierState(
            retention_ratios=ratios,
            calibration_target_per_ratio=int(
                get_nested(
                    cfg,
                    "scopd.trajectory_weighting.calibration_target_per_ratio",
                    64,
                )
            ),
            ema_half_life_per_ratio=float(
                get_nested(
                    cfg,
                    "scopd.trajectory_weighting.ema_half_life_per_ratio",
                    64.0,
                )
            ),
            seed=int(
                get_nested(
                    cfg,
                    "paired_sampling.ratio_seed",
                    get_nested(cfg, "training.seed", 42),
                )
            ),
            namespace=str(
                get_nested(cfg, "paired_sampling.namespace", "scopd_pair_v1")
            )
            + ":adaptive_budget_frontier",
            eps=float(get_nested(cfg, "scopd.trajectory_weighting.eps", 1e-8)),
        )
    if mode == "progress_adaptive_robust_frontier_batch":
        calibration = get_nested(cfg, "scopd.trajectory_weighting.calibration", None)
        if not isinstance(calibration, dict):
            raise ValueError(
                "progress_adaptive_robust_frontier_batch requires frozen calibration."
            )
        initial_tau = float(calibration["initial_tau"])
        initial_mean = float(calibration["initial_robust_need_mean"])
        return ProgressAdaptiveFrontierState(
            initial_tau=initial_tau,
            initial_robust_need_mean=initial_mean,
            ema_robust_need_mean=initial_mean,
            ema_half_life_trajectories=float(
                get_nested(
                    cfg,
                    "scopd.trajectory_weighting.ema_half_life_trajectories",
                    256.0,
                )
            ),
        )
    if mode == "sensitivity_frontier":
        return SensitivityFrontierState(
            calibration_target_per_ratio=int(
                get_nested(cfg, "scopd.trajectory_weighting.calibration_target_per_ratio", 64)
            ),
            ema_half_life_trajectories=float(
                get_nested(cfg, "scopd.trajectory_weighting.ema_half_life_trajectories", 256.0)
            ),
            progress_drop_scale=float(
                get_nested(cfg, "scopd.trajectory_weighting.progress_drop_scale", 0.5)
            ),
            progress_power=float(
                get_nested(cfg, "scopd.trajectory_weighting.progress_power", 2.0)
            ),
        )
    if mode != "robustness_gated_curriculum":
        return None
    calibration = get_nested(cfg, "scopd.trajectory_weighting.calibration", None)
    if not isinstance(calibration, dict):
        raise ValueError(
            "robustness_gated_curriculum requires a frozen trajectory_weighting.calibration mapping."
        )
    initial_gap = float(calibration["initial_teacher_gap_mean"])
    return RobustnessGatedCurriculumState(
        initial_teacher_gap_mean=initial_gap,
        ema_teacher_gap_mean=initial_gap,
        ema_half_life_trajectories=float(
            get_nested(cfg, "scopd.trajectory_weighting.ema_half_life_trajectories", 256.0)
        ),
        progress_power=float(get_nested(cfg, "scopd.trajectory_weighting.progress_power", 3.0)),
    )


def apply_distributed_trajectory_weighting(
    batch_losses: list[torch.Tensor],
    batch_metrics: list[dict[str, Any]],
    cfg: dict[str, Any],
    *,
    distributed: bool,
    rank: int,
    world_size: int,
    curriculum_state: (
        AdaptiveBudgetFrontierState
        |
        ProgressAdaptiveFrontierState
        | RobustnessGatedCurriculumState
        | SensitivityFrontierState
        | None
    ) = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Redistribute whole-trajectory SCOPD gradients across a synchronized DDP block."""

    enabled = bool(get_nested(cfg, "scopd.trajectory_weighting.enabled", False))
    vanilla = torch.stack(batch_losses).mean()
    if not enabled:
        return vanilla, {"trajectory_weighting_enabled": False}
    mode = str(get_nested(cfg, "scopd.trajectory_weighting.mode", "closure_rank")).strip().lower()
    probability_modes = {
        "global_calibrated_counterfactual_teachability_batch",
        "global_f_intermediate_curriculum_batch",
        "jsd_over_current_kl_batch",
        "jsd_over_current_kl_direct_inverse_batch",
        "jsd_over_current_kl_softmax_batch",
        "jsd_over_step0_kl_batch",
        "ratio_group_counterfactual_teachability_batch",
        "trajectory_counterfactual_teachability_softmax_batch",
        "progress_adaptive_robust_frontier_batch",
        "adaptive_budget_frontier_sampler_batch",
    }
    if mode not in probability_modes and (
        not distributed or not dist.is_initialized() or world_size < 2
    ):
        raise ValueError("Trajectory rank weighting requires synchronized multi-rank DDP.")

    signal_spec = {
        "closure_rank": ("native_trajectory_budget_closure_mass", True),
        "teacher_gap_rank": ("native_teacher_gap_b_mean", True),
        "robustness_rank": ("native_sensitivity_mean", False),
        "relative_robustness_rank": ("derived_relative_sensitivity", False),
        "residual_robustness_rank": ("derived_residual_sensitivity", False),
        "residual_robustness_soft": ("derived_residual_sensitivity", False),
        "robustness_gated_curriculum": ("native_trajectory_mass_robustness", True),
        "sensitivity_frontier": ("native_trajectory_teacher_gap_sensitivity", False),
        "jsd_over_current_kl_batch": ("derived_jsd_over_current_kl", False),
        "jsd_over_current_kl_direct_inverse_batch": (
            "derived_jsd_over_current_kl",
            False,
        ),
        "jsd_over_current_kl_softmax_batch": (
            "derived_jsd_over_current_kl",
            False,
        ),
        "jsd_over_step0_kl_batch": ("derived_jsd_over_step0_kl", False),
        "ratio_group_counterfactual_teachability_batch": (
            "native_trajectory_budget_explained_fraction",
            True,
        ),
        "global_calibrated_counterfactual_teachability_batch": (
            "native_trajectory_budget_explained_fraction",
            True,
        ),
        "global_f_intermediate_curriculum_batch": (
            "native_trajectory_budget_explained_fraction",
            True,
        ),
        "trajectory_counterfactual_teachability_softmax_batch": (
            "native_trajectory_budget_explained_fraction",
            True,
        ),
        "progress_adaptive_robust_frontier_batch": (
            "derived_progress_adaptive_robust_need",
            True,
        ),
        "adaptive_budget_frontier_sampler_batch": (
            "derived_adaptive_budget_frontier_signal",
            True,
        ),
    }
    if mode not in signal_spec:
        raise ValueError(
            f"Unknown trajectory weighting mode {mode!r}; expected one of {sorted(signal_spec)}."
        )
    signal_key, higher_is_better = signal_spec[mode]
    if len(batch_losses) != len(batch_metrics):
        raise AssertionError("Every trajectory loss must have one metrics record.")
    device = batch_losses[0].device
    required_keys = {signal_key}
    if mode == "progress_adaptive_robust_frontier_batch":
        required_keys = {
            "native_teacher_gap_b_mean",
            "native_trajectory_budget_explained_fraction",
        }
    elif mode == "adaptive_budget_frontier_sampler_batch":
        if not isinstance(curriculum_state, AdaptiveBudgetFrontierState):
            raise ValueError(
                "adaptive_budget_frontier_sampler_batch requires initialized state."
            )
        required_keys = {
            "native_teacher_gap_b_mean",
            "native_trajectory_budget_explained_fraction",
            "sampled_b",
        }
    elif mode in {
        "relative_robustness_rank",
        "residual_robustness_rank",
        "residual_robustness_soft",
    }:
        required_keys = {"native_sensitivity_mean", "native_teacher_gap_b_mean"}
    if mode in {"residual_robustness_rank", "residual_robustness_soft"}:
        required_keys.add("sampled_b")
    if mode == "progress_adaptive_robust_frontier_batch":
        if not isinstance(curriculum_state, ProgressAdaptiveFrontierState):
            raise ValueError(
                "progress_adaptive_robust_frontier_batch requires initialized state."
            )
    elif mode == "robustness_gated_curriculum":
        required_keys = {
            "native_trajectory_mass_robustness",
            "native_teacher_gap_b_mean",
        }
    if mode == "sensitivity_frontier":
        required_keys = {
            "native_trajectory_teacher_gap_sensitivity",
            "sampled_b",
        }
    if mode in {
        "jsd_over_current_kl_batch",
        "jsd_over_current_kl_direct_inverse_batch",
        "jsd_over_current_kl_softmax_batch",
    }:
        required_keys = {
            "native_student_budget_jsd_mean",
            "native_teacher_gap_b_mean",
        }
    if mode == "jsd_over_step0_kl_batch":
        required_keys = {
            "native_student_budget_jsd_mean",
            "sampled_b",
        }
    if mode in {
        "ratio_group_counterfactual_teachability_batch",
        "global_calibrated_counterfactual_teachability_batch",
        "global_f_intermediate_curriculum_batch",
    }:
        required_keys = {
            "native_trajectory_budget_explained_fraction",
            "native_trajectory_budget_projection_mass",
            "native_trajectory_teacher_js_mass",
            "sampled_b",
        }
        if ratio_group_statistic(cfg) in {
            "teacher_directed_projection_cosine",
            "teacher_directed_projection_cosine_sample_normalized",
        }:
            required_keys.add("native_student_budget_jsd_mean")
    for required_key in sorted(required_keys):
        missing = [index for index, item in enumerate(batch_metrics) if required_key not in item]
        if missing:
            raise KeyError(
                f"Trajectory signal input {required_key!r} missing for local items {missing}."
            )

    local_losses = torch.stack([item.detach().float() for item in batch_losses])
    local_teacher_gaps: torch.Tensor | None = None
    local_ratios: torch.Tensor | None = None
    local_projection_mass: torch.Tensor | None = None
    local_teacher_js_mass: torch.Tensor | None = None
    local_budget_js_mass: torch.Tensor | None = None
    if mode == "progress_adaptive_robust_frontier_batch":
        local_signals = torch.tensor(
            [
                float(item["native_teacher_gap_b_mean"])
                * (
                    1.0
                    - min(
                        max(
                            float(
                                item[
                                    "native_trajectory_budget_explained_fraction"
                                ]
                            ),
                            0.0,
                        ),
                        1.0,
                    )
                )
                for item in batch_metrics
            ],
            dtype=torch.float32,
            device=device,
        )
    elif mode == "adaptive_budget_frontier_sampler_batch":
        sampler_metric = str(
            get_nested(
                cfg,
                "scopd.trajectory_weighting.sampler_metric",
                "robust_need",
            )
        ).strip().lower()
        if sampler_metric == "robust_need":
            local_signals = torch.tensor(
                [
                    float(item["native_teacher_gap_b_mean"])
                    * (
                        1.0
                        - min(
                            max(
                                float(
                                    item[
                                        "native_trajectory_budget_explained_fraction"
                                    ]
                                ),
                                0.0,
                            ),
                            1.0,
                        )
                    )
                    for item in batch_metrics
                ],
                dtype=torch.float32,
                device=device,
            )
        elif sampler_metric == "teacher_kl":
            local_signals = torch.tensor(
                [float(item["native_teacher_gap_b_mean"]) for item in batch_metrics],
                dtype=torch.float32,
                device=device,
            )
        else:
            raise ValueError(f"Unsupported adaptive sampler metric: {sampler_metric!r}.")
        local_ratios = torch.tensor(
            [float(item["sampled_b"]) for item in batch_metrics],
            dtype=torch.float32,
            device=device,
        )
    elif mode == "robustness_gated_curriculum":
        if curriculum_state is None:
            raise ValueError("robustness_gated_curriculum requires initialized curriculum state.")
        local_signals = torch.tensor(
            [float(item["native_trajectory_mass_robustness"]) for item in batch_metrics],
            dtype=torch.float32,
            device=device,
        )
        local_teacher_gaps = torch.tensor(
            [max(float(item["native_teacher_gap_b_mean"]), 0.0) for item in batch_metrics],
            dtype=torch.float32,
            device=device,
        )
    elif mode == "sensitivity_frontier":
        if not isinstance(curriculum_state, SensitivityFrontierState):
            raise ValueError("sensitivity_frontier requires initialized frontier state.")
        local_signals = torch.tensor(
            [float(item["native_trajectory_teacher_gap_sensitivity"]) for item in batch_metrics],
            dtype=torch.float32,
            device=device,
        )
        local_ratios = torch.tensor(
            [float(item["sampled_b"]) for item in batch_metrics],
            dtype=torch.float32,
            device=device,
        )
    elif mode in {
        "ratio_group_counterfactual_teachability_batch",
        "global_calibrated_counterfactual_teachability_batch",
        "global_f_intermediate_curriculum_batch",
    }:
        if mode in DIRECT_GLOBAL_F_MODES:
            local_signals = torch.tensor(
                [
                    float(item["native_trajectory_budget_projection_mass"])
                    / max(float(item["native_trajectory_teacher_js_mass"]), 1e-8)
                    for item in batch_metrics
                ],
                dtype=torch.float32,
                device=device,
            )
        else:
            local_signals = torch.tensor(
                [
                    float(item["native_trajectory_budget_explained_fraction"])
                    for item in batch_metrics
                ],
                dtype=torch.float32,
                device=device,
            )
        local_ratios = torch.tensor(
            [float(item["sampled_b"]) for item in batch_metrics],
            dtype=torch.float32,
            device=device,
        )
        local_projection_mass = torch.tensor(
            [
                float(item["native_trajectory_budget_projection_mass"])
                for item in batch_metrics
            ],
            dtype=torch.float32,
            device=device,
        )
        local_teacher_js_mass = torch.tensor(
            [float(item["native_trajectory_teacher_js_mass"]) for item in batch_metrics],
            dtype=torch.float32,
            device=device,
        )
        local_budget_js_mass = torch.tensor(
            [float(item.get("native_student_budget_jsd_mean", 0.0)) for item in batch_metrics],
            dtype=torch.float32,
            device=device,
        )
    elif mode == "relative_robustness_rank":
        local_signals = torch.tensor(
            [
                float(item["native_sensitivity_mean"])
                / max(float(item["native_teacher_gap_b_mean"]), 1e-8)
                for item in batch_metrics
            ],
            dtype=torch.float32,
            device=device,
        )
    elif mode in {
        "jsd_over_current_kl_batch",
        "jsd_over_current_kl_direct_inverse_batch",
        "jsd_over_current_kl_softmax_batch",
    }:
        eps = float(get_nested(cfg, "scopd.trajectory_weighting.eps", 1e-8))
        local_signals = torch.tensor(
            [
                max(float(item["native_student_budget_jsd_mean"]), 0.0)
                / max(float(item["native_teacher_gap_b_mean"]), eps)
                for item in batch_metrics
            ],
            dtype=torch.float32,
            device=device,
        )
    elif mode == "jsd_over_step0_kl_batch":
        calibration = get_nested(
            cfg,
            "scopd.trajectory_weighting.step0_teacher_kl_by_ratio",
            None,
        )
        if not isinstance(calibration, dict):
            raise ValueError(
                "jsd_over_step0_kl_batch requires step0_teacher_kl_by_ratio calibration."
            )

        def step0_kl(item: dict[str, Any]) -> float:
            key = f"{float(item['sampled_b']):.2f}"
            if key not in calibration:
                raise KeyError(f"Missing step-0 teacher KL calibration for ratio {key}.")
            value = float(calibration[key])
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"Invalid step-0 teacher KL calibration for ratio {key}: {value}.")
            return value

        local_signals = torch.tensor(
            [
                max(float(item["native_student_budget_jsd_mean"]), 0.0)
                / step0_kl(item)
                for item in batch_metrics
            ],
            dtype=torch.float32,
            device=device,
        )
    elif mode in {"residual_robustness_rank", "residual_robustness_soft"}:
        calibration = get_nested(cfg, "scopd.trajectory_weighting.residual_calibration", None)
        if not isinstance(calibration, dict):
            raise ValueError("Residual robustness requires a frozen residual_calibration mapping.")
        local_signals = residualized_budget_sensitivity(
            torch.tensor(
                [float(item["native_sensitivity_mean"]) for item in batch_metrics],
                dtype=torch.float32,
                device=device,
            ),
            torch.tensor(
                [float(item["native_teacher_gap_b_mean"]) for item in batch_metrics],
                dtype=torch.float32,
                device=device,
            ),
            torch.tensor(
                [float(item["sampled_b"]) for item in batch_metrics],
                dtype=torch.float32,
                device=device,
            ),
            ratio_intercepts={
                str(key): float(value) for key, value in calibration["ratio_intercepts"].items()
            },
            ratio_log_teacher_gap_coefficients={
                str(key): float(value)
                for key, value in calibration["ratio_log_teacher_gap_coefficients"].items()
            },
            ratio_scales={str(key): float(value) for key, value in calibration["ratio_scales"].items()},
            eps=float(get_nested(cfg, "scopd.trajectory_weighting.eps", 1e-8)),
        )
    else:
        local_signals = torch.tensor(
            [float(item[signal_key]) for item in batch_metrics],
            dtype=torch.float32,
            device=device,
        )
    use_distributed_gather = distributed and dist.is_initialized() and world_size > 1
    if use_distributed_gather:
        gathered_losses = [torch.empty_like(local_losses) for _ in range(world_size)]
        gathered_signals = [torch.empty_like(local_signals) for _ in range(world_size)]
        dist.all_gather(gathered_losses, local_losses)
        dist.all_gather(gathered_signals, local_signals)
        global_losses = torch.cat(gathered_losses)
        global_signals = torch.cat(gathered_signals)
    else:
        global_losses = local_losses
        global_signals = local_signals
    global_teacher_gaps: torch.Tensor | None = None
    if local_teacher_gaps is not None:
        gathered_teacher_gaps = [torch.empty_like(local_teacher_gaps) for _ in range(world_size)]
        dist.all_gather(gathered_teacher_gaps, local_teacher_gaps)
        global_teacher_gaps = torch.cat(gathered_teacher_gaps)
    global_ratios: torch.Tensor | None = None
    if local_ratios is not None:
        if use_distributed_gather:
            gathered_ratios = [torch.empty_like(local_ratios) for _ in range(world_size)]
            dist.all_gather(gathered_ratios, local_ratios)
            global_ratios = torch.cat(gathered_ratios)
        else:
            global_ratios = local_ratios
    global_projection_mass: torch.Tensor | None = None
    global_teacher_js_mass: torch.Tensor | None = None
    global_budget_js_mass: torch.Tensor | None = None
    if (
        local_projection_mass is not None
        and local_teacher_js_mass is not None
        and local_budget_js_mass is not None
    ):
        if use_distributed_gather:
            gathered_projection_mass = [
                torch.empty_like(local_projection_mass) for _ in range(world_size)
            ]
            gathered_teacher_js_mass = [
                torch.empty_like(local_teacher_js_mass) for _ in range(world_size)
            ]
            gathered_budget_js_mass = [
                torch.empty_like(local_budget_js_mass) for _ in range(world_size)
            ]
            dist.all_gather(gathered_projection_mass, local_projection_mass)
            dist.all_gather(gathered_teacher_js_mass, local_teacher_js_mass)
            dist.all_gather(gathered_budget_js_mass, local_budget_js_mass)
            global_projection_mass = torch.cat(gathered_projection_mass)
            global_teacher_js_mass = torch.cat(gathered_teacher_js_mass)
            global_budget_js_mass = torch.cat(gathered_budget_js_mass)
        else:
            global_projection_mass = local_projection_mass
            global_teacher_js_mass = local_teacher_js_mass
            global_budget_js_mass = local_budget_js_mass
    strength = float(get_nested(cfg, "scopd.trajectory_weighting.downweight_strength", 0.25))
    eps = float(get_nested(cfg, "scopd.trajectory_weighting.eps", 1e-8))
    curriculum_metrics: dict[str, Any] = {}
    frontier_ratio_gate: torch.Tensor | None = None
    frontier_local_robustness: torch.Tensor | None = None
    if mode in probability_modes:
        weight_transform = "median_inverse"
        weight_temperature: float | None = None
        if mode == "global_calibrated_counterfactual_teachability_batch":
            calibration = global_trajectory_calibration(cfg)
            probability = globally_calibrated_trajectory_weights(
                global_signals,
                q05=calibration["q05"],
                q95=calibration["q95"],
                calibration_mean=calibration["normalized_mean"],
                coefficient=calibration["coefficient"],
                eps=eps,
            )
            batch_tau = None
            group_signal = global_signals
            weight_transform = "fixed_global_robust_affine"
            curriculum_metrics = {
                "trajectory_global_calibration_q05": calibration["q05"],
                "trajectory_global_calibration_q95": calibration["q95"],
                "trajectory_global_calibration_normalized_mean": calibration[
                    "normalized_mean"
                ],
                "trajectory_global_calibration_coefficient": calibration[
                    "coefficient"
                ],
                "trajectory_global_normalized_signal_min": float(
                    probability.normalized_signal.min().cpu()
                ),
                "trajectory_global_normalized_signal_mean": float(
                    probability.normalized_signal.mean().cpu()
                ),
                "trajectory_global_normalized_signal_max": float(
                    probability.normalized_signal.max().cpu()
                ),
                "trajectory_global_objective_weight_min": float(
                    probability.objective_weight.min().cpu()
                ),
                "trajectory_global_objective_weight_mean": float(
                    probability.objective_weight.mean().cpu()
                ),
                "trajectory_global_objective_weight_max": float(
                    probability.objective_weight.max().cpu()
                ),
                "trajectory_global_batch_renormalized": False,
            }
        elif mode == "global_f_intermediate_curriculum_batch":
            gamma = float(get_nested(cfg, "scopd.trajectory_weighting.gamma", 4.0))
            probability = global_f_curriculum_trajectory_weights(
                global_signals,
                gamma=gamma,
            )
            batch_tau = None
            group_signal = global_signals
            weight_transform = "fixed_global_gamma_f_one_minus_f"
            curriculum_metrics = {
                "trajectory_global_curriculum_gamma": gamma,
                "trajectory_global_clipped_signal_min": float(
                    probability.clipped_signal.min().cpu()
                ),
                "trajectory_global_clipped_signal_mean": float(
                    probability.clipped_signal.mean().cpu()
                ),
                "trajectory_global_clipped_signal_max": float(
                    probability.clipped_signal.max().cpu()
                ),
                "trajectory_global_objective_weight_min": float(
                    probability.objective_weight.min().cpu()
                ),
                "trajectory_global_objective_weight_mean": float(
                    probability.objective_weight.mean().cpu()
                ),
                "trajectory_global_objective_weight_max": float(
                    probability.objective_weight.max().cpu()
                ),
                "trajectory_global_batch_renormalized": False,
            }
        elif mode == "progress_adaptive_robust_frontier_batch":
            if not isinstance(curriculum_state, ProgressAdaptiveFrontierState):
                raise AssertionError(
                    "Progress-adaptive robust frontier requires its state."
                )
            batch_tau = curriculum_state.tau
            probability = competence_frontier_probability_weights(
                global_losses,
                global_signals,
                tau=batch_tau,
                eps=eps,
            )
            group_signal = global_signals
            weight_transform = "progress_adaptive_robust_competence_frontier"
            state_update = curriculum_state.update(
                float(global_signals.mean().cpu()), int(global_signals.numel())
            )
            curriculum_metrics = {
                "trajectory_frontier_tau": float(batch_tau),
                "trajectory_frontier_tau_after_update": float(
                    state_update["tau_after"]
                ),
                "trajectory_frontier_ema_robust_need": float(
                    state_update["ema_robust_need_mean"]
                ),
                "trajectory_frontier_loss_mass_scale": float(
                    probability.loss_mass_scale.cpu()
                ),
            }
        elif mode == "adaptive_budget_frontier_sampler_batch":
            if not isinstance(curriculum_state, AdaptiveBudgetFrontierState):
                raise AssertionError("Adaptive budget sampler state is unavailable.")
            if global_ratios is None:
                raise AssertionError("Adaptive budget sampler ratios are unavailable.")
            probabilities_before = curriculum_state.probabilities()
            tau_before = curriculum_state.tau
            probability = uniform_trajectory_probability_weights(
                int(global_signals.numel()), device=device
            )
            state_update = curriculum_state.update(
                [float(value) for value in global_ratios.detach().cpu().tolist()],
                [float(value) for value in global_signals.detach().cpu().tolist()],
            )
            batch_tau = tau_before
            group_signal = global_signals
            weight_transform = "adaptive_budget_frontier_sampling_with_vanilla_scopd_loss"
            curriculum_metrics = {
                "budget_sampler_metric": str(
                    get_nested(
                        cfg,
                        "scopd.trajectory_weighting.sampler_metric",
                        "robust_need",
                    )
                ).strip().lower(),
                "budget_sampler_ready_before": bool(state_update["ready_before"]),
                "budget_sampler_ready_after": bool(state_update["ready_after"]),
                "budget_sampler_tau": tau_before,
                "budget_sampler_tau_after_update": state_update["tau_after"],
                "budget_sampler_initial_tau": state_update["initial_tau"],
                "budget_sampler_initial_metric_mean": state_update[
                    "initial_metric_mean"
                ],
                "budget_sampler_probabilities_before": {
                    f"r{int(round(ratio * 100)):03d}": float(value)
                    for ratio, value in zip(
                        curriculum_state.retention_ratios, probabilities_before
                    )
                },
                "budget_sampler_probabilities_after": {
                    f"r{int(round(ratio * 100)):03d}": float(value)
                    for ratio, value in zip(
                        curriculum_state.retention_ratios,
                        state_update["probabilities_after"],
                    )
                },
                "budget_sampler_initial_group_metric": state_update[
                    "initial_group_metric"
                ],
                "budget_sampler_ema_group_metric": state_update[
                    "ema_group_metric"
                ],
                "budget_sampler_calibration_counts": state_update[
                    "calibration_counts"
                ],
                "budget_sampler_calibration_complete_at_trajectory": state_update[
                    "calibration_complete_at_trajectory"
                ],
                "budget_sampler_updates": int(state_update["updates"]),
                "budget_sampler_trajectories_seen": int(
                    state_update["trajectories_seen"]
                ),
                "budget_sampler_loss_is_vanilla_mean": True,
                "budget_sampler_state_update_scope": "synchronized_microbatch",
            }
        elif mode == "ratio_group_counterfactual_teachability_batch":
            if (
                global_ratios is None
                or global_projection_mass is None
                or global_teacher_js_mass is None
                or global_budget_js_mass is None
            ):
                raise AssertionError(
                    "Ratio-group teachability requires synchronized ratios and geometry masses."
                )
            group_transform, group_temperature = ratio_group_weight_transform(cfg)
            group_statistic = ratio_group_statistic(cfg)
            if group_statistic == "teacher_directed_projection_cosine":
                if group_transform != "softmax" or group_temperature is None:
                    raise ValueError("Ratio-group angle weighting requires softmax.")
                probability = ratio_group_angle_probability_weights(
                    global_projection_mass,
                    global_teacher_js_mass,
                    global_budget_js_mass,
                    global_ratios,
                    temperature=group_temperature,
                    eps=eps,
                )
                weight_transform = "ratio_group_softmax_angle"
            elif group_statistic == "teacher_directed_projection_cosine_sample_normalized":
                if group_transform != "softmax" or group_temperature is None:
                    raise ValueError(
                        "Sample-normalized ratio-group angle weighting requires softmax."
                    )
                probability = ratio_group_angle_sample_probability_weights(
                    global_projection_mass,
                    global_teacher_js_mass,
                    global_budget_js_mass,
                    global_ratios,
                    temperature=group_temperature,
                    eps=eps,
                )
                weight_transform = "ratio_group_softmax_angle_sample_normalized"
            elif group_statistic == "teacher_directed_projection_fraction_exact":
                if group_transform != "softmax" or group_temperature is None:
                    raise ValueError("Exact ratio-group projection weighting requires softmax.")
                probability = ratio_group_projection_probability_weights(
                    global_projection_mass,
                    global_teacher_js_mass,
                    global_ratios,
                    temperature=group_temperature,
                    eps=eps,
                )
                weight_transform = "ratio_group_softmax_projection_exact"
            else:
                probability = ratio_group_fraction_probability_weights(
                    global_projection_mass,
                    global_teacher_js_mass,
                    global_ratios,
                    transform=group_transform,
                    temperature=group_temperature,
                    eps=eps,
                )
                weight_transform = f"ratio_group_{group_transform}_projection"
            batch_tau: float | None = None
            group_signal = probability.group_signal
            weight_temperature = group_temperature
        elif mode == "trajectory_counterfactual_teachability_softmax_batch":
            weight_temperature = float(
                get_nested(cfg, "scopd.trajectory_weighting.temperature")
            )
            probability = softmax_trajectory_signal_probability_weights(
                global_signals,
                temperature=weight_temperature,
            )
            batch_tau = None
            group_signal = global_signals
            weight_transform = "trajectory_softmax_projection"
        elif mode == "jsd_over_current_kl_direct_inverse_batch":
            probability = direct_inverse_sensitivity_probability_weights(
                global_signals, eps=eps
            )
            batch_tau = None
            group_signal = global_signals
            weight_transform = "direct_inverse"
        elif mode == "jsd_over_current_kl_softmax_batch":
            probability = softmax_inverse_sensitivity_probability_weights(
                global_signals,
                temperature=float(
                    get_nested(cfg, "scopd.trajectory_weighting.temperature")
                ),
            )
            batch_tau = None
            group_signal = global_signals
            weight_transform = "softmax_negative_sensitivity"
            weight_temperature = float(
                get_nested(cfg, "scopd.trajectory_weighting.temperature")
            )
        else:
            probability = inverse_sensitivity_probability_weights(global_signals, eps=eps)
            batch_tau = float(probability.tau.cpu())
            group_signal = global_signals
        start = rank * len(batch_losses) if use_distributed_gather else 0
        end = start + len(batch_losses)
        local_probability = probability.probability_weight[start:end].to(device=device)
        local_loss_vector = torch.stack(batch_losses)
        ddp_scale = float(world_size) if use_distributed_gather else 1.0
        weighted = ddp_scale * (local_probability * local_loss_vector).sum()
        global_weighted_kl = (
            probability.probability_weight * global_losses
        ).sum()
        global_unweighted_kl = global_losses.mean()
        effective_multiplier = probability.probability_weight * global_losses.numel()
        return weighted, {
            "trajectory_weighting_enabled": True,
            "trajectory_weighting_mode": mode,
            "trajectory_signal_key": signal_key,
            "trajectory_signal": float(local_signals.mean().cpu()),
            "trajectory_signal_min": float(global_signals.min().cpu()),
            "trajectory_signal_mean": float(global_signals.mean().cpu()),
            "trajectory_signal_max": float(global_signals.max().cpu()),
            "trajectory_batch_tau": batch_tau,
            "trajectory_weight_transform": weight_transform,
            "trajectory_weight_temperature": weight_temperature,
            "trajectory_ratio_group_signal": float(
                group_signal[start:end].mean().cpu()
            ),
            "trajectory_raw_weight": float(probability.raw_weight[start:end].mean().cpu()),
            "trajectory_probability_weight": float(local_probability.mean().cpu()),
            "trajectory_probability_weight_sum": float(
                probability.probability_weight.sum().cpu()
            ),
            "trajectory_effective_multiplier": float(
                effective_multiplier[start:end].mean().cpu()
            ),
            "trajectory_global_unweighted_kl": float(global_unweighted_kl.cpu()),
            "trajectory_global_weighted_kl": float(global_weighted_kl.cpu()),
            "trajectory_global_scalar_error": float(
                (global_weighted_kl - global_unweighted_kl).cpu()
            ),
            "trajectory_loss_scale_ratio": float(
                (global_weighted_kl / global_unweighted_kl.clamp_min(eps)).cpu()
            ),
            "trajectory_rank_block_size": int(global_losses.numel()),
            "trajectory_normalization": (
                "fixed_global_calibration_no_batch_renormalization"
                if mode == "global_calibrated_counterfactual_teachability_batch"
                else "fixed_global_f_curriculum_no_batch_renormalization"
                if mode == "global_f_intermediate_curriculum_batch"
                else "synchronized_kl_mass_preserving"
                if mode == "progress_adaptive_robust_frontier_batch"
                else "vanilla_mean_after_adaptive_ratio_sampling"
                if mode == "adaptive_budget_frontier_sampler_batch"
                else "synchronized_probability_sum_one"
            ),
            "trajectory_ddp_objective_scale": ddp_scale,
            "trajectory_weight_detached": not local_probability.requires_grad,
            **curriculum_metrics,
        }
    if mode == "robustness_gated_curriculum":
        if curriculum_state is None or global_teacher_gaps is None:
            raise AssertionError("Curriculum state and teacher gaps must be available.")
        calibration = get_nested(cfg, "scopd.trajectory_weighting.calibration", None)
        if not isinstance(calibration, dict):
            raise ValueError("Missing frozen robustness-gated curriculum calibration.")
        stage = curriculum_state.stage
        result = robustness_gated_curriculum_weights(
            global_losses,
            global_signals,
            global_teacher_gaps,
            curriculum_stage=stage,
            log_teacher_gap_center=float(calibration["log_teacher_gap_center"]),
            log_teacher_gap_scale=float(calibration["log_teacher_gap_scale"]),
            weight_floor=float(get_nested(cfg, "scopd.trajectory_weighting.weight_floor", 0.1)),
            eps=eps,
        )
        state_update = curriculum_state.update(
            float(global_teacher_gaps.mean().cpu()),
            int(global_teacher_gaps.numel()),
        )
        curriculum_metrics = {
            "trajectory_curriculum_stage": float(stage),
            "trajectory_curriculum_stage_after_update": float(state_update["stage_after"]),
            "trajectory_curriculum_ema_teacher_gap": float(
                state_update["ema_teacher_gap_mean"]
            ),
            "trajectory_curriculum_initial_teacher_gap": float(
                curriculum_state.initial_teacher_gap_mean
            ),
            "trajectory_curriculum_ema_decay": float(state_update["ema_decay"]),
            "trajectory_curriculum_updates": int(state_update["updates"]),
            "trajectory_curriculum_trajectories_seen": int(
                state_update["trajectories_seen"]
            ),
            "trajectory_global_teacher_gap": float(global_teacher_gaps.mean().cpu()),
        }
    elif mode == "sensitivity_frontier":
        if not isinstance(curriculum_state, SensitivityFrontierState) or global_ratios is None:
            raise AssertionError("Sensitivity frontier state and ratios must be available.")
        ready_before = curriculum_state.ready
        progress_before = curriculum_state.progress
        frontier_before = curriculum_state.frontier_index
        weight_floor = float(get_nested(cfg, "scopd.trajectory_weighting.weight_floor", 0.1))
        if ready_before:
            result = sensitivity_frontier_weights(
                global_losses,
                global_signals,
                global_ratios,
                curriculum_state,
                weight_floor=weight_floor,
                eps=eps,
            )
            frontier_ratio_gate = result.ratio_gate
            frontier_local_robustness = result.local_robustness
        else:
            result = trajectory_priority_downweights(
                global_losses,
                torch.ones_like(global_losses),
                downweight_strength=1.0 - weight_floor,
                eps=eps,
            )
            frontier_ratio_gate = torch.zeros_like(global_losses)
            frontier_local_robustness = torch.ones_like(global_losses)
        state_update = curriculum_state.update(
            [float(value) for value in global_ratios.detach().cpu().tolist()],
            [float(value) for value in global_signals.detach().cpu().tolist()],
        )
        ratio_weight_means: dict[str, float] = {}
        ratio_gate_means: dict[str, float] = {}
        for ratio in curriculum_state.retention_ratios:
            ratio_mask = torch.isclose(
                global_ratios,
                torch.tensor(ratio, dtype=global_ratios.dtype, device=global_ratios.device),
                rtol=0.0,
                atol=1e-6,
            )
            if bool(ratio_mask.any()):
                key = f"{ratio:.2f}"
                ratio_weight_means[key] = float(result.weight[ratio_mask].mean().cpu())
                ratio_gate_means[key] = float(frontier_ratio_gate[ratio_mask].mean().cpu())
        curriculum_metrics = {
            "trajectory_frontier_ready_before": ready_before,
            "trajectory_frontier_ready_after": bool(state_update["ready_after"]),
            "trajectory_frontier_progress": float(progress_before),
            "trajectory_frontier_progress_after_update": float(state_update["progress_after"]),
            "trajectory_frontier_index": float(frontier_before),
            "trajectory_frontier_index_after_update": float(state_update["frontier_after"]),
            "trajectory_frontier_unresolved_sensitivity": float(
                curriculum_state.unresolved_sensitivity
            ),
            "trajectory_frontier_calibration_counts": dict(
                curriculum_state.calibration_counts
            ),
            "trajectory_frontier_initial_sensitivity": dict(
                curriculum_state.initial_sensitivity
            ),
            "trajectory_frontier_ema_sensitivity": dict(curriculum_state.ema_sensitivity),
            "trajectory_frontier_calibration_complete_at": (
                curriculum_state.calibration_complete_at_trajectory
            ),
            "trajectory_frontier_ratio_weight_means": ratio_weight_means,
            "trajectory_frontier_ratio_gate_means": ratio_gate_means,
            "trajectory_frontier_updates": int(curriculum_state.updates),
            "trajectory_frontier_trajectories_seen": int(
                curriculum_state.trajectories_seen
            ),
            "trajectory_frontier_progress_drop_scale": float(
                curriculum_state.progress_drop_scale
            ),
            "trajectory_frontier_ema_half_life": float(
                curriculum_state.ema_half_life_trajectories
            ),
        }
    elif mode == "residual_robustness_soft":
        result = trajectory_sigmoid_downweights(
            global_losses,
            global_signals,
            downweight_strength=strength,
            temperature=float(
                get_nested(cfg, "scopd.trajectory_weighting.residual_temperature", 1.0)
            ),
            eps=eps,
        )
    else:
        result = trajectory_rank_downweights(
            global_losses,
            global_signals,
            downweight_strength=strength,
            higher_is_better=higher_is_better,
            eps=eps,
        )
    start = rank * len(batch_losses)
    local_weights = result.weight[start : start + len(batch_losses)].to(device=device)
    weighted = torch.stack(
        [weight * loss for weight, loss in zip(local_weights, batch_losses)]
    ).mean()
    scalar_error = result.weighted_loss_mass - result.unweighted_loss_mass
    output_metrics = {
        "trajectory_weighting_enabled": True,
        "trajectory_weighting_mode": mode,
        "trajectory_signal_key": signal_key,
        "trajectory_signal": float(local_signals.mean().cpu()),
        "trajectory_priority": float(result.priority[start : start + len(batch_losses)].mean().cpu()),
        "trajectory_raw_weight": float(result.raw_weight[start : start + len(batch_losses)].mean().cpu()),
        "trajectory_weight": float(local_weights.mean().cpu()),
        "trajectory_loss_mass_scale": float(result.loss_mass_scale.cpu()),
        "trajectory_global_unweighted_kl": float(
            (result.unweighted_loss_mass / global_losses.numel()).cpu()
        ),
        "trajectory_global_weighted_kl": float(
            (result.weighted_loss_mass / global_losses.numel()).cpu()
        ),
        "trajectory_global_scalar_error": float(scalar_error.cpu()),
        "trajectory_rank_block_size": int(global_losses.numel()),
        "trajectory_downweight_strength": float(
            1.0 - float(get_nested(cfg, "scopd.trajectory_weighting.weight_floor", 0.1))
            if mode == "sensitivity_frontier"
            else get_nested(cfg, "scopd.trajectory_weighting.downweight_strength", 0.25)
        ),
        "trajectory_residual_temperature": (
            float(get_nested(cfg, "scopd.trajectory_weighting.residual_temperature", 1.0))
            if mode == "residual_robustness_soft"
            else None
        ),
        "trajectory_weight_detached": not local_weights.requires_grad,
    }
    if mode == "robustness_gated_curriculum":
        current_slice = slice(start, start + len(batch_losses))
        output_metrics.update(
            {
                "trajectory_robustness": float(global_signals[current_slice].mean().cpu()),
                "trajectory_difficulty": float(result.difficulty[current_slice].mean().cpu()),
                "trajectory_focus": float(result.focus[current_slice].mean().cpu()),
                "trajectory_mean_normalized_weight": float(
                    result.mean_normalized_weight[current_slice].mean().cpu()
                ),
                "trajectory_weight_floor": float(
                    get_nested(cfg, "scopd.trajectory_weighting.weight_floor", 0.1)
                ),
                **curriculum_metrics,
            }
        )
    if mode == "sensitivity_frontier":
        if frontier_ratio_gate is None or frontier_local_robustness is None:
            raise AssertionError("Sensitivity frontier diagnostics were not initialized.")
        current_slice = slice(start, start + len(batch_losses))
        output_metrics.update(
            {
                "trajectory_ratio_gate": float(
                    frontier_ratio_gate[current_slice].mean().cpu()
                ),
                "trajectory_local_robustness": float(
                    frontier_local_robustness[current_slice].mean().cpu()
                ),
                "trajectory_weight_floor": float(
                    get_nested(cfg, "scopd.trajectory_weighting.weight_floor", 0.1)
                ),
                **curriculum_metrics,
            }
        )
    return weighted, output_metrics


def validate_paired_native_budget_config(
    cfg: dict[str, Any],
    method: str,
    parameter_scope: str,
    pruning_method: str,
) -> None:
    paired = bool(get_nested(cfg, "paired_sampling.enabled", False))
    weighted = bool(get_nested(cfg, "scopd.native_budget_weighting.enabled", False))
    if not paired and not weighted:
        return
    if method != "scopd_nogt":
        raise ValueError("Paired native-budget training is implemented only for training.method=scopd_nogt.")
    if parameter_scope != "language_decoder_only":
        raise ValueError("Paired native-budget training requires LLM-only LoRA scope.")
    if pruning_method not in {"visionzip", "random"}:
        raise ValueError(
            "Native budget sensitivity requires the native VisionZip or RandomPruner backend."
        )
    if float(get_nested(cfg, "training.lora_dropout", -1.0)) != 0.0:
        raise ValueError("Paired native-budget training requires training.lora_dropout=0.")
    ratio_schedule = str(
        get_nested(cfg, "pruning.retention_ratio_schedule", "")
    ).strip().lower()
    trajectory_mode = str(
        get_nested(cfg, "scopd.trajectory_weighting.mode", "")
    ).strip().lower()
    adaptive_sampler = (
        trajectory_mode == "adaptive_budget_frontier_sampler_batch"
        and ratio_schedule == "adaptive_budget_frontier"
    )
    if ratio_schedule != "paired_deterministic_uniform" and not adaptive_sampler:
        raise ValueError(
            "Paired runs require pruning.retention_ratio_schedule="
            "paired_deterministic_uniform, except the explicit adaptive budget sampler."
        )
    ratios = [float(value) for value in get_nested(cfg, "pruning.train_retention_ratios", [])]
    allow_custom_ratios = bool(
        get_nested(cfg, "paired_sampling.allow_custom_retention_ratios", False)
    )
    if ratios != [0.1, 0.2, 0.3, 0.4] and not allow_custom_ratios:
        raise ValueError(f"Paired runs require ratios [0.1, 0.2, 0.3, 0.4]; got {ratios}.")
    if allow_custom_ratios and (
        not ratios
        or len(ratios) != len(set(ratios))
        or any(not math.isfinite(ratio) or ratio <= 0.0 or ratio >= 1.0 for ratio in ratios)
    ):
        raise ValueError(
            "Custom paired retention ratios must be unique finite values strictly between 0 and 1; "
            f"got {ratios}."
        )
    if int(get_nested(cfg, "training.max_sample_retries", 0) or 0) != 0:
        raise ValueError("Paired runs require max_sample_retries=0 so corresponding sample order cannot diverge.")
    if weighted:
        weighting_mode = str(
            get_nested(cfg, "scopd.native_budget_weighting.mode", "inverse_student_gap")
        ).strip().lower()
        if weighting_mode not in {
            TOKEN_HELLINGER_CURRICULUM_MODE,
            TOKEN_RANDOM_KEEP_MODE,
            TOKEN_FORWARD_KL_TOP_MODE,
        }:
            delta_mode = str(
                get_nested(cfg, "scopd.native_budget_weighting.budget_delta_mode", "absolute")
            ).strip().lower()
            if delta_mode == "absolute":
                delta = float(
                    get_nested(cfg, "scopd.native_budget_weighting.budget_delta", float("nan"))
                )
                allowed_deltas = (0.01, 0.02, 0.03, 0.05, 0.075, 0.10)
                if not any(
                    math.isclose(delta, allowed, rel_tol=0.0, abs_tol=1e-12)
                    for allowed in allowed_deltas
                ):
                    raise ValueError(
                        f"Native budget weighting requires budget_delta in {allowed_deltas}; got {delta}."
                    )
            elif delta_mode == "relative":
                fraction = float(
                    get_nested(
                        cfg,
                        "scopd.native_budget_weighting.budget_delta_fraction",
                        float("nan"),
                    )
                )
                if not math.isfinite(fraction) or fraction <= 0.0:
                    raise ValueError("Relative native budget expansion must be finite and positive.")
                if max(ratios) * (1.0 + fraction) >= 1.0:
                    raise ValueError("Relative native budget expansion must remain pruned.")
            else:
                raise ValueError(f"Unsupported native budget delta mode: {delta_mode!r}.")
            if float(
                get_nested(cfg, "scopd.native_budget_weighting.sensitivity_temperature", 1.0)
            ) <= 0.0:
                raise ValueError("Sensitivity KL temperature must be positive.")
        allowed_modes = {
            "trajectory_probe",
            "symmetric_teacher_gap_stability",
            "inverse_student_gap",
            "max_kl_fraction_inverse_jsd",
            "max_kl_fraction_softmax_inverse_jsd",
            "max_kl_fraction_softmax_inverse_jsd_group_balanced",
            "teacher_gap_persistence",
            "counterfactual_rescue_amplification",
            "native_budget_rescue_grouped",
            "counterfactual_teachability_grouped",
            "teacher_gap_grouped_control",
            "counterfactual_teachability_mixture",
            "counterfactual_teachability_modulation",
            "conditional_rescue_residual",
            "budget_consistent_rank",
            "budget_residual_hardness",
            "budget_gradient_consensus",
            "counterfactual_budget_bridge",
            "budget_gradient_aligned_bridge",
            "counterfactual_gradient_residual",
            "budget_tangent_residual",
            "budget_counterfactual_teachability",
            "budget_contrastive_target",
            "dual_budget_decomposition",
            "token_projection_fraction_top20",
            "token_projection_fraction_bottom80",
            TOKEN_RANDOM_DROP_MODE,
            TOKEN_RANDOM_KEEP_MODE,
            TOKEN_FORWARD_KL_TOP_MODE,
            TOKEN_PROJECTION_MASS_GROUPED_MODE,
            TOKEN_PROJECTION_FRACTION_GROUPED_MODE,
            TOKEN_BUDGET_JSD_ABS_F_GROUPED_MODE,
            TOKEN_PROJECTION_BOTTOM_DROP_MODE,
            TOKEN_PROJECTION_TOP_DROP_MODE,
            TOKEN_HELLINGER_CURRICULUM_MODE,
            TOKEN_TIP_KL_ABS_PROJECTION_MODE,
        }
        if weighting_mode not in allowed_modes:
            raise ValueError(
                f"Native budget weighting mode must be one of {sorted(allowed_modes)}; "
                f"got {weighting_mode!r}."
            )
        if weighting_mode == TOKEN_HELLINGER_CURRICULUM_MODE:
            curriculum_scale = float(
                get_nested(
                    cfg,
                    "scopd.native_budget_weighting.curriculum_scale",
                    float("nan"),
                )
            )
            if not math.isfinite(curriculum_scale) or curriculum_scale <= 0.0:
                raise ValueError(
                    "Token Hellinger curriculum requires a finite positive curriculum_scale."
                )
        if weighting_mode in {
            "max_kl_fraction_inverse_jsd",
            "max_kl_fraction_softmax_inverse_jsd",
            "max_kl_fraction_softmax_inverse_jsd_group_balanced",
        }:
            max_kl_fraction = float(
                get_nested(
                    cfg,
                    "scopd.native_budget_weighting.max_kl_fraction",
                    float("nan"),
                )
            )
            if not math.isfinite(max_kl_fraction) or not 0.0 < max_kl_fraction < 1.0:
                raise ValueError(
                    "Max-KL-fraction JSD weighting requires max_kl_fraction in (0, 1)."
                )
        if weighting_mode in {
            *TOKEN_PROJECTION_PARTITION_MODES,
            TOKEN_RANDOM_DROP_MODE,
            TOKEN_RANDOM_KEEP_MODE,
            TOKEN_FORWARD_KL_TOP_MODE,
        }:
            top_fraction = float(
                get_nested(
                    cfg,
                    "scopd.native_budget_weighting.top_fraction",
                    float("nan"),
                )
            )
            min_teacher_kl = float(
                get_nested(
                    cfg,
                    "scopd.native_budget_weighting.min_teacher_kl",
                    float("nan"),
                )
            )
            if not math.isfinite(top_fraction) or not 0.0 < top_fraction < 1.0:
                raise ValueError("Token projection partition requires top_fraction in (0, 1).")
            if not math.isfinite(min_teacher_kl) or min_teacher_kl < 0.0:
                raise ValueError(
                    "Token projection partition requires finite nonnegative min_teacher_kl."
                )
        if weighting_mode == TOKEN_TIP_KL_ABS_PROJECTION_MODE:
            top_fraction = float(
                get_nested(
                    cfg,
                    "scopd.native_budget_weighting.top_fraction",
                    float("nan"),
                )
            )
            projection_clip_quantile = float(
                get_nested(
                    cfg,
                    "scopd.native_budget_weighting.projection_clip_quantile",
                    float("nan"),
                )
            )
            if not math.isfinite(top_fraction) or not 0.0 < top_fraction < 1.0:
                raise ValueError("KL/absolute-P SoftOR requires top_fraction in (0, 1).")
            if (
                not math.isfinite(projection_clip_quantile)
                or not 0.0 < projection_clip_quantile <= 1.0
            ):
                raise ValueError(
                    "KL/absolute-P SoftOR requires projection_clip_quantile in (0, 1]."
                )
        if weighting_mode == TOKEN_PROJECTION_MASS_GROUPED_MODE:
            top_fraction = float(
                get_nested(
                    cfg,
                    "scopd.native_budget_weighting.top_fraction",
                    float("nan"),
                )
            )
            high_group_lambda = float(
                get_nested(
                    cfg,
                    "scopd.native_budget_weighting.high_group_lambda",
                    float("nan"),
                )
            )
            if not math.isfinite(top_fraction) or not 0.0 < top_fraction < 1.0:
                raise ValueError("Projection-mass grouping requires top_fraction in (0, 1).")
            if not math.isfinite(high_group_lambda) or not 0.0 < high_group_lambda < 1.0:
                raise ValueError(
                    "Projection-mass grouping requires high_group_lambda in (0, 1)."
                )
        if weighting_mode == TOKEN_PROJECTION_FRACTION_GROUPED_MODE:
            group_fraction = float(
                get_nested(
                    cfg,
                    "scopd.native_budget_weighting.top_fraction",
                    float("nan"),
                )
            )
            selected_group_lambda = float(
                get_nested(
                    cfg,
                    "scopd.native_budget_weighting.selected_group_lambda",
                    float("nan"),
                )
            )
            selection = str(
                get_nested(cfg, "scopd.native_budget_weighting.selection", "")
            ).strip().lower()
            min_teacher_kl = float(
                get_nested(
                    cfg,
                    "scopd.native_budget_weighting.min_teacher_kl",
                    float("nan"),
                )
            )
            if not math.isfinite(group_fraction) or not 0.0 < group_fraction < 1.0:
                raise ValueError("Projection-F grouping requires top_fraction in (0, 1).")
            if (
                not math.isfinite(selected_group_lambda)
                or not 0.0 < selected_group_lambda < 1.0
            ):
                raise ValueError(
                    "Projection-F grouping requires selected_group_lambda in (0, 1)."
                )
            if selection not in {"top", "bottom"}:
                raise ValueError("Projection-F grouping selection must be top or bottom.")
            if not math.isfinite(min_teacher_kl) or min_teacher_kl < 0.0:
                raise ValueError(
                    "Projection-F grouping requires finite nonnegative min_teacher_kl."
                )
        if weighting_mode == TOKEN_BUDGET_JSD_ABS_F_GROUPED_MODE:
            budget_top_fraction = float(
                get_nested(
                    cfg,
                    "scopd.native_budget_weighting.budget_top_fraction",
                    float("nan"),
                )
            )
            within_budget_fraction = float(
                get_nested(
                    cfg,
                    "scopd.native_budget_weighting.within_budget_fraction",
                    float("nan"),
                )
            )
            selected_group_lambda = float(
                get_nested(
                    cfg,
                    "scopd.native_budget_weighting.selected_group_lambda",
                    float("nan"),
                )
            )
            selection = str(
                get_nested(cfg, "scopd.native_budget_weighting.selection", "")
            ).strip().lower()
            if not math.isfinite(budget_top_fraction) or not 0.0 < budget_top_fraction < 1.0:
                raise ValueError("B-JSD prefilter requires budget_top_fraction in (0, 1).")
            if (
                not math.isfinite(within_budget_fraction)
                or not 0.0 < within_budget_fraction < 1.0
            ):
                raise ValueError("Absolute-F grouping requires within_budget_fraction in (0, 1).")
            if (
                not math.isfinite(selected_group_lambda)
                or not 0.0 < selected_group_lambda < 1.0
            ):
                raise ValueError(
                    "B-JSD/absolute-F grouping requires selected_group_lambda in (0, 1)."
                )
            if selection not in {"top", "bottom"}:
                raise ValueError("Absolute-F grouping selection must be top or bottom.")
        if weighting_mode in {
            TOKEN_PROJECTION_BOTTOM_DROP_MODE,
            TOKEN_PROJECTION_TOP_DROP_MODE,
        }:
            drop_fraction = float(
                get_nested(
                    cfg,
                    "scopd.native_budget_weighting.drop_fraction",
                    float("nan"),
                )
            )
            if not math.isfinite(drop_fraction) or not 0.0 < drop_fraction < 1.0:
                raise ValueError("F-token dropping requires drop_fraction in (0, 1).")
        if weighting_mode in {
            "max_kl_fraction_softmax_inverse_jsd",
            "max_kl_fraction_softmax_inverse_jsd_group_balanced",
        }:
            softmax_temperature = float(
                get_nested(
                    cfg,
                    "scopd.native_budget_weighting.softmax_temperature",
                    float("nan"),
                )
            )
            if not math.isfinite(softmax_temperature) or softmax_temperature <= 0.0:
                raise ValueError(
                    "Softmax max-KL-fraction JSD weighting requires a positive finite "
                    "softmax_temperature."
                )
        if weighting_mode == "max_kl_fraction_softmax_inverse_jsd_group_balanced":
            high_group_coefficient = float(
                get_nested(
                    cfg,
                    "scopd.native_budget_weighting.high_group_coefficient",
                    float("nan"),
                )
            )
            if (
                not math.isfinite(high_group_coefficient)
                or not 0.0 < high_group_coefficient < 1.0
            ):
                raise ValueError(
                    "Group-balanced max-KL weighting requires high_group_coefficient in (0, 1)."
                )
        trajectory_enabled = bool(get_nested(cfg, "scopd.trajectory_weighting.enabled", False))
        if trajectory_enabled:
            if weighting_mode != "trajectory_probe":
                raise ValueError(
                    "Trajectory weighting requires native_budget_weighting.mode=trajectory_probe "
                    "so tokenwise SCOPD remains unchanged."
                )
            trajectory_mode = str(
                get_nested(cfg, "scopd.trajectory_weighting.mode", "closure_rank")
            ).strip().lower()
            if trajectory_mode not in {
                "closure_rank",
                "teacher_gap_rank",
                "robustness_rank",
                "relative_robustness_rank",
                "residual_robustness_rank",
                "residual_robustness_soft",
                "robustness_gated_curriculum",
                "sensitivity_frontier",
                "jsd_over_current_kl_batch",
                "jsd_over_current_kl_direct_inverse_batch",
                "jsd_over_current_kl_softmax_batch",
                "jsd_over_step0_kl_batch",
                "ratio_group_counterfactual_teachability_batch",
                "global_calibrated_counterfactual_teachability_batch",
                "trajectory_counterfactual_teachability_softmax_batch",
                "trajectory_projection_fraction_top20_batch",
                "trajectory_projection_fraction_bottom80_batch",
                "global_f_intermediate_curriculum_batch",
                "progress_adaptive_robust_frontier_batch",
                "adaptive_budget_frontier_sampler_batch",
            }:
                raise ValueError(f"Unsupported trajectory weighting mode: {trajectory_mode!r}.")
            if trajectory_mode == "jsd_over_step0_kl_batch":
                calibration = get_nested(
                    cfg,
                    "scopd.trajectory_weighting.step0_teacher_kl_by_ratio",
                    None,
                )
                expected_keys = {"0.10", "0.20", "0.30", "0.40"}
                if not isinstance(calibration, dict) or set(calibration) != expected_keys:
                    raise ValueError(
                        "jsd_over_step0_kl_batch requires positive calibration values for "
                        f"{sorted(expected_keys)}."
                    )
                if any(
                    not math.isfinite(float(value)) or float(value) <= 0.0
                    for value in calibration.values()
                ):
                    raise ValueError("Step-0 teacher KL calibration values must be finite and positive.")
            if trajectory_mode == "jsd_over_current_kl_softmax_batch":
                temperature = float(
                    get_nested(cfg, "scopd.trajectory_weighting.temperature", float("nan"))
                )
                if not math.isfinite(temperature) or temperature <= 0.0:
                    raise ValueError(
                        "jsd_over_current_kl_softmax_batch requires a finite positive temperature."
                    )
            if trajectory_mode == "ratio_group_counterfactual_teachability_batch":
                ratio_group_weight_transform(cfg)
                statistic = ratio_group_statistic(cfg)
                if (
                    statistic
                    in {
                        "teacher_directed_projection_cosine",
                        "teacher_directed_projection_cosine_sample_normalized",
                    }
                    and str(
                        get_nested(
                            cfg,
                            "scopd.trajectory_weighting.group_transform",
                            "linear",
                        )
                    ).strip().lower()
                    != "softmax"
                ):
                    raise ValueError(
                        "Ratio-group cosine statistics require group_transform=softmax."
                    )
            if trajectory_mode == "trajectory_counterfactual_teachability_softmax_batch":
                temperature = float(
                    get_nested(cfg, "scopd.trajectory_weighting.temperature", float("nan"))
                )
                if not math.isfinite(temperature) or temperature <= 0.0:
                    raise ValueError(
                        "trajectory_counterfactual_teachability_softmax_batch requires "
                        "a finite positive temperature."
                    )
            if trajectory_mode == "global_calibrated_counterfactual_teachability_batch":
                global_trajectory_calibration(cfg)
                normalization = str(
                    get_nested(cfg, "scopd.trajectory_weighting.normalization", "")
                ).strip().lower()
                normalization_scope = str(
                    get_nested(
                        cfg,
                        "scopd.trajectory_weighting.normalization_scope",
                        "",
                    )
                ).strip().lower()
                if normalization != "fixed_global_centered_scale":
                    raise ValueError(
                        "Global F weighting requires normalization=fixed_global_centered_scale."
                    )
                if normalization_scope != "frozen_training_calibration":
                    raise ValueError(
                        "Global F weighting requires "
                        "normalization_scope=frozen_training_calibration."
                    )
            if trajectory_mode in {
                "trajectory_projection_fraction_top20_batch",
                "trajectory_projection_fraction_bottom80_batch",
            }:
                top_fraction = float(
                    get_nested(
                        cfg,
                        "scopd.trajectory_weighting.top_fraction",
                        float("nan"),
                    )
                )
                if not math.isfinite(top_fraction) or not 0.0 < top_fraction < 1.0:
                    raise ValueError(
                        "Trajectory projection partition requires top_fraction in (0, 1)."
                    )
            if trajectory_mode == "global_f_intermediate_curriculum_batch":
                gamma = float(
                    get_nested(cfg, "scopd.trajectory_weighting.gamma", float("nan"))
                )
                if not math.isfinite(gamma) or gamma <= 0.0:
                    raise ValueError(
                        "Global F intermediate curriculum requires finite positive gamma."
                    )
                normalization = str(
                    get_nested(cfg, "scopd.trajectory_weighting.normalization", "")
                ).strip().lower()
                normalization_scope = str(
                    get_nested(cfg, "scopd.trajectory_weighting.normalization_scope", "")
                ).strip().lower()
                if normalization != "fixed_global_gate_no_batch_renormalization":
                    raise ValueError(
                        "Global F intermediate curriculum requires "
                        "normalization=fixed_global_gate_no_batch_renormalization."
                    )
                if normalization_scope != "global_formula":
                    raise ValueError(
                        "Global F intermediate curriculum requires "
                        "normalization_scope=global_formula."
                    )
            if trajectory_mode == "progress_adaptive_robust_frontier_batch":
                calibration = get_nested(
                    cfg, "scopd.trajectory_weighting.calibration", None
                )
                if not isinstance(calibration, dict):
                    raise ValueError(
                        "progress_adaptive_robust_frontier_batch requires a frozen "
                        "calibration mapping."
                    )
                for key in ("initial_tau", "initial_robust_need_mean"):
                    value = float(calibration.get(key, float("nan")))
                    if not math.isfinite(value) or value <= 0.0:
                        raise ValueError(
                            f"Progress-adaptive frontier calibration {key} must be "
                            f"finite and positive; got {value}."
                        )
                half_life = float(
                    get_nested(
                        cfg,
                        "scopd.trajectory_weighting.ema_half_life_trajectories",
                        256.0,
                    )
                )
                if not math.isfinite(half_life) or half_life <= 0.0:
                    raise ValueError(
                        "Progress-adaptive frontier EMA half-life must be finite and positive."
                    )
                normalization = str(
                    get_nested(cfg, "scopd.trajectory_weighting.normalization", "")
                ).strip().lower()
                if normalization != "kl_mass_preserving":
                    raise ValueError(
                        "Progress-adaptive frontier requires normalization=kl_mass_preserving."
                    )
            elif trajectory_mode == "adaptive_budget_frontier_sampler_batch":
                if ratio_schedule != "adaptive_budget_frontier":
                    raise ValueError(
                        "Adaptive budget sampling requires "
                        "pruning.retention_ratio_schedule=adaptive_budget_frontier."
                    )
                sampler_metric = str(
                    get_nested(
                        cfg,
                        "scopd.trajectory_weighting.sampler_metric",
                        "robust_need",
                    )
                ).strip().lower()
                if sampler_metric not in {"robust_need", "teacher_kl"}:
                    raise ValueError(
                        "Adaptive budget sampler_metric must be robust_need or teacher_kl."
                    )
                if int(
                    get_nested(
                        cfg,
                        "scopd.trajectory_weighting.calibration_target_per_ratio",
                        64,
                    )
                ) <= 0:
                    raise ValueError(
                        "Adaptive budget calibration target must be positive."
                    )
                half_life = float(
                    get_nested(
                        cfg,
                        "scopd.trajectory_weighting.ema_half_life_per_ratio",
                        64.0,
                    )
                )
                if not math.isfinite(half_life) or half_life <= 0.0:
                    raise ValueError(
                        "Adaptive budget EMA half-life must be finite and positive."
                    )
                normalization = str(
                    get_nested(cfg, "scopd.trajectory_weighting.normalization", "")
                ).strip().lower()
                normalization_scope = str(
                    get_nested(
                        cfg,
                        "scopd.trajectory_weighting.normalization_scope",
                        "",
                    )
                ).strip().lower()
                if normalization != "probability_sum_one":
                    raise ValueError(
                        "Adaptive budget sampling requires normalization=probability_sum_one."
                    )
                if normalization_scope != "effective_batch":
                    raise ValueError(
                        "Adaptive budget sampling requires normalization_scope=effective_batch."
                    )
            elif (
                trajectory_mode in EFFECTIVE_BATCH_PROBABILITY_MODES
                and trajectory_mode not in DIRECT_GLOBAL_F_MODES
            ):
                normalization = str(
                    get_nested(cfg, "scopd.trajectory_weighting.normalization", "")
                ).strip().lower()
                normalization_scope = str(
                    get_nested(cfg, "scopd.trajectory_weighting.normalization_scope", "")
                ).strip().lower()
                if normalization != "probability_sum_one":
                    raise ValueError(
                        "JSD trajectory weighting requires normalization=probability_sum_one."
                    )
                if normalization_scope != "effective_batch":
                    raise ValueError(
                        "JSD trajectory weighting requires normalization_scope=effective_batch."
                    )
            if trajectory_mode in {"residual_robustness_rank", "residual_robustness_soft"}:
                calibration = get_nested(cfg, "scopd.trajectory_weighting.residual_calibration", None)
                if not isinstance(calibration, dict):
                    raise ValueError(
                        "residual_robustness_rank requires a frozen residual_calibration mapping."
                    )
            if trajectory_mode == "robustness_gated_curriculum":
                calibration = get_nested(cfg, "scopd.trajectory_weighting.calibration", None)
                if not isinstance(calibration, dict):
                    raise ValueError(
                        "robustness_gated_curriculum requires a frozen calibration mapping."
                    )
                for key in (
                    "initial_teacher_gap_mean",
                    "log_teacher_gap_center",
                    "log_teacher_gap_scale",
                ):
                    if key not in calibration or not math.isfinite(float(calibration[key])):
                        raise ValueError(f"Invalid curriculum calibration value for {key!r}.")
                if float(calibration["initial_teacher_gap_mean"]) <= 0.0:
                    raise ValueError("Curriculum initial_teacher_gap_mean must be positive.")
                if float(calibration["log_teacher_gap_scale"]) <= 0.0:
                    raise ValueError("Curriculum log_teacher_gap_scale must be positive.")
                if float(get_nested(cfg, "scopd.trajectory_weighting.progress_power", 3.0)) <= 0.0:
                    raise ValueError("Curriculum progress_power must be positive.")
                if float(
                    get_nested(cfg, "scopd.trajectory_weighting.ema_half_life_trajectories", 256.0)
                ) <= 0.0:
                    raise ValueError("Curriculum EMA half-life must be positive.")
                weight_floor = float(get_nested(cfg, "scopd.trajectory_weighting.weight_floor", 0.1))
                if not 0.0 < weight_floor <= 1.0:
                    raise ValueError("Curriculum weight_floor must be in (0, 1].")
            if trajectory_mode == "sensitivity_frontier":
                fraction = float(
                    get_nested(
                        cfg,
                        "scopd.native_budget_weighting.budget_delta_fraction",
                        float("nan"),
                    )
                )
                if delta_mode != "relative" or not math.isclose(
                    fraction, 0.25, rel_tol=0.0, abs_tol=1e-12
                ):
                    raise ValueError(
                        "sensitivity_frontier requires a 25% relative native budget expansion."
                    )
                if int(
                    get_nested(cfg, "scopd.trajectory_weighting.calibration_target_per_ratio", 64)
                ) <= 0:
                    raise ValueError("Sensitivity frontier calibration target must be positive.")
                for key, default in (
                    ("ema_half_life_trajectories", 256.0),
                    ("progress_drop_scale", 0.5),
                    ("progress_power", 2.0),
                ):
                    value = float(get_nested(cfg, f"scopd.trajectory_weighting.{key}", default))
                    if not math.isfinite(value) or value <= 0.0:
                        raise ValueError(f"Sensitivity frontier {key} must be finite and positive.")
                if float(
                    get_nested(cfg, "scopd.trajectory_weighting.progress_drop_scale", 0.5)
                ) > 1.0:
                    raise ValueError("Sensitivity frontier progress_drop_scale must be at most one.")
                weight_floor = float(
                    get_nested(cfg, "scopd.trajectory_weighting.weight_floor", 0.1)
                )
                if not 0.0 < weight_floor <= 1.0:
                    raise ValueError("Sensitivity frontier weight_floor must be in (0, 1].")
            strength = float(get_nested(cfg, "scopd.trajectory_weighting.downweight_strength", 0.25))
            if not 0.0 <= strength < 1.0:
                raise ValueError(
                    "scopd.trajectory_weighting.downweight_strength must be in [0, 1)."
                )
        if weighting_mode == "symmetric_teacher_gap_stability":
            alpha = float(get_nested(cfg, "scopd.native_budget_weighting.alpha", 0.25))
            if not 0.0 <= alpha < 1.0:
                raise ValueError(
                    f"Symmetric teacher-gap stability alpha must be in [0, 1); got {alpha}."
                )
        elif weighting_mode == "teacher_gap_persistence":
            alpha = float(get_nested(cfg, "scopd.native_budget_weighting.alpha", 0.5))
            if not 0.0 <= alpha <= 1.0:
                raise ValueError(f"Teacher-gap persistence alpha must be in [0, 1]; got {alpha}.")
        elif weighting_mode == "counterfactual_rescue_amplification":
            alpha = float(get_nested(cfg, "scopd.native_budget_weighting.alpha", 0.5))
            if not 0.0 <= alpha <= 4.0:
                raise ValueError(
                    "Counterfactual rescue amplification alpha must be in [0, 4]; "
                    f"got {alpha}."
                )
        elif weighting_mode in {
            "native_budget_rescue_grouped",
            "counterfactual_teachability_grouped",
            "teacher_gap_grouped_control",
        }:
            top_fraction = float(
                get_nested(cfg, "scopd.native_budget_weighting.top_fraction", 0.2)
            )
            high_group_mass = float(
                get_nested(cfg, "scopd.native_budget_weighting.high_group_mass", 0.5)
            )
            if not 0.0 < top_fraction < 1.0:
                raise ValueError(
                    f"Grouped top_fraction must be in (0, 1); got {top_fraction}."
                )
            if not 0.0 < high_group_mass < 1.0:
                raise ValueError(
                    "Grouped high_group_mass must be in (0, 1); "
                    f"got {high_group_mass}."
                )
            if weighting_mode == "counterfactual_teachability_grouped":
                rescue_modulation = float(
                    get_nested(
                        cfg,
                        "scopd.native_budget_weighting.rescue_modulation",
                        0.1,
                    )
                )
                if not 0.0 <= rescue_modulation <= 1.0:
                    raise ValueError(
                        "Grouped rescue_modulation must be in [0, 1]; "
                        f"got {rescue_modulation}."
                    )
        elif weighting_mode == "counterfactual_teachability_mixture":
            alpha = float(get_nested(cfg, "scopd.native_budget_weighting.alpha", 0.5))
            rescue_mix = float(
                get_nested(cfg, "scopd.native_budget_weighting.rescue_mix", 0.1)
            )
            if not 0.0 <= alpha <= 4.0:
                raise ValueError(
                    "Counterfactual teachability alpha must be in [0, 4]; "
                    f"got {alpha}."
                )
            if not 0.0 <= rescue_mix <= 1.0:
                raise ValueError(
                    "Counterfactual teachability rescue_mix must be in [0, 1]; "
                    f"got {rescue_mix}."
                )
        elif weighting_mode == "counterfactual_teachability_modulation":
            alpha = float(get_nested(cfg, "scopd.native_budget_weighting.alpha", 0.5))
            rescue_modulation = float(
                get_nested(
                    cfg,
                    "scopd.native_budget_weighting.rescue_modulation",
                    0.1,
                )
            )
            if not 0.0 <= alpha <= 4.0:
                raise ValueError(
                    "Counterfactual teachability alpha must be in [0, 4]; "
                    f"got {alpha}."
                )
            if not 0.0 <= rescue_modulation <= 1.0:
                raise ValueError(
                    "Counterfactual teachability rescue_modulation must be in [0, 1]; "
                    f"got {rescue_modulation}."
                )
        elif weighting_mode == "conditional_rescue_residual":
            alpha = float(get_nested(cfg, "scopd.native_budget_weighting.alpha", 0.1))
            difficulty_bins = int(
                get_nested(cfg, "scopd.native_budget_weighting.difficulty_bins", 5)
            )
            if not 0.0 <= alpha < 1.0:
                raise ValueError(
                    "Conditional rescue residual alpha must be in [0, 1); "
                    f"got {alpha}."
                )
            if difficulty_bins < 2:
                raise ValueError(
                    "Conditional rescue residual difficulty_bins must be at least 2; "
                    f"got {difficulty_bins}."
                )
        elif weighting_mode == "budget_consistent_rank":
            alpha = float(get_nested(cfg, "scopd.native_budget_weighting.alpha", 1.0))
            if not 0.0 <= alpha <= 4.0:
                raise ValueError(f"Budget-consistent rank alpha must be in [0, 4]; got {alpha}.")
        elif weighting_mode == "budget_residual_hardness":
            alpha = float(get_nested(cfg, "scopd.native_budget_weighting.alpha", 1.0))
            persistence_mix = float(
                get_nested(cfg, "scopd.native_budget_weighting.persistence_mix", 0.1)
            )
            if not 0.0 <= alpha <= 4.0:
                raise ValueError(f"Budget-residual alpha must be in [0, 4]; got {alpha}.")
            if not 0.0 <= persistence_mix <= 1.0:
                raise ValueError(
                    "Budget-residual persistence_mix must be in [0, 1]; "
                    f"got {persistence_mix}."
                )
        elif weighting_mode == "budget_gradient_consensus":
            alpha = float(get_nested(cfg, "scopd.native_budget_weighting.alpha", 0.5))
            if not 0.0 <= alpha <= 4.0:
                raise ValueError(f"Budget-gradient consensus alpha must be in [0, 4]; got {alpha}.")
        elif weighting_mode == "counterfactual_budget_bridge":
            fraction = float(
                get_nested(cfg, "scopd.native_budget_weighting.max_bridge_fraction", 0.5)
            )
            if not 0.0 <= fraction <= 1.0:
                raise ValueError(
                    "Counterfactual max_bridge_fraction must be in [0, 1]; "
                    f"got {fraction}."
                )
        elif weighting_mode == "budget_gradient_aligned_bridge":
            fraction = float(
                get_nested(cfg, "scopd.native_budget_weighting.max_bridge_fraction", 0.5)
            )
            if not 0.0 <= fraction <= 1.0:
                raise ValueError(
                    "Gradient-aligned max_bridge_fraction must be in [0, 1]; "
                    f"got {fraction}."
                )
        elif weighting_mode == "counterfactual_gradient_residual":
            strength = float(
                get_nested(cfg, "scopd.native_budget_weighting.cancellation_strength", 0.5)
            )
            max_projection = float(
                get_nested(
                    cfg,
                    "scopd.native_budget_weighting.max_projection_coefficient",
                    1.0,
                )
            )
            if not 0.0 <= strength <= 1.0:
                raise ValueError(
                    "Counterfactual cancellation_strength must be in [0, 1]; "
                    f"got {strength}."
                )
            if max_projection <= 0.0:
                raise ValueError(
                    "Counterfactual max_projection_coefficient must be positive; "
                    f"got {max_projection}."
                )
            cancellation_schedule = str(
                get_nested(
                    cfg,
                    "scopd.native_budget_weighting.cancellation_schedule",
                    "constant",
                )
            ).strip().lower()
            if cancellation_schedule not in {"constant", "linear_to_zero"}:
                raise ValueError(
                    "Counterfactual cancellation_schedule must be constant or "
                    f"linear_to_zero; got {cancellation_schedule!r}."
                )
            decay_fraction = float(
                get_nested(
                    cfg,
                    "scopd.native_budget_weighting.cancellation_decay_fraction",
                    0.5,
                )
            )
            if not 0.0 < decay_fraction <= 1.0:
                raise ValueError(
                    "Counterfactual cancellation_decay_fraction must be in (0, 1]; "
                    f"got {decay_fraction}."
                )
        elif weighting_mode == "budget_tangent_residual":
            alpha = float(get_nested(cfg, "scopd.native_budget_weighting.alpha", 1.0))
            if not 0.0 <= alpha <= 4.0:
                raise ValueError(f"Budget-tangent residual alpha must be in [0, 4]; got {alpha}.")
        elif weighting_mode == "budget_counterfactual_teachability":
            alpha = float(get_nested(cfg, "scopd.native_budget_weighting.alpha", 1.0))
            support_top_k = int(get_nested(cfg, "scopd.native_budget_weighting.support_top_k", 32))
            if not 0.0 <= alpha <= 4.0:
                raise ValueError(
                    f"Budget-counterfactual teachability alpha must be in [0, 4]; got {alpha}."
                )
            if support_top_k <= 0:
                raise ValueError(
                    f"Budget-counterfactual support_top_k must be positive; got {support_top_k}."
                )
        elif weighting_mode == "budget_contrastive_target":
            beta_max = float(get_nested(cfg, "scopd.native_budget_weighting.beta_max", 0.5))
            advantage_clip = float(get_nested(cfg, "scopd.native_budget_weighting.advantage_clip", 2.0))
            if not 0.0 <= beta_max <= 2.0:
                raise ValueError(f"Budget-contrastive beta_max must be in [0, 2]; got {beta_max}.")
            if advantage_clip <= 0.0:
                raise ValueError(
                    f"Budget-contrastive advantage_clip must be positive; got {advantage_clip}."
                )
        elif weighting_mode == "dual_budget_decomposition":
            alpha = float(get_nested(cfg, "scopd.native_budget_weighting.alpha", 1.0))
            persistence_mix = float(
                get_nested(cfg, "scopd.native_budget_weighting.persistence_mix", 0.1)
            )
            beta_max = float(get_nested(cfg, "scopd.native_budget_weighting.beta_max", 0.5))
            advantage_clip = float(get_nested(cfg, "scopd.native_budget_weighting.advantage_clip", 2.0))
            if not 0.0 <= alpha <= 4.0:
                raise ValueError(f"Dual-budget alpha must be in [0, 4]; got {alpha}.")
            if not 0.0 <= persistence_mix <= 1.0:
                raise ValueError(
                    f"Dual-budget persistence_mix must be in [0, 1]; got {persistence_mix}."
                )
            if not 0.0 <= beta_max <= 2.0:
                raise ValueError(f"Dual-budget beta_max must be in [0, 2]; got {beta_max}.")
            if advantage_clip <= 0.0:
                raise ValueError(
                    f"Dual-budget advantage_clip must be positive; got {advantage_clip}."
                )


def checkpoint_contract(cfg: dict[str, Any]) -> dict[str, Any]:
    contract = copy.deepcopy(cfg)
    contract.pop("output_dir", None)
    training = contract.setdefault("training", {})
    training.pop("start_step", None)
    training.pop("adapter_path", None)
    checkpointing = contract.setdefault("checkpointing", {})
    checkpointing.pop("resume_from", None)
    checkpointing.pop("stop_at_step", None)
    return contract


def checkpoint_contract_sha256(cfg: dict[str, Any]) -> str:
    payload = json.dumps(checkpoint_contract(cfg), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _torch_load_full(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def trim_jsonl_to_step(path: Path, maximum_step: int) -> None:
    if not path.is_file():
        return
    kept: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row.get("step", -1)) <= int(maximum_step):
            kept.append(json.dumps(row, ensure_ascii=False))
    temporary = path.with_name(f".{path.name}.trim.{os.getpid()}")
    temporary.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    os.replace(temporary, path)


def distributed_barrier(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.barrier()


def capture_rank_rng_state(
    ratio_rng: random.Random,
    global_step: int,
    curriculum_state: (
        AdaptiveBudgetFrontierState
        | ProgressAdaptiveFrontierState
        | RobustnessGatedCurriculumState
        | SensitivityFrontierState
        | None
    ) = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "global_step": int(global_step),
        "python_global_rng": random.getstate(),
        "ratio_rng": ratio_rng.getstate(),
        "torch_cpu_rng": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_rng"] = torch.cuda.get_rng_state(torch.cuda.current_device())
    if curriculum_state is not None:
        state["trajectory_curriculum_state"] = curriculum_state.state_dict()
    return state


def restore_rank_rng_state(
    state: dict[str, Any], ratio_rng: random.Random, expected_step: int
) -> dict[str, Any] | None:
    if int(state.get("global_step", -1)) != int(expected_step):
        raise ValueError(
            f"RNG checkpoint step mismatch: {state.get('global_step')} vs expected {expected_step}."
        )
    random.setstate(state["python_global_rng"])
    ratio_rng.setstate(state["ratio_rng"])
    torch.set_rng_state(state["torch_cpu_rng"])
    if torch.cuda.is_available() and "torch_cuda_rng" in state:
        torch.cuda.set_rng_state(state["torch_cuda_rng"], torch.cuda.current_device())
    curriculum_state = state.get("trajectory_curriculum_state")
    return curriculum_state if isinstance(curriculum_state, dict) else None


def prepare_resume_config(cfg: dict[str, Any], output_dir: Path) -> tuple[Path | None, dict[str, Any] | None]:
    raw = str(get_nested(cfg, "checkpointing.resume_from", "") or "").strip()
    if not raw:
        return None, None
    checkpoint = Path(raw).resolve()
    if not (checkpoint / "COMPLETE").is_file():
        raise FileNotFoundError(f"Resume checkpoint is missing COMPLETE marker: {checkpoint}")
    metadata_path = checkpoint / "trainer_state.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint is missing trainer_state.json: {checkpoint}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_contract = checkpoint_contract_sha256(cfg)
    if metadata.get("config_contract_sha256") != expected_contract:
        raise ValueError(
            "Resume config does not match checkpoint contract: "
            f"checkpoint={metadata.get('config_contract_sha256')} current={expected_contract}."
        )
    expected_output = checkpoint.parent.parent.resolve()
    if output_dir.resolve() != expected_output:
        raise ValueError(f"Resume output must remain {expected_output}; got {output_dir.resolve()}.")
    global_step = int(metadata["global_step"])
    configured_start = int(get_nested(cfg, "training.start_step", 0) or 0)
    if configured_start not in {0, global_step}:
        raise ValueError(
            f"Configured start_step={configured_start} conflicts with checkpoint step={global_step}."
        )
    set_nested(cfg, "training.start_step", global_step)
    set_nested(cfg, "training.adapter_path", str(checkpoint))
    return checkpoint, metadata


def save_lora_evaluation_snapshot(
    model: Any,
    output_dir: Path,
    global_step: int,
    cfg: dict[str, Any],
    rank: int,
    distributed: bool,
) -> Path:
    target = output_dir / "eval_snapshots" / f"step_{int(global_step):06d}"
    distributed_barrier(distributed)
    if is_main_process(rank) and not (target / "COMPLETE").is_file():
        if target.exists():
            raise FileExistsError(f"Incomplete evaluation snapshot already exists: {target}")
        temporary = target.with_name(f".{target.name}.tmp")
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        save_checkpoint(model, temporary, ema_shadow=None)
        _atomic_write_json(
            temporary / "snapshot_metadata.json",
            {
                "checkpoint_type": "lightweight_lora_evaluation_snapshot",
                "global_step": int(global_step),
                "config_contract_sha256": checkpoint_contract_sha256(cfg),
            },
        )
        (temporary / "COMPLETE").write_text("complete\n", encoding="utf-8")
        os.replace(temporary, target)
    distributed_barrier(distributed)
    return target


def save_full_resumable_checkpoint(
    model: Any,
    optimizer: torch.optim.Optimizer,
    ema_shadow: dict[str, torch.Tensor] | None,
    ratio_rng: random.Random,
    output_dir: Path,
    global_step: int,
    cfg: dict[str, Any],
    rank: int,
    world_size: int,
    distributed: bool,
    curriculum_state: (
        AdaptiveBudgetFrontierState
        | ProgressAdaptiveFrontierState
        | RobustnessGatedCurriculumState
        | SensitivityFrontierState
        | None
    ) = None,
) -> Path:
    target = output_dir / "resume_checkpoints" / f"step_{int(global_step):06d}"
    temporary = target.with_name(f".{target.name}.tmp")
    distributed_barrier(distributed)
    if (target / "COMPLETE").is_file():
        distributed_barrier(distributed)
        return target
    if is_main_process(rank):
        if target.exists():
            raise FileExistsError(f"Incomplete resumable checkpoint already exists: {target}")
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        save_checkpoint(model, temporary, ema_shadow=ema_shadow)
        _atomic_torch_save(optimizer.state_dict(), temporary / "optimizer.pt")
        _atomic_write_json(
            temporary / "trainer_state.json",
            {
                "checkpoint_type": "full_resumable_lora_training_checkpoint",
                "global_step": int(global_step),
                "world_size": int(world_size),
                "gradient_accumulation_remainder": 0,
                "config_contract_sha256": checkpoint_contract_sha256(cfg),
                "trajectory_curriculum_state": (
                    curriculum_state.state_dict() if curriculum_state is not None else None
                ),
            },
        )
        source_config = output_dir / "config_resolved.yaml"
        if source_config.is_file():
            shutil.copy2(source_config, temporary / "config_resolved.yaml")
    distributed_barrier(distributed)
    _atomic_torch_save(
        capture_rank_rng_state(ratio_rng, global_step, curriculum_state),
        temporary / "rank_rng_states" / f"rank_{rank:02d}.pt",
    )
    distributed_barrier(distributed)
    if is_main_process(rank):
        rank_states = sorted((temporary / "rank_rng_states").glob("rank_*.pt"))
        if len(rank_states) != int(world_size):
            raise RuntimeError(f"Expected {world_size} rank RNG states, found {len(rank_states)}.")
        (temporary / "COMPLETE").write_text("complete\n", encoding="utf-8")
        os.replace(temporary, target)
    distributed_barrier(distributed)
    return target


def point_final_checkpoint_at(output_dir: Path, checkpoint: Path, rank: int, distributed: bool) -> None:
    distributed_barrier(distributed)
    if is_main_process(rank):
        final = output_dir / "final"
        if final.exists() or final.is_symlink():
            if final.is_symlink() and final.resolve() == checkpoint.resolve():
                pass
            else:
                raise FileExistsError(f"Refusing to replace existing final checkpoint: {final}")
        else:
            final.symlink_to(checkpoint.relative_to(output_dir), target_is_directory=True)
    distributed_barrier(distributed)


def train(cfg: dict[str, Any]) -> Path:
    distributed, rank, local_rank, world_size = setup_distributed()
    method = str(get_nested(cfg, "training.method", "sft")).lower()
    parameter_scope = str(get_nested(cfg, "experiment.parameter_scope", "") or "").strip()
    pruning_method = configure_pruning_backend(cfg)
    if pruning_method == "random" and parameter_scope != "language_decoder_only":
        raise ValueError(
            "The RandomPruner backend currently detaches vision-encoder outputs and is therefore "
            "restricted to experiment.parameter_scope=language_decoder_only."
        )
    prompt_mode = prompt_mode_from_config(cfg)
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}.")
    token_outlier_exclusion_enabled = bool(
        get_nested(cfg, "scopd.token_outlier_exclusion.enabled", False)
    )
    token_kl_floor_filter_enabled = bool(
        get_nested(cfg, "scopd.token_kl_floor_filter.enabled", False)
    )
    if token_kl_floor_filter_enabled:
        if method != "scopd_nogt":
            raise ValueError("Token KL floor filtering is restricted to training.method=scopd_nogt.")
        min_kl = float(get_nested(cfg, "scopd.token_kl_floor_filter.min_kl", float("nan")))
        if not math.isfinite(min_kl) or min_kl < 0.0:
            raise ValueError("scopd.token_kl_floor_filter.min_kl must be finite and nonnegative.")
        reduction = str(
            get_nested(cfg, "scopd.token_kl_floor_filter.reduction", "mean_kept")
        ).strip().lower()
        if reduction != "mean_kept":
            raise ValueError("Token KL floor filtering requires reduction=mean_kept.")
        direction = str(
            get_nested(
                cfg,
                "scopd.token_kl_floor_filter.kl_direction",
                "teacher_to_student",
            )
        ).strip().lower()
        if direction != "teacher_to_student":
            raise ValueError(
                "Token KL floor filtering must preserve SCOPD KL(q_teacher || p_student)."
            )
        kl_chunk_size = int(
            get_nested(cfg, "scopd.token_kl_floor_filter.kl_chunk_size", 32)
        )
        if kl_chunk_size <= 0:
            raise ValueError("scopd.token_kl_floor_filter.kl_chunk_size must be positive.")
        if token_outlier_exclusion_enabled:
            raise ValueError("Token KL floor filtering cannot be combined with top-k exclusion.")
        if bool(get_nested(cfg, "scopd.native_budget_weighting.enabled", False)):
            raise ValueError("Token KL floor filtering cannot be combined with native budget weighting.")
        if bool(get_nested(cfg, "scopd.trajectory_weighting.enabled", False)):
            raise ValueError("Token KL floor filtering cannot be combined with trajectory weighting.")
    if token_outlier_exclusion_enabled:
        if method != "scopd_nogt":
            raise ValueError("Token outlier exclusion is restricted to training.method=scopd_nogt.")
        top_k = int(get_nested(cfg, "scopd.token_outlier_exclusion.top_k", 0) or 0)
        top_k_by_ratio = get_nested(
            cfg, "scopd.token_outlier_exclusion.top_k_by_ratio", None
        )
        if top_k_by_ratio is None and top_k <= 0:
            raise ValueError(
                "scopd.token_outlier_exclusion.top_k must be positive when exclusion is enabled."
            )
        if top_k_by_ratio is not None:
            train_ratios = get_nested(cfg, "pruning.train_retention_ratios", [])
            resolved_top_k = [
                resolve_token_outlier_top_k(top_k, top_k_by_ratio, float(ratio))
                for ratio in train_ratios
            ]
            if not resolved_top_k or not any(value > 0 for value in resolved_top_k):
                raise ValueError(
                    "Ratio-specific token outlier exclusion requires at least one positive top_k."
                )
        ranking_direction = str(
            get_nested(
                cfg,
                "scopd.token_outlier_exclusion.ranking_kl_direction",
                "teacher_to_student",
            )
        ).strip().lower()
        if ranking_direction != "teacher_to_student":
            raise ValueError(
                "Token outlier exclusion must rank the same forward KL as SCOPD and requires "
                "ranking_kl_direction=teacher_to_student."
            )
        training_direction = str(
            get_nested(
                cfg,
                "scopd.token_outlier_exclusion.training_kl_direction",
                "teacher_to_student",
            )
        ).strip().lower()
        if training_direction != "teacher_to_student":
            raise ValueError(
                "Token outlier exclusion preserves SCOPD and requires "
                "training_kl_direction=teacher_to_student."
            )
        kl_chunk_size = int(
            get_nested(cfg, "scopd.token_outlier_exclusion.kl_chunk_size", 32)
        )
        if kl_chunk_size <= 0:
            raise ValueError("scopd.token_outlier_exclusion.kl_chunk_size must be positive.")
        if not bool(
            get_nested(cfg, "scopd.token_outlier_exclusion.renormalize_remaining_mean", True)
        ):
            raise ValueError(
                "Token outlier exclusion requires renormalize_remaining_mean=true to avoid loss-scale shrinkage."
            )
        if bool(get_nested(cfg, "scopd.native_budget_weighting.enabled", False)):
            raise ValueError("Token outlier exclusion cannot be combined with native budget weighting.")
        if bool(get_nested(cfg, "scopd.trajectory_weighting.enabled", False)):
            raise ValueError("Token outlier exclusion cannot be combined with trajectory weighting.")
    validate_paired_native_budget_config(cfg, method, parameter_scope, pruning_method)
    validate_phase_ratio_scaling_config(
        get_nested(cfg, "scopd.phase_ratio_scaling", None),
        method=method,
        train_retention_ratios=get_nested(cfg, "pruning.train_retention_ratios", []),
    )
    curriculum_state = initialize_trajectory_curriculum_state(cfg)
    teacher_ground_truth_access = bool(get_nested(cfg, "scopd.teacher_ground_truth_access", False))
    if method == "scopd_gt_prompt" and not teacher_ground_truth_access:
        raise ValueError("training.method=scopd_gt_prompt requires scopd.teacher_ground_truth_access=true.")
    if method == "scopd_nogt" and teacher_ground_truth_access:
        raise ValueError("training.method=scopd_nogt requires scopd.teacher_ground_truth_access=false.")
    if method == "epic_official":
        required_values = {
            "epic.alpha": 0.5,
            "epic.temperature": 2.0,
            "epic.warmup_ratio": 0.03,
            "epic.vision_learning_rate": 2e-6,
            "training.learning_rate": 2e-5,
            "training.weight_decay": 0.0,
            "training.max_grad_norm": 1.0,
            "training.model_max_length": 2048,
        }
        for dotted, expected in required_values.items():
            actual = float(get_nested(cfg, dotted, float("nan")))
            if actual != expected:
                raise ValueError(f"Official EPIC parity requires {dotted}={expected}; got {actual}.")
        if pruning_method != "visionzip":
            raise ValueError(f"This official EPIC adaptation is scoped to VisionZip; got {pruning_method!r}.")
        if str(get_nested(cfg, "epic.teacher_retention_policy", "")) != "official_dynamic_gap":
            raise ValueError("Official EPIC requires epic.teacher_retention_policy=official_dynamic_gap.")
        if parameter_scope not in {"language_decoder_only", "vision_encoder_plus_llm"}:
            raise ValueError(
                "Official EPIC paired runs require parameter_scope=language_decoder_only or "
                f"vision_encoder_plus_llm; got {parameter_scope!r}."
            )
        if parameter_scope == "vision_encoder_plus_llm":
            if get_nested(cfg, "training.lora_layers_to_transform", None) is not None:
                raise ValueError("Official EPIC joint scope cannot restrict LoRA to language-model layers.")
            if get_nested(cfg, "training.lora_layers_pattern", None) is not None:
                raise ValueError("Official EPIC joint scope cannot set a language-only LoRA layers pattern.")
        else:
            expected_layers = list(range(28))
            if get_nested(cfg, "training.lora_layers_to_transform", None) != expected_layers:
                raise ValueError("Official EPIC LLM-only scope must explicitly select Qwen decoder layers 0..27.")
            if get_nested(cfg, "training.lora_layers_pattern", None) != "layers":
                raise ValueError("Official EPIC LLM-only scope requires lora_layers_pattern=layers.")
        if not bool(get_nested(cfg, "training.gradient_checkpointing", False)):
            raise ValueError("Official EPIC requires training.gradient_checkpointing=true.")
        if bool(get_nested(cfg, "training.gradient_checkpointing_use_reentrant", True)):
            raise ValueError(
                "The Qwen LoRA adaptation requires non-reentrant gradient checkpointing so frozen-base "
                "visual activations retain LoRA gradients."
            )
        if not bool(get_nested(cfg, "training.tf32", False)):
            raise ValueError("Official EPIC requires training.tf32=true.")
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    if method == "scopd_fixed_teacher":
        set_nested(cfg, "scopd.teacher_strategy", "fixed_base")
        set_nested(cfg, "scopd.fixed_teacher", True)
    elif (
        method == "scopd"
        and not str(get_nested(cfg, "scopd.teacher_strategy", "") or "").strip()
        and not str(get_nested(cfg, "scopd.teacher_adapter_path", "") or "").strip()
    ):
        set_nested(cfg, "scopd.teacher_strategy", "dynamic_shared_current")
        set_nested(cfg, "scopd.fixed_teacher", False)
    output_dir = Path(str(cfg.get("output_dir", OUTPUT_ROOT / "checkpoints" / method)))
    resume_checkpoint, resume_metadata = prepare_resume_config(cfg, output_dir)
    log_path = output_dir / "training_log.jsonl"
    student_text_log_path = output_dir / "student_text_outputs.jsonl"
    ratio_group_monitor_log_path = output_dir / "ratio_group_weight_monitor.jsonl"
    trajectory_weight_monitor_log_path = output_dir / "trajectory_weight_monitor.jsonl"
    max_steps = int(get_nested(cfg, "training.max_steps", 1000))
    start_step = int(get_nested(cfg, "training.start_step", 0) or 0)
    stop_at_step = int(get_nested(cfg, "checkpointing.stop_at_step", max_steps) or max_steps)
    if start_step < 0 or start_step > max_steps:
        raise ValueError(f"training.start_step must be in [0, max_steps]; got {start_step} with max_steps={max_steps}.")
    if stop_at_step <= start_step or stop_at_step > max_steps:
        raise ValueError(
            "checkpointing.stop_at_step must satisfy start_step < stop_at_step <= max_steps; "
            f"got start_step={start_step}, stop_at_step={stop_at_step}, max_steps={max_steps}."
        )
    if is_main_process(rank) and log_path.exists() and start_step == 0:
        log_path.unlink()
    if is_main_process(rank) and student_text_log_path.exists() and start_step == 0:
        student_text_log_path.unlink()
    if is_main_process(rank) and ratio_group_monitor_log_path.exists() and start_step == 0:
        ratio_group_monitor_log_path.unlink()
    if is_main_process(rank) and trajectory_weight_monitor_log_path.exists() and start_step == 0:
        trajectory_weight_monitor_log_path.unlink()
    native_rank_log_path = output_dir / f"rank{rank}_native_budget_metrics.jsonl"
    outlier_rank_log_path = output_dir / f"rank{rank}_token_outlier_metrics.jsonl"
    kl_floor_rank_log_path = output_dir / f"rank{rank}_token_kl_floor_metrics.jsonl"
    paired_rank_log_path = output_dir / f"rank{rank}_paired_sampling.jsonl"
    assignment_rank_log_path = output_dir / f"rank{rank}_sample_assignments.jsonl"
    if start_step == 0 and native_rank_log_path.exists():
        native_rank_log_path.unlink()
    if start_step == 0 and outlier_rank_log_path.exists():
        outlier_rank_log_path.unlink()
    if start_step == 0 and kl_floor_rank_log_path.exists():
        kl_floor_rank_log_path.unlink()
    if start_step == 0 and paired_rank_log_path.exists():
        paired_rank_log_path.unlink()
    if start_step == 0 and assignment_rank_log_path.exists():
        assignment_rank_log_path.unlink()
    if resume_checkpoint is not None:
        if is_main_process(rank):
            trim_jsonl_to_step(log_path, start_step)
            trim_jsonl_to_step(student_text_log_path, start_step)
            trim_jsonl_to_step(ratio_group_monitor_log_path, start_step)
        trim_jsonl_to_step(native_rank_log_path, start_step)
        trim_jsonl_to_step(outlier_rank_log_path, start_step)
        trim_jsonl_to_step(kl_floor_rank_log_path, start_step)
        trim_jsonl_to_step(paired_rank_log_path, start_step)
        trim_jsonl_to_step(assignment_rank_log_path, start_step)
        distributed_barrier(distributed)
    if is_main_process(rank):
        save_resolved_config(cfg, output_dir)
    dataset_verification = verify_decontaminated_dataset(cfg)
    if is_main_process(rank) and dataset_verification is not None:
        (output_dir / "training_data_verification.json").write_text(
            json.dumps(dataset_verification, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    rng = random.Random(int(get_nested(cfg, "training.seed", 42)) + rank)
    torch.manual_seed(int(get_nested(cfg, "training.seed", 42)) + rank)
    dataset = load_aokvqa_dataset(
        get_nested(cfg, "dataset.name", "HuggingFaceM4/A-OKVQA"),
        splits=list(get_nested(cfg, "dataset.use_splits", ["train", "validation"])),
        limit=int(get_nested(cfg, "dataset.limit", 0) or 0),
        seed=int(get_nested(cfg, "training.seed", 42)),
        prompt_mode=prompt_mode,
        shuffle=bool(get_nested(cfg, "dataset.shuffle", True)),
    )
    dataset = apply_selected_ids(dataset, get_nested(cfg, "dataset.selected_ids_path", ""))
    if not dataset:
        raise ValueError("Training dataset is empty.")
    if distributed and max_steps % world_size != 0:
        raise ValueError(f"DDP requires max_steps divisible by world_size; got max_steps={max_steps}, world_size={world_size}.")
    if distributed and start_step % world_size != 0:
        raise ValueError(f"DDP requires start_step divisible by world_size; got start_step={start_step}, world_size={world_size}.")
    if distributed and stop_at_step % world_size != 0:
        raise ValueError(
            f"DDP requires stop_at_step divisible by world_size; got {stop_at_step} and world_size={world_size}."
        )
    remaining_steps = stop_at_step - start_step
    if distributed and remaining_steps % world_size != 0:
        raise ValueError(
            f"DDP requires remaining steps divisible by world_size; got remaining={remaining_steps}, world_size={world_size}."
        )
    device_map = get_nested(cfg, "training.device_map", "auto")
    if distributed:
        device_map = {"": local_rank}
        stagger = float(os.environ.get("SCOPD_DDP_STAGGER_LOAD_SECONDS", "0"))
        if stagger > 0:
            time.sleep(float(local_rank) * stagger)
    min_pixels, max_pixels = image_pixel_bounds_from_config(cfg)
    model, processor = load_qwen_model_and_processor(
        str(get_nested(cfg, "base_model", "Qwen/Qwen2.5-VL-7B-Instruct")),
        bf16=bool(get_nested(cfg, "training.bf16", True)),
        attn_implementation=str(get_nested(cfg, "training.attn_implementation", "flash_attention_2")),
        device_map=device_map,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    if bool(get_nested(cfg, "training.use_lora", True)):
        model = apply_lora(
            model,
            r=int(get_nested(cfg, "training.lora_r", 16)),
            alpha=int(get_nested(cfg, "training.lora_alpha", 32)),
            dropout=float(get_nested(cfg, "training.lora_dropout", 0.05)),
            target_modules=list(get_nested(cfg, "training.target_modules", [])) or None,
            adapter_path=str(get_nested(cfg, "training.adapter_path", "")),
            layers_to_transform=get_nested(cfg, "training.lora_layers_to_transform", None),
            layers_pattern=get_nested(cfg, "training.lora_layers_pattern", None),
        )
    visual_checkpoint_input_module = ""
    gradient_checkpointing_enabled = bool(
        get_nested(cfg, "training.gradient_checkpointing", False)
    )
    if gradient_checkpointing_enabled:
        if not hasattr(model, "gradient_checkpointing_enable"):
            raise RuntimeError("Loaded Qwen model does not support requested gradient checkpointing.")
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": bool(get_nested(cfg, "training.gradient_checkpointing_use_reentrant", False))
            }
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if parameter_scope == "vision_encoder_plus_llm":
            visual_checkpoint_input_module = enable_visual_checkpoint_input_grads(model)
        else:
            visual_checkpoint_input_module = "not_required_language_decoder_only"
    student_adapter_name = active_lora_adapter_name(model)
    teacher_model = None
    teacher_adapter_name = ""
    ema_shadow: dict[str, torch.Tensor] | None = None
    ema_parameter_names: list[str] = []
    teacher_adapter_path = str(get_nested(cfg, "scopd.teacher_adapter_path", "") or "").strip()
    ema_teacher_enabled = bool(get_nested(cfg, "scopd.use_ema_teacher", False)) or (
        str(get_nested(cfg, "scopd.teacher_strategy", "") or "").strip().lower() in SCOPD_EMA_TEACHER_ALIASES
    )
    ema_settings = resolve_ema_update_settings(cfg) if ema_teacher_enabled else {}
    if teacher_adapter_path:
        if not Path(teacher_adapter_path).exists():
            raise FileNotFoundError(f"SCOPD teacher adapter path does not exist: {teacher_adapter_path}")
        teacher_adapter_name = str(get_nested(cfg, "scopd.teacher_adapter_name", DEFAULT_TEACHER_ADAPTER_NAME) or "")
        if not teacher_adapter_name:
            raise ValueError("scopd.teacher_adapter_name must be non-empty when scopd.teacher_adapter_path is set.")
        if teacher_adapter_name == student_adapter_name:
            raise ValueError(
                f"scopd.teacher_adapter_name={teacher_adapter_name!r} conflicts with the student adapter name."
            )
        load_shared_teacher_lora_adapter(model, teacher_adapter_path, teacher_adapter_name)
    if ema_teacher_enabled:
        ema_parameter_names = trainable_parameter_names(model)
        adapter_path = str(get_nested(cfg, "training.adapter_path", "") or "").strip()
        ema_shadow = load_ema_shadow(adapter_path, model, ema_parameter_names)
        if ema_shadow is None and not bool(ema_settings.get("lazy_init", True)):
            ema_shadow = create_ema_shadow(model, ema_parameter_names)
    if distributed:
        dist.barrier()
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    model.train()
    if teacher_model is not None:
        teacher_model.eval()
    trainable_named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    trainable = [parameter for _, parameter in trainable_named]
    if bool(get_nested(cfg, "paired_sampling.enabled", False)):
        stochastic_dropout = [
            (name, float(module.p))
            for name, module in unwrap_model(model).named_modules()
            if isinstance(module, torch.nn.Dropout) and float(module.p) != 0.0
        ]
        if stochastic_dropout:
            raise ValueError(
                "Paired sensitivity scoring requires deterministic forwards, but nonzero dropout remains: "
                f"{stochastic_dropout[:10]}"
            )
        unexpected_trainable = [
            name
            for name, _ in trainable_named
            if "lora_" not in name or ".language_model.layers." not in name or ".visual." in name
        ]
        if unexpected_trainable:
            raise ValueError(
                "Paired LLM-only SCOPD found trainable parameters outside decoder LoRA: "
                f"{unexpected_trainable[:10]}"
            )
        expected_tensors = int(get_nested(cfg, "training.expected_trainable_tensors", 392))
        expected_parameters = int(get_nested(cfg, "training.expected_trainable_parameters", 40_370_176))
        actual_parameters = sum(parameter.numel() for parameter in trainable)
        if len(trainable_named) != expected_tensors or actual_parameters != expected_parameters:
            raise ValueError(
                "Paired LLM-only LoRA scope mismatch: "
                f"tensors={len(trainable_named)} (expected {expected_tensors}), "
                f"parameters={actual_parameters} (expected {expected_parameters})."
            )
    if is_main_process(rank):
        (output_dir / "trainable_params.txt").write_text(
            f"trainable={sum(p.numel() for p in trainable)}\n"
            f"total={sum(p.numel() for p in model.parameters())}\n"
            f"distributed={distributed}\nworld_size={world_size}\n"
            f"prompt_mode={prompt_mode}\n"
            f"pruning_method={pruning_method}\n"
            f"fastv_tokens_anchor={os.environ.get('SCOPD_FASTV_TOKENS_ANCHOR', '')}\n"
            f"fastv_tokens_prune_layers={os.environ.get('SCOPD_FASTV_TOKENS_PRUNE_LAYERS', '')}\n"
            f"dataset_min_pixels={min_pixels}\n"
            f"dataset_max_pixels={max_pixels}\n"
            f"generation_max_new_tokens={get_nested(cfg, 'generation.max_new_tokens', 128)}\n"
            f"generation_max_unparseable_new_tokens={get_nested(cfg, 'generation.max_unparseable_new_tokens', '')}\n"
            f"scopd_teacher_adapter_path={teacher_adapter_path}\n"
            f"scopd_shared_teacher_adapter_name={teacher_adapter_name}\n"
            f"scopd_student_adapter_name={student_adapter_name}\n"
            f"scopd_teacher_strategy={get_nested(cfg, 'scopd.teacher_strategy', '')}\n"
            f"scopd_use_ema_teacher={ema_teacher_enabled}\n"
            f"scopd_ema_mode={ema_settings.get('mode', '')}\n"
            f"scopd_ema_decay={ema_settings.get('decay', '')}\n"
            f"scopd_ema_alpha={ema_settings.get('alpha', '')}\n"
            f"scopd_ema_lazy_init={ema_settings.get('lazy_init', '')}\n"
            f"scopd_ema_parameter_count={len(ema_parameter_names)}\n"
            f"scopd_ema_shadow={ema_shadow is not None}\n"
            f"scopd_token_outlier_exclusion={token_outlier_exclusion_enabled}\n"
            f"scopd_token_outlier_top_k={get_nested(cfg, 'scopd.token_outlier_exclusion.top_k', 0)}\n"
            f"scopd_token_outlier_top_k_by_ratio={get_nested(cfg, 'scopd.token_outlier_exclusion.top_k_by_ratio', None)}\n"
            f"scopd_token_outlier_ranking_kl_direction={get_nested(cfg, 'scopd.token_outlier_exclusion.ranking_kl_direction', '')}\n"
            f"scopd_token_kl_floor_filter={token_kl_floor_filter_enabled}\n"
            f"scopd_token_kl_floor_min_kl={get_nested(cfg, 'scopd.token_kl_floor_filter.min_kl', '')}\n"
            f"scopd_token_kl_floor_reduction={get_nested(cfg, 'scopd.token_kl_floor_filter.reduction', '')}\n"
            f"gradient_checkpointing={gradient_checkpointing_enabled}\n"
            f"gradient_checkpointing_use_reentrant={get_nested(cfg, 'training.gradient_checkpointing_use_reentrant', '') if gradient_checkpointing_enabled else ''}\n"
            f"epic_upstream_repository={EPIC_UPSTREAM_REPOSITORY if method == 'epic_official' else ''}\n"
            f"epic_upstream_commit={EPIC_UPSTREAM_COMMIT if method == 'epic_official' else ''}\n"
            f"epic_upstream_trainer_sha256={EPIC_UPSTREAM_TRAINER_SHA256 if method == 'epic_official' else ''}\n"
            f"epic_gradient_checkpointing={get_nested(cfg, 'training.gradient_checkpointing', False) if method == 'epic_official' else ''}\n"
            f"epic_gradient_checkpointing_use_reentrant={get_nested(cfg, 'training.gradient_checkpointing_use_reentrant', '') if method == 'epic_official' else ''}\n"
            f"epic_max_grad_norm={get_nested(cfg, 'training.max_grad_norm', '') if method == 'epic_official' else ''}\n"
            f"epic_tf32={get_nested(cfg, 'training.tf32', False) if method == 'epic_official' else ''}\n"
            f"epic_model_max_length={get_nested(cfg, 'training.model_max_length', '') if method == 'epic_official' else ''}\n"
            f"epic_visual_checkpoint_input_module={visual_checkpoint_input_module}\n",
            encoding="utf-8",
        )
    learning_rate = float(get_nested(cfg, "training.learning_rate", 2e-5))
    weight_decay = float(get_nested(cfg, "training.weight_decay", 0.0))
    vision_learning_rate_value = get_nested(cfg, "training.vision_learning_rate", None)
    if method == "epic_official" and parameter_scope == "vision_encoder_plus_llm":
        vision_learning_rate_value = get_nested(cfg, "epic.vision_learning_rate", 2e-6)
    elif method == "epic_official":
        # Keep the upstream vision LR in the config for parity documentation,
        # but no visual optimizer group exists in the controlled LLM-only run.
        vision_learning_rate_value = None

    vision_learning_rate = (
        float(vision_learning_rate_value) if vision_learning_rate_value is not None else None
    )
    if vision_learning_rate is not None:
        visual_trainable = [parameter for name, parameter in trainable_named if ".visual." in name]
        nonvisual_trainable = [parameter for name, parameter in trainable_named if ".visual." not in name]
        if not visual_trainable or not nonvisual_trainable:
            raise ValueError(
                "A separate vision learning rate requires both visual and non-visual trainable parameters; "
                f"found visual={len(visual_trainable)}, nonvisual={len(nonvisual_trainable)}."
            )
        optimizer = torch.optim.AdamW(
            [
                {"params": nonvisual_trainable, "lr": learning_rate, "initial_lr": learning_rate},
                {"params": visual_trainable, "lr": vision_learning_rate, "initial_lr": vision_learning_rate},
            ],
            weight_decay=weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)
    save_every = int(get_nested(cfg, "training.save_every", 500))
    grad_accum = int(get_nested(cfg, "training.gradient_accumulation_steps", 1))
    micro_batch_size = int(get_nested(cfg, "training.micro_batch_size", 1) or 1)
    if micro_batch_size < 1:
        raise ValueError(f"training.micro_batch_size must be >= 1; got {micro_batch_size}.")
    global_step_unit = world_size * micro_batch_size if distributed else micro_batch_size
    effective_batch_size = global_step_unit * grad_accum
    probability_mode = trajectory_probability_mode(cfg)
    trajectory_mode = str(
        get_nested(cfg, "scopd.trajectory_weighting.mode", "")
    ).strip().lower()
    adaptive_sampler_online = (
        bool(get_nested(cfg, "scopd.trajectory_weighting.enabled", False))
        and trajectory_mode == "adaptive_budget_frontier_sampler_batch"
    )
    probability_scope = str(
        get_nested(cfg, "scopd.trajectory_weighting.normalization_scope", "synchronized_block")
    ).strip().lower()
    effective_batch_probability_weighting = (
        probability_mode is not None and probability_scope == "effective_batch"
    )
    replay_cfg: dict[str, Any] | None = None
    if effective_batch_probability_weighting:
        if method != "scopd_nogt":
            raise ValueError("Effective-batch JSD weighting requires training.method=scopd_nogt.")
        if micro_batch_size != 1:
            raise ValueError("Effective-batch JSD weighting currently requires micro_batch_size=1.")
        if not bool(get_nested(cfg, "paired_sampling.enabled", False)):
            raise ValueError("Effective-batch JSD weighting requires deterministic paired sampling.")
        if int(get_nested(cfg, "training.max_sample_retries", 0) or 0) != 0:
            raise ValueError("Effective-batch JSD weighting requires max_sample_retries=0.")
        replay_cfg = copy.deepcopy(cfg)
        set_nested(replay_cfg, "scopd.native_budget_weighting.enabled", False)
        set_nested(replay_cfg, "scopd.trajectory_weighting.enabled", False)
    checkpointing_enabled = bool(get_nested(cfg, "checkpointing.enabled", False))
    eval_snapshot_every = int(get_nested(cfg, "checkpointing.eval_snapshot_every", 0) or 0)
    resumable_every = int(get_nested(cfg, "checkpointing.resumable_every", 0) or 0)
    if checkpointing_enabled:
        for label, interval in (
            ("eval_snapshot_every", eval_snapshot_every),
            ("resumable_every", resumable_every),
        ):
            if interval <= 0 or interval % effective_batch_size != 0:
                raise ValueError(
                    f"checkpointing.{label} must be positive and divisible by effective batch size "
                    f"{effective_batch_size}; got {interval}."
                )
        if (
            max_steps % effective_batch_size != 0
            or start_step % effective_batch_size != 0
            or stop_at_step % effective_batch_size != 0
        ):
            raise ValueError(
                "Full paired runs require max_steps/start_step/stop_at_step on optimizer boundaries; "
                f"got max_steps={max_steps}, start_step={start_step}, stop_at_step={stop_at_step}, "
                f"effective_batch_size={effective_batch_size}."
            )
        if save_every != 0:
            raise ValueError("Set training.save_every=0 when structured checkpointing is enabled.")
        if resume_checkpoint is not None:
            if resume_metadata is None:
                raise AssertionError("Resume metadata was not loaded.")
            if int(resume_metadata.get("world_size", -1)) != int(world_size):
                raise ValueError(
                    f"Resume world size mismatch: checkpoint={resume_metadata.get('world_size')} current={world_size}."
                )
            optimizer.load_state_dict(_torch_load_full(resume_checkpoint / "optimizer.pt"))
            rank_state_path = resume_checkpoint / "rank_rng_states" / f"rank_{rank:02d}.pt"
            if not rank_state_path.is_file():
                raise FileNotFoundError(f"Missing rank RNG state: {rank_state_path}")
            restored_curriculum_state = restore_rank_rng_state(
                _torch_load_full(rank_state_path), rng, expected_step=start_step
            )
            if curriculum_state is not None:
                if restored_curriculum_state is None:
                    raise ValueError("Resume checkpoint is missing trajectory curriculum state.")
                curriculum_state.load_state_dict(restored_curriculum_state)
        elif bool(get_nested(cfg, "checkpointing.save_step_zero", False)):
            save_lora_evaluation_snapshot(
                model,
                output_dir,
                global_step=0,
                cfg=cfg,
                rank=rank,
                distributed=distributed,
            )
    official_epic_rng: random.Random | None = None
    official_epic_total_optimizer_steps = 0
    official_epic_optimizer_steps_completed = 0
    official_epic_scheduler: Any | None = None
    if method == "epic_official":
        if micro_batch_size != 1:
            raise ValueError("Official EPIC parity currently requires training.micro_batch_size=1.")
        schedule = str(get_nested(cfg, "pruning.retention_ratio_schedule", "") or "").strip().lower()
        if schedule != "epic_official_progressive_continuous":
            raise ValueError(
                "Official EPIC requires pruning.retention_ratio_schedule="
                "epic_official_progressive_continuous."
            )
        effective_batch_size = global_step_unit * grad_accum
        if max_steps % effective_batch_size != 0 or start_step % effective_batch_size != 0:
            raise ValueError(
                "Official EPIC requires max_steps and start_step aligned to optimizer-update boundaries; "
                f"got max_steps={max_steps}, start_step={start_step}, effective_batch_size={effective_batch_size}."
            )
        official_epic_total_optimizer_steps = max_steps // effective_batch_size
        official_epic_optimizer_steps_completed = start_step // effective_batch_size
        official_epic_rng = random.Random(int(get_nested(cfg, "training.seed", 42)))
        prior_local_calls = start_step // global_step_unit
        for prior_local_call in range(prior_local_calls):
            sample_official_epic_curriculum(
                official_epic_rng,
                optimizer_step=prior_local_call // grad_accum,
                total_optimizer_steps=official_epic_total_optimizer_steps,
            )
        if start_step != 0:
            raise ValueError(
                "Official EPIC resume requires optimizer and scheduler state, which this adapter-only checkpoint "
                "format does not save. Start a fresh run with training.start_step=0."
            )
        from transformers.optimization import get_cosine_schedule_with_warmup

        warmup_ratio = float(get_nested(cfg, "epic.warmup_ratio", 0.03))
        warmup_steps = int(math.ceil(official_epic_total_optimizer_steps * warmup_ratio))
        official_epic_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=official_epic_total_optimizer_steps,
        )
    max_sample_retries = int(get_nested(cfg, "training.max_sample_retries", 0) or 0)
    if max_sample_retries < 0:
        raise ValueError(f"training.max_sample_retries must be >= 0; got {max_sample_retries}.")
    if remaining_steps % global_step_unit != 0:
        raise ValueError(
            "DDP requires remaining steps divisible by world_size * training.micro_batch_size; "
            f"got remaining={remaining_steps}, world_size={world_size}, micro_batch_size={micro_batch_size}."
        )
    local_steps = remaining_steps // global_step_unit
    step = 0
    accum = 0
    effective_batch_window: list[dict[str, Any]] = []
    effective_batch_summary: dict[str, Any] = {}
    start = time.time()
    try:
        while step < local_steps:
            sample: FormattedAOKVQASample | None = None
            ratio: float | None = None
            try:
                if effective_batch_probability_weighting and accum == 0:
                    effective_batch_window, effective_batch_summary = (
                        prepare_effective_batch_probability_window(
                            model,
                            processor,
                            dataset,
                            cfg,
                            rng,
                            teacher_model=teacher_model,
                            ema_shadow=ema_shadow,
                            teacher_adapter_name=teacher_adapter_name,
                            start_step=start_step,
                            local_step_start=step,
                            accumulation_steps=grad_accum,
                            max_steps=max_steps,
                            distributed=distributed,
                            rank=rank,
                            world_size=world_size,
                            curriculum_state=curriculum_state,
                        )
                    )
                    if len(effective_batch_window) != grad_accum:
                        raise AssertionError("Effective-batch probe window has the wrong local size.")
                    if probability_mode == "progress_adaptive_robust_frontier_batch":
                        if not math.isclose(
                            float(
                                effective_batch_summary[
                                    "trajectory_global_weighted_kl"
                                ]
                            ),
                            float(
                                effective_batch_summary[
                                    "trajectory_global_unweighted_kl"
                                ]
                            ),
                            rel_tol=2e-6,
                            abs_tol=2e-7,
                        ):
                            raise AssertionError(
                                "Progress-adaptive frontier did not preserve effective-batch KL mass."
                            )
                    elif (
                        probability_mode not in DIRECT_GLOBAL_F_MODES
                        and not math.isclose(
                        float(effective_batch_summary["trajectory_probability_weight_sum"]),
                        1.0,
                        rel_tol=0.0,
                        abs_tol=1e-6,
                        )
                    ):
                        raise AssertionError("Effective-batch trajectory weights do not sum to one.")
                    elif probability_mode in DIRECT_GLOBAL_F_MODES and (
                        not math.isfinite(
                            float(
                                effective_batch_summary[
                                    "trajectory_probability_weight_sum"
                                ]
                            )
                        )
                        or float(
                            effective_batch_summary["trajectory_probability_weight_sum"]
                        )
                        < 0.0
                    ):
                        raise AssertionError(
                            "Direct global trajectory objective has invalid detached weight mass."
                        )
                    if is_main_process(rank):
                        probe_log = dict(effective_batch_summary)
                        probe_log.update(
                            {
                                "optimizer_batch_start_step": int(
                                    start_step + step * global_step_unit
                                ),
                                "optimizer_batch_end_step": int(
                                    start_step + (step + grad_accum) * global_step_unit
                                ),
                            }
                        )
                        write_jsonl(
                            output_dir / "effective_batch_probe_metrics.jsonl",
                            probe_log,
                        )
                        if probability_mode in COUNTERFACTUAL_TEACHABILITY_MODES:
                            monitor_row = {
                                key: value
                                for key, value in probe_log.items()
                                if key != "global_records"
                            }
                            monitor_row["sample_weights"] = [
                                {
                                    "global_index": int(record["global_index"]),
                                    "sample_id": str(record["sample_id"]),
                                    "ratio": float(record["ratio"]),
                                    "sampled_b_plus": float(record["sampled_b_plus"]),
                                    "ratio_group_signal": float(
                                        record["ratio_group_signal"]
                                    ),
                                    "weight_signal": float(record["weight_signal"]),
                                    "probability_weight": float(
                                        record["probability_weight"]
                                    ),
                                    "effective_multiplier": float(
                                        record["probability_weight"]
                                    )
                                    * int(
                                        monitor_row["trajectory_effective_batch_size"]
                                    ),
                                    "unweighted_kl": float(record["probe_loss"]),
                                }
                                for record in probe_log["global_records"]
                            ]
                            monitor_row["step"] = int(
                                monitor_row["optimizer_batch_end_step"]
                            )
                            write_jsonl(
                                ratio_group_monitor_log_path
                                if probability_mode
                                == "ratio_group_counterfactual_teachability_batch"
                                else trajectory_weight_monitor_log_path,
                                monitor_row,
                            )
                            groups = monitor_row.get("trajectory_ratio_groups", {})
                            direct_mode = (
                                probability_mode
                                == "trajectory_counterfactual_teachability_softmax_batch"
                            )
                            angle_mode = monitor_row.get(
                                "trajectory_weight_transform"
                            ) in {
                                "ratio_group_softmax_angle",
                                "ratio_group_softmax_angle_sample_normalized",
                            }
                            compact = " ".join(
                                f"{name}:n={stats['count']},"
                                f"{'r' if direct_mode else 'cos' if angle_mode else 'g'}="
                                f"{stats['trajectory_signal_mean'] if direct_mode else stats['projection_cosine'] if angle_mode else stats['projection_fraction']:.4f},"
                                f"w={stats['mean_multiplier']:.3f},kl={stats['unweighted_kl_mean']:.5f}"
                                for name, stats in sorted(groups.items())
                            )
                            print(
                                "[trajectory-monitor] "
                                f"samples={monitor_row['optimizer_batch_start_step']}:"
                                f"{monitor_row['optimizer_batch_end_step']} "
                                f"ess={monitor_row['trajectory_effective_sample_size']:.2f}/"
                                f"{monitor_row['trajectory_effective_batch_size']} "
                                f"loss_scale={monitor_row['trajectory_loss_scale_ratio']:.4f} "
                                f"{compact}",
                                flush=True,
                            )
                batch_losses: list[torch.Tensor] = []
                batch_metrics: list[dict[str, Any]] = []
                batch_samples: list[FormattedAOKVQASample] = []
                batch_global_indices: list[int] = []
                batch_ratios: list[float] = []
                batch_rollout_seeds: list[int | None] = []
                batch_text_logs: list[dict[str, Any]] = []
                for micro_idx in range(micro_batch_size):
                    if distributed:
                        base_global_index = start_step + (step * micro_batch_size + micro_idx) * world_size + rank
                    else:
                        base_global_index = start_step + step * micro_batch_size + micro_idx
                    for sample_attempt in range(max_sample_retries + 1):
                        global_index = base_global_index + sample_attempt * global_step_unit
                        sample = dataset[global_index % len(dataset)]
                        official_epic_sample = None
                        if method == "epic_official":
                            if official_epic_rng is None:
                                raise AssertionError("Official EPIC RNG was not initialized.")
                            current_local_call = start_step // global_step_unit + step
                            official_epic_sample = sample_official_epic_curriculum(
                                official_epic_rng,
                                optimizer_step=current_local_call // grad_accum,
                                total_optimizer_steps=official_epic_total_optimizer_steps,
                            )
                            ratio = official_epic_sample.student_retention_ratio
                        elif effective_batch_probability_weighting:
                            effective_assignment = effective_batch_window[accum]
                            if int(effective_assignment["global_index"]) != int(
                                global_index
                            ):
                                raise AssertionError(
                                    "Adaptive probe/replay global index mismatch."
                                )
                            if str(effective_assignment["sample_id"]) != str(
                                sample.sample_id
                            ):
                                raise AssertionError(
                                    "Adaptive probe/replay sample mismatch."
                                )
                            ratio = float(effective_assignment["ratio"])
                        elif adaptive_sampler_online:
                            if not isinstance(
                                curriculum_state, AdaptiveBudgetFrontierState
                            ):
                                raise AssertionError(
                                    "Adaptive budget sampler state is unavailable."
                                )
                            ratio = curriculum_state.select_ratio(
                                global_index, sample.sample_id
                            )
                        else:
                            ratio = sample_retention_ratio(
                                cfg,
                                rng,
                                progress_step=global_index,
                                total_steps=max_steps,
                                sample_id=sample.sample_id,
                            )
                        rollout_seed = (
                            paired_rollout_seed(
                                seed=int(
                                    get_nested(
                                        cfg,
                                        "paired_sampling.rollout_seed",
                                        get_nested(cfg, "training.seed", 42),
                                    )
                                ),
                                global_index=global_index,
                                sample_id=sample.sample_id,
                                namespace=str(get_nested(cfg, "paired_sampling.namespace", "scopd_pair_v1")),
                            )
                            if bool(get_nested(cfg, "paired_sampling.enabled", False))
                            else None
                        )
                        try:
                            if method == "sft":
                                sample_loss, sample_metrics = sft_like_step(model, processor, sample, cfg, ratio, sample.target)
                            elif method == "epic":
                                sample_loss, sample_metrics = epic_tcd_step(model, processor, sample, cfg, ratio)
                            elif method == "epic_official":
                                if official_epic_sample is None:
                                    raise AssertionError("Official EPIC curriculum sample was not initialized.")
                                sample_loss, sample_metrics = epic_tcd_step(
                                    model,
                                    processor,
                                    sample,
                                    cfg,
                                    ratio,
                                    teacher_retention_override=official_epic_sample.teacher_retention_ratio,
                                    official_logit_alignment=True,
                                )
                                sample_metrics.update(official_epic_sample.metrics())
                            elif method == "grpo":
                                sample_loss, sample_metrics = grpo_step(model, processor, sample, cfg, ratio)
                            elif method in {"scopd", "scopd_fixed_teacher"}:
                                sample_loss, sample_metrics = scopd_step(
                                    model,
                                    processor,
                                    sample,
                                    cfg,
                                    ratio,
                                    teacher_model=teacher_model,
                                    ema_shadow=ema_shadow,
                                    teacher_adapter_name=teacher_adapter_name,
                                )
                            elif method in {"scopd_nogt", "scopd_gt_prompt"}:
                                effective_record = (
                                    effective_batch_window[accum]
                                    if effective_batch_probability_weighting
                                    else None
                                )
                                if effective_record is not None:
                                    if int(effective_record["global_index"]) != int(global_index):
                                        raise AssertionError("Probe/replay global index mismatch.")
                                    if str(effective_record["sample_id"]) != str(sample.sample_id):
                                        raise AssertionError("Probe/replay sample mismatch.")
                                    if not math.isclose(
                                        float(effective_record["ratio"]),
                                        float(ratio),
                                        rel_tol=0.0,
                                        abs_tol=1e-12,
                                    ):
                                        raise AssertionError("Probe/replay retention-ratio mismatch.")
                                    if int(effective_record["rollout_seed"]) != int(rollout_seed):
                                        raise AssertionError("Probe/replay rollout-seed mismatch.")
                                    rollout_cache = effective_record["rollout"]
                                    active_cfg = replay_cfg
                                else:
                                    rollout_cache = None
                                    active_cfg = cfg
                                if active_cfg is None:
                                    raise AssertionError("Effective-batch replay config was not initialized.")
                                sample_loss, sample_metrics = scopd_nogt_step(
                                    model,
                                    processor,
                                    sample,
                                    active_cfg,
                                    ratio,
                                    teacher_model=teacher_model,
                                    ema_shadow=ema_shadow,
                                    teacher_adapter_name=teacher_adapter_name,
                                    teacher_uses_ground_truth=method == "scopd_gt_prompt",
                                    rollout_seed=rollout_seed,
                                    progress_step=global_index,
                                    total_steps=max_steps,
                                    fixed_rollout_token_ids=(
                                        rollout_cache["token_ids"]
                                        if rollout_cache is not None
                                        else None
                                    ),
                                    fixed_rollout_text=(
                                        rollout_cache["text"]
                                        if rollout_cache is not None
                                        else None
                                    ),
                                    fixed_rollout_metadata=(
                                        rollout_cache["generation_metadata"]
                                        if rollout_cache is not None
                                        else None
                                    ),
                                )
                                if effective_record is not None:
                                    replay_text_log = sample_metrics.get(STUDENT_TEXT_LOG_KEY)
                                    if not isinstance(replay_text_log, dict):
                                        raise AssertionError("Replay did not return a student text record.")
                                    if replay_text_log.get("student_text") != rollout_cache["text"]:
                                        raise AssertionError("Probe/replay generated text mismatch.")
                                    replay_error = abs(
                                        float(sample_loss.detach().float().cpu())
                                        - float(effective_record["probe_loss"])
                                    )
                                    sample_metrics.update(
                                        {
                                            "native_student_budget_jsd_mean": float(
                                                effective_record["jsd"]
                                            ),
                                            "native_teacher_gap_b_mean": float(
                                                effective_record["current_teacher_kl"]
                                            ),
                                            "native_teacher_gap_b_plus_mean": float(
                                                effective_record["b_plus_teacher_kl"]
                                            ),
                                            "sampled_b_plus": float(
                                                effective_record["sampled_b_plus"]
                                            ),
                                            "native_b_plus_num_kept_visual_tokens": int(
                                                effective_record["b_plus_visual_tokens"]
                                            ),
                                            "native_b_plus_random_mask_hash": effective_record.get(
                                                "b_plus_random_mask_hash"
                                            ),
                                            "native_random_b_subset_b_plus": effective_record.get(
                                                "random_b_subset_b_plus"
                                            ),
                                            "effective_batch_probe_native_budget_weighting_enabled": True,
                                            "effective_batch_sensitivity": float(
                                                effective_record["signal"]
                                            ),
                                            "effective_batch_probability_weight": float(
                                                effective_record["probability_weight"]
                                            ),
                                            "effective_batch_probe_replay_kl_abs_error": replay_error,
                                            "effective_batch_rollout_token_ids_sha256": rollout_cache[
                                                "token_ids_sha256"
                                            ],
                                            "effective_batch_fixed_prefix_replay": True,
                                        }
                                    )
                                    if probability_mode in COUNTERFACTUAL_TEACHABILITY_MODES:
                                        sample_metrics.update(
                                            {
                                                "native_trajectory_budget_explained_fraction": float(
                                                    effective_record[
                                                        "budget_explained_fraction"
                                                    ]
                                                ),
                                                "effective_batch_ratio_group_signal": float(
                                                    effective_record["ratio_group_signal"]
                                                ),
                                                "effective_batch_weight_signal": float(
                                                    effective_record["weight_signal"]
                                                ),
                                                "native_trajectory_budget_projection_mass": float(
                                                    effective_record["projection_mass"]
                                                ),
                                                "native_trajectory_teacher_js_mass": float(
                                                    effective_record["teacher_js_mass"]
                                                ),
                                            }
                                        )
                                        if (
                                            probability_mode
                                            == "progress_adaptive_robust_frontier_batch"
                                        ):
                                            sample_metrics[
                                                "derived_progress_adaptive_robust_need"
                                            ] = float(effective_record["signal"])
                            elif method == "offpolicy":
                                sample_loss, sample_metrics = offpolicy_step(model, processor, sample, cfg, ratio)
                            else:
                                raise AssertionError(method)
                            phase_ratio_config = get_nested(cfg, "scopd.phase_ratio_scaling", None)
                            if isinstance(phase_ratio_config, dict) and bool(
                                phase_ratio_config.get("enabled", False)
                            ):
                                phase_ratio_scale = resolve_phase_ratio_scale(
                                    phase_ratio_config,
                                    retention_ratio=float(ratio),
                                    progress_step=int(global_index),
                                    total_steps=int(max_steps),
                                )
                                unscaled_sample_loss = sample_loss
                                sample_loss = unscaled_sample_loss * phase_ratio_scale.scale
                                sample_metrics.update(
                                    {
                                        **phase_ratio_scale.metrics(),
                                        "phase_ratio_unweighted_loss": float(
                                            unscaled_sample_loss.detach().cpu()
                                        ),
                                        "phase_ratio_weighted_loss": float(sample_loss.detach().cpu()),
                                        "phase_ratio_weight_detached": True,
                                    }
                                )
                            sample_text_log = sample_metrics.pop(STUDENT_TEXT_LOG_KEY, None)
                            if isinstance(sample_text_log, dict):
                                batch_text_logs.append(
                                    {
                                        "global_index": global_index,
                                        "micro_idx": micro_idx,
                                        "retry_attempt": sample_attempt,
                                        **sample_text_log,
                                    }
                                )
                            break
                        except Exception as exc:
                            if sample_attempt >= max_sample_retries or not is_retryable_training_error(exc):
                                raise
                            write_jsonl(
                                output_dir / f"rank{rank}_skipped_samples.jsonl",
                                {
                                    "step": start_step + (step + 1) * global_step_unit,
                                    "local_step": step + 1,
                                    "rank": rank,
                                    "method": method,
                                    "sample_id": sample.sample_id,
                                    "global_index": global_index,
                                    "retention_ratio": ratio,
                                    "retry_attempt": sample_attempt + 1,
                                    "error": repr(exc),
                                },
                            )
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                    batch_losses.append(sample_loss)
                    batch_metrics.append(sample_metrics)
                    batch_samples.append(sample)
                    batch_global_indices.append(global_index)
                    batch_ratios.append(float(ratio))
                    batch_rollout_seeds.append(rollout_seed)
                if effective_batch_probability_weighting:
                    if len(batch_losses) != 1:
                        raise AssertionError("Effective-batch probability weighting expects one local loss.")
                    effective_record = effective_batch_window[accum]
                    probability_weight = torch.tensor(
                        float(effective_record["probability_weight"]),
                        dtype=batch_losses[0].dtype,
                        device=batch_losses[0].device,
                    ).detach()
                    loss = effective_batch_local_objective(
                        batch_losses[0],
                        probability_weight,
                        effective_batch_size=effective_batch_size,
                    )
                    summary_metrics = {
                        key: value
                        for key, value in effective_batch_summary.items()
                        if key not in {"global_records", "trajectory_ratio_groups"}
                    }
                    trajectory_metrics = {
                        "trajectory_weighting_enabled": True,
                        **summary_metrics,
                        "trajectory_probability_weight": float(probability_weight.cpu()),
                        "trajectory_effective_multiplier": float(
                            probability_weight.cpu() * effective_batch_size
                        ),
                        "trajectory_ddp_accumulation_objective_scale": float(
                            effective_batch_size
                        ),
                        "trajectory_weight_detached": not probability_weight.requires_grad,
                    }
                else:
                    loss, trajectory_metrics = apply_distributed_trajectory_weighting(
                        batch_losses,
                        batch_metrics,
                        cfg,
                        distributed=distributed,
                        rank=rank,
                        world_size=world_size,
                        curriculum_state=curriculum_state,
                    )
                metrics = aggregate_microbatch_metrics(batch_metrics)
                metrics.update(trajectory_metrics)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss: {loss}")
                sync_context = model.no_sync() if distributed and hasattr(model, "no_sync") and (accum + 1) < grad_accum else nullcontext()
                with sync_context:
                    (loss / grad_accum).backward()
                accum += 1
                ema_update_metrics: dict[str, Any] = {}
                gradient_norm_value: float | None = None
                if accum >= grad_accum:
                    if method == "epic_official":
                        max_grad_norm = float(get_nested(cfg, "training.max_grad_norm", 1.0))
                        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                        gradient_norm_value = float(gradient_norm.detach().cpu())
                    optimizer.step()
                    if official_epic_scheduler is not None:
                        official_epic_scheduler.step()
                        official_epic_optimizer_steps_completed += 1
                    if ema_teacher_enabled and teacher_adapter_name:
                        update_ema_adapter(
                            model,
                            ema_parameter_names,
                            source_adapter=student_adapter_name,
                            target_adapter=teacher_adapter_name,
                            decay=float(ema_settings["decay"]),
                        )
                        ema_update_metrics = {
                            "scopd_ema_update": "updated_teacher_adapter",
                            "scopd_ema_mode": ema_settings["mode"],
                            "scopd_ema_decay": float(ema_settings["decay"]),
                            "scopd_ema_teacher_adapter": teacher_adapter_name,
                        }
                    elif ema_teacher_enabled and ema_shadow is not None:
                        update_ema_shadow(
                            model,
                            ema_shadow,
                            ema_parameter_names,
                            decay=float(ema_settings["decay"]),
                        )
                        ema_update_metrics = {
                            "scopd_ema_update": "updated",
                            "scopd_ema_mode": ema_settings["mode"],
                            "scopd_ema_decay": float(ema_settings["decay"]),
                        }
                    elif ema_teacher_enabled and ema_shadow is None and teacher_model is None:
                        ema_shadow = create_ema_shadow(model, ema_parameter_names)
                        ema_update_metrics = {
                            "scopd_ema_update": "initialized",
                            "scopd_ema_mode": ema_settings["mode"],
                            "scopd_ema_decay": float(ema_settings["decay"]),
                        }
                    elif ema_teacher_enabled and teacher_model is not None:
                        update_ema_teacher(
                            model,
                            teacher_model,
                            ema_parameter_names,
                            decay=float(ema_settings["decay"]),
                        )
                        ema_update_metrics = {
                            "scopd_ema_update": "updated_external_teacher",
                            "scopd_ema_mode": ema_settings["mode"],
                            "scopd_ema_decay": float(ema_settings["decay"]),
                        }
                        teacher_model.eval()
                    optimizer.zero_grad(set_to_none=True)
                    accum = 0
                step += 1
                global_step = min(stop_at_step, start_step + step * global_step_unit)
                row = {
                    "step": global_step,
                    "local_step": step,
                    "rank": rank,
                    "world_size": world_size,
                    "method": method,
                    "pruning_method": pruning_method,
                    "prompt_mode": prompt_mode,
                    "generation_max_new_tokens": int(get_nested(cfg, "generation.max_new_tokens", 128)),
                    "generation_max_unparseable_new_tokens": get_nested(cfg, "generation.max_unparseable_new_tokens", None),
                    "generation_stop_on_parse": generation_stop_on_parse(cfg),
                    "micro_batch_size": micro_batch_size,
                    "sample_id": batch_samples[0].sample_id,
                    "sample_ids": [item.sample_id for item in batch_samples],
                    "global_index": batch_global_indices[0],
                    "global_indices": batch_global_indices,
                    "retention_ratio": batch_ratios[0],
                    "retention_ratios": batch_ratios,
                    "rollout_seed": batch_rollout_seeds[0],
                    "rollout_seeds": batch_rollout_seeds,
                    "retention_ratio_schedule": str(get_nested(cfg, "pruning.retention_ratio_schedule", "random") or "random"),
                    "loss": float(loss.detach().cpu()),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "vision_learning_rate": (
                        float(optimizer.param_groups[1]["lr"])
                        if vision_learning_rate is not None and len(optimizer.param_groups) > 1
                        else None
                    ),
                    "optimizer_steps_completed": (
                        official_epic_optimizer_steps_completed if method == "epic_official" else None
                    ),
                    "gradient_norm": gradient_norm_value,
                    "micro_batch_losses": [float(item.detach().cpu()) for item in batch_losses],
                    "elapsed_seconds": time.time() - start,
                    **metrics,
                    **ema_update_metrics,
                }
                if is_main_process(rank):
                    write_jsonl(log_path, row)
                    for text_row in batch_text_logs:
                        write_jsonl(
                            student_text_log_path,
                            {
                                "step": global_step,
                                "local_step": step,
                                "rank": rank,
                                "world_size": world_size,
                                "method": method,
                                "pruning_method": pruning_method,
                                "prompt_mode": prompt_mode,
                                "generation_max_new_tokens": int(get_nested(cfg, "generation.max_new_tokens", 128)),
                                "generation_max_unparseable_new_tokens": get_nested(
                                    cfg, "generation.max_unparseable_new_tokens", None
                                ),
                                "generation_stop_on_parse": generation_stop_on_parse(cfg),
                                **text_row,
                            },
                        )
                    print(json.dumps(row, ensure_ascii=False), flush=True)
                    if not checkpointing_enabled and save_every > 0 and global_step % save_every == 0:
                        save_checkpoint(model, output_dir / f"step_{global_step}", ema_shadow=ema_shadow)
                if bool(get_nested(cfg, "scopd.native_budget_weighting.enabled", False)):
                    write_jsonl(output_dir / f"rank{rank}_native_budget_metrics.jsonl", row)
                if bool(get_nested(cfg, "scopd.token_outlier_exclusion.enabled", False)):
                    write_jsonl(outlier_rank_log_path, row)
                if bool(get_nested(cfg, "scopd.token_kl_floor_filter.enabled", False)):
                    write_jsonl(kl_floor_rank_log_path, row)
                if bool(get_nested(cfg, "scopd.phase_ratio_scaling.enabled", False)):
                    write_jsonl(output_dir / f"rank{rank}_phase_ratio_scaling.jsonl", row)
                if bool(get_nested(cfg, "paired_sampling.enabled", False)):
                    write_jsonl(
                        paired_rank_log_path,
                        {
                            "step": global_step,
                            "rank": rank,
                            "global_indices": batch_global_indices,
                            "sample_ids": [item.sample_id for item in batch_samples],
                            "retention_ratios": batch_ratios,
                            "rollout_seeds": batch_rollout_seeds,
                        },
                    )
                if bool(get_nested(cfg, "checkpointing.log_sample_assignments", False)):
                    write_jsonl(
                        assignment_rank_log_path,
                        {
                            "step": global_step,
                            "rank": rank,
                            "global_indices": batch_global_indices,
                            "sample_ids": [item.sample_id for item in batch_samples],
                            "retention_ratios": batch_ratios,
                            "rollout_seeds": batch_rollout_seeds,
                        },
                    )
                if checkpointing_enabled and global_step % eval_snapshot_every == 0:
                    if accum != 0:
                        raise RuntimeError(f"Evaluation snapshot step {global_step} is not an optimizer boundary.")
                    save_lora_evaluation_snapshot(
                        model,
                        output_dir,
                        global_step=global_step,
                        cfg=cfg,
                        rank=rank,
                        distributed=distributed,
                    )
                if checkpointing_enabled and global_step % resumable_every == 0:
                    if accum != 0:
                        raise RuntimeError(f"Resumable checkpoint step {global_step} is not an optimizer boundary.")
                    save_full_resumable_checkpoint(
                        model,
                        optimizer,
                        ema_shadow,
                        rng,
                        output_dir,
                        global_step=global_step,
                        cfg=cfg,
                        rank=rank,
                        world_size=world_size,
                        distributed=distributed,
                        curriculum_state=curriculum_state,
                    )
            except Exception as exc:
                row = {
                    "step": start_step + (step + 1) * world_size if distributed else start_step + step + 1,
                    "local_step": step + 1,
                    "rank": rank,
                    "method": method,
                    "pruning_method": pruning_method,
                    "sample_id": sample.sample_id if sample is not None else "",
                    "retention_ratio": ratio,
                    "error": repr(exc),
                }
                write_jsonl(output_dir / f"rank{rank}_errors.jsonl", row)
                if is_main_process(rank):
                    write_jsonl(log_path, row)
                raise
        if checkpointing_enabled:
            final_checkpoint = save_full_resumable_checkpoint(
                model,
                optimizer,
                ema_shadow,
                rng,
                output_dir,
                global_step=stop_at_step,
                cfg=cfg,
                rank=rank,
                world_size=world_size,
                distributed=distributed,
                curriculum_state=curriculum_state,
            )
            if stop_at_step == max_steps:
                point_final_checkpoint_at(output_dir, final_checkpoint, rank, distributed)
            else:
                distributed_barrier(distributed)
                if is_main_process(rank):
                    _atomic_write_json(
                        output_dir / f"segment_complete_step_{stop_at_step:06d}.json",
                        {
                            "status": "segment_complete",
                            "global_step": int(stop_at_step),
                            "training_target_global_step": int(max_steps),
                            "resume_checkpoint": str(final_checkpoint.resolve()),
                            "config_contract_sha256": checkpoint_contract_sha256(cfg),
                        },
                    )
                distributed_barrier(distributed)
        elif is_main_process(rank):
            save_checkpoint(model, output_dir / "final", ema_shadow=ema_shadow)
    finally:
        cleanup_distributed(distributed)
    return output_dir


def save_checkpoint(model: Any, path: Path, ema_shadow: dict[str, torch.Tensor] | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model = unwrap_model(model)
    if hasattr(model, "save_pretrained"):
        if hasattr(model, "peft_config"):
            model.save_pretrained(path, save_embedding_layers=False)
        else:
            model.save_pretrained(path)
    else:
        torch.save(model.state_dict(), path / "pytorch_model.bin")
    save_ema_shadow(path, ema_shadow)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = resolve_config(args)
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
