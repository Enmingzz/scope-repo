"""Pruning-aware self-distillation utilities for Qwen2.5-VL."""

from .losses import compute_ce_loss, compute_kd_loss, teacher_confidence_weights
from .pruners import (
    BaseVisualTokenPruner,
    DivPruneLitePruner,
    ExistingMethodPruner,
    GridPruner,
    KeepAllPruner,
    RandomPruner,
    VScanStage1Pruner,
    build_pruner,
)
from .qwen25_pruned_forward import (
    build_pruned_inputs_embeds,
    compute_full_position_ids,
    extract_next_token_logits,
    get_qwen25_visual_embeds,
    validate_single_image_qwen_inputs,
)

__all__ = [
    "BaseVisualTokenPruner",
    "DivPruneLitePruner",
    "ExistingMethodPruner",
    "GridPruner",
    "KeepAllPruner",
    "RandomPruner",
    "VScanStage1Pruner",
    "build_pruner",
    "build_pruned_inputs_embeds",
    "compute_ce_loss",
    "compute_full_position_ids",
    "compute_kd_loss",
    "extract_next_token_logits",
    "get_qwen25_visual_embeds",
    "teacher_confidence_weights",
    "validate_single_image_qwen_inputs",
]
