from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NativeBudgetWeights:
    sensitivity: torch.Tensor
    tau: torch.Tensor
    robustness: torch.Tensor
    raw_weight: torch.Tensor
    weight: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class HellingerCurriculumWeights:
    affinity: torch.Tensor
    difficulty: torch.Tensor
    raw_weight: torch.Tensor
    weight: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class TeacherGapPersistenceWeights:
    teacher_gap_b: torch.Tensor
    teacher_gap_b_plus: torch.Tensor
    tau_teacher_gap: torch.Tensor
    rescue_fraction: torch.Tensor
    persistence: torch.Tensor
    confidence: torch.Tensor
    priority: torch.Tensor
    raw_weight: torch.Tensor
    loss_mass_scale: torch.Tensor
    weight: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class SymmetricTeacherGapStabilityWeights:
    teacher_gap_b: torch.Tensor
    teacher_gap_b_plus: torch.Tensor
    signed_budget_change: torch.Tensor
    normalized_budget_change: torch.Tensor
    robustness: torch.Tensor
    raw_weight: torch.Tensor
    loss_mass_scale: torch.Tensor
    weight: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class CounterfactualRescueAmplificationWeights:
    teacher_gap_b: torch.Tensor
    teacher_gap_b_plus: torch.Tensor
    rescue_fraction: torch.Tensor
    raw_weight: torch.Tensor
    loss_mass_scale: torch.Tensor
    weight: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class CounterfactualTeachabilityMixtureWeights:
    teacher_gap_b: torch.Tensor
    teacher_gap_b_plus: torch.Tensor
    teacher_gap_rank: torch.Tensor
    rescue_fraction: torch.Tensor
    priority: torch.Tensor
    raw_weight: torch.Tensor
    loss_mass_scale: torch.Tensor
    weight: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class CounterfactualTeachabilityModulationWeights:
    teacher_gap_b: torch.Tensor
    teacher_gap_b_plus: torch.Tensor
    teacher_gap_rank: torch.Tensor
    rescue_fraction: torch.Tensor
    priority: torch.Tensor
    raw_weight: torch.Tensor
    loss_mass_scale: torch.Tensor
    weight: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class ConditionalRescueResidualWeights:
    teacher_gap_b: torch.Tensor
    teacher_gap_b_plus: torch.Tensor
    teacher_gap_rank: torch.Tensor
    rescue_fraction: torch.Tensor
    expected_rescue: torch.Tensor
    rescue_residual: torch.Tensor
    raw_weight: torch.Tensor
    loss_mass_scale: torch.Tensor
    weight: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class GroupedKLMassWeights:
    ranking_signal: torch.Tensor
    high_group_mask: torch.Tensor
    raw_weight: torch.Tensor
    loss_mass_scale: torch.Tensor
    weight: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class MaxKLFractionInverseJSDWeights:
    teacher_gap: torch.Tensor
    sensitivity: torch.Tensor
    threshold: torch.Tensor
    high_group_mask: torch.Tensor
    raw_weight: torch.Tensor
    weight: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class GroupBalancedMaxKLFractionWeights:
    teacher_gap: torch.Tensor
    sensitivity: torch.Tensor
    threshold: torch.Tensor
    high_group_mask: torch.Tensor
    within_high_weight: torch.Tensor
    balanced_unweighted_weight: torch.Tensor
    weight: torch.Tensor
    high_group_coefficient: float
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class BudgetConsistentRankWeights:
    teacher_gap_b: torch.Tensor
    teacher_gap_b_plus: torch.Tensor
    persistent_gap: torch.Tensor
    persistent_rank: torch.Tensor
    raw_weight: torch.Tensor
    loss_mass_scale: torch.Tensor
    weight: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class BudgetResidualHardnessWeights:
    teacher_gap_b: torch.Tensor
    teacher_gap_b_plus: torch.Tensor
    persistent_gap: torch.Tensor
    teacher_gap_rank: torch.Tensor
    persistent_rank: torch.Tensor
    priority: torch.Tensor
    raw_weight: torch.Tensor
    loss_mass_scale: torch.Tensor
    weight: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class CounterfactualBudgetBridge:
    teacher_gap_b: torch.Tensor
    teacher_gap_b_plus: torch.Tensor
    bridge_gap: torch.Tensor
    tau_teacher_gap: torch.Tensor
    rescue_fraction: torch.Tensor
    confidence: torch.Tensor
    bridge_fraction: torch.Tensor
    loss_mass_scale: torch.Tensor
    full_teacher_weight: torch.Tensor
    bridge_teacher_weight: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class BudgetContrastiveGate:
    teacher_gap_b: torch.Tensor
    teacher_gap_b_plus: torch.Tensor
    tau_teacher_gap: torch.Tensor
    rescue_fraction: torch.Tensor
    confidence: torch.Tensor
    shaping_strength: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class BudgetGradientAlignedBridgeGate:
    teacher_gap_b: torch.Tensor
    teacher_gap_b_plus: torch.Tensor
    gradient_alignment: torch.Tensor
    positive_alignment: torch.Tensor
    tau_teacher_gap: torch.Tensor
    rescue_fraction: torch.Tensor
    confidence: torch.Tensor
    bridge_fraction: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class CounterfactualGradientResidualGate:
    teacher_gap_b: torch.Tensor
    teacher_gap_b_plus: torch.Tensor
    gradient_alignment: torch.Tensor
    raw_projection_coefficient: torch.Tensor
    clipped_projection_coefficient: torch.Tensor
    budget_rescue_indicator: torch.Tensor
    cancellation_coefficient: torch.Tensor
    residual_gradient_norm_ratio: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class BudgetTangentResidualWeights:
    teacher_gap_b: torch.Tensor
    gradient_alignment: torch.Tensor
    budget_explained_fraction: torch.Tensor
    budget_residual_fraction: torch.Tensor
    teacher_gap_rank: torch.Tensor
    priority: torch.Tensor
    raw_weight: torch.Tensor
    loss_mass_scale: torch.Tensor
    weight: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class BudgetGradientConsensusWeights:
    teacher_gap_b: torch.Tensor
    teacher_gap_rank: torch.Tensor
    gradient_consensus: torch.Tensor
    gradient_norm_consistency: torch.Tensor
    invariant_priority: torch.Tensor
    raw_weight: torch.Tensor
    loss_mass_scale: torch.Tensor
    weight: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class BudgetCounterfactualTeachabilityWeights:
    teacher_gap_b: torch.Tensor
    gradient_alignment: torch.Tensor
    budget_explained_fraction: torch.Tensor
    budget_residual_fraction: torch.Tensor
    teacher_support_coverage: torch.Tensor
    teacher_gap_rank: torch.Tensor
    support_coverage_rank: torch.Tensor
    priority: torch.Tensor
    raw_weight: torch.Tensor
    loss_mass_scale: torch.Tensor
    weight: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class LossMassNormalization:
    reference_loss: torch.Tensor
    candidate_loss: torch.Tensor
    raw_weight: torch.Tensor
    loss_mass_scale: torch.Tensor
    weight: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class ProjectionFractionTokenPartition:
    teacher_js: torch.Tensor
    budget_js: torch.Tensor
    teacher_plus_js: torch.Tensor
    projection_mass: torch.Tensor
    projection_fraction: torch.Tensor
    eligible_mask: torch.Tensor
    top_mask: torch.Tensor
    selected_mask: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class ProjectionFractionBottomDropPartition:
    teacher_js: torch.Tensor
    budget_js: torch.Tensor
    teacher_plus_js: torch.Tensor
    projection_mass: torch.Tensor
    projection_fraction: torch.Tensor
    dropped_mask: torch.Tensor
    kept_mask: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class RandomTokenDropPartition:
    eligible_mask: torch.Tensor
    dropped_mask: torch.Tensor
    selected_mask: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class RandomTokenKeepPartition:
    eligible_mask: torch.Tensor
    selected_mask: torch.Tensor
    excluded_mask: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class ForwardKLTopFractionPartition:
    ranking_kl: torch.Tensor
    eligible_mask: torch.Tensor
    selected_mask: torch.Tensor
    excluded_mask: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class KLAbsProjectionSoftORPartition:
    teacher_js: torch.Tensor
    budget_js: torch.Tensor
    teacher_plus_js: torch.Tensor
    projection_mass: torch.Tensor
    absolute_projection_mass: torch.Tensor
    clipped_absolute_projection_mass: torch.Tensor
    normalized_kl: torch.Tensor
    normalized_absolute_projection: torch.Tensor
    soft_or_score: torch.Tensor
    selected_mask: torch.Tensor
    valid_mask: torch.Tensor
    projection_clip_value: torch.Tensor
    kl_degenerate: bool
    projection_degenerate: bool
    uniform_fallback: bool


@dataclass(frozen=True)
class ProjectionMassGroupedWeights:
    teacher_js: torch.Tensor
    budget_js: torch.Tensor
    teacher_plus_js: torch.Tensor
    projection_mass: torch.Tensor
    positive_projection_mass: torch.Tensor
    high_mask: torch.Tensor
    low_mask: torch.Tensor
    raw_weight: torch.Tensor
    loss_mass_scale: torch.Tensor
    weight: torch.Tensor
    valid_mask: torch.Tensor
    high_group_lambda: float
    degenerate: bool


@dataclass(frozen=True)
class ProjectionFractionGroupedWeights:
    teacher_js: torch.Tensor
    budget_js: torch.Tensor
    teacher_plus_js: torch.Tensor
    projection_mass: torch.Tensor
    projection_fraction: torch.Tensor
    eligible_mask: torch.Tensor
    selected_mask: torch.Tensor
    complement_mask: torch.Tensor
    raw_weight: torch.Tensor
    loss_mass_scale: torch.Tensor
    weight: torch.Tensor
    valid_mask: torch.Tensor
    selection: str
    selected_group_lambda: float
    degenerate: bool


@dataclass(frozen=True)
class BudgetJSDConditionedAbsProjectionGroupedWeights:
    teacher_js: torch.Tensor
    budget_js: torch.Tensor
    teacher_plus_js: torch.Tensor
    projection_mass: torch.Tensor
    projection_fraction: torch.Tensor
    abs_projection_fraction: torch.Tensor
    budget_sensitive_mask: torch.Tensor
    selected_mask: torch.Tensor
    complement_mask: torch.Tensor
    raw_weight: torch.Tensor
    loss_mass_scale: torch.Tensor
    weight: torch.Tensor
    valid_mask: torch.Tensor
    selection: str
    selected_group_lambda: float
    degenerate: bool


def counterfactual_cancellation_strength(
    base_strength: float,
    *,
    schedule: str = "constant",
    progress_step: int | None = None,
    total_steps: int | None = None,
    decay_fraction: float = 0.5,
) -> float:
    """Resolve a deterministic early counterfactual-gradient curriculum."""

    strength = float(base_strength)
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"base_strength must be in [0, 1], got {base_strength}.")
    normalized_schedule = str(schedule).strip().lower()
    if normalized_schedule == "constant":
        return strength
    if normalized_schedule != "linear_to_zero":
        raise ValueError(f"Unsupported cancellation schedule: {schedule!r}.")
    if progress_step is None or total_steps is None:
        raise ValueError("linear_to_zero requires progress_step and total_steps.")
    if int(total_steps) <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}.")
    if int(progress_step) < 0:
        raise ValueError(f"progress_step must be nonnegative, got {progress_step}.")
    fraction = float(decay_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"decay_fraction must be in (0, 1], got {decay_fraction}.")
    cutoff = max(1.0, fraction * float(total_steps))
    remaining = 1.0 - min(float(progress_step) / cutoff, 1.0)
    return strength * remaining


def generated_token_valid_mask(generated_ids: torch.Tensor) -> torch.Tensor:
    """Mark emitted positions as valid without treating Qwen's EOS/pad alias as padding."""

    token_ids = generated_ids.detach().reshape(-1)
    if token_ids.numel() <= 0:
        raise ValueError("At least one generated token is required.")
    return torch.ones_like(token_ids, dtype=torch.bool)


def hellinger_curriculum_weights(
    affinity: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    scale: float = 8.0,
) -> HellingerCurriculumWeights:
    """Weight intermediate teacher/student mismatch without normalization."""

    with torch.no_grad():
        coefficient = affinity.detach().float().reshape(-1)
        valid = valid_mask.detach().to(device=coefficient.device, dtype=torch.bool).reshape(-1)
        if coefficient.shape != valid.shape:
            raise ValueError("Affinity and valid mask must align.")
        if not valid.any():
            raise ValueError("Hellinger curriculum requires a valid generated token.")
        if not torch.isfinite(coefficient[valid]).all():
            raise FloatingPointError("Affinity must be finite on valid positions.")
        tolerance = 1e-5
        if (coefficient[valid] < -tolerance).any() or (coefficient[valid] > 1.0 + tolerance).any():
            raise ValueError("Bhattacharyya affinity must lie in [0, 1].")
        curriculum_scale = float(scale)
        if not math.isfinite(curriculum_scale) or curriculum_scale <= 0.0:
            raise ValueError(f"Curriculum scale must be finite and positive, got {scale}.")
        coefficient = coefficient.clamp(0.0, 1.0)
        difficulty = 1.0 - coefficient
        raw_weight = curriculum_scale * difficulty * (1.0 - difficulty)
        weight = torch.zeros_like(raw_weight)
        weight[valid] = raw_weight[valid]
        return HellingerCurriculumWeights(
            affinity=coefficient,
            difficulty=difficulty,
            raw_weight=raw_weight,
            weight=weight.detach(),
            valid_mask=valid,
        )


def projection_fraction_token_partition(
    teacher_js: torch.Tensor,
    budget_js: torch.Tensor,
    teacher_plus_js: torch.Tensor,
    ranking_kl: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    top_fraction: float = 0.2,
    min_kl: float = 1e-5,
    select: str = "top",
    eps: float = 1e-8,
) -> ProjectionFractionTokenPartition:
    """Rank above-floor tokens by F and select the top set or its valid complement."""

    with torch.no_grad():
        tensors = [
            value.detach().float().reshape(-1)
            for value in (teacher_js, budget_js, teacher_plus_js, ranking_kl)
        ]
        teacher, budget, teacher_plus, kl = tensors
        valid = valid_mask.detach().to(device=teacher.device, dtype=torch.bool).reshape(-1)
        if not all(value.shape == teacher.shape for value in (*tensors[1:], valid)):
            raise ValueError("Token projection inputs and valid mask must align.")
        if not valid.any():
            raise ValueError("Token projection partition requires a valid generated token.")
        if not all(torch.isfinite(value[valid]).all() for value in tensors):
            raise FloatingPointError("Token projection inputs must be finite on valid positions.")
        if (
            (teacher[valid] < 0).any()
            or (budget[valid] < 0).any()
            or (teacher_plus[valid] < 0).any()
        ):
            raise ValueError("JSD inputs must be nonnegative.")
        fraction = float(top_fraction)
        threshold = float(min_kl)
        eps = float(eps)
        if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
            raise ValueError(f"top_fraction must be in (0, 1), got {top_fraction}.")
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError(f"min_kl must be finite and nonnegative, got {min_kl}.")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError(f"eps must be finite and positive, got {eps}.")
        selection = str(select).strip().lower()
        if selection not in {"top", "bottom"}:
            raise ValueError(f"select must be 'top' or 'bottom', got {select!r}.")

        projection = 0.5 * (teacher + budget - teacher_plus)
        projection_fraction = projection / (teacher + eps)
        eligible = valid & (kl >= threshold)
        eligible_indices = torch.nonzero(eligible, as_tuple=False).reshape(-1)
        eligible_count = int(eligible_indices.numel())
        top_count = (
            min(eligible_count, max(1, int(math.ceil(fraction * eligible_count))))
            if eligible_count
            else 0
        )
        top = torch.zeros_like(valid)
        if top_count:
            ranked_local = torch.argsort(
                projection_fraction[eligible_indices],
                descending=True,
                stable=True,
            )
            top[eligible_indices[ranked_local[:top_count]]] = True
        # The KL floor controls only which tokens may enter the top-F set. The
        # bottom partition is the complement over every valid response token,
        # including tokens below the ranking floor.
        selected = top if selection == "top" else valid & ~top

    return ProjectionFractionTokenPartition(
        teacher_js=teacher.detach(),
        budget_js=budget.detach(),
        teacher_plus_js=teacher_plus.detach(),
        projection_mass=projection.detach(),
        projection_fraction=projection_fraction.detach(),
        eligible_mask=eligible.detach(),
        top_mask=top.detach(),
        selected_mask=selected.detach(),
        valid_mask=valid.detach(),
    )


def _projection_fraction_drop_partition(
    teacher_js: torch.Tensor,
    budget_js: torch.Tensor,
    teacher_plus_js: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    drop: str,
    drop_fraction: float = 0.10,
    eps: float = 1e-8,
) -> ProjectionFractionBottomDropPartition:
    """Drop one F tail over all valid tokens without a KL eligibility floor."""

    with torch.no_grad():
        selection = str(drop).strip().lower()
        if selection not in {"bottom", "top"}:
            raise ValueError(f"drop must be 'bottom' or 'top', got {drop!r}.")
        teacher, budget, teacher_plus = [
            value.detach().float().reshape(-1)
            for value in (teacher_js, budget_js, teacher_plus_js)
        ]
        valid = valid_mask.detach().to(device=teacher.device, dtype=torch.bool).reshape(-1)
        if not all(value.shape == teacher.shape for value in (budget, teacher_plus, valid)):
            raise ValueError("F-drop inputs and valid mask must align.")
        if not valid.any():
            raise ValueError("F dropping requires a valid generated token.")
        if not all(torch.isfinite(value[valid]).all() for value in (teacher, budget, teacher_plus)):
            raise FloatingPointError("F-drop JSD inputs must be finite on valid positions.")
        if (
            (teacher[valid] < 0).any()
            or (budget[valid] < 0).any()
            or (teacher_plus[valid] < 0).any()
        ):
            raise ValueError("F-drop JSD inputs must be nonnegative.")
        fraction = float(drop_fraction)
        eps = float(eps)
        if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
            raise ValueError(f"drop_fraction must be in (0, 1), got {drop_fraction}.")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError(f"eps must be finite and positive, got {eps}.")

        projection = 0.5 * (teacher + budget - teacher_plus)
        projection_fraction = projection / (teacher + eps)
        valid_indices = torch.nonzero(valid, as_tuple=False).reshape(-1)
        valid_count = int(valid_indices.numel())
        drop_count = min(
            max(0, valid_count - 1),
            max(1, int(math.ceil(fraction * valid_count))),
        )
        dropped = torch.zeros_like(valid)
        if drop_count:
            ranked_local = torch.argsort(
                projection_fraction[valid_indices],
                descending=selection == "top",
                stable=True,
            )
            dropped[valid_indices[ranked_local[:drop_count]]] = True
        kept = valid & ~dropped

    return ProjectionFractionBottomDropPartition(
        teacher_js=teacher.detach(),
        budget_js=budget.detach(),
        teacher_plus_js=teacher_plus.detach(),
        projection_mass=projection.detach(),
        projection_fraction=projection_fraction.detach(),
        dropped_mask=dropped.detach(),
        kept_mask=kept.detach(),
        valid_mask=valid.detach(),
    )


def projection_fraction_bottom_drop_partition(
    teacher_js: torch.Tensor,
    budget_js: torch.Tensor,
    teacher_plus_js: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    drop_fraction: float = 0.10,
    eps: float = 1e-8,
) -> ProjectionFractionBottomDropPartition:
    """Drop the lowest-F response tokens without applying a KL eligibility floor."""

    return _projection_fraction_drop_partition(
        teacher_js,
        budget_js,
        teacher_plus_js,
        valid_mask,
        drop="bottom",
        drop_fraction=drop_fraction,
        eps=eps,
    )


def projection_fraction_top_drop_partition(
    teacher_js: torch.Tensor,
    budget_js: torch.Tensor,
    teacher_plus_js: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    drop_fraction: float = 0.10,
    eps: float = 1e-8,
) -> ProjectionFractionBottomDropPartition:
    """Drop the highest-F response tokens without applying a KL eligibility floor."""

    return _projection_fraction_drop_partition(
        teacher_js,
        budget_js,
        teacher_plus_js,
        valid_mask,
        drop="top",
        drop_fraction=drop_fraction,
        eps=eps,
    )


def deterministic_random_token_drop_partition(
    ranking_kl: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    sample_key: str,
    seed: int = 42,
    drop_fraction: float = 0.2,
    min_kl: float = 1e-5,
) -> RandomTokenDropPartition:
    """Drop an exact deterministic random fraction of eligible response tokens."""

    with torch.no_grad():
        kl = ranking_kl.detach().float().reshape(-1)
        valid = valid_mask.detach().to(device=kl.device, dtype=torch.bool).reshape(-1)
        if kl.shape != valid.shape:
            raise ValueError("Token KL and valid mask must align.")
        if not valid.any():
            raise ValueError("Random token dropping requires a valid generated token.")
        if not torch.isfinite(kl[valid]).all():
            raise FloatingPointError("Token KL must be finite on valid positions.")
        fraction = float(drop_fraction)
        threshold = float(min_kl)
        if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
            raise ValueError(f"drop_fraction must be in (0, 1), got {drop_fraction}.")
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError(f"min_kl must be finite and nonnegative, got {min_kl}.")
        key = str(sample_key)
        if not key:
            raise ValueError("sample_key must be nonempty.")

        # A zero threshold means no eligibility floor. Do not let tiny
        # negative roundoff in an otherwise nonnegative KL exclude tokens.
        eligible = valid if threshold == 0.0 else valid & (kl >= threshold)
        eligible_indices = torch.nonzero(eligible, as_tuple=False).reshape(-1)
        eligible_count = int(eligible_indices.numel())
        drop_count = (
            min(eligible_count, max(1, int(math.ceil(fraction * eligible_count))))
            if eligible_count
            else 0
        )
        dropped = torch.zeros_like(valid)
        if drop_count:
            indexed_scores = []
            for index in eligible_indices.detach().cpu().tolist():
                digest = hashlib.sha256(
                    f"{int(seed)}:{key}:{int(index)}".encode("utf-8")
                ).digest()
                indexed_scores.append((int.from_bytes(digest[:8], "big"), int(index)))
            indexed_scores.sort()
            dropped_indices = torch.tensor(
                [index for _, index in indexed_scores[:drop_count]],
                dtype=torch.long,
                device=kl.device,
            )
            dropped[dropped_indices] = True
        selected = eligible & ~dropped

    return RandomTokenDropPartition(
        eligible_mask=eligible.detach(),
        dropped_mask=dropped.detach(),
        selected_mask=selected.detach(),
        valid_mask=valid.detach(),
    )


def deterministic_random_token_keep_partition(
    ranking_kl: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    sample_key: str,
    seed: int = 42,
    keep_fraction: float = 0.2,
    min_kl: float = 0.0,
) -> RandomTokenKeepPartition:
    """Select an exact deterministic random fraction of eligible response tokens."""

    partition = deterministic_random_token_drop_partition(
        ranking_kl,
        valid_mask,
        sample_key=sample_key,
        seed=seed,
        drop_fraction=keep_fraction,
        min_kl=min_kl,
    )
    return RandomTokenKeepPartition(
        eligible_mask=partition.eligible_mask,
        selected_mask=partition.dropped_mask,
        excluded_mask=partition.selected_mask,
        valid_mask=partition.valid_mask,
    )


def forward_kl_top_fraction_partition(
    ranking_kl: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    top_fraction: float = 0.2,
    min_kl: float = 0.0,
) -> ForwardKLTopFractionPartition:
    """Select the largest detached forward-KL values within one response."""

    with torch.no_grad():
        kl = ranking_kl.detach().float().reshape(-1)
        valid = valid_mask.detach().to(device=kl.device, dtype=torch.bool).reshape(-1)
        if kl.shape != valid.shape:
            raise ValueError("Token KL and valid mask must align.")
        if not valid.any():
            raise ValueError("Forward-KL top-fraction selection requires a valid token.")
        if not torch.isfinite(kl[valid]).all():
            raise FloatingPointError("Token KL must be finite on valid positions.")
        fraction = float(top_fraction)
        threshold = float(min_kl)
        if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError(f"top_fraction must be in (0, 1], got {top_fraction}.")
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError(f"min_kl must be finite and nonnegative, got {min_kl}.")

        eligible = valid if threshold == 0.0 else valid & (kl >= threshold)
        eligible_indices = torch.nonzero(eligible, as_tuple=False).reshape(-1)
        eligible_count = int(eligible_indices.numel())
        selected_count = (
            min(eligible_count, max(1, int(math.ceil(fraction * eligible_count))))
            if eligible_count
            else 0
        )
        selected = torch.zeros_like(valid)
        if selected_count:
            ranked_local = torch.argsort(
                kl[eligible_indices],
                descending=True,
                stable=True,
            )
            selected[eligible_indices[ranked_local[:selected_count]]] = True
        excluded = valid & ~selected

    return ForwardKLTopFractionPartition(
        ranking_kl=kl.detach(),
        eligible_mask=eligible.detach(),
        selected_mask=selected.detach(),
        excluded_mask=excluded.detach(),
        valid_mask=valid.detach(),
    )


def kl_abs_projection_soft_or_topk_partition(
    teacher_js: torch.Tensor,
    budget_js: torch.Tensor,
    teacher_plus_js: torch.Tensor,
    ranking_kl: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    top_fraction: float = 0.20,
    projection_clip_quantile: float = 0.98,
    eps: float = 1e-8,
) -> KLAbsProjectionSoftORPartition:
    """Select response tokens using TIP-style SoftOR over KL and ``abs(P)``.

    Both signals are normalized independently within one response.  Only the
    absolute projection signal is winsorized; the differentiable OPSD KL is
    detached for ranking and remains unchanged in the selected-token loss.
    """

    with torch.no_grad():
        teacher, budget, teacher_plus, kl = [
            value.detach().float().reshape(-1)
            for value in (teacher_js, budget_js, teacher_plus_js, ranking_kl)
        ]
        valid = valid_mask.detach().to(device=teacher.device, dtype=torch.bool).reshape(-1)
        if not all(value.shape == teacher.shape for value in (budget, teacher_plus, kl, valid)):
            raise ValueError("KL/absolute-P SoftOR inputs and valid mask must align.")
        if not valid.any():
            raise ValueError("KL/absolute-P SoftOR requires a valid generated token.")
        if not all(
            torch.isfinite(value[valid]).all()
            for value in (teacher, budget, teacher_plus, kl)
        ):
            raise FloatingPointError("KL/absolute-P SoftOR inputs must be finite.")
        fraction = float(top_fraction)
        clip_quantile = float(projection_clip_quantile)
        eps = float(eps)
        if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
            raise ValueError(f"top_fraction must be in (0, 1), got {top_fraction}.")
        if not math.isfinite(clip_quantile) or not 0.0 < clip_quantile <= 1.0:
            raise ValueError(
                "projection_clip_quantile must be in (0, 1], got "
                f"{projection_clip_quantile}."
            )
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError(f"eps must be finite and positive, got {eps}.")

        # A full-vocabulary FP32 reduction can put a mathematically zero KL a
        # few ulps below zero.  Accept only roundoff-scale negatives; larger
        # violations still indicate a broken divergence input.
        negative_tolerance = max(1e-6, 100.0 * eps)
        divergence_minima = {
            "teacher_js": float(teacher[valid].min().cpu()),
            "budget_js": float(budget[valid].min().cpu()),
            "teacher_plus_js": float(teacher_plus[valid].min().cpu()),
            "ranking_kl": float(kl[valid].min().cpu()),
        }
        if any(value < -negative_tolerance for value in divergence_minima.values()):
            raise ValueError(
                "KL/absolute-P SoftOR divergence below numerical tolerance: "
                f"{divergence_minima}."
            )

        teacher = teacher.clamp_min(0.0)
        budget = budget.clamp_min(0.0)
        teacher_plus = teacher_plus.clamp_min(0.0)
        # Ranking is detached, so this does not alter the differentiable OPSD loss.
        kl = kl.clamp_min(0.0)
        projection = 0.5 * (teacher + budget - teacher_plus)
        absolute_projection = projection.abs()
        valid_abs_projection = absolute_projection[valid]
        clip_value = torch.quantile(valid_abs_projection, clip_quantile)
        clipped_absolute_projection = absolute_projection.clone()
        clipped_absolute_projection[valid] = valid_abs_projection.clamp_max(clip_value)

        def normalize(values: torch.Tensor) -> tuple[torch.Tensor, bool]:
            valid_values = values[valid]
            minimum = valid_values.min()
            maximum = valid_values.max()
            span = maximum - minimum
            degenerate = bool(span <= eps)
            normalized = torch.zeros_like(values)
            if not degenerate:
                normalized[valid] = ((valid_values - minimum) / span).clamp(0.0, 1.0)
            return normalized, degenerate

        normalized_kl, kl_degenerate = normalize(kl)
        normalized_projection, projection_degenerate = normalize(
            clipped_absolute_projection
        )
        score = torch.zeros_like(kl)
        score[valid] = (
            normalized_kl[valid]
            + normalized_projection[valid]
            - normalized_kl[valid] * normalized_projection[valid]
        )

        uniform_fallback = kl_degenerate and projection_degenerate
        selected = torch.zeros_like(valid)
        if uniform_fallback:
            # Preserve vanilla OPSD instead of selecting arbitrary token positions.
            selected[valid] = True
        else:
            valid_indices = torch.nonzero(valid, as_tuple=False).reshape(-1)
            selected_count = max(1, int(math.floor(fraction * int(valid_indices.numel()))))
            ranked_local = torch.argsort(
                score[valid_indices],
                descending=True,
                stable=True,
            )
            selected[valid_indices[ranked_local[:selected_count]]] = True

    return KLAbsProjectionSoftORPartition(
        teacher_js=teacher.detach(),
        budget_js=budget.detach(),
        teacher_plus_js=teacher_plus.detach(),
        projection_mass=projection.detach(),
        absolute_projection_mass=absolute_projection.detach(),
        clipped_absolute_projection_mass=clipped_absolute_projection.detach(),
        normalized_kl=normalized_kl.detach(),
        normalized_absolute_projection=normalized_projection.detach(),
        soft_or_score=score.detach(),
        selected_mask=selected.detach(),
        valid_mask=valid.detach(),
        projection_clip_value=clip_value.detach(),
        kl_degenerate=kl_degenerate,
        projection_degenerate=projection_degenerate,
        uniform_fallback=uniform_fallback,
    )


def projection_mass_grouped_weights(
    teacher_js: torch.Tensor,
    budget_js: torch.Tensor,
    teacher_plus_js: torch.Tensor,
    per_token_loss: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    top_fraction: float = 0.10,
    high_group_lambda: float = 0.30,
    preserve_loss_mass: bool = False,
    eps: float = 1e-8,
) -> ProjectionMassGroupedWeights:
    """Build VA-OPD-style grouped weights from positive projection mass.

    Tokens are ranked by ``relu((A + B - C) / 2)`` within one rollout. The
    raw weights exactly reconstruct ``lambda * mean(high) + (1-lambda) *
    mean(low)`` when consumed by a mean over valid tokens. The primary objective
    uses these raw weights directly. ``preserve_loss_mass`` is retained only as
    an explicit ablation; it is not the VA-OPD-style grouped objective.
    """

    fraction = float(top_fraction)
    coefficient = float(high_group_lambda)
    eps = float(eps)
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError(f"top_fraction must be in (0, 1), got {top_fraction}.")
    if not math.isfinite(coefficient) or not 0.0 < coefficient < 1.0:
        raise ValueError(
            f"high_group_lambda must be in (0, 1), got {high_group_lambda}."
        )
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"eps must be finite and positive, got {eps}.")

    with torch.no_grad():
        teacher, budget, teacher_plus, loss = [
            value.detach().float().reshape(-1)
            for value in (teacher_js, budget_js, teacher_plus_js, per_token_loss)
        ]
        valid = valid_mask.detach().to(device=teacher.device, dtype=torch.bool).reshape(-1)
        if not all(value.shape == teacher.shape for value in (budget, teacher_plus, loss, valid)):
            raise ValueError("Projection-mass grouped inputs and valid mask must align.")
        if not valid.any():
            raise ValueError("Projection-mass grouping requires a valid generated token.")
        for name, values in (
            ("teacher_js", teacher),
            ("budget_js", budget),
            ("teacher_plus_js", teacher_plus),
            ("per_token_loss", loss),
        ):
            if not torch.isfinite(values[valid]).all():
                raise ValueError(f"{name} must be finite on valid positions.")
        if (
            (teacher[valid] < -1e-6).any()
            or (budget[valid] < -1e-6).any()
            or (teacher_plus[valid] < -1e-6).any()
            or (loss[valid] < -1e-6).any()
        ):
            raise ValueError("JSD and OPSD loss inputs must be nonnegative.")

        projection = 0.5 * (teacher + budget - teacher_plus)
        positive_projection = projection.clamp_min(0.0)
        valid_indices = torch.nonzero(valid, as_tuple=False).reshape(-1)
        valid_count = int(valid_indices.numel())
        high_count = min(
            valid_count - 1,
            max(1, int(math.ceil(fraction * valid_count))),
        ) if valid_count > 1 else valid_count
        ranked = torch.argsort(
            positive_projection[valid_indices], descending=True, stable=True
        )
        high = torch.zeros_like(valid)
        high[valid_indices[ranked[:high_count]]] = True
        low = valid & ~high
        degenerate = bool(
            valid_count < 2 or float(positive_projection[valid].max()) <= eps
        )

        raw_weight = torch.zeros_like(loss)
        if degenerate:
            raw_weight[valid] = 1.0
        else:
            low_count = valid_count - high_count
            raw_weight[high] = coefficient * valid_count / high_count
            raw_weight[low] = (1.0 - coefficient) * valid_count / low_count
            if not torch.allclose(
                raw_weight[valid].mean(),
                torch.ones((), device=loss.device, dtype=torch.float32),
                rtol=1e-6,
                atol=1e-7,
            ):
                raise FloatingPointError("Grouped raw token weights must have mean one.")

        reference_mass = loss[valid].sum()
        grouped_mass = (raw_weight[valid] * loss[valid]).sum()
        if preserve_loss_mass and float(reference_mass) > eps:
            if not torch.isfinite(grouped_mass) or float(grouped_mass) <= 0.0:
                raise FloatingPointError(f"Invalid grouped OPSD loss mass: {float(grouped_mass)}")
            loss_mass_scale = reference_mass / grouped_mass
        else:
            loss_mass_scale = torch.ones((), device=loss.device, dtype=torch.float32)
        weight = torch.zeros_like(loss)
        weight[valid] = raw_weight[valid] * loss_mass_scale

        if preserve_loss_mass:
            normalized_mass = (weight[valid] * loss[valid]).sum()
            if not torch.allclose(
                normalized_mass, reference_mass, rtol=2e-6, atol=2e-7
            ):
                raise FloatingPointError(
                    "Projection-mass grouped loss normalization failed: "
                    f"weighted={float(normalized_mass)}, reference={float(reference_mass)}."
                )

    return ProjectionMassGroupedWeights(
        teacher_js=teacher.detach(),
        budget_js=budget.detach(),
        teacher_plus_js=teacher_plus.detach(),
        projection_mass=projection.detach(),
        positive_projection_mass=positive_projection.detach(),
        high_mask=high.detach(),
        low_mask=low.detach(),
        raw_weight=raw_weight.detach(),
        loss_mass_scale=loss_mass_scale.detach(),
        weight=weight.detach(),
        valid_mask=valid.detach(),
        high_group_lambda=coefficient,
        degenerate=degenerate,
    )


def projection_fraction_grouped_weights(
    teacher_js: torch.Tensor,
    budget_js: torch.Tensor,
    teacher_plus_js: torch.Tensor,
    per_token_loss: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    group_fraction: float,
    selected_group_lambda: float,
    selection: str = "top",
    min_kl: float = 1e-5,
    preserve_loss_mass: bool = False,
    eps: float = 1e-8,
) -> ProjectionFractionGroupedWeights:
    """Mix selected-F and complement token losses with an explicit coefficient.

    The selected group is the requested top or bottom fraction of eligible
    tokens ranked by ``F = ((A + B - C) / 2) / A``. The KL floor only controls
    ranking eligibility; every valid generated token belongs to exactly one of
    the two loss groups. Raw weights reconstruct
    ``lambda * mean(selected) + (1-lambda) * mean(complement)`` when consumed
    by a mean over valid tokens.
    """

    fraction = float(group_fraction)
    coefficient = float(selected_group_lambda)
    threshold = float(min_kl)
    eps = float(eps)
    selection = str(selection).strip().lower()
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError(f"group_fraction must be in (0, 1), got {group_fraction}.")
    if not math.isfinite(coefficient) or not 0.0 < coefficient < 1.0:
        raise ValueError(
            "selected_group_lambda must be in (0, 1), got "
            f"{selected_group_lambda}."
        )
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError(f"min_kl must be finite and nonnegative, got {min_kl}.")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"eps must be finite and positive, got {eps}.")
    if selection not in {"top", "bottom"}:
        raise ValueError(f"selection must be 'top' or 'bottom', got {selection!r}.")

    with torch.no_grad():
        teacher, budget, teacher_plus, loss = [
            value.detach().float().reshape(-1)
            for value in (teacher_js, budget_js, teacher_plus_js, per_token_loss)
        ]
        valid = valid_mask.detach().to(device=teacher.device, dtype=torch.bool).reshape(-1)
        if not all(value.shape == teacher.shape for value in (budget, teacher_plus, loss, valid)):
            raise ValueError("Projection-fraction grouped inputs and valid mask must align.")
        if not valid.any():
            raise ValueError("Projection-fraction grouping requires a valid generated token.")
        for name, values in (
            ("teacher_js", teacher),
            ("budget_js", budget),
            ("teacher_plus_js", teacher_plus),
            ("per_token_loss", loss),
        ):
            if not torch.isfinite(values[valid]).all():
                raise ValueError(f"{name} must be finite on valid positions.")
        if (
            (teacher[valid] < -1e-6).any()
            or (budget[valid] < -1e-6).any()
            or (teacher_plus[valid] < -1e-6).any()
            or (loss[valid] < -1e-6).any()
        ):
            raise ValueError("JSD and OPSD loss inputs must be nonnegative.")

        projection = 0.5 * (teacher + budget - teacher_plus)
        projection_fraction = projection / teacher.clamp_min(eps)
        # A zero threshold explicitly means rank every valid response token.
        # This avoids excluding tiny negative round-off values from FP32 KL.
        eligible = valid if threshold == 0.0 else valid & (loss >= threshold)
        eligible_indices = torch.nonzero(eligible, as_tuple=False).reshape(-1)
        eligible_count = int(eligible_indices.numel())
        selected_count = (
            min(eligible_count, max(1, int(math.ceil(fraction * eligible_count))))
            if eligible_count
            else 0
        )
        selected = torch.zeros_like(valid)
        if selected_count:
            ranked = torch.argsort(
                projection_fraction[eligible_indices],
                descending=selection == "top",
                stable=True,
            )
            selected[eligible_indices[ranked[:selected_count]]] = True
        complement = valid & ~selected
        valid_count = int(valid.sum())
        complement_count = int(complement.sum())
        degenerate = selected_count == 0 or complement_count == 0

        raw_weight = torch.zeros_like(loss)
        if degenerate:
            raw_weight[valid] = 1.0
        else:
            raw_weight[selected] = coefficient * valid_count / selected_count
            raw_weight[complement] = (
                (1.0 - coefficient) * valid_count / complement_count
            )
            if not torch.allclose(
                raw_weight[valid].mean(),
                torch.ones((), device=loss.device, dtype=torch.float32),
                rtol=1e-6,
                atol=1e-7,
            ):
                raise FloatingPointError("Grouped raw token weights must have mean one.")

        reference_mass = loss[valid].sum()
        grouped_mass = (raw_weight[valid] * loss[valid]).sum()
        if preserve_loss_mass and float(reference_mass) > eps:
            if not torch.isfinite(grouped_mass) or float(grouped_mass) <= 0.0:
                raise FloatingPointError(
                    f"Invalid grouped OPSD loss mass: {float(grouped_mass)}"
                )
            loss_mass_scale = reference_mass / grouped_mass
        else:
            loss_mass_scale = torch.ones((), device=loss.device, dtype=torch.float32)
        weight = torch.zeros_like(loss)
        weight[valid] = raw_weight[valid] * loss_mass_scale

    return ProjectionFractionGroupedWeights(
        teacher_js=teacher.detach(),
        budget_js=budget.detach(),
        teacher_plus_js=teacher_plus.detach(),
        projection_mass=projection.detach(),
        projection_fraction=projection_fraction.detach(),
        eligible_mask=eligible.detach(),
        selected_mask=selected.detach(),
        complement_mask=complement.detach(),
        raw_weight=raw_weight.detach(),
        loss_mass_scale=loss_mass_scale.detach(),
        weight=weight.detach(),
        valid_mask=valid.detach(),
        selection=selection,
        selected_group_lambda=coefficient,
        degenerate=degenerate,
    )


def budget_jsd_conditioned_abs_projection_grouped_weights(
    teacher_js: torch.Tensor,
    budget_js: torch.Tensor,
    teacher_plus_js: torch.Tensor,
    per_token_loss: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    budget_top_fraction: float,
    within_budget_fraction: float,
    selected_group_lambda: float,
    selection: str,
    preserve_loss_mass: bool = False,
    eps: float = 1e-8,
) -> BudgetJSDConditionedAbsProjectionGroupedWeights:
    """Group tokens by absolute F only after a high-B-JSD prefilter.

    For each trajectory, the first stage keeps the top ``budget_top_fraction``
    valid tokens ranked by ``B = JSD(p_b, p_b_plus)``. The second stage selects
    the requested top or bottom ``within_budget_fraction`` of that set ranked by
    ``abs(F)``, where ``F = ((A + B - C) / 2) / A``. The selected group is mixed
    against every remaining valid token using an explicit group-level lambda.
    """

    budget_fraction = float(budget_top_fraction)
    within_fraction = float(within_budget_fraction)
    coefficient = float(selected_group_lambda)
    eps = float(eps)
    selection = str(selection).strip().lower()
    for name, value in (
        ("budget_top_fraction", budget_fraction),
        ("within_budget_fraction", within_fraction),
        ("selected_group_lambda", coefficient),
    ):
        if not math.isfinite(value) or not 0.0 < value < 1.0:
            raise ValueError(f"{name} must be in (0, 1), got {value}.")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"eps must be finite and positive, got {eps}.")
    if selection not in {"top", "bottom"}:
        raise ValueError(f"selection must be 'top' or 'bottom', got {selection!r}.")

    with torch.no_grad():
        teacher, budget, teacher_plus, loss = [
            value.detach().float().reshape(-1)
            for value in (teacher_js, budget_js, teacher_plus_js, per_token_loss)
        ]
        valid = valid_mask.detach().to(device=teacher.device, dtype=torch.bool).reshape(-1)
        if not all(value.shape == teacher.shape for value in (budget, teacher_plus, loss, valid)):
            raise ValueError("B-JSD/absolute-F grouped inputs and valid mask must align.")
        if not valid.any():
            raise ValueError("B-JSD/absolute-F grouping requires a valid generated token.")
        for name, values in (
            ("teacher_js", teacher),
            ("budget_js", budget),
            ("teacher_plus_js", teacher_plus),
            ("per_token_loss", loss),
        ):
            if not torch.isfinite(values[valid]).all():
                raise ValueError(f"{name} must be finite on valid positions.")
        if (
            (teacher[valid] < -1e-6).any()
            or (budget[valid] < -1e-6).any()
            or (teacher_plus[valid] < -1e-6).any()
            or (loss[valid] < -1e-6).any()
        ):
            raise ValueError("JSD and OPSD loss inputs must be nonnegative.")

        projection = 0.5 * (teacher + budget - teacher_plus)
        projection_fraction = projection / teacher.clamp_min(eps)
        absolute_fraction = projection_fraction.abs()

        valid_indices = torch.nonzero(valid, as_tuple=False).reshape(-1)
        valid_count = int(valid_indices.numel())
        budget_count = min(
            valid_count,
            max(1, int(math.ceil(budget_fraction * valid_count))),
        )
        budget_order = torch.argsort(
            budget[valid_indices], descending=True, stable=True
        )
        budget_sensitive = torch.zeros_like(valid)
        budget_sensitive[valid_indices[budget_order[:budget_count]]] = True

        budget_indices = torch.nonzero(budget_sensitive, as_tuple=False).reshape(-1)
        selected_count = min(
            budget_count,
            max(1, int(math.ceil(within_fraction * budget_count))),
        )
        f_order = torch.argsort(
            absolute_fraction[budget_indices],
            descending=selection == "top",
            stable=True,
        )
        selected = torch.zeros_like(valid)
        selected[budget_indices[f_order[:selected_count]]] = True
        complement = valid & ~selected
        complement_count = int(complement.sum())
        degenerate = selected_count == 0 or complement_count == 0

        raw_weight = torch.zeros_like(loss)
        if degenerate:
            raw_weight[valid] = 1.0
        else:
            raw_weight[selected] = coefficient * valid_count / selected_count
            raw_weight[complement] = (
                (1.0 - coefficient) * valid_count / complement_count
            )
            if not torch.allclose(
                raw_weight[valid].mean(),
                torch.ones((), device=loss.device, dtype=torch.float32),
                rtol=1e-6,
                atol=1e-7,
            ):
                raise FloatingPointError("B-JSD/absolute-F raw token weights must have mean one.")

        reference_mass = loss[valid].sum()
        grouped_mass = (raw_weight[valid] * loss[valid]).sum()
        if preserve_loss_mass and float(reference_mass) > eps:
            if not torch.isfinite(grouped_mass) or float(grouped_mass) <= 0.0:
                raise FloatingPointError(
                    f"Invalid B-JSD/absolute-F grouped OPSD loss mass: {float(grouped_mass)}"
                )
            loss_mass_scale = reference_mass / grouped_mass
        else:
            loss_mass_scale = torch.ones((), device=loss.device, dtype=torch.float32)
        weight = torch.zeros_like(loss)
        weight[valid] = raw_weight[valid] * loss_mass_scale

        if preserve_loss_mass:
            normalized_mass = (weight[valid] * loss[valid]).sum()
            if not torch.allclose(normalized_mass, reference_mass, rtol=2e-6, atol=2e-7):
                raise FloatingPointError(
                    "B-JSD/absolute-F grouped loss normalization failed: "
                    f"weighted={float(normalized_mass)}, reference={float(reference_mass)}."
                )

    return BudgetJSDConditionedAbsProjectionGroupedWeights(
        teacher_js=teacher.detach(),
        budget_js=budget.detach(),
        teacher_plus_js=teacher_plus.detach(),
        projection_mass=projection.detach(),
        projection_fraction=projection_fraction.detach(),
        abs_projection_fraction=absolute_fraction.detach(),
        budget_sensitive_mask=budget_sensitive.detach(),
        selected_mask=selected.detach(),
        complement_mask=complement.detach(),
        raw_weight=raw_weight.detach(),
        loss_mass_scale=loss_mass_scale.detach(),
        weight=weight.detach(),
        valid_mask=valid.detach(),
        selection=selection,
        selected_group_lambda=coefficient,
        degenerate=degenerate,
    )


def native_budget_robustness_weights(
    sensitivity: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-8,
) -> NativeBudgetWeights:
    """Convert detached native-budget sensitivity into per-sample mean-one weights."""

    with torch.no_grad():
        values = sensitivity.detach().float().reshape(-1)
        valid = valid_mask.detach().to(device=values.device, dtype=torch.bool).reshape(-1)
        if values.shape != valid.shape:
            raise ValueError(f"Sensitivity and valid mask must align: {values.shape} vs {valid.shape}.")
        if not valid.any():
            raise ValueError("At least one valid generated-token position is required.")
        if not torch.isfinite(values[valid]).all():
            raise ValueError("Sensitivity KL must be finite on all valid positions.")
        if (values[valid] < -1e-6).any():
            raise ValueError("Sensitivity KL must be nonnegative.")
        values = values.clamp_min(0.0)
        tau = torch.quantile(values[valid], 0.5)
        robustness = torch.zeros_like(values)
        robustness[valid] = tau / (tau + values[valid] + float(eps))
        raw_weight = torch.zeros_like(values)
        raw_weight[valid] = 1.0 + robustness[valid]
        mean_raw = raw_weight[valid].mean()
        if not torch.isfinite(mean_raw) or float(mean_raw) <= 0.0:
            raise FloatingPointError(f"Invalid mean raw weight: {float(mean_raw)}")
        weight = torch.zeros_like(values)
        weight[valid] = raw_weight[valid] / mean_raw
    return NativeBudgetWeights(
        sensitivity=values.detach(),
        tau=tau.detach(),
        robustness=robustness.detach(),
        raw_weight=raw_weight.detach(),
        weight=weight.detach(),
        valid_mask=valid.detach(),
    )


def teacher_gap_persistence_weights(
    teacher_gap_b: torch.Tensor,
    teacher_gap_b_plus: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    alpha: float = 0.5,
    eps: float = 1e-8,
) -> TeacherGapPersistenceWeights:
    """Prioritize hard teacher gaps that an extra visual budget does not rescue.

    The final scale preserves the detached, per-sample KL mass exactly. This
    keeps the scalar loss scale matched to vanilla OPSD while redistributing
    gradients across valid generated-token positions.
    """

    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError(f"alpha must be in [0, 1]; got {alpha}.")
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be positive; got {eps}.")

    with torch.no_grad():
        gap_b = teacher_gap_b.detach().float().reshape(-1)
        gap_b_plus = teacher_gap_b_plus.detach().float().reshape(-1)
        valid = valid_mask.detach().to(device=gap_b.device, dtype=torch.bool).reshape(-1)
        if gap_b.shape != gap_b_plus.shape or gap_b.shape != valid.shape:
            raise ValueError(
                "Teacher gaps and valid mask must align: "
                f"{gap_b.shape}, {gap_b_plus.shape}, {valid.shape}."
            )
        if not valid.any():
            raise ValueError("At least one valid generated-token position is required.")
        if not torch.isfinite(gap_b[valid]).all() or not torch.isfinite(gap_b_plus[valid]).all():
            raise ValueError("Teacher KL must be finite on all valid positions.")
        if (gap_b[valid] < -1e-6).any() or (gap_b_plus[valid] < -1e-6).any():
            raise ValueError("Teacher KL must be nonnegative.")

        gap_b = gap_b.clamp_min(0.0)
        gap_b_plus = gap_b_plus.clamp_min(0.0)
        tau = torch.quantile(gap_b[valid], 0.5)

        rescue = torch.zeros_like(gap_b)
        rescue[valid] = ((gap_b[valid] - gap_b_plus[valid]) / (gap_b[valid] + float(eps))).clamp(0.0, 1.0)
        persistence = torch.zeros_like(gap_b)
        persistence[valid] = 1.0 - rescue[valid]
        confidence = torch.zeros_like(gap_b)
        confidence[valid] = gap_b[valid] / (gap_b[valid] + tau + float(eps))
        priority = torch.zeros_like(gap_b)
        priority[valid] = persistence[valid] * confidence[valid]

        raw_weight = torch.zeros_like(gap_b)
        raw_weight[valid] = 1.0 + float(alpha) * priority[valid]
        unweighted_mass = gap_b[valid].sum()
        raw_weighted_mass = (raw_weight[valid] * gap_b[valid]).sum()
        if float(unweighted_mass) <= float(eps):
            loss_mass_scale = torch.ones((), device=gap_b.device, dtype=torch.float32)
            weight = torch.zeros_like(gap_b)
            weight[valid] = 1.0
        else:
            if not torch.isfinite(raw_weighted_mass) or float(raw_weighted_mass) <= 0.0:
                raise FloatingPointError(f"Invalid raw weighted KL mass: {float(raw_weighted_mass)}")
            loss_mass_scale = unweighted_mass / raw_weighted_mass
            weight = torch.zeros_like(gap_b)
            weight[valid] = raw_weight[valid] * loss_mass_scale

        normalized_mass = (weight[valid] * gap_b[valid]).sum()
        if not torch.allclose(normalized_mass, unweighted_mass, rtol=2e-6, atol=2e-7):
            raise FloatingPointError(
                "KL-mass normalization failed: "
                f"weighted={float(normalized_mass)}, unweighted={float(unweighted_mass)}."
            )

    return TeacherGapPersistenceWeights(
        teacher_gap_b=gap_b.detach(),
        teacher_gap_b_plus=gap_b_plus.detach(),
        tau_teacher_gap=tau.detach(),
        rescue_fraction=rescue.detach(),
        persistence=persistence.detach(),
        confidence=confidence.detach(),
        priority=priority.detach(),
        raw_weight=raw_weight.detach(),
        loss_mass_scale=loss_mass_scale.detach(),
        weight=weight.detach(),
        valid_mask=valid.detach(),
    )


def symmetric_teacher_gap_stability_weights(
    teacher_gap_b: torch.Tensor,
    teacher_gap_b_plus: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    alpha: float = 0.25,
    eps: float = 1e-8,
) -> SymmetricTeacherGapStabilityWeights:
    """Downweight tokens whose teacher gap changes under a nearby visual budget.

    The symmetric relative change removes the absolute KL scale. The detached
    KL-mass normalization preserves each sample's vanilla OPSD scalar loss and
    changes only the distribution of gradients across generated tokens.
    """

    if not 0.0 <= float(alpha) < 1.0:
        raise ValueError(f"alpha must be in [0, 1); got {alpha}.")
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be positive; got {eps}.")

    with torch.no_grad():
        gap_b = teacher_gap_b.detach().float().reshape(-1)
        gap_b_plus = teacher_gap_b_plus.detach().float().reshape(-1)
        valid = valid_mask.detach().to(device=gap_b.device, dtype=torch.bool).reshape(-1)
        if gap_b.shape != gap_b_plus.shape or gap_b.shape != valid.shape:
            raise ValueError(
                "Teacher gaps and valid mask must align: "
                f"{gap_b.shape}, {gap_b_plus.shape}, {valid.shape}."
            )
        if not valid.any():
            raise ValueError("At least one valid generated-token position is required.")
        if not torch.isfinite(gap_b[valid]).all() or not torch.isfinite(gap_b_plus[valid]).all():
            raise ValueError("Teacher KL must be finite on all valid positions.")
        if (gap_b[valid] < -1e-6).any() or (gap_b_plus[valid] < -1e-6).any():
            raise ValueError("Teacher KL must be nonnegative.")

        gap_b = gap_b.clamp_min(0.0)
        gap_b_plus = gap_b_plus.clamp_min(0.0)
        signed_change = torch.zeros_like(gap_b)
        signed_change[valid] = (gap_b[valid] - gap_b_plus[valid]) / (
            gap_b[valid] + gap_b_plus[valid] + float(eps)
        )
        normalized_change = torch.zeros_like(gap_b)
        normalized_change[valid] = signed_change[valid].abs().clamp(0.0, 1.0)
        robustness = torch.zeros_like(gap_b)
        robustness[valid] = 1.0 - normalized_change[valid]

        raw_weight = torch.zeros_like(gap_b)
        raw_weight[valid] = 1.0 - float(alpha) * normalized_change[valid]
        unweighted_mass = gap_b[valid].sum()
        raw_weighted_mass = (raw_weight[valid] * gap_b[valid]).sum()
        if float(unweighted_mass) <= float(eps):
            loss_mass_scale = torch.ones((), device=gap_b.device, dtype=torch.float32)
            weight = torch.zeros_like(gap_b)
            weight[valid] = 1.0
        else:
            if not torch.isfinite(raw_weighted_mass) or float(raw_weighted_mass) <= 0.0:
                raise FloatingPointError(f"Invalid raw weighted KL mass: {float(raw_weighted_mass)}")
            loss_mass_scale = unweighted_mass / raw_weighted_mass
            weight = torch.zeros_like(gap_b)
            weight[valid] = raw_weight[valid] * loss_mass_scale

        normalized_mass = (weight[valid] * gap_b[valid]).sum()
        if not torch.allclose(normalized_mass, unweighted_mass, rtol=2e-6, atol=2e-7):
            raise FloatingPointError(
                "KL-mass normalization failed: "
                f"weighted={float(normalized_mass)}, unweighted={float(unweighted_mass)}."
            )

    return SymmetricTeacherGapStabilityWeights(
        teacher_gap_b=gap_b.detach(),
        teacher_gap_b_plus=gap_b_plus.detach(),
        signed_budget_change=signed_change.detach(),
        normalized_budget_change=normalized_change.detach(),
        robustness=robustness.detach(),
        raw_weight=raw_weight.detach(),
        loss_mass_scale=loss_mass_scale.detach(),
        weight=weight.detach(),
        valid_mask=valid.detach(),
    )


def counterfactual_rescue_amplification_weights(
    teacher_gap_b: torch.Tensor,
    teacher_gap_b_plus: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    alpha: float = 0.5,
    eps: float = 1e-8,
) -> CounterfactualRescueAmplificationWeights:
    """Amplify full-teacher OPSD where extra visual budget closes the gap.

    The adjacent native budget is used only as a detached teachability probe.
    It never replaces or mixes the full-token teacher target. A bounded rescue
    fraction mildly amplifies the original forward-KL gradient, while detached
    per-sample KL-mass normalization preserves vanilla OPSD's scalar loss.
    """

    if not 0.0 <= float(alpha) <= 4.0:
        raise ValueError(f"alpha must be in [0, 4]; got {alpha}.")
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be positive; got {eps}.")
    with torch.no_grad():
        gap_b, gap_b_plus, valid = _validate_gap_inputs(
            teacher_gap_b, teacher_gap_b_plus, valid_mask
        )
        rescue = torch.zeros_like(gap_b)
        rescue[valid] = (
            (gap_b[valid] - gap_b_plus[valid])
            / (gap_b[valid] + float(eps))
        ).clamp(0.0, 1.0)
        raw_weight = torch.zeros_like(gap_b)
        raw_weight[valid] = 1.0 + float(alpha) * rescue[valid]
        normalized = normalize_candidate_loss_mass(
            gap_b,
            gap_b,
            raw_weight,
            valid,
            eps=eps,
        )
    return CounterfactualRescueAmplificationWeights(
        teacher_gap_b=gap_b.detach(),
        teacher_gap_b_plus=gap_b_plus.detach(),
        rescue_fraction=rescue.detach(),
        raw_weight=raw_weight.detach(),
        loss_mass_scale=normalized.loss_mass_scale.detach(),
        weight=normalized.weight.detach(),
        valid_mask=valid.detach(),
    )


def counterfactual_teachability_mixture_weights(
    teacher_gap_b: torch.Tensor,
    teacher_gap_b_plus: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    alpha: float = 0.5,
    rescue_mix: float = 0.1,
    eps: float = 1e-8,
) -> CounterfactualTeachabilityMixtureWeights:
    """Prioritize hard corrections with a mild adjacent-budget teachability signal.

    The deployed-budget teacher gap supplies the primary within-response
    difficulty rank. The positive fraction closed by ``b_plus`` is a bounded
    counterfactual probe of whether the same model can absorb that correction
    when given slightly more visual evidence. The full-token teacher remains
    the sole target and detached KL-mass normalization preserves the vanilla
    OPSD scalar loss for every sample.
    """

    if not 0.0 <= float(alpha) <= 4.0:
        raise ValueError(f"alpha must be in [0, 4]; got {alpha}.")
    if not 0.0 <= float(rescue_mix) <= 1.0:
        raise ValueError(f"rescue_mix must be in [0, 1]; got {rescue_mix}.")
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be positive; got {eps}.")

    with torch.no_grad():
        gap_b, gap_b_plus, valid = _validate_gap_inputs(
            teacher_gap_b, teacher_gap_b_plus, valid_mask
        )
        gap_rank = torch.zeros_like(gap_b)
        gap_rank[valid] = _ordinal_percentile_rank(gap_b[valid])
        rescue = torch.zeros_like(gap_b)
        rescue[valid] = (
            (gap_b[valid] - gap_b_plus[valid])
            / (gap_b[valid] + float(eps))
        ).clamp(0.0, 1.0)
        priority = torch.zeros_like(gap_b)
        priority[valid] = (
            (1.0 - float(rescue_mix)) * gap_rank[valid]
            + float(rescue_mix) * rescue[valid]
        )
        raw_weight = torch.zeros_like(gap_b)
        raw_weight[valid] = 1.0 + float(alpha) * priority[valid]
        normalized = normalize_candidate_loss_mass(
            gap_b,
            gap_b,
            raw_weight,
            valid,
            eps=eps,
        )

    return CounterfactualTeachabilityMixtureWeights(
        teacher_gap_b=gap_b.detach(),
        teacher_gap_b_plus=gap_b_plus.detach(),
        teacher_gap_rank=gap_rank.detach(),
        rescue_fraction=rescue.detach(),
        priority=priority.detach(),
        raw_weight=raw_weight.detach(),
        loss_mass_scale=normalized.loss_mass_scale.detach(),
        weight=normalized.weight.detach(),
        valid_mask=valid.detach(),
    )


def counterfactual_teachability_modulation_weights(
    teacher_gap_b: torch.Tensor,
    teacher_gap_b_plus: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    alpha: float = 0.5,
    rescue_modulation: float = 0.1,
    eps: float = 1e-8,
) -> CounterfactualTeachabilityModulationWeights:
    """Modulate hard full-teacher corrections by adjacent-budget teachability.

    The deployed-budget teacher-gap rank remains the base priority. Positive
    gap closure under the native ``b_plus`` intervention can only increase
    that priority by a bounded multiplicative factor, so an easy token cannot
    outrank a hard correction solely because its relative rescue is large.
    The probe and all weights are detached, the full-token teacher remains the
    sole target, and per-sample KL-mass normalization preserves vanilla OPSD's
    scalar loss exactly.
    """

    if not 0.0 <= float(alpha) <= 4.0:
        raise ValueError(f"alpha must be in [0, 4]; got {alpha}.")
    if not 0.0 <= float(rescue_modulation) <= 1.0:
        raise ValueError(
            "rescue_modulation must be in [0, 1]; "
            f"got {rescue_modulation}."
        )
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be positive; got {eps}.")

    with torch.no_grad():
        gap_b, gap_b_plus, valid = _validate_gap_inputs(
            teacher_gap_b, teacher_gap_b_plus, valid_mask
        )
        gap_rank = torch.zeros_like(gap_b)
        gap_rank[valid] = _ordinal_percentile_rank(gap_b[valid])
        rescue = torch.zeros_like(gap_b)
        rescue[valid] = (
            (gap_b[valid] - gap_b_plus[valid])
            / (gap_b[valid] + float(eps))
        ).clamp(0.0, 1.0)
        priority = torch.zeros_like(gap_b)
        priority[valid] = gap_rank[valid] * (
            1.0 + float(rescue_modulation) * rescue[valid]
        )
        raw_weight = torch.zeros_like(gap_b)
        raw_weight[valid] = 1.0 + float(alpha) * priority[valid]
        normalized = normalize_candidate_loss_mass(
            gap_b,
            gap_b,
            raw_weight,
            valid,
            eps=eps,
        )

    return CounterfactualTeachabilityModulationWeights(
        teacher_gap_b=gap_b.detach(),
        teacher_gap_b_plus=gap_b_plus.detach(),
        teacher_gap_rank=gap_rank.detach(),
        rescue_fraction=rescue.detach(),
        priority=priority.detach(),
        raw_weight=raw_weight.detach(),
        loss_mass_scale=normalized.loss_mass_scale.detach(),
        weight=normalized.weight.detach(),
        valid_mask=valid.detach(),
    )


def conditional_rescue_residual_weights(
    teacher_gap_b: torch.Tensor,
    teacher_gap_b_plus: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    alpha: float = 0.1,
    difficulty_bins: int = 5,
    eps: float = 1e-8,
) -> ConditionalRescueResidualWeights:
    """Reweight only budget rescue unexplained by within-response difficulty.

    Raw positive rescue is strongly correlated with the deployed-budget
    teacher gap. We remove that first-order confound by subtracting the median
    rescue in each within-response teacher-gap quantile. The residual is then
    centered over valid tokens and used as a small symmetric perturbation of
    unit weights. Detached KL-mass normalization preserves vanilla OPSD's
    scalar loss exactly for every response.
    """

    if not 0.0 <= float(alpha) < 1.0:
        raise ValueError(f"alpha must be in [0, 1); got {alpha}.")
    if int(difficulty_bins) < 2:
        raise ValueError(f"difficulty_bins must be at least 2; got {difficulty_bins}.")
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be positive; got {eps}.")

    with torch.no_grad():
        gap_b, gap_b_plus, valid = _validate_gap_inputs(
            teacher_gap_b, teacher_gap_b_plus, valid_mask
        )
        gap_rank = torch.zeros_like(gap_b)
        gap_rank[valid] = _ordinal_percentile_rank(gap_b[valid])
        rescue = torch.zeros_like(gap_b)
        rescue[valid] = (
            (gap_b[valid] - gap_b_plus[valid])
            / (gap_b[valid] + float(eps))
        ).clamp(0.0, 1.0)

        valid_ranks = gap_rank[valid]
        bins = (
            torch.ceil(valid_ranks * float(difficulty_bins)).long() - 1
        ).clamp(0, int(difficulty_bins) - 1)
        expected = torch.zeros_like(gap_b)
        valid_expected = torch.empty_like(rescue[valid])
        valid_rescue = rescue[valid]
        for bin_index in range(int(difficulty_bins)):
            in_bin = bins == bin_index
            if in_bin.any():
                valid_expected[in_bin] = valid_rescue[in_bin].median()
        expected[valid] = valid_expected

        residual = torch.zeros_like(gap_b)
        centered = valid_rescue - valid_expected
        centered = (centered - centered.mean()).clamp(-1.0, 1.0)
        residual[valid] = centered
        raw_weight = torch.zeros_like(gap_b)
        raw_weight[valid] = 1.0 + float(alpha) * residual[valid]
        normalized = normalize_candidate_loss_mass(
            gap_b,
            gap_b,
            raw_weight,
            valid,
            eps=eps,
        )

    return ConditionalRescueResidualWeights(
        teacher_gap_b=gap_b.detach(),
        teacher_gap_b_plus=gap_b_plus.detach(),
        teacher_gap_rank=gap_rank.detach(),
        rescue_fraction=rescue.detach(),
        expected_rescue=expected.detach(),
        rescue_residual=residual.detach(),
        raw_weight=raw_weight.detach(),
        loss_mass_scale=normalized.loss_mass_scale.detach(),
        weight=normalized.weight.detach(),
        valid_mask=valid.detach(),
    )


def grouped_kl_mass_weights(
    reference_loss: torch.Tensor,
    ranking_signal: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    top_fraction: float = 0.2,
    high_group_mass: float = 0.5,
    eps: float = 1e-8,
) -> GroupedKLMassWeights:
    """Allocate fixed loss mass to a ranked token group without scale drift.

    This is the token-group normalization used for the native-budget rescue
    candidate and its teacher-gap-only control. The detached ranking signal
    chooses the high group; the full-teacher KL remains the sole objective.
    A final detached KL-mass correction makes the forward scalar exactly match
    vanilla OPSD for each response while retaining the grouped gradient ratio.
    """

    if not 0.0 < float(top_fraction) < 1.0:
        raise ValueError(f"top_fraction must be in (0, 1); got {top_fraction}.")
    if not 0.0 < float(high_group_mass) < 1.0:
        raise ValueError(
            f"high_group_mass must be in (0, 1); got {high_group_mass}."
        )
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be positive; got {eps}.")

    with torch.no_grad():
        loss = reference_loss.detach().float().reshape(-1).clamp_min(0.0)
        signal = ranking_signal.detach().float().reshape(-1)
        valid = valid_mask.detach().to(device=loss.device, dtype=torch.bool).reshape(-1)
        if loss.shape != signal.shape or loss.shape != valid.shape:
            raise ValueError(
                "Reference loss, ranking signal, and valid mask must align: "
                f"{loss.shape}, {signal.shape}, {valid.shape}."
            )
        if not valid.any():
            raise ValueError("At least one valid generated-token position is required.")
        if not torch.isfinite(loss[valid]).all() or not torch.isfinite(signal[valid]).all():
            raise ValueError("Grouped loss and ranking signal must be finite.")
        if (loss[valid] < -1e-6).any():
            raise ValueError("Reference KL must be nonnegative.")

        valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
        n_valid = int(valid_indices.numel())
        n_high = (
            max(1, min(n_valid - 1, math.ceil(float(top_fraction) * n_valid)))
            if n_valid > 1
            else 1
        )
        order = torch.argsort(signal[valid], descending=True, stable=True)
        high_indices = valid_indices[order[:n_high]]
        high = torch.zeros_like(valid)
        high[high_indices] = True
        low = valid & ~high

        raw_weight = torch.zeros_like(loss)
        if low.any():
            raw_weight[high] = float(high_group_mass) * n_valid / float(n_high)
            raw_weight[low] = (
                (1.0 - float(high_group_mass))
                * n_valid
                / float(n_valid - n_high)
            )
        else:
            raw_weight[high] = 1.0
        normalized = normalize_candidate_loss_mass(
            loss,
            loss,
            raw_weight,
            valid,
            eps=eps,
        )

    return GroupedKLMassWeights(
        ranking_signal=signal.detach(),
        high_group_mask=high.detach(),
        raw_weight=raw_weight.detach(),
        loss_mass_scale=normalized.loss_mass_scale.detach(),
        weight=normalized.weight.detach(),
        valid_mask=valid.detach(),
    )


def max_kl_fraction_inverse_jsd_weights(
    teacher_gap: torch.Tensor,
    sensitivity: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    max_kl_fraction: float = 0.10,
    eps: float = 1e-8,
) -> MaxKLFractionInverseJSDWeights:
    """Apply mean-one inverse-JSD weights inside a high teacher-KL group.

    The high group contains valid tokens whose forward OPSD KL is strictly
    greater than ``max_kl_fraction`` times the trajectory maximum. Tokens
    outside the group retain unit weight. Inverse-JSD weights are normalized
    to mean one inside the group, which also makes the complete trajectory's
    valid-token mean weight one.
    """

    fraction = float(max_kl_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"max_kl_fraction must be in (0, 1); got {fraction}.")
    eps = float(eps)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"eps must be finite and positive; got {eps}.")

    with torch.no_grad():
        gap = teacher_gap.detach().float().reshape(-1)
        jsd = sensitivity.detach().float().reshape(-1)
        valid = valid_mask.detach().to(device=gap.device, dtype=torch.bool).reshape(-1)
        if gap.shape != jsd.shape or gap.shape != valid.shape:
            raise ValueError(
                "Teacher gap, sensitivity, and valid mask must align: "
                f"{gap.shape}, {jsd.shape}, {valid.shape}."
            )
        if not valid.any():
            raise ValueError("At least one valid generated-token position is required.")
        if not torch.isfinite(gap[valid]).all() or not torch.isfinite(jsd[valid]).all():
            raise ValueError("Teacher KL and JSD must be finite on valid positions.")
        if (gap[valid] < -1e-6).any() or (jsd[valid] < -1e-6).any():
            raise ValueError("Teacher KL and JSD must be nonnegative.")

        gap = gap.clamp_min(0.0)
        jsd = jsd.clamp_min(0.0)
        threshold = gap[valid].max() * fraction
        high = valid & (gap > threshold)

        raw_weight = torch.zeros_like(gap)
        weight = torch.zeros_like(gap)
        raw_weight[valid] = 1.0
        weight[valid] = 1.0
        if high.any():
            inverse = (jsd[high] + eps).reciprocal()
            raw_weight[high] = inverse
            weight[high] = inverse / inverse.mean().clamp_min(eps)
            # Remove only floating-point accumulation drift; this is exactly
            # one in real arithmetic because both groups have mean-one weight.
            weight[valid] = weight[valid] / weight[valid].mean().clamp_min(eps)

    return MaxKLFractionInverseJSDWeights(
        teacher_gap=gap.detach(),
        sensitivity=jsd.detach(),
        threshold=threshold.detach(),
        high_group_mask=high.detach(),
        raw_weight=raw_weight.detach(),
        weight=weight.detach(),
        valid_mask=valid.detach(),
    )


def max_kl_fraction_softmax_inverse_jsd_weights(
    teacher_gap: torch.Tensor,
    sensitivity: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    max_kl_fraction: float = 0.10,
    temperature: float = 0.05,
) -> MaxKLFractionInverseJSDWeights:
    """Apply mean-one softmax inverse-JSD weights inside a high-KL group."""

    fraction = float(max_kl_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"max_kl_fraction must be in (0, 1); got {fraction}.")
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"temperature must be finite and positive; got {temperature}.")

    with torch.no_grad():
        gap = teacher_gap.detach().float().reshape(-1)
        jsd = sensitivity.detach().float().reshape(-1)
        valid = valid_mask.detach().to(device=gap.device, dtype=torch.bool).reshape(-1)
        if gap.shape != jsd.shape or gap.shape != valid.shape:
            raise ValueError(
                "Teacher gap, sensitivity, and valid mask must align: "
                f"{gap.shape}, {jsd.shape}, {valid.shape}."
            )
        if not valid.any():
            raise ValueError("At least one valid generated-token position is required.")
        if not torch.isfinite(gap[valid]).all() or not torch.isfinite(jsd[valid]).all():
            raise ValueError("Teacher KL and JSD must be finite on valid positions.")
        if (gap[valid] < -1e-6).any() or (jsd[valid] < -1e-6).any():
            raise ValueError("Teacher KL and JSD must be nonnegative.")

        gap = gap.clamp_min(0.0)
        jsd = jsd.clamp_min(0.0)
        threshold = gap[valid].max() * fraction
        high = valid & (gap > threshold)

        raw_weight = torch.zeros_like(gap)
        weight = torch.zeros_like(gap)
        raw_weight[valid] = 1.0
        weight[valid] = 1.0
        if high.any():
            logits = -jsd[high] / temperature
            shifted = logits - logits.max()
            softmax_raw = shifted.exp()
            raw_weight[high] = softmax_raw
            weight[high] = softmax_raw / softmax_raw.mean()
            weight[valid] = weight[valid] / weight[valid].mean()

    return MaxKLFractionInverseJSDWeights(
        teacher_gap=gap.detach(),
        sensitivity=jsd.detach(),
        threshold=threshold.detach(),
        high_group_mask=high.detach(),
        raw_weight=raw_weight.detach(),
        weight=weight.detach(),
        valid_mask=valid.detach(),
    )


def max_kl_fraction_softmax_inverse_jsd_group_balanced_weights(
    teacher_gap: torch.Tensor,
    sensitivity: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    max_kl_fraction: float = 0.10,
    temperature: float = 0.05,
    high_group_coefficient: float = 0.10,
) -> GroupBalancedMaxKLFractionWeights:
    """Build a detached high/low group objective with inverse-JSD ordering.

    The returned weights satisfy

    ``mean(weight * loss) = lambda * mean_H(w_jsd * loss)
                            + (1-lambda) * mean_L(loss)``.

    Both the JSD-weighted and unweighted group-balanced weight vectors have
    mean one over valid positions. Degenerate one-group trajectories fall back
    to vanilla token averaging.
    """

    coefficient = float(high_group_coefficient)
    if not math.isfinite(coefficient) or not 0.0 < coefficient < 1.0:
        raise ValueError(
            "high_group_coefficient must be finite and in (0, 1); "
            f"got {high_group_coefficient}."
        )
    internal = max_kl_fraction_softmax_inverse_jsd_weights(
        teacher_gap,
        sensitivity,
        valid_mask,
        max_kl_fraction=max_kl_fraction,
        temperature=temperature,
    )
    with torch.no_grad():
        valid = internal.valid_mask
        high = internal.high_group_mask
        low = valid & ~high
        balanced = torch.zeros_like(internal.weight)
        weighted = torch.zeros_like(internal.weight)
        if high.any() and low.any():
            n_valid = float(valid.sum())
            n_high = float(high.sum())
            n_low = float(low.sum())
            high_scale = coefficient * n_valid / n_high
            low_scale = (1.0 - coefficient) * n_valid / n_low
            balanced[high] = high_scale
            balanced[low] = low_scale
            weighted[high] = high_scale * internal.weight[high]
            weighted[low] = low_scale
        else:
            balanced[valid] = 1.0
            weighted[valid] = 1.0

    return GroupBalancedMaxKLFractionWeights(
        teacher_gap=internal.teacher_gap,
        sensitivity=internal.sensitivity,
        threshold=internal.threshold,
        high_group_mask=high.detach(),
        within_high_weight=internal.weight.detach(),
        balanced_unweighted_weight=balanced.detach(),
        weight=weighted.detach(),
        high_group_coefficient=coefficient,
        valid_mask=valid.detach(),
    )


def _validate_gap_inputs(
    teacher_gap_b: torch.Tensor,
    teacher_gap_b_plus: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gap_b = teacher_gap_b.detach().float().reshape(-1)
    gap_b_plus = teacher_gap_b_plus.detach().float().reshape(-1)
    valid = valid_mask.detach().to(device=gap_b.device, dtype=torch.bool).reshape(-1)
    if gap_b.shape != gap_b_plus.shape or gap_b.shape != valid.shape:
        raise ValueError(
            "Teacher gaps and valid mask must align: "
            f"{gap_b.shape}, {gap_b_plus.shape}, {valid.shape}."
        )
    if not valid.any():
        raise ValueError("At least one valid generated-token position is required.")
    if not torch.isfinite(gap_b[valid]).all() or not torch.isfinite(gap_b_plus[valid]).all():
        raise ValueError("Teacher KL must be finite on all valid positions.")
    if (gap_b[valid] < -1e-6).any() or (gap_b_plus[valid] < -1e-6).any():
        raise ValueError("Teacher KL must be nonnegative.")
    return gap_b.clamp_min(0.0), gap_b_plus.clamp_min(0.0), valid


def _ordinal_percentile_rank(values: torch.Tensor) -> torch.Tensor:
    """Return deterministic ranks in (0, 1], used only on detached signals."""

    if values.ndim != 1 or values.numel() <= 0:
        raise ValueError("Rank input must be a nonempty vector.")
    order = torch.argsort(values, stable=True)
    ordinal = torch.empty_like(order)
    ordinal[order] = torch.arange(1, values.numel() + 1, device=values.device)
    return ordinal.float() / float(values.numel())


def normalize_candidate_loss_mass(
    reference_loss: torch.Tensor,
    candidate_loss: torch.Tensor,
    raw_weight: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> LossMassNormalization:
    """Match a detached weighted candidate objective to reference KL mass."""

    if float(eps) <= 0.0:
        raise ValueError(f"eps must be positive; got {eps}.")
    with torch.no_grad():
        reference = reference_loss.detach().float().reshape(-1)
        candidate = candidate_loss.detach().float().reshape(-1)
        raw = raw_weight.detach().float().reshape(-1)
        valid = valid_mask.detach().to(device=reference.device, dtype=torch.bool).reshape(-1)
        if not (reference.shape == candidate.shape == raw.shape == valid.shape):
            raise ValueError(
                "Reference loss, candidate loss, raw weight, and valid mask must align: "
                f"{reference.shape}, {candidate.shape}, {raw.shape}, {valid.shape}."
            )
        if not valid.any():
            raise ValueError("At least one valid token is required for loss-mass normalization.")
        for name, values in (("reference", reference), ("candidate", candidate), ("weight", raw)):
            if not torch.isfinite(values[valid]).all():
                raise ValueError(f"{name} values must be finite on valid positions.")
        if (reference[valid] < -1e-6).any() or (candidate[valid] < -1e-6).any():
            raise ValueError("KL losses must be nonnegative on valid positions.")
        if (raw[valid] <= 0.0).any():
            raise ValueError("Raw weights must be positive on valid positions.")
        reference = reference.clamp_min(0.0)
        candidate = candidate.clamp_min(0.0)
        reference_mass = reference[valid].sum()
        candidate_mass = (raw[valid] * candidate[valid]).sum()
        if float(reference_mass) <= float(eps):
            scale = torch.zeros((), device=reference.device, dtype=torch.float32)
            weight = torch.zeros_like(reference)
        else:
            if not torch.isfinite(candidate_mass) or float(candidate_mass) <= 0.0:
                raise FloatingPointError(f"Invalid weighted candidate loss mass: {float(candidate_mass)}")
            scale = reference_mass / candidate_mass
            weight = torch.zeros_like(reference)
            weight[valid] = raw[valid] * scale
        normalized = (weight[valid] * candidate[valid]).sum()
        if not torch.allclose(normalized, reference_mass, rtol=2e-6, atol=2e-7):
            raise FloatingPointError(
                "Candidate loss-mass normalization failed: "
                f"normalized={float(normalized)}, reference={float(reference_mass)}."
            )
    return LossMassNormalization(
        reference_loss=reference.detach(),
        candidate_loss=candidate.detach(),
        raw_weight=raw.detach(),
        loss_mass_scale=scale.detach(),
        weight=weight.detach(),
        valid_mask=valid.detach(),
    )


def budget_consistent_rank_weights(
    teacher_gap_b: torch.Tensor,
    teacher_gap_b_plus: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    alpha: float = 1.0,
    eps: float = 1e-8,
) -> BudgetConsistentRankWeights:
    """Prioritize divergence that remains high at both native token budgets.

    ``min(K_b, K_b_plus)`` is a conservative lower bound on the teacher gap
    across the two budgets. Per-sample ranks remove absolute KL-scale and
    response-length confounds. KL-mass normalization keeps the forward scalar
    loss identical to vanilla OPSD while changing token-level gradients.
    """

    if not 0.0 <= float(alpha) <= 4.0:
        raise ValueError(f"alpha must be in [0, 4]; got {alpha}.")
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be positive; got {eps}.")
    with torch.no_grad():
        gap_b, gap_b_plus, valid = _validate_gap_inputs(teacher_gap_b, teacher_gap_b_plus, valid_mask)
        persistent_gap = torch.zeros_like(gap_b)
        persistent_gap[valid] = torch.minimum(gap_b[valid], gap_b_plus[valid])
        persistent_rank = torch.zeros_like(gap_b)
        persistent_rank[valid] = _ordinal_percentile_rank(persistent_gap[valid])
        raw_weight = torch.zeros_like(gap_b)
        raw_weight[valid] = 1.0 + float(alpha) * persistent_rank[valid]

        unweighted_mass = gap_b[valid].sum()
        raw_weighted_mass = (raw_weight[valid] * gap_b[valid]).sum()
        if float(unweighted_mass) <= float(eps):
            loss_mass_scale = torch.ones((), device=gap_b.device, dtype=torch.float32)
            weight = torch.zeros_like(gap_b)
            weight[valid] = 1.0
        else:
            if not torch.isfinite(raw_weighted_mass) or float(raw_weighted_mass) <= 0.0:
                raise FloatingPointError(f"Invalid raw weighted KL mass: {float(raw_weighted_mass)}")
            loss_mass_scale = unweighted_mass / raw_weighted_mass
            weight = torch.zeros_like(gap_b)
            weight[valid] = raw_weight[valid] * loss_mass_scale
        if not torch.allclose(
            (weight[valid] * gap_b[valid]).sum(), unweighted_mass, rtol=2e-6, atol=2e-7
        ):
            raise FloatingPointError("Budget-consistent rank KL-mass normalization failed.")

    return BudgetConsistentRankWeights(
        teacher_gap_b=gap_b.detach(),
        teacher_gap_b_plus=gap_b_plus.detach(),
        persistent_gap=persistent_gap.detach(),
        persistent_rank=persistent_rank.detach(),
        raw_weight=raw_weight.detach(),
        loss_mass_scale=loss_mass_scale.detach(),
        weight=weight.detach(),
        valid_mask=valid.detach(),
    )


def budget_residual_hardness_weights(
    teacher_gap_b: torch.Tensor,
    teacher_gap_b_plus: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    alpha: float = 1.0,
    persistence_mix: float = 0.1,
    eps: float = 1e-8,
) -> BudgetResidualHardnessWeights:
    """Reweight hard tokens while requiring evidence across adjacent budgets.

    The deployed-budget teacher gap remains the primary signal. A small,
    preregistered persistence term favors gaps that remain large after adding
    native visual tokens. Per-sample ranks make the two terms commensurate;
    KL-mass normalization preserves vanilla OPSD's detached scalar loss.
    """

    if not 0.0 <= float(alpha) <= 4.0:
        raise ValueError(f"alpha must be in [0, 4]; got {alpha}.")
    if not 0.0 <= float(persistence_mix) <= 1.0:
        raise ValueError(
            f"persistence_mix must be in [0, 1]; got {persistence_mix}."
        )
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be positive; got {eps}.")

    with torch.no_grad():
        gap_b, gap_b_plus, valid = _validate_gap_inputs(
            teacher_gap_b, teacher_gap_b_plus, valid_mask
        )
        persistent_gap = torch.zeros_like(gap_b)
        persistent_gap[valid] = (
            2.0
            * gap_b[valid]
            * gap_b_plus[valid]
            / (gap_b[valid] + gap_b_plus[valid] + float(eps))
        )
        teacher_gap_rank = torch.zeros_like(gap_b)
        teacher_gap_rank[valid] = _ordinal_percentile_rank(gap_b[valid])
        persistent_rank = torch.zeros_like(gap_b)
        persistent_rank[valid] = _ordinal_percentile_rank(persistent_gap[valid])
        priority = torch.zeros_like(gap_b)
        priority[valid] = (
            (1.0 - float(persistence_mix)) * teacher_gap_rank[valid]
            + float(persistence_mix) * persistent_rank[valid]
        )
        raw_weight = torch.zeros_like(gap_b)
        raw_weight[valid] = 1.0 + float(alpha) * priority[valid]

        unweighted_mass = gap_b[valid].sum()
        raw_weighted_mass = (raw_weight[valid] * gap_b[valid]).sum()
        if float(unweighted_mass) <= float(eps):
            loss_mass_scale = torch.ones((), device=gap_b.device, dtype=torch.float32)
            weight = torch.zeros_like(gap_b)
            weight[valid] = 1.0
        else:
            if not torch.isfinite(raw_weighted_mass) or float(raw_weighted_mass) <= 0.0:
                raise FloatingPointError(
                    f"Invalid budget-residual weighted KL mass: {float(raw_weighted_mass)}"
                )
            loss_mass_scale = unweighted_mass / raw_weighted_mass
            weight = torch.zeros_like(gap_b)
            weight[valid] = raw_weight[valid] * loss_mass_scale
        if not torch.allclose(
            (weight[valid] * gap_b[valid]).sum(),
            unweighted_mass,
            rtol=2e-6,
            atol=2e-7,
        ):
            raise FloatingPointError(
                "Budget-residual hardness KL-mass normalization failed."
            )

    return BudgetResidualHardnessWeights(
        teacher_gap_b=gap_b.detach(),
        teacher_gap_b_plus=gap_b_plus.detach(),
        persistent_gap=persistent_gap.detach(),
        teacher_gap_rank=teacher_gap_rank.detach(),
        persistent_rank=persistent_rank.detach(),
        priority=priority.detach(),
        raw_weight=raw_weight.detach(),
        loss_mass_scale=loss_mass_scale.detach(),
        weight=weight.detach(),
        valid_mask=valid.detach(),
    )


def counterfactual_budget_bridge(
    teacher_gap_b: torch.Tensor,
    teacher_gap_b_plus: torch.Tensor,
    bridge_gap: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    max_bridge_fraction: float = 0.5,
    eps: float = 1e-8,
) -> CounterfactualBudgetBridge:
    """Route budget-rescuable tokens toward a proximal higher-budget teacher.

    The bridge is enabled only where the native ``b_plus`` student is closer
    to the full teacher than the deployed-budget student. The routed loss is
    KL-mass normalized so its detached scalar value equals vanilla OPSD.
    """

    if not 0.0 <= float(max_bridge_fraction) <= 1.0:
        raise ValueError(
            f"max_bridge_fraction must be in [0, 1]; got {max_bridge_fraction}."
        )
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be positive; got {eps}.")
    with torch.no_grad():
        gap_b, gap_b_plus, valid = _validate_gap_inputs(teacher_gap_b, teacher_gap_b_plus, valid_mask)
        bridge = bridge_gap.detach().float().reshape(-1)
        if bridge.shape != gap_b.shape:
            raise ValueError(f"Bridge gap must align with teacher gaps: {bridge.shape} vs {gap_b.shape}.")
        if not torch.isfinite(bridge[valid]).all() or (bridge[valid] < -1e-6).any():
            raise ValueError("Bridge KL must be finite and nonnegative on valid positions.")
        bridge = bridge.clamp_min(0.0)

        tau = torch.quantile(gap_b[valid], 0.5)
        rescue = torch.zeros_like(gap_b)
        rescue[valid] = ((gap_b[valid] - gap_b_plus[valid]) / (gap_b[valid] + float(eps))).clamp(0.0, 1.0)
        confidence = torch.zeros_like(gap_b)
        confidence[valid] = gap_b[valid] / (gap_b[valid] + tau + float(eps))
        bridge_fraction = torch.zeros_like(gap_b)
        bridge_fraction[valid] = float(max_bridge_fraction) * rescue[valid] * confidence[valid]

        routed = torch.zeros_like(gap_b)
        routed[valid] = (
            (1.0 - bridge_fraction[valid]) * gap_b[valid]
            + bridge_fraction[valid] * bridge[valid]
        )
        unweighted_mass = gap_b[valid].sum()
        routed_mass = routed[valid].sum()
        if float(unweighted_mass) <= float(eps):
            loss_mass_scale = torch.ones((), device=gap_b.device, dtype=torch.float32)
        else:
            if not torch.isfinite(routed_mass) or float(routed_mass) <= 0.0:
                raise FloatingPointError(f"Invalid routed KL mass: {float(routed_mass)}")
            loss_mass_scale = unweighted_mass / routed_mass
        full_teacher_weight = torch.zeros_like(gap_b)
        bridge_teacher_weight = torch.zeros_like(gap_b)
        full_teacher_weight[valid] = loss_mass_scale * (1.0 - bridge_fraction[valid])
        bridge_teacher_weight[valid] = loss_mass_scale * bridge_fraction[valid]
        normalized = (
            full_teacher_weight[valid] * gap_b[valid]
            + bridge_teacher_weight[valid] * bridge[valid]
        ).sum()
        if not torch.allclose(normalized, unweighted_mass, rtol=2e-6, atol=2e-7):
            raise FloatingPointError("Counterfactual bridge KL-mass normalization failed.")

    return CounterfactualBudgetBridge(
        teacher_gap_b=gap_b.detach(),
        teacher_gap_b_plus=gap_b_plus.detach(),
        bridge_gap=bridge.detach(),
        tau_teacher_gap=tau.detach(),
        rescue_fraction=rescue.detach(),
        confidence=confidence.detach(),
        bridge_fraction=bridge_fraction.detach(),
        loss_mass_scale=loss_mass_scale.detach(),
        full_teacher_weight=full_teacher_weight.detach(),
        bridge_teacher_weight=bridge_teacher_weight.detach(),
        valid_mask=valid.detach(),
    )


def budget_contrastive_gate(
    teacher_gap_b: torch.Tensor,
    teacher_gap_b_plus: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    beta_max: float = 0.5,
    eps: float = 1e-8,
) -> BudgetContrastiveGate:
    """Gate budget-contrastive target shaping by observed teacher-gap rescue."""

    if not 0.0 <= float(beta_max) <= 2.0:
        raise ValueError(f"beta_max must be in [0, 2], got {beta_max}.")
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}.")
    with torch.no_grad():
        gap_b, gap_b_plus, valid = _validate_gap_inputs(teacher_gap_b, teacher_gap_b_plus, valid_mask)
        tau = torch.quantile(gap_b[valid], 0.5)
        rescue = torch.zeros_like(gap_b)
        rescue[valid] = ((gap_b[valid] - gap_b_plus[valid]) / (gap_b[valid] + float(eps))).clamp(0.0, 1.0)
        confidence = torch.zeros_like(gap_b)
        confidence[valid] = gap_b[valid] / (gap_b[valid] + tau + float(eps))
        strength = torch.zeros_like(gap_b)
        strength[valid] = float(beta_max) * rescue[valid] * confidence[valid]
    return BudgetContrastiveGate(
        teacher_gap_b=gap_b.detach(),
        teacher_gap_b_plus=gap_b_plus.detach(),
        tau_teacher_gap=tau.detach(),
        rescue_fraction=rescue.detach(),
        confidence=confidence.detach(),
        shaping_strength=strength.detach(),
        valid_mask=valid.detach(),
    )


def budget_gradient_aligned_bridge_gate(
    teacher_gap_b: torch.Tensor,
    teacher_gap_b_plus: torch.Tensor,
    gradient_alignment: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    max_bridge_fraction: float = 0.5,
    eps: float = 1e-8,
) -> BudgetGradientAlignedBridgeGate:
    """Gate a proximal budget bridge by causal improvement and gradient agreement."""

    if not 0.0 <= float(max_bridge_fraction) <= 1.0:
        raise ValueError(
            f"max_bridge_fraction must be in [0, 1]; got {max_bridge_fraction}."
        )
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}.")
    with torch.no_grad():
        gap_b, gap_b_plus, valid = _validate_gap_inputs(
            teacher_gap_b, teacher_gap_b_plus, valid_mask
        )
        alignment = gradient_alignment.detach().float().reshape(-1)
        if alignment.shape != gap_b.shape:
            raise ValueError(
                f"Gradient alignment must match teacher gaps: {alignment.shape} vs {gap_b.shape}."
            )
        if not torch.isfinite(alignment[valid]).all():
            raise ValueError("Gradient alignment must be finite on valid positions.")
        if (alignment[valid].abs() > 1.0 + 1e-5).any():
            raise ValueError("Gradient alignment must lie in [-1, 1].")
        alignment = alignment.clamp(-1.0, 1.0)
        positive_alignment = torch.zeros_like(alignment)
        positive_alignment[valid] = alignment[valid].clamp(0.0, 1.0)

        tau = torch.quantile(gap_b[valid], 0.5)
        rescue = torch.zeros_like(gap_b)
        rescue[valid] = (
            (gap_b[valid] - gap_b_plus[valid]) / (gap_b[valid] + float(eps))
        ).clamp(0.0, 1.0)
        confidence = torch.zeros_like(gap_b)
        confidence[valid] = gap_b[valid] / (gap_b[valid] + tau + float(eps))
        bridge_fraction = torch.zeros_like(gap_b)
        bridge_fraction[valid] = (
            float(max_bridge_fraction)
            * rescue[valid]
            * confidence[valid]
            * positive_alignment[valid]
        )

    return BudgetGradientAlignedBridgeGate(
        teacher_gap_b=gap_b.detach(),
        teacher_gap_b_plus=gap_b_plus.detach(),
        gradient_alignment=alignment.detach(),
        positive_alignment=positive_alignment.detach(),
        tau_teacher_gap=tau.detach(),
        rescue_fraction=rescue.detach(),
        confidence=confidence.detach(),
        bridge_fraction=bridge_fraction.detach(),
        valid_mask=valid.detach(),
    )


def counterfactual_gradient_residual_gate(
    teacher_gap_b: torch.Tensor,
    teacher_gap_b_plus: torch.Tensor,
    gradient_alignment: torch.Tensor,
    projection_coefficient: torch.Tensor,
    teacher_gradient_norm_sq: torch.Tensor,
    budget_gradient_norm_sq: torch.Tensor,
    gradient_dot_product: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    cancellation_strength: float = 0.5,
    max_projection_coefficient: float = 1.0,
    eps: float = 1e-8,
) -> CounterfactualGradientResidualGate:
    """Remove a bounded portion of the teacher gradient explained by ``b_plus``.

    A token is eligible only when the native adjacent-budget intervention
    lowers its full-teacher KL. The cancellation coefficient never exceeds
    the positive orthogonal-projection coefficient, so the residual cannot
    reverse the teacher gradient along the budget direction. All values are
    detached; this gate is used with a stop-gradient scalar correction that
    preserves the vanilla OPSD forward loss exactly.
    """

    if not 0.0 <= float(cancellation_strength) <= 1.0:
        raise ValueError(
            f"cancellation_strength must be in [0, 1]; got {cancellation_strength}."
        )
    if float(max_projection_coefficient) <= 0.0:
        raise ValueError(
            "max_projection_coefficient must be positive; "
            f"got {max_projection_coefficient}."
        )
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be positive; got {eps}.")

    with torch.no_grad():
        gap_b, gap_b_plus, valid = _validate_gap_inputs(
            teacher_gap_b, teacher_gap_b_plus, valid_mask
        )
        alignment = gradient_alignment.detach().float().reshape(-1)
        raw_projection = projection_coefficient.detach().float().reshape(-1)
        teacher_norm_sq = teacher_gradient_norm_sq.detach().float().reshape(-1)
        budget_norm_sq = budget_gradient_norm_sq.detach().float().reshape(-1)
        dot_product = gradient_dot_product.detach().float().reshape(-1)
        if not (
            gap_b.shape
            == alignment.shape
            == raw_projection.shape
            == teacher_norm_sq.shape
            == budget_norm_sq.shape
            == dot_product.shape
        ):
            raise ValueError("Counterfactual gradient-geometry tensors must align by token.")
        for name, values in (
            ("gradient alignment", alignment),
            ("projection coefficient", raw_projection),
            ("teacher gradient norm", teacher_norm_sq),
            ("budget gradient norm", budget_norm_sq),
            ("gradient dot product", dot_product),
        ):
            if not torch.isfinite(values[valid]).all():
                raise ValueError(f"{name} must be finite on valid positions.")
        if (teacher_norm_sq[valid] < -1e-7).any() or (budget_norm_sq[valid] < -1e-7).any():
            raise ValueError("Squared gradient norms must be nonnegative.")

        alignment = alignment.clamp(-1.0, 1.0)
        clipped_projection = torch.zeros_like(raw_projection)
        clipped_projection[valid] = raw_projection[valid].clamp(
            0.0, float(max_projection_coefficient)
        )
        rescue_indicator = torch.zeros_like(gap_b)
        rescue_indicator[valid] = (gap_b[valid] > gap_b_plus[valid]).float()
        coefficient = torch.zeros_like(gap_b)
        coefficient[valid] = (
            float(cancellation_strength)
            * clipped_projection[valid]
            * rescue_indicator[valid]
        )

        residual_norm_sq = (
            teacher_norm_sq
            - 2.0 * coefficient * dot_product
            + coefficient.square() * budget_norm_sq
        ).clamp_min(0.0)
        residual_ratio = torch.ones_like(gap_b)
        nonzero_teacher = valid & (teacher_norm_sq > float(eps))
        residual_ratio[nonzero_teacher] = (
            residual_norm_sq[nonzero_teacher]
            / teacher_norm_sq[nonzero_teacher].clamp_min(float(eps))
        ).sqrt()

        # Since coefficient <= the positive projection coefficient, the
        # residual must not have a larger component along g_budget.
        residual_dot = dot_product - coefficient * budget_norm_sq
        eligible = valid & (coefficient > 0.0)
        if eligible.any() and (residual_dot[eligible] < -2e-6).any():
            raise FloatingPointError("Gradient cancellation reversed the budget-direction component.")

    return CounterfactualGradientResidualGate(
        teacher_gap_b=gap_b.detach(),
        teacher_gap_b_plus=gap_b_plus.detach(),
        gradient_alignment=alignment.detach(),
        raw_projection_coefficient=raw_projection.detach(),
        clipped_projection_coefficient=clipped_projection.detach(),
        budget_rescue_indicator=rescue_indicator.detach(),
        cancellation_coefficient=coefficient.detach(),
        residual_gradient_norm_ratio=residual_ratio.detach(),
        valid_mask=valid.detach(),
    )


def budget_tangent_residual_weights(
    teacher_gap_b: torch.Tensor,
    gradient_alignment: torch.Tensor,
    budget_explained_fraction: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    alpha: float = 1.0,
    eps: float = 1e-8,
) -> BudgetTangentResidualWeights:
    """Prioritize hard teacher residuals not explained by a +delta budget step.

    The adjacent native VisionZip distribution defines a finite-difference
    tangent in vocabulary-probability space. A token receives high priority
    only when its deployed-budget teacher KL is high and that tangent explains
    little of the full-teacher gradient. Detached KL-mass normalization keeps
    the scalar objective exactly equal to vanilla OPSD for each sample.
    """

    if not 0.0 <= float(alpha) <= 4.0:
        raise ValueError(f"alpha must be in [0, 4]; got {alpha}.")
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}.")
    with torch.no_grad():
        gap = teacher_gap_b.detach().float().reshape(-1).clamp_min(0.0)
        alignment = gradient_alignment.detach().float().reshape(-1)
        explained = budget_explained_fraction.detach().float().reshape(-1)
        valid = valid_mask.detach().to(device=gap.device, dtype=torch.bool).reshape(-1)
        if gap.shape != alignment.shape or gap.shape != explained.shape or gap.shape != valid.shape:
            raise ValueError(
                "Teacher gap, gradient geometry, and valid mask must align: "
                f"{gap.shape}, {alignment.shape}, {explained.shape}, {valid.shape}."
            )
        if not valid.any():
            raise ValueError("At least one valid generated-token position is required.")
        if not torch.isfinite(gap[valid]).all() or not torch.isfinite(alignment[valid]).all():
            raise ValueError("Teacher gap and gradient alignment must be finite.")
        if not torch.isfinite(explained[valid]).all():
            raise ValueError("Budget-explained fraction must be finite.")
        if (alignment[valid].abs() > 1.0 + 1e-5).any():
            raise ValueError("Gradient alignment must lie in [-1, 1].")
        if ((explained[valid] < -1e-6) | (explained[valid] > 1.0 + 1e-6)).any():
            raise ValueError("Budget-explained fraction must lie in [0, 1].")

        alignment = alignment.clamp(-1.0, 1.0)
        explained = explained.clamp(0.0, 1.0)
        residual = torch.zeros_like(gap)
        residual[valid] = 1.0 - explained[valid]
        gap_rank = torch.zeros_like(gap)
        gap_rank[valid] = _ordinal_percentile_rank(gap[valid])
        priority = torch.zeros_like(gap)
        priority[valid] = gap_rank[valid] * residual[valid]
        raw_weight = torch.zeros_like(gap)
        raw_weight[valid] = 1.0 + float(alpha) * priority[valid]

        unweighted_mass = gap[valid].sum()
        weighted_mass = (raw_weight[valid] * gap[valid]).sum()
        if float(unweighted_mass) <= float(eps):
            loss_mass_scale = torch.ones((), device=gap.device, dtype=torch.float32)
            weight = torch.zeros_like(gap)
            weight[valid] = 1.0
        else:
            if not torch.isfinite(weighted_mass) or float(weighted_mass) <= 0.0:
                raise FloatingPointError(f"Invalid tangent-residual weighted KL mass: {float(weighted_mass)}")
            loss_mass_scale = unweighted_mass / weighted_mass
            weight = torch.zeros_like(gap)
            weight[valid] = raw_weight[valid] * loss_mass_scale
        if not torch.allclose(
            (weight[valid] * gap[valid]).sum(),
            unweighted_mass,
            rtol=2e-6,
            atol=2e-7,
        ):
            raise FloatingPointError("Budget-tangent residual KL-mass normalization failed.")

    return BudgetTangentResidualWeights(
        teacher_gap_b=gap.detach(),
        gradient_alignment=alignment.detach(),
        budget_explained_fraction=explained.detach(),
        budget_residual_fraction=residual.detach(),
        teacher_gap_rank=gap_rank.detach(),
        priority=priority.detach(),
        raw_weight=raw_weight.detach(),
        loss_mass_scale=loss_mass_scale.detach(),
        weight=weight.detach(),
        valid_mask=valid.detach(),
    )


def budget_gradient_consensus_weights(
    teacher_gap_b: torch.Tensor,
    gradient_consensus: torch.Tensor,
    gradient_norm_consistency: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    alpha: float = 0.5,
    eps: float = 1e-8,
) -> BudgetGradientConsensusWeights:
    """Prioritize hard teacher corrections invariant to a native budget probe."""

    if not 0.0 <= float(alpha) <= 4.0:
        raise ValueError(f"alpha must be in [0, 4]; got {alpha}.")
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}.")
    with torch.no_grad():
        gap = teacher_gap_b.detach().float().reshape(-1).clamp_min(0.0)
        consensus = gradient_consensus.detach().float().reshape(-1)
        norm_consistency = gradient_norm_consistency.detach().float().reshape(-1)
        valid = valid_mask.detach().to(device=gap.device, dtype=torch.bool).reshape(-1)
        if not (gap.shape == consensus.shape == norm_consistency.shape == valid.shape):
            raise ValueError("Gradient-consensus weighting tensors must align by token.")
        if not valid.any():
            raise ValueError("At least one valid generated-token position is required.")
        for name, values in (
            ("teacher gap", gap),
            ("gradient consensus", consensus),
            ("gradient norm consistency", norm_consistency),
        ):
            if not torch.isfinite(values[valid]).all():
                raise ValueError(f"{name} must be finite on valid positions.")
        if (consensus[valid].abs() > 1.0 + 1e-5).any():
            raise ValueError("Gradient consensus must lie in [-1, 1].")
        if ((norm_consistency[valid] < -1e-6) | (norm_consistency[valid] > 1.0 + 1e-6)).any():
            raise ValueError("Gradient norm consistency must lie in [0, 1].")

        consensus = consensus.clamp(-1.0, 1.0)
        norm_consistency = norm_consistency.clamp(0.0, 1.0)
        gap_rank = torch.zeros_like(gap)
        gap_rank[valid] = _ordinal_percentile_rank(gap[valid])
        priority = torch.zeros_like(gap)
        priority[valid] = (
            gap_rank[valid]
            * consensus[valid].clamp(0.0, 1.0)
            * norm_consistency[valid]
        )
        raw_weight = torch.zeros_like(gap)
        raw_weight[valid] = 1.0 + float(alpha) * priority[valid]
        normalized = normalize_candidate_loss_mass(
            gap,
            gap,
            raw_weight,
            valid,
            eps=eps,
        )
    return BudgetGradientConsensusWeights(
        teacher_gap_b=gap.detach(),
        teacher_gap_rank=gap_rank.detach(),
        gradient_consensus=consensus.detach(),
        gradient_norm_consistency=norm_consistency.detach(),
        invariant_priority=priority.detach(),
        raw_weight=raw_weight.detach(),
        loss_mass_scale=normalized.loss_mass_scale.detach(),
        weight=normalized.weight.detach(),
        valid_mask=valid.detach(),
    )


def budget_counterfactual_teachability_weights(
    teacher_gap_b: torch.Tensor,
    gradient_alignment: torch.Tensor,
    budget_explained_fraction: torch.Tensor,
    teacher_support_coverage: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    alpha: float = 1.0,
    eps: float = 1e-8,
) -> BudgetCounterfactualTeachabilityWeights:
    """Weight hard, locally teachable gaps unexplained by a +delta budget.

    Local support coverage follows the compatibility diagnostic of
    Teachability-Aware OPD. The additional adjacent-budget residual is a
    pruning-specific counterfactual: it suppresses teacher corrections that
    are already explained by adding visual evidence while prioritizing
    support-compatible utilization gaps. All signals and weights are detached,
    and the per-sample KL mass matches vanilla OPSD exactly.
    """

    if not 0.0 <= float(alpha) <= 4.0:
        raise ValueError(f"alpha must be in [0, 4]; got {alpha}.")
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}.")
    with torch.no_grad():
        gap = teacher_gap_b.detach().float().reshape(-1).clamp_min(0.0)
        alignment = gradient_alignment.detach().float().reshape(-1)
        explained = budget_explained_fraction.detach().float().reshape(-1)
        coverage = teacher_support_coverage.detach().float().reshape(-1)
        valid = valid_mask.detach().to(device=gap.device, dtype=torch.bool).reshape(-1)
        if not (gap.shape == alignment.shape == explained.shape == coverage.shape == valid.shape):
            raise ValueError(
                "Counterfactual teachability inputs must align: "
                f"{gap.shape}, {alignment.shape}, {explained.shape}, {coverage.shape}, {valid.shape}."
            )
        if not valid.any():
            raise ValueError("At least one valid generated-token position is required.")
        for name, values in (
            ("teacher gap", gap),
            ("gradient alignment", alignment),
            ("budget explained fraction", explained),
            ("teacher support coverage", coverage),
        ):
            if not torch.isfinite(values[valid]).all():
                raise ValueError(f"{name} must be finite on valid positions.")
        if (alignment[valid].abs() > 1.0 + 1e-5).any():
            raise ValueError("Gradient alignment must lie in [-1, 1].")
        if ((explained[valid] < -1e-6) | (explained[valid] > 1.0 + 1e-6)).any():
            raise ValueError("Budget-explained fraction must lie in [0, 1].")
        if ((coverage[valid] < -1e-6) | (coverage[valid] > 1.0 + 1e-6)).any():
            raise ValueError("Teacher support coverage must lie in [0, 1].")

        alignment = alignment.clamp(-1.0, 1.0)
        explained = explained.clamp(0.0, 1.0)
        coverage = coverage.clamp(0.0, 1.0)
        residual = torch.zeros_like(gap)
        residual[valid] = 1.0 - explained[valid]
        gap_rank = torch.zeros_like(gap)
        gap_rank[valid] = _ordinal_percentile_rank(gap[valid])
        coverage_rank = torch.zeros_like(gap)
        coverage_rank[valid] = _ordinal_percentile_rank(coverage[valid])
        priority = torch.zeros_like(gap)
        priority[valid] = gap_rank[valid] * coverage_rank[valid] * residual[valid]
        raw_weight = torch.zeros_like(gap)
        raw_weight[valid] = 1.0 + float(alpha) * priority[valid]

        unweighted_mass = gap[valid].sum()
        weighted_mass = (raw_weight[valid] * gap[valid]).sum()
        if float(unweighted_mass) <= float(eps):
            loss_mass_scale = torch.ones((), device=gap.device, dtype=torch.float32)
            weight = torch.zeros_like(gap)
            weight[valid] = 1.0
        else:
            if not torch.isfinite(weighted_mass) or float(weighted_mass) <= 0.0:
                raise FloatingPointError(
                    f"Invalid counterfactual-teachability weighted KL mass: {float(weighted_mass)}"
                )
            loss_mass_scale = unweighted_mass / weighted_mass
            weight = torch.zeros_like(gap)
            weight[valid] = raw_weight[valid] * loss_mass_scale
        if not torch.allclose(
            (weight[valid] * gap[valid]).sum(),
            unweighted_mass,
            rtol=2e-6,
            atol=2e-7,
        ):
            raise FloatingPointError("Counterfactual-teachability KL-mass normalization failed.")

    return BudgetCounterfactualTeachabilityWeights(
        teacher_gap_b=gap.detach(),
        gradient_alignment=alignment.detach(),
        budget_explained_fraction=explained.detach(),
        budget_residual_fraction=residual.detach(),
        teacher_support_coverage=coverage.detach(),
        teacher_gap_rank=gap_rank.detach(),
        support_coverage_rank=coverage_rank.detach(),
        priority=priority.detach(),
        raw_weight=raw_weight.detach(),
        loss_mass_scale=loss_mass_scale.detach(),
        weight=weight.detach(),
        valid_mask=valid.detach(),
    )
