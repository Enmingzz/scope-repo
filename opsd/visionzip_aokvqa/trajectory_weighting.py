"""Detached trajectory-level weighting utilities for paired OPSD pilots."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math

import torch


@dataclass(frozen=True)
class TrajectoryRankWeights:
    priority: torch.Tensor
    raw_weight: torch.Tensor
    weight: torch.Tensor
    loss_mass_scale: torch.Tensor
    unweighted_loss_mass: torch.Tensor
    weighted_loss_mass: torch.Tensor


@dataclass(frozen=True)
class TrajectoryProbabilityWeights:
    sensitivity: torch.Tensor
    tau: torch.Tensor
    raw_weight: torch.Tensor
    probability_weight: torch.Tensor


@dataclass(frozen=True)
class DirectInverseTrajectoryProbabilityWeights:
    sensitivity: torch.Tensor
    raw_weight: torch.Tensor
    probability_weight: torch.Tensor


@dataclass(frozen=True)
class SoftmaxTrajectoryProbabilityWeights:
    sensitivity: torch.Tensor
    temperature: torch.Tensor
    raw_weight: torch.Tensor
    probability_weight: torch.Tensor


@dataclass(frozen=True)
class TrajectorySignalProbabilityWeights:
    """Detached probabilities induced by a higher-is-better signal."""

    signal: torch.Tensor
    temperature: torch.Tensor
    raw_weight: torch.Tensor
    probability_weight: torch.Tensor


@dataclass(frozen=True)
class GloballyCalibratedTrajectoryWeights:
    """Fixed-calibration trajectory weights without batch competition."""

    signal: torch.Tensor
    normalized_signal: torch.Tensor
    raw_weight: torch.Tensor
    objective_weight: torch.Tensor
    probability_weight: torch.Tensor


@dataclass(frozen=True)
class CompetenceFrontierProbabilityWeights:
    """Detached frontier weights with exact effective-batch KL-mass control."""

    signal: torch.Tensor
    tau: torch.Tensor
    raw_weight: torch.Tensor
    objective_weight: torch.Tensor
    probability_weight: torch.Tensor
    loss_mass_scale: torch.Tensor
    unweighted_loss: torch.Tensor
    weighted_loss: torch.Tensor


@dataclass(frozen=True)
class RatioGroupProbabilityWeights:
    """Detached trajectory probabilities induced by ratio-level signals."""

    signal: torch.Tensor
    retention_ratio: torch.Tensor
    group_signal: torch.Tensor
    raw_weight: torch.Tensor
    probability_weight: torch.Tensor


@dataclass(frozen=True)
class UniformTrajectoryProbabilityWeights:
    """Uniform detached coefficients used after adaptive ratio selection."""

    raw_weight: torch.Tensor
    probability_weight: torch.Tensor


@dataclass(frozen=True)
class HardTrajectoryPartitionWeights:
    signal: torch.Tensor
    top_mask: torch.Tensor
    selected_mask: torch.Tensor
    raw_weight: torch.Tensor
    probability_weight: torch.Tensor
    top_count: int


@dataclass(frozen=True)
class GlobalCurriculumTrajectoryWeights:
    signal: torch.Tensor
    clipped_signal: torch.Tensor
    raw_weight: torch.Tensor
    objective_weight: torch.Tensor
    probability_weight: torch.Tensor
    gamma: float


def uniform_trajectory_probability_weights(
    count: int,
    *,
    device: torch.device | str | None = None,
) -> UniformTrajectoryProbabilityWeights:
    """Return detached coefficients whose weighted sum is the vanilla mean."""

    count = int(count)
    if count <= 0:
        raise ValueError(f"count must be positive; got {count}.")
    probability = torch.full(
        (count,),
        1.0 / float(count),
        dtype=torch.float32,
        device=device,
    ).detach()
    raw_weight = torch.ones_like(probability).detach()
    return UniformTrajectoryProbabilityWeights(
        raw_weight=raw_weight,
        probability_weight=probability,
    )


def hard_trajectory_partition_weights(
    signal: torch.Tensor,
    *,
    top_fraction: float,
    batch_ordinal: int,
    select: str,
) -> HardTrajectoryPartitionWeights:
    """Give uniform loss mass to a top-F partition or its complement.

    With an effective batch of 32 and ``top_fraction=0.2``, the top count
    alternates deterministically between six and seven. Every five batches
    therefore select exactly 32 of 160 trajectories.
    """

    values = signal.detach().float().reshape(-1)
    if values.numel() < 2:
        raise ValueError("Trajectory partition requires at least two trajectories.")
    if not torch.isfinite(values).all():
        raise FloatingPointError("Trajectory partition signal must be finite.")
    fraction = float(top_fraction)
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError(f"top_fraction must be in (0, 1), got {top_fraction}.")
    ordinal = int(batch_ordinal)
    if ordinal < 0:
        raise ValueError(f"batch_ordinal must be nonnegative, got {batch_ordinal}.")
    selection = str(select).strip().lower()
    if selection not in {"top", "bottom"}:
        raise ValueError(f"select must be 'top' or 'bottom', got {select!r}.")

    count = int(values.numel())
    exact_count = fraction * count
    base_count = int(math.floor(exact_count))
    fractional_count = exact_count - base_count
    extra = int(math.floor((ordinal + 1) * fractional_count)) - int(
        math.floor(ordinal * fractional_count)
    )
    top_count = max(1, min(count - 1, base_count + extra))
    ranked = torch.argsort(values, descending=True, stable=True)
    top = torch.zeros(count, dtype=torch.bool, device=values.device)
    top[ranked[:top_count]] = True
    selected = top if selection == "top" else ~top
    selected_count = int(selected.sum())
    if selected_count <= 0:
        raise AssertionError("Trajectory partition produced an empty objective.")
    probability = selected.float() / float(selected_count)
    if not torch.allclose(
        probability.sum(),
        torch.ones((), dtype=torch.float32, device=probability.device),
        rtol=0.0,
        atol=1e-6,
    ):
        raise FloatingPointError("Trajectory partition probabilities do not sum to one.")
    return HardTrajectoryPartitionWeights(
        signal=values.detach(),
        top_mask=top.detach(),
        selected_mask=selected.detach(),
        raw_weight=selected.float().detach(),
        probability_weight=probability.detach(),
        top_count=top_count,
    )


def global_f_curriculum_trajectory_weights(
    signal: torch.Tensor,
    *,
    gamma: float = 4.0,
) -> GlobalCurriculumTrajectoryWeights:
    """Apply detached global weights ``gamma * F * (1 - F)`` without renormalizing."""

    values = signal.detach().float().reshape(-1)
    if values.numel() == 0:
        raise ValueError("Global F curriculum requires at least one trajectory.")
    if not torch.isfinite(values).all():
        raise FloatingPointError("Global F curriculum signal must be finite.")
    gamma = float(gamma)
    if not math.isfinite(gamma) or gamma <= 0.0:
        raise ValueError(f"gamma must be finite and positive, got {gamma}.")
    clipped = values.clamp(0.0, 1.0)
    objective = gamma * clipped * (1.0 - clipped)
    probability = objective / float(values.numel())
    return GlobalCurriculumTrajectoryWeights(
        signal=values.detach(),
        clipped_signal=clipped.detach(),
        raw_weight=objective.detach(),
        objective_weight=objective.detach(),
        probability_weight=probability.detach(),
        gamma=gamma,
    )


def globally_calibrated_trajectory_weights(
    signal: torch.Tensor,
    *,
    q05: float,
    q95: float,
    calibration_mean: float,
    coefficient: float = 1.0,
    eps: float = 1e-8,
) -> GloballyCalibratedTrajectoryWeights:
    """Apply a frozen robust affine map without normalizing the current batch.

    ``signal`` is mapped to ``[0, 1]`` using calibration-set quantiles. The
    fixed calibration mean centers the multiplier at one on the calibration
    distribution:

    ``w_i = 1 + coefficient * (normalized_i - calibration_mean)``.

    The executor consumes coefficients of a weighted sum, so
    ``probability_weight`` is ``w_i / N``. Its sum is deliberately allowed to
    differ from one after calibration; this preserves the requested objective
    ``mean_i(w_i * loss_i)`` rather than silently dividing by the current
    batch's mean weight.
    """

    values = signal.detach().float().reshape(-1)
    if values.numel() == 0:
        raise ValueError("Global-calibration signal must contain at least one value.")
    if not torch.isfinite(values).all():
        raise FloatingPointError("Global-calibration signal must be finite.")
    q05 = float(q05)
    q95 = float(q95)
    calibration_mean = float(calibration_mean)
    coefficient = float(coefficient)
    eps = float(eps)
    if not all(math.isfinite(value) for value in (q05, q95, calibration_mean, coefficient, eps)):
        raise ValueError("Global-calibration parameters must be finite.")
    if q95 <= q05:
        raise ValueError(f"Expected q95 > q05; got q05={q05}, q95={q95}.")
    if not 0.0 <= calibration_mean <= 1.0:
        raise ValueError(
            f"calibration_mean must be in [0, 1]; got {calibration_mean}."
        )
    if coefficient < 0.0:
        raise ValueError(f"coefficient must be nonnegative; got {coefficient}.")
    if eps <= 0.0:
        raise ValueError(f"eps must be positive; got {eps}.")

    normalized = ((values - q05) / (q95 - q05 + eps)).clamp(0.0, 1.0)
    objective_weight = 1.0 + coefficient * (normalized - calibration_mean)
    if not torch.isfinite(objective_weight).all():
        raise FloatingPointError("Global-calibration weights must be finite.")
    if bool((objective_weight <= 0.0).any()):
        raise ValueError(
            "Global-calibration weights must remain positive; reduce coefficient "
            f"or calibration_mean (minimum={float(objective_weight.min())})."
        )
    probability_weight = objective_weight / float(values.numel())
    return GloballyCalibratedTrajectoryWeights(
        signal=values.detach(),
        normalized_signal=normalized.detach(),
        raw_weight=objective_weight.detach(),
        objective_weight=objective_weight.detach(),
        probability_weight=probability_weight.detach(),
    )


def ratio_group_fraction_probability_weights(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    retention_ratio: torch.Tensor,
    *,
    transform: str = "linear",
    temperature: float | None = None,
    eps: float = 1e-8,
) -> RatioGroupProbabilityWeights:
    """Build probabilities from a ratio-level mass fraction.

    Each group's signal is ``clip(sum(numerator) / sum(denominator), 0, 1)``.
    This is used for the direct-sum root-JS projection fraction: numerator is
    teacher-directed native-budget projection mass and denominator is the
    original teacher-discrepancy mass.  Reducing before taking the ratio keeps
    near-zero-discrepancy trajectories from receiving equal votes.
    """

    numerators = numerator.detach().float().reshape(-1)
    denominators = denominator.detach().float().reshape(-1)
    ratios = retention_ratio.detach().float().reshape(-1)
    if (
        numerators.numel() == 0
        or numerators.shape != denominators.shape
        or numerators.shape != ratios.shape
    ):
        raise ValueError("Ratio-group fraction inputs must align and be non-empty.")
    if not all(torch.isfinite(values).all() for values in (numerators, denominators, ratios)):
        raise FloatingPointError("Ratio-group fraction inputs must be finite.")
    if (denominators < 0.0).any():
        raise ValueError("Ratio-group fraction denominators must be nonnegative.")
    eps = float(eps)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"eps must be finite and positive; got {eps}.")

    group_signal = torch.empty_like(numerators)
    for ratio in torch.unique(ratios, sorted=True):
        members = torch.isclose(ratios, ratio, rtol=0.0, atol=1e-6)
        signal = numerators[members].sum() / denominators[members].sum().clamp_min(eps)
        group_signal[members] = signal.clamp(0.0, 1.0)
    transform = str(transform).strip().lower()
    if transform == "linear":
        raw_weight = group_signal.clamp_min(eps)
    elif transform == "softmax":
        if temperature is None:
            raise ValueError("Ratio-group softmax weighting requires a temperature.")
        temperature = float(temperature)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError(
                f"Ratio-group softmax temperature must be finite and positive; got {temperature}."
            )
        logits = group_signal / temperature
        raw_weight = torch.exp(logits - logits.max())
    else:
        raise ValueError(
            f"Unknown ratio-group weight transform {transform!r}; expected 'linear' or 'softmax'."
        )
    probability_weight = raw_weight / raw_weight.sum().clamp_min(eps)
    return RatioGroupProbabilityWeights(
        signal=numerators.detach(),
        retention_ratio=ratios.detach(),
        group_signal=group_signal.detach(),
        raw_weight=raw_weight.detach(),
        probability_weight=probability_weight.detach(),
    )


def ratio_group_angle_probability_weights(
    projection_mass: torch.Tensor,
    teacher_js_mass: torch.Tensor,
    budget_js_mass: torch.Tensor,
    retention_ratio: torch.Tensor,
    *,
    temperature: float,
    eps: float = 1e-8,
) -> RatioGroupProbabilityWeights:
    """Apply an exact four-group softmax to the budget-alignment cosine.

    For each observed retention group, the score is
    ``sum(P) / sqrt(sum(A) * sum(B))``.  Softmax is applied once across
    groups, then each group's probability mass is divided uniformly among
    its trajectories.  Consequently, group counts cannot bias the ratio-level
    objective: each group receives exactly its softmax coefficient.
    """

    projection = projection_mass.detach().float().reshape(-1)
    teacher = teacher_js_mass.detach().float().reshape(-1)
    budget = budget_js_mass.detach().float().reshape(-1)
    ratios = retention_ratio.detach().float().reshape(-1)
    if (
        projection.numel() == 0
        or projection.shape != teacher.shape
        or projection.shape != budget.shape
        or projection.shape != ratios.shape
    ):
        raise ValueError("Ratio-group angle inputs must align and be non-empty.")
    if not all(torch.isfinite(values).all() for values in (projection, teacher, budget, ratios)):
        raise FloatingPointError("Ratio-group angle inputs must be finite.")
    if (teacher < 0.0).any() or (budget < 0.0).any():
        raise ValueError("Ratio-group angle JSD masses must be nonnegative.")
    eps = float(eps)
    temperature = float(temperature)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"eps must be finite and positive; got {eps}.")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(
            f"Ratio-group angle temperature must be finite and positive; got {temperature}."
        )

    unique_ratios = torch.unique(ratios, sorted=True)
    group_scores: list[torch.Tensor] = []
    group_members: list[torch.Tensor] = []
    group_signal = torch.empty_like(projection)
    for ratio in unique_ratios:
        members = torch.isclose(ratios, ratio, rtol=0.0, atol=1e-6)
        teacher_sum = teacher[members].sum()
        budget_sum = budget[members].sum()
        denominator = (teacher_sum * budget_sum).clamp_min(0.0).sqrt()
        score = torch.where(
            denominator > eps,
            projection[members].sum() / denominator.clamp_min(eps),
            torch.zeros_like(denominator),
        ).clamp(-1.0, 1.0)
        group_signal[members] = score
        group_scores.append(score)
        group_members.append(members)

    scores = torch.stack(group_scores)
    group_logits = scores / temperature
    group_raw_weight = torch.exp(group_logits - group_logits.max())
    group_probability = group_raw_weight / group_raw_weight.sum().clamp_min(eps)

    raw_weight = torch.empty_like(projection)
    probability_weight = torch.empty_like(projection)
    for raw, probability, members in zip(
        group_raw_weight, group_probability, group_members
    ):
        count = members.sum().to(dtype=torch.float32)
        raw_weight[members] = raw / count
        probability_weight[members] = probability / count

    if not torch.allclose(
        probability_weight.sum(),
        torch.ones((), dtype=torch.float32, device=probability_weight.device),
        rtol=1e-6,
        atol=1e-7,
    ):
        raise FloatingPointError("Ratio-group angle probabilities do not sum to one.")
    return RatioGroupProbabilityWeights(
        signal=projection.detach(),
        retention_ratio=ratios.detach(),
        group_signal=group_signal.detach(),
        raw_weight=raw_weight.detach(),
        probability_weight=probability_weight.detach(),
    )


def ratio_group_angle_sample_probability_weights(
    projection_mass: torch.Tensor,
    teacher_js_mass: torch.Tensor,
    budget_js_mass: torch.Tensor,
    retention_ratio: torch.Tensor,
    *,
    temperature: float,
    eps: float = 1e-8,
) -> RatioGroupProbabilityWeights:
    """Broadcast ratio-level angle scores, then normalize over trajectories.

    For each observed retention group, the score is
    ``sum(P) / sqrt(sum(A) * sum(B))``. The score is copied to every
    trajectory in that group before softmax normalization over the complete
    effective batch. Consequently, equal group scores recover the vanilla
    sample mean exactly, even when the sampled group counts are unequal.
    """

    projection = projection_mass.detach().float().reshape(-1)
    teacher = teacher_js_mass.detach().float().reshape(-1)
    budget = budget_js_mass.detach().float().reshape(-1)
    ratios = retention_ratio.detach().float().reshape(-1)
    if (
        projection.numel() == 0
        or projection.shape != teacher.shape
        or projection.shape != budget.shape
        or projection.shape != ratios.shape
    ):
        raise ValueError("Ratio-group angle inputs must align and be non-empty.")
    if not all(
        torch.isfinite(values).all()
        for values in (projection, teacher, budget, ratios)
    ):
        raise FloatingPointError("Ratio-group angle inputs must be finite.")
    if (teacher < 0.0).any() or (budget < 0.0).any():
        raise ValueError("Ratio-group angle JSD masses must be nonnegative.")
    eps = float(eps)
    temperature = float(temperature)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"eps must be finite and positive; got {eps}.")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(
            "Ratio-group angle temperature must be finite and positive; "
            f"got {temperature}."
        )

    group_signal = torch.empty_like(projection)
    for ratio in torch.unique(ratios, sorted=True):
        members = torch.isclose(ratios, ratio, rtol=0.0, atol=1e-6)
        teacher_sum = teacher[members].sum()
        budget_sum = budget[members].sum()
        denominator = (teacher_sum * budget_sum).clamp_min(0.0).sqrt()
        score = torch.where(
            denominator > eps,
            projection[members].sum() / denominator.clamp_min(eps),
            torch.zeros_like(denominator),
        ).clamp(-1.0, 1.0)
        group_signal[members] = score

    logits = group_signal / temperature
    raw_weight = torch.exp(logits - logits.max())
    probability_weight = raw_weight / raw_weight.sum().clamp_min(eps)
    if not torch.allclose(
        probability_weight.sum(),
        torch.ones((), dtype=torch.float32, device=probability_weight.device),
        rtol=1e-6,
        atol=1e-7,
    ):
        raise FloatingPointError(
            "Sample-normalized ratio-group angle probabilities do not sum to one."
        )
    return RatioGroupProbabilityWeights(
        signal=projection.detach(),
        retention_ratio=ratios.detach(),
        group_signal=group_signal.detach(),
        raw_weight=raw_weight.detach(),
        probability_weight=probability_weight.detach(),
    )


def ratio_group_projection_probability_weights(
    projection_mass: torch.Tensor,
    teacher_js_mass: torch.Tensor,
    retention_ratio: torch.Tensor,
    *,
    temperature: float,
    eps: float = 1e-8,
) -> RatioGroupProbabilityWeights:
    """Apply an exact ratio-group softmax to the projection fraction.

    The score for retention group ``b`` is ``sum(P_i) / sum(A_i)``. Softmax
    is applied once across observed groups, and each group coefficient is then
    divided uniformly among that group's trajectories. This implements
    ``sum_b alpha_b * mean_{i in b}(loss_i)`` without group-count bias.
    """

    projection = projection_mass.detach().float().reshape(-1)
    teacher = teacher_js_mass.detach().float().reshape(-1)
    ratios = retention_ratio.detach().float().reshape(-1)
    if (
        projection.numel() == 0
        or projection.shape != teacher.shape
        or projection.shape != ratios.shape
    ):
        raise ValueError("Ratio-group projection inputs must align and be non-empty.")
    if not all(torch.isfinite(values).all() for values in (projection, teacher, ratios)):
        raise FloatingPointError("Ratio-group projection inputs must be finite.")
    if (teacher < 0.0).any():
        raise ValueError("Ratio-group projection teacher JSD masses must be nonnegative.")
    eps = float(eps)
    temperature = float(temperature)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"eps must be finite and positive; got {eps}.")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(
            f"Ratio-group projection temperature must be finite and positive; got {temperature}."
        )

    unique_ratios = torch.unique(ratios, sorted=True)
    group_scores: list[torch.Tensor] = []
    group_members: list[torch.Tensor] = []
    group_signal = torch.empty_like(projection)
    for ratio in unique_ratios:
        members = torch.isclose(ratios, ratio, rtol=0.0, atol=1e-6)
        teacher_sum = teacher[members].sum()
        score = torch.where(
            teacher_sum > eps,
            projection[members].sum() / teacher_sum.clamp_min(eps),
            torch.zeros_like(teacher_sum),
        ).clamp(0.0, 1.0)
        group_signal[members] = score
        group_scores.append(score)
        group_members.append(members)

    scores = torch.stack(group_scores)
    group_logits = scores / temperature
    group_raw_weight = torch.exp(group_logits - group_logits.max())
    group_probability = group_raw_weight / group_raw_weight.sum().clamp_min(eps)

    raw_weight = torch.empty_like(projection)
    probability_weight = torch.empty_like(projection)
    for raw, probability, members in zip(
        group_raw_weight, group_probability, group_members
    ):
        count = members.sum().to(dtype=torch.float32)
        raw_weight[members] = raw / count
        probability_weight[members] = probability / count

    if not torch.allclose(
        probability_weight.sum(),
        torch.ones((), dtype=torch.float32, device=probability_weight.device),
        rtol=1e-6,
        atol=1e-7,
    ):
        raise FloatingPointError("Ratio-group projection probabilities do not sum to one.")
    return RatioGroupProbabilityWeights(
        signal=projection.detach(),
        retention_ratio=ratios.detach(),
        group_signal=group_signal.detach(),
        raw_weight=raw_weight.detach(),
        probability_weight=probability_weight.detach(),
    )


def softmax_trajectory_signal_probability_weights(
    signal: torch.Tensor,
    *,
    temperature: float,
) -> TrajectorySignalProbabilityWeights:
    """Allocate loss mass with ``softmax(signal / temperature)``.

    The input is detached before FP32 normalization. A constant signal exactly
    recovers uniform effective-batch weighting.
    """

    values = signal.detach().float().reshape(-1)
    if values.numel() == 0:
        raise ValueError("Trajectory signal must contain at least one value.")
    if not torch.isfinite(values).all():
        raise FloatingPointError("Trajectory signal must be finite.")
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(
            f"Softmax trajectory temperature must be finite and positive; got {temperature}."
        )

    logits = values / temperature
    raw_weight = torch.exp(logits - logits.max())
    probability_weight = raw_weight / raw_weight.sum()
    return TrajectorySignalProbabilityWeights(
        signal=values,
        temperature=torch.tensor(temperature, dtype=torch.float32).detach(),
        raw_weight=raw_weight.detach(),
        probability_weight=probability_weight.detach(),
    )


def competence_frontier_probability_weights(
    losses: torch.Tensor,
    signal: torch.Tensor,
    *,
    tau: float,
    eps: float = 1e-8,
) -> CompetenceFrontierProbabilityWeights:
    """Focus on intermediate robust need while preserving scalar OPSD loss.

    The bounded frontier ``4*tau*x/(tau+x)^2`` is one at ``x=tau`` and tends
    to zero for solved and currently too-hard trajectories. A detached scalar
    rescales the frontier so ``sum(w_i * loss_i) == sum(loss_i)``. The
    returned probability coefficients are ``w_i / N`` because the effective
    batch executor expects coefficients whose weighted sum is the batch mean.
    """

    values = signal.detach().float().reshape(-1)
    detached_losses = losses.detach().float().reshape(-1)
    if values.numel() == 0 or values.shape != detached_losses.shape:
        raise ValueError("Frontier signals and losses must align and be non-empty.")
    if not torch.isfinite(values).all() or bool((values < 0.0).any()):
        raise FloatingPointError("Frontier signals must be finite and nonnegative.")
    if not torch.isfinite(detached_losses).all() or bool((detached_losses < 0.0).any()):
        raise FloatingPointError("Frontier losses must be finite and nonnegative.")
    tau = float(tau)
    eps = float(eps)
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError(f"tau must be finite and positive; got {tau}.")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"eps must be finite and positive; got {eps}.")

    tau_tensor = torch.tensor(tau, dtype=torch.float32, device=values.device)
    raw_weight = (
        4.0
        * tau_tensor
        * values
        / (tau_tensor + values).square().clamp_min(eps)
    ).clamp(0.0, 1.0)
    unweighted_mass = detached_losses.sum()
    raw_weighted_mass = (raw_weight * detached_losses).sum()
    if float(unweighted_mass) <= eps:
        loss_mass_scale = torch.ones((), dtype=torch.float32, device=values.device)
        objective_weight = torch.ones_like(values)
    else:
        if not torch.isfinite(raw_weighted_mass) or float(raw_weighted_mass) <= eps:
            raise FloatingPointError(
                "Competence frontier assigned zero mass to a nonzero OPSD batch."
            )
        loss_mass_scale = unweighted_mass / raw_weighted_mass
        objective_weight = raw_weight * loss_mass_scale
    probability_weight = objective_weight / float(values.numel())
    unweighted_loss = detached_losses.mean()
    weighted_loss = (probability_weight * detached_losses).sum()
    if not torch.allclose(weighted_loss, unweighted_loss, rtol=2e-6, atol=2e-7):
        raise FloatingPointError(
            "Competence-frontier KL-mass normalization failed: "
            f"weighted={float(weighted_loss)}, unweighted={float(unweighted_loss)}."
        )
    return CompetenceFrontierProbabilityWeights(
        signal=values.detach(),
        tau=tau_tensor.detach(),
        raw_weight=raw_weight.detach(),
        objective_weight=objective_weight.detach(),
        probability_weight=probability_weight.detach(),
        loss_mass_scale=loss_mass_scale.detach(),
        unweighted_loss=unweighted_loss.detach(),
        weighted_loss=weighted_loss.detach(),
    )


def ratio_group_signal_probability_weights(
    signal: torch.Tensor,
    retention_ratio: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> RatioGroupProbabilityWeights:
    """Weight each trajectory by its retention group's mean positive signal.

    This deliberately discards within-ratio sample ranking.  It turns a noisy
    sample-level counterfactual into a ratio-level curriculum while preserving
    the data order and normalizing the effective-batch objective to a
    probability-weighted mean.
    """

    values = signal.detach().float().reshape(-1)
    ratios = retention_ratio.detach().float().reshape(-1)
    if values.numel() == 0 or values.shape != ratios.shape:
        raise ValueError("Ratio-group signals and retention ratios must align and be non-empty.")
    if not torch.isfinite(values).all() or not torch.isfinite(ratios).all():
        raise FloatingPointError("Ratio-group inputs must be finite.")
    if (values < 0.0).any():
        raise ValueError("Ratio-group signals must be nonnegative.")
    eps = float(eps)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"eps must be finite and positive; got {eps}.")

    group_signal = torch.empty_like(values)
    for ratio in torch.unique(ratios, sorted=True):
        members = torch.isclose(ratios, ratio, rtol=0.0, atol=1e-6)
        group_signal[members] = values[members].mean()
    raw_weight = group_signal.clamp_min(eps)
    probability_weight = raw_weight / raw_weight.sum().clamp_min(eps)
    return RatioGroupProbabilityWeights(
        signal=values.detach(),
        retention_ratio=ratios.detach(),
        group_signal=group_signal.detach(),
        raw_weight=raw_weight.detach(),
        probability_weight=probability_weight.detach(),
    )


def inverse_sensitivity_probability_weights(
    sensitivity: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> TrajectoryProbabilityWeights:
    """Turn detached trajectory sensitivity into probabilities summing to one.

    The synchronized-block median supplies a robust, scale-equivariant pivot.
    Lower sensitivity receives larger weight; no KL-mass correction is applied.
    """

    values = sensitivity.detach().float().reshape(-1)
    if values.numel() == 0:
        raise ValueError("Trajectory sensitivity must contain at least one value.")
    if not torch.isfinite(values).all():
        raise FloatingPointError("Trajectory sensitivity must be finite.")
    if (values < 0.0).any():
        raise ValueError("Trajectory sensitivity must be nonnegative.")
    eps = float(eps)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"eps must be finite and positive; got {eps}.")

    tau = values.median()
    raw_weight = (tau + eps) / (tau + values + eps)
    probability_weight = raw_weight / raw_weight.sum().clamp_min(eps)
    return TrajectoryProbabilityWeights(
        sensitivity=values,
        tau=tau.detach(),
        raw_weight=raw_weight.detach(),
        probability_weight=probability_weight.detach(),
    )


def direct_inverse_sensitivity_probability_weights(
    sensitivity: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> DirectInverseTrajectoryProbabilityWeights:
    """Normalize direct inverse-sensitivity weights over an effective batch.

    This variant intentionally has no median pivot or temperature. Lower
    sensitivity receives proportionally larger loss mass through
    ``1 / (sensitivity + eps)``. The returned probabilities sum to one, so a
    uniform signal exactly recovers the ordinary effective-batch mean.
    """

    values = sensitivity.detach().float().reshape(-1)
    if values.numel() == 0:
        raise ValueError("Trajectory sensitivity must contain at least one value.")
    if not torch.isfinite(values).all():
        raise FloatingPointError("Trajectory sensitivity must be finite.")
    if (values < 0.0).any():
        raise ValueError("Trajectory sensitivity must be nonnegative.")
    eps = float(eps)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"eps must be finite and positive; got {eps}.")

    raw_weight = values.add(eps).reciprocal()
    probability_weight = raw_weight / raw_weight.sum().clamp_min(eps)
    return DirectInverseTrajectoryProbabilityWeights(
        sensitivity=values,
        raw_weight=raw_weight.detach(),
        probability_weight=probability_weight.detach(),
    )


def softmax_inverse_sensitivity_probability_weights(
    sensitivity: torch.Tensor,
    *,
    temperature: float,
) -> SoftmaxTrajectoryProbabilityWeights:
    """Allocate trajectory loss mass with ``softmax(-sensitivity / T)``."""

    values = sensitivity.detach().float().reshape(-1)
    if values.numel() == 0:
        raise ValueError("Trajectory sensitivity must contain at least one value.")
    if not torch.isfinite(values).all():
        raise FloatingPointError("Trajectory sensitivity must be finite.")
    if (values < 0.0).any():
        raise ValueError("Trajectory sensitivity must be nonnegative.")
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(
            f"Softmax trajectory temperature must be finite and positive; got {temperature}."
        )

    logits = -values / temperature
    shifted_logits = logits - logits.max()
    raw_weight = shifted_logits.exp()
    probability_weight = raw_weight / raw_weight.sum()
    return SoftmaxTrajectoryProbabilityWeights(
        sensitivity=values,
        temperature=torch.tensor(temperature, dtype=torch.float32).detach(),
        raw_weight=raw_weight.detach(),
        probability_weight=probability_weight.detach(),
    )


def effective_batch_local_objective(
    loss: torch.Tensor,
    probability_weight: torch.Tensor | float,
    *,
    effective_batch_size: int,
) -> torch.Tensor:
    """Scale one local loss so DDP averaging and accumulation yield sum(w_i L_i)."""

    if loss.numel() != 1:
        raise ValueError("Each trajectory must contribute one scalar loss.")
    effective_batch_size = int(effective_batch_size)
    if effective_batch_size <= 0:
        raise ValueError("effective_batch_size must be positive.")
    weight = torch.as_tensor(
        probability_weight,
        dtype=loss.dtype,
        device=loss.device,
    ).detach()
    if weight.numel() != 1 or not torch.isfinite(weight) or float(weight) < 0.0:
        raise ValueError("A trajectory probability weight must be one finite nonnegative scalar.")
    return loss * weight * float(effective_batch_size)


@dataclass(frozen=True)
class RobustnessGatedCurriculumWeights:
    difficulty: torch.Tensor
    focus: torch.Tensor
    priority: torch.Tensor
    raw_weight: torch.Tensor
    mean_normalized_weight: torch.Tensor
    weight: torch.Tensor
    loss_mass_scale: torch.Tensor
    unweighted_loss_mass: torch.Tensor
    weighted_loss_mass: torch.Tensor


@dataclass(frozen=True)
class SensitivityFrontierWeights:
    ratio_gate: torch.Tensor
    local_robustness: torch.Tensor
    priority: torch.Tensor
    raw_weight: torch.Tensor
    mean_normalized_weight: torch.Tensor
    weight: torch.Tensor
    loss_mass_scale: torch.Tensor
    unweighted_loss_mass: torch.Tensor
    weighted_loss_mass: torch.Tensor


@dataclass
class ProgressAdaptiveFrontierState:
    """EMA state for a ratio-agnostic, progress-adaptive competence frontier."""

    initial_tau: float
    initial_robust_need_mean: float
    ema_robust_need_mean: float
    ema_half_life_trajectories: float = 256.0
    updates: int = 0
    trajectories_seen: int = 0

    def __post_init__(self) -> None:
        for name in (
            "initial_tau",
            "initial_robust_need_mean",
            "ema_robust_need_mean",
            "ema_half_life_trajectories",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive; got {value}.")
        if int(self.updates) < 0 or int(self.trajectories_seen) < 0:
            raise ValueError("Progress-adaptive frontier counters must be nonnegative.")

    @property
    def progress(self) -> float:
        return max(
            float(self.initial_robust_need_mean) - float(self.ema_robust_need_mean),
            0.0,
        )

    @property
    def tau(self) -> float:
        return float(self.initial_tau) + self.progress

    def update(self, robust_need_mean: float, trajectory_count: int) -> dict[str, float | int]:
        robust_need_mean = float(robust_need_mean)
        trajectory_count = int(trajectory_count)
        if not math.isfinite(robust_need_mean) or robust_need_mean < 0.0:
            raise ValueError(
                f"robust_need_mean must be finite and nonnegative; got {robust_need_mean}."
            )
        if trajectory_count <= 0:
            raise ValueError(f"trajectory_count must be positive; got {trajectory_count}.")
        tau_before = self.tau
        decay = math.exp(
            -math.log(2.0)
            * float(trajectory_count)
            / float(self.ema_half_life_trajectories)
        )
        self.ema_robust_need_mean = (
            decay * float(self.ema_robust_need_mean)
            + (1.0 - decay) * robust_need_mean
        )
        self.updates += 1
        self.trajectories_seen += trajectory_count
        return {
            "tau_before": tau_before,
            "tau_after": self.tau,
            "progress_after": self.progress,
            "ema_decay": decay,
            "ema_robust_need_mean": float(self.ema_robust_need_mean),
            "updates": int(self.updates),
            "trajectories_seen": int(self.trajectories_seen),
        }

    def state_dict(self) -> dict[str, float | int]:
        return {
            "initial_tau": float(self.initial_tau),
            "initial_robust_need_mean": float(self.initial_robust_need_mean),
            "ema_robust_need_mean": float(self.ema_robust_need_mean),
            "ema_half_life_trajectories": float(self.ema_half_life_trajectories),
            "updates": int(self.updates),
            "trajectories_seen": int(self.trajectories_seen),
        }

    def load_state_dict(self, state: dict[str, float | int]) -> None:
        for key in (
            "initial_tau",
            "initial_robust_need_mean",
            "ema_half_life_trajectories",
        ):
            if not math.isclose(
                float(state[key]), float(getattr(self, key)), rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    f"Progress-adaptive frontier resume changed {key}: "
                    f"checkpoint={state[key]} config={getattr(self, key)}."
                )
        self.ema_robust_need_mean = float(state["ema_robust_need_mean"])
        self.updates = int(state["updates"])
        self.trajectories_seen = int(state["trajectories_seen"])
        self.__post_init__()


@dataclass
class AdaptiveBudgetFrontierState:
    """Training-only state for adaptive retention-budget sampling.

    A balanced warmup estimates the initial robust-need distribution.  The
    state then tracks one EMA per retention ratio and samples the next budget
    from a bounded competence frontier.  It never changes the OPSD loss after
    a trajectory has been selected.
    """

    retention_ratios: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4)
    calibration_target_per_ratio: int = 64
    ema_half_life_per_ratio: float = 64.0
    seed: int = 42
    namespace: str = "adaptive_budget_frontier_v1"
    eps: float = 1e-8
    calibration_values: dict[str, list[float]] = field(default_factory=dict)
    initial_tau: float | None = None
    initial_metric_mean: float | None = None
    initial_group_metric: dict[str, float] = field(default_factory=dict)
    ema_group_metric: dict[str, float] = field(default_factory=dict)
    updates: int = 0
    trajectories_seen: int = 0
    calibration_complete_at_trajectory: int | None = None

    def __post_init__(self) -> None:
        ratios = tuple(float(value) for value in self.retention_ratios)
        if ratios != tuple(sorted(ratios)) or len(ratios) < 2 or len(set(ratios)) != len(ratios):
            raise ValueError("Adaptive budget ratios must be unique and sorted.")
        if any(not math.isfinite(value) or not 0.0 < value < 1.0 for value in ratios):
            raise ValueError(f"Invalid adaptive budget ratios: {ratios}.")
        self.retention_ratios = ratios
        if int(self.calibration_target_per_ratio) <= 0:
            raise ValueError("calibration_target_per_ratio must be positive.")
        if not math.isfinite(float(self.ema_half_life_per_ratio)) or float(
            self.ema_half_life_per_ratio
        ) <= 0.0:
            raise ValueError("ema_half_life_per_ratio must be finite and positive.")
        if not str(self.namespace):
            raise ValueError("namespace must be non-empty.")
        if not math.isfinite(float(self.eps)) or float(self.eps) <= 0.0:
            raise ValueError("eps must be finite and positive.")
        if int(self.updates) < 0 or int(self.trajectories_seen) < 0:
            raise ValueError("Adaptive budget counters must be nonnegative.")

        keys = {self._key(ratio) for ratio in ratios}
        if not self.calibration_values:
            self.calibration_values = {key: [] for key in keys}
        if set(self.calibration_values) != keys:
            raise ValueError("Adaptive budget calibration has inconsistent ratio keys.")
        target = int(self.calibration_target_per_ratio)
        for key, values in self.calibration_values.items():
            if len(values) > target:
                raise ValueError(f"Too many calibration values for ratio {key}.")
            cleaned = [float(value) for value in values]
            if any(not math.isfinite(value) or value < 0.0 for value in cleaned):
                raise ValueError(f"Invalid calibration value for ratio {key}.")
            self.calibration_values[key] = cleaned

        initialized = self.initial_tau is not None
        expected_initialized = bool(self.initial_group_metric or self.ema_group_metric)
        if initialized != expected_initialized:
            raise ValueError("Adaptive budget scalar and group initialization disagree.")
        if initialized:
            if set(self.initial_group_metric) != keys or set(self.ema_group_metric) != keys:
                raise ValueError("Initialized adaptive budget state must contain every ratio.")
            for name in ("initial_tau", "initial_metric_mean"):
                value = float(getattr(self, name))
                if not math.isfinite(value) or value <= 0.0:
                    raise ValueError(f"{name} must be finite and positive; got {value}.")
            for mapping_name in ("initial_group_metric", "ema_group_metric"):
                mapping = getattr(self, mapping_name)
                for key, value in mapping.items():
                    value = float(value)
                    if not math.isfinite(value) or value < 0.0:
                        raise ValueError(f"{mapping_name}[{key}] is invalid.")
                    mapping[key] = value
        if self.ready != initialized:
            raise ValueError("Adaptive budget calibration completion is inconsistent.")

    @staticmethod
    def _key(ratio: float) -> str:
        return f"{float(ratio):.4f}"

    def _ratio_index(self, ratio: float) -> int:
        for index, expected in enumerate(self.retention_ratios):
            if math.isclose(float(ratio), expected, rel_tol=0.0, abs_tol=1e-6):
                return index
        raise ValueError(f"Unsupported adaptive retention ratio {ratio}.")

    @property
    def warmup_trajectories(self) -> int:
        return int(self.calibration_target_per_ratio) * len(self.retention_ratios)

    @property
    def ready(self) -> bool:
        target = int(self.calibration_target_per_ratio)
        return all(len(values) == target for values in self.calibration_values.values())

    @property
    def tau(self) -> float | None:
        if not self.ready:
            return None
        return float(self.initial_tau) + max(
            float(self.initial_metric_mean)
            - sum(self.ema_group_metric.values()) / len(self.ema_group_metric),
            0.0,
        )

    def probabilities(self) -> tuple[float, ...]:
        if not self.ready:
            uniform = 1.0 / len(self.retention_ratios)
            return tuple(uniform for _ in self.retention_ratios)
        tau = float(self.tau)
        scores = []
        for ratio in self.retention_ratios:
            value = max(float(self.ema_group_metric[self._key(ratio)]), 0.0)
            score = 4.0 * tau * value / max((tau + value) ** 2, float(self.eps))
            scores.append(max(score, float(self.eps)))
        total = sum(scores)
        return tuple(score / total for score in scores)

    def _stable_uint64(self, label: str, *parts: object) -> int:
        payload = "\0".join(
            [str(self.namespace), label, str(int(self.seed)), *(str(part) for part in parts)]
        ).encode("utf-8")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")

    def select_ratio(self, global_index: int, sample_id: str) -> float:
        global_index = int(global_index)
        if global_index < 0:
            raise ValueError("global_index must be nonnegative.")
        if global_index < self.warmup_trajectories:
            group = global_index // len(self.retention_ratios)
            offset = global_index % len(self.retention_ratios)
            order = sorted(
                range(len(self.retention_ratios)),
                key=lambda index: self._stable_uint64("warmup", group, index),
            )
            return float(self.retention_ratios[order[offset]])
        if not self.ready:
            raise RuntimeError(
                "Adaptive budget warmup indices are exhausted before calibration completed."
            )
        unit = self._stable_uint64("sample", global_index, sample_id) / float(2**64)
        cumulative = 0.0
        probabilities = self.probabilities()
        for ratio, probability in zip(self.retention_ratios, probabilities):
            cumulative += probability
            if unit < cumulative:
                return float(ratio)
        return float(self.retention_ratios[-1])

    def update(
        self,
        ratios: list[float] | tuple[float, ...],
        metric_values: list[float] | tuple[float, ...],
    ) -> dict[str, object]:
        if len(ratios) != len(metric_values) or not ratios:
            raise ValueError("Adaptive budget update requires paired non-empty values.")
        pairs: list[tuple[str, float]] = []
        for ratio, value in zip(ratios, metric_values):
            index = self._ratio_index(ratio)
            value = float(value)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"Adaptive budget metric must be nonnegative; got {value}.")
            pairs.append((self._key(self.retention_ratios[index]), value))

        probabilities_before = self.probabilities()
        tau_before = self.tau
        ready_before = self.ready
        target = int(self.calibration_target_per_ratio)
        if not ready_before:
            for position, (key, value) in enumerate(pairs):
                if len(self.calibration_values[key]) < target:
                    self.calibration_values[key].append(value)
                if self.ready and self.initial_tau is None:
                    flattened = [
                        item
                        for ratio_values in self.calibration_values.values()
                        for item in ratio_values
                    ]
                    ordered = sorted(flattened)
                    middle = len(ordered) // 2
                    median = (
                        ordered[middle]
                        if len(ordered) % 2
                        else 0.5 * (ordered[middle - 1] + ordered[middle])
                    )
                    self.initial_tau = max(
                        float(median),
                        float(self.eps),
                    )
                    self.initial_metric_mean = max(
                        sum(flattened) / len(flattened), float(self.eps)
                    )
                    self.initial_group_metric = {
                        current: sum(self.calibration_values[current]) / target
                        for current in sorted(self.calibration_values)
                    }
                    self.ema_group_metric = dict(self.initial_group_metric)
                    self.calibration_complete_at_trajectory = (
                        int(self.trajectories_seen) + position + 1
                    )
        else:
            grouped: dict[str, list[float]] = {
                self._key(ratio): [] for ratio in self.retention_ratios
            }
            for key, value in pairs:
                grouped[key].append(value)
            for key, values in grouped.items():
                if not values:
                    continue
                decay = math.exp(
                    -math.log(2.0)
                    * len(values)
                    / float(self.ema_half_life_per_ratio)
                )
                self.ema_group_metric[key] = (
                    decay * float(self.ema_group_metric[key])
                    + (1.0 - decay) * (sum(values) / len(values))
                )

        self.updates += 1
        self.trajectories_seen += len(pairs)
        return {
            "ready_before": ready_before,
            "ready_after": self.ready,
            "probabilities_before": list(probabilities_before),
            "probabilities_after": list(self.probabilities()),
            "tau_before": tau_before,
            "tau_after": self.tau,
            "initial_tau": self.initial_tau,
            "initial_metric_mean": self.initial_metric_mean,
            "initial_group_metric": dict(self.initial_group_metric),
            "ema_group_metric": dict(self.ema_group_metric),
            "calibration_counts": {
                key: len(values) for key, values in self.calibration_values.items()
            },
            "calibration_complete_at_trajectory": self.calibration_complete_at_trajectory,
            "updates": int(self.updates),
            "trajectories_seen": int(self.trajectories_seen),
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "retention_ratios": list(self.retention_ratios),
            "calibration_target_per_ratio": int(self.calibration_target_per_ratio),
            "ema_half_life_per_ratio": float(self.ema_half_life_per_ratio),
            "seed": int(self.seed),
            "namespace": str(self.namespace),
            "eps": float(self.eps),
            "calibration_values": {
                key: list(values) for key, values in self.calibration_values.items()
            },
            "initial_tau": self.initial_tau,
            "initial_metric_mean": self.initial_metric_mean,
            "initial_group_metric": dict(self.initial_group_metric),
            "ema_group_metric": dict(self.ema_group_metric),
            "updates": int(self.updates),
            "trajectories_seen": int(self.trajectories_seen),
            "calibration_complete_at_trajectory": self.calibration_complete_at_trajectory,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        expected = {
            "retention_ratios": list(self.retention_ratios),
            "calibration_target_per_ratio": int(self.calibration_target_per_ratio),
            "ema_half_life_per_ratio": float(self.ema_half_life_per_ratio),
            "seed": int(self.seed),
            "namespace": str(self.namespace),
            "eps": float(self.eps),
        }
        for key, current in expected.items():
            if state[key] != current:
                raise ValueError(
                    f"Adaptive budget resume changed {key}: checkpoint={state[key]} config={current}."
                )
        self.calibration_values = {
            str(key): [float(value) for value in values]
            for key, values in dict(state["calibration_values"]).items()
        }
        self.initial_tau = (
            float(state["initial_tau"]) if state.get("initial_tau") is not None else None
        )
        self.initial_metric_mean = (
            float(state["initial_metric_mean"])
            if state.get("initial_metric_mean") is not None
            else None
        )
        self.initial_group_metric = {
            str(key): float(value)
            for key, value in dict(state["initial_group_metric"]).items()
        }
        self.ema_group_metric = {
            str(key): float(value)
            for key, value in dict(state["ema_group_metric"]).items()
        }
        self.updates = int(state["updates"])
        self.trajectories_seen = int(state["trajectories_seen"])
        complete = state.get("calibration_complete_at_trajectory")
        self.calibration_complete_at_trajectory = (
            int(complete) if complete is not None else None
        )
        self.__post_init__()


@dataclass
class SensitivityFrontierState:
    """Online calibration and EMA state for a sensitivity-driven ratio frontier."""

    retention_ratios: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4)
    calibration_target_per_ratio: int = 64
    ema_half_life_trajectories: float = 256.0
    progress_drop_scale: float = 0.5
    progress_power: float = 2.0
    calibration_sums: dict[str, float] = field(default_factory=dict)
    calibration_counts: dict[str, int] = field(default_factory=dict)
    initial_sensitivity: dict[str, float] = field(default_factory=dict)
    ema_sensitivity: dict[str, float] = field(default_factory=dict)
    updates: int = 0
    trajectories_seen: int = 0
    calibration_complete_at_trajectory: int | None = None

    def __post_init__(self) -> None:
        ratios = tuple(float(value) for value in self.retention_ratios)
        if ratios != (0.1, 0.2, 0.3, 0.4):
            raise ValueError(f"Sensitivity frontier requires ratios (0.1, 0.2, 0.3, 0.4); got {ratios}.")
        self.retention_ratios = ratios
        if int(self.calibration_target_per_ratio) <= 0:
            raise ValueError("calibration_target_per_ratio must be positive.")
        for name in ("ema_half_life_trajectories", "progress_drop_scale", "progress_power"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive; got {value}.")
        if float(self.progress_drop_scale) > 1.0:
            raise ValueError("progress_drop_scale must be at most one.")
        if int(self.updates) < 0 or int(self.trajectories_seen) < 0:
            raise ValueError("Sensitivity frontier counters must be nonnegative.")
        keys = {self._key(ratio) for ratio in self.retention_ratios}
        if not self.calibration_sums:
            self.calibration_sums = {key: 0.0 for key in keys}
        if not self.calibration_counts:
            self.calibration_counts = {key: 0 for key in keys}
        if set(self.calibration_sums) != keys or set(self.calibration_counts) != keys:
            raise ValueError("Sensitivity frontier calibration state has inconsistent ratio keys.")
        target = int(self.calibration_target_per_ratio)
        for key in keys:
            value = float(self.calibration_sums[key])
            count = int(self.calibration_counts[key])
            if not math.isfinite(value) or value < 0.0 or not 0 <= count <= target:
                raise ValueError(f"Invalid calibration state for ratio {key}.")
            self.calibration_sums[key] = value
            self.calibration_counts[key] = count
        if self.initial_sensitivity or self.ema_sensitivity:
            if set(self.initial_sensitivity) != keys or set(self.ema_sensitivity) != keys:
                raise ValueError("Initialized sensitivity state must contain every ratio.")
            for mapping_name in ("initial_sensitivity", "ema_sensitivity"):
                mapping = getattr(self, mapping_name)
                for key, value in mapping.items():
                    value = float(value)
                    if not math.isfinite(value) or value <= 0.0:
                        raise ValueError(f"{mapping_name}[{key}] must be finite and positive.")
                    mapping[key] = value
        if self.ready != bool(self.initial_sensitivity and self.ema_sensitivity):
            raise ValueError("Sensitivity frontier calibration and initialized state disagree.")

    @staticmethod
    def _key(ratio: float) -> str:
        return f"{float(ratio):.2f}"

    def _validated_key(self, ratio: float) -> str:
        for expected in self.retention_ratios:
            # Ratios cross DDP as FP32 tensors, so accept their representational noise.
            if math.isclose(float(ratio), expected, rel_tol=0.0, abs_tol=1e-6):
                return self._key(expected)
        raise ValueError(f"Unsupported retention ratio {ratio}.")

    @property
    def ready(self) -> bool:
        target = int(self.calibration_target_per_ratio)
        return all(int(value) == target for value in self.calibration_counts.values())

    @property
    def unresolved_sensitivity(self) -> float:
        if not self.ready:
            return 1.0
        normalized = [
            float(self.ema_sensitivity[key]) / float(self.initial_sensitivity[key])
            for key in sorted(self.initial_sensitivity)
        ]
        return sum(normalized) / len(normalized)

    @property
    def progress(self) -> float:
        contraction = max(0.0, 1.0 - self.unresolved_sensitivity)
        normalized = min(contraction / float(self.progress_drop_scale), 1.0)
        return normalized ** float(self.progress_power)

    @property
    def frontier_index(self) -> float:
        return 3.0 * self.progress

    def ratio_gate(self, ratio: float) -> float:
        key = self._validated_key(ratio)
        descending = tuple(reversed(self.retention_ratios))
        index = [self._key(value) for value in descending].index(key)
        return max(0.0, 1.0 - abs(float(index) - self.frontier_index))

    def local_robustness(self, ratio: float, sensitivity: float, eps: float = 1e-8) -> float:
        if not self.ready:
            raise RuntimeError("Sensitivity frontier is not calibrated.")
        key = self._validated_key(ratio)
        sensitivity = float(sensitivity)
        if not math.isfinite(sensitivity) or sensitivity < 0.0:
            raise ValueError(f"Sensitivity must be finite and nonnegative; got {sensitivity}.")
        initial = float(self.initial_sensitivity[key])
        return initial / (initial + sensitivity + float(eps))

    def update(
        self,
        ratios: list[float] | tuple[float, ...],
        sensitivities: list[float] | tuple[float, ...],
    ) -> dict[str, object]:
        if len(ratios) != len(sensitivities) or not ratios:
            raise ValueError("Sensitivity frontier update requires paired non-empty values.")
        pairs: list[tuple[str, float]] = []
        for ratio, sensitivity in zip(ratios, sensitivities):
            key = self._validated_key(float(ratio))
            sensitivity = float(sensitivity)
            if not math.isfinite(sensitivity) or sensitivity < 0.0:
                raise ValueError(f"Sensitivity must be finite and nonnegative; got {sensitivity}.")
            pairs.append((key, sensitivity))

        ready_before = self.ready
        progress_before = self.progress
        frontier_before = self.frontier_index
        if not ready_before:
            target = int(self.calibration_target_per_ratio)
            for position, (key, sensitivity) in enumerate(pairs):
                if self.calibration_counts[key] < target:
                    self.calibration_sums[key] += sensitivity
                    self.calibration_counts[key] += 1
                if self.ready and not self.initial_sensitivity:
                    self.initial_sensitivity = {
                        current: self.calibration_sums[current] / target
                        for current in sorted(self.calibration_sums)
                    }
                    if any(value <= 0.0 for value in self.initial_sensitivity.values()):
                        raise FloatingPointError("Calibration produced a nonpositive ratio scale.")
                    self.ema_sensitivity = dict(self.initial_sensitivity)
                    self.calibration_complete_at_trajectory = (
                        int(self.trajectories_seen) + position + 1
                    )
        else:
            decay = math.exp(
                -math.log(2.0) / float(self.ema_half_life_trajectories)
            )
            for key, sensitivity in pairs:
                self.ema_sensitivity[key] = (
                    decay * float(self.ema_sensitivity[key])
                    + (1.0 - decay) * sensitivity
                )

        self.updates += 1
        self.trajectories_seen += len(pairs)
        return {
            "ready_before": ready_before,
            "ready_after": self.ready,
            "progress_before": progress_before,
            "progress_after": self.progress,
            "frontier_before": frontier_before,
            "frontier_after": self.frontier_index,
            "updates": int(self.updates),
            "trajectories_seen": int(self.trajectories_seen),
            "calibration_counts": dict(self.calibration_counts),
            "initial_sensitivity": dict(self.initial_sensitivity),
            "ema_sensitivity": dict(self.ema_sensitivity),
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "retention_ratios": list(self.retention_ratios),
            "calibration_target_per_ratio": int(self.calibration_target_per_ratio),
            "ema_half_life_trajectories": float(self.ema_half_life_trajectories),
            "progress_drop_scale": float(self.progress_drop_scale),
            "progress_power": float(self.progress_power),
            "calibration_sums": dict(self.calibration_sums),
            "calibration_counts": dict(self.calibration_counts),
            "initial_sensitivity": dict(self.initial_sensitivity),
            "ema_sensitivity": dict(self.ema_sensitivity),
            "updates": int(self.updates),
            "trajectories_seen": int(self.trajectories_seen),
            "calibration_complete_at_trajectory": self.calibration_complete_at_trajectory,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        for key in (
            "retention_ratios",
            "calibration_target_per_ratio",
            "ema_half_life_trajectories",
            "progress_drop_scale",
            "progress_power",
        ):
            current = list(self.retention_ratios) if key == "retention_ratios" else getattr(self, key)
            if state[key] != current:
                raise ValueError(
                    f"Sensitivity frontier resume changed {key}: checkpoint={state[key]} config={current}."
                )
        self.calibration_sums = {
            str(key): float(value) for key, value in dict(state["calibration_sums"]).items()
        }
        self.calibration_counts = {
            str(key): int(value) for key, value in dict(state["calibration_counts"]).items()
        }
        self.initial_sensitivity = {
            str(key): float(value) for key, value in dict(state["initial_sensitivity"]).items()
        }
        self.ema_sensitivity = {
            str(key): float(value) for key, value in dict(state["ema_sensitivity"]).items()
        }
        self.updates = int(state["updates"])
        self.trajectories_seen = int(state["trajectories_seen"])
        value = state.get("calibration_complete_at_trajectory")
        self.calibration_complete_at_trajectory = int(value) if value is not None else None
        self.__post_init__()


@dataclass
class RobustnessGatedCurriculumState:
    """Detached learning-progress state shared identically by every DDP rank."""

    initial_teacher_gap_mean: float
    ema_teacher_gap_mean: float
    ema_half_life_trajectories: float
    progress_power: float
    updates: int = 0
    trajectories_seen: int = 0

    def __post_init__(self) -> None:
        for name in ("initial_teacher_gap_mean", "ema_teacher_gap_mean"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive; got {value}.")
        if (
            not math.isfinite(float(self.ema_half_life_trajectories))
            or float(self.ema_half_life_trajectories) <= 0.0
        ):
            raise ValueError("ema_half_life_trajectories must be finite and positive.")
        if not math.isfinite(float(self.progress_power)) or float(self.progress_power) <= 0.0:
            raise ValueError("progress_power must be finite and positive.")
        if int(self.updates) < 0 or int(self.trajectories_seen) < 0:
            raise ValueError("Curriculum counters must be nonnegative.")

    @property
    def stage(self) -> float:
        ratio = float(self.ema_teacher_gap_mean) / float(self.initial_teacher_gap_mean)
        return min(max(1.0 - ratio ** float(self.progress_power), 0.0), 1.0)

    def update(self, teacher_gap_mean: float, trajectory_count: int) -> dict[str, float | int]:
        teacher_gap_mean = float(teacher_gap_mean)
        trajectory_count = int(trajectory_count)
        if not math.isfinite(teacher_gap_mean) or teacher_gap_mean < 0.0:
            raise ValueError(
                f"teacher_gap_mean must be finite and nonnegative; got {teacher_gap_mean}."
            )
        if trajectory_count <= 0:
            raise ValueError(f"trajectory_count must be positive; got {trajectory_count}.")
        stage_before = self.stage
        decay = math.exp(
            -math.log(2.0)
            * float(trajectory_count)
            / float(self.ema_half_life_trajectories)
        )
        self.ema_teacher_gap_mean = (
            decay * float(self.ema_teacher_gap_mean)
            + (1.0 - decay) * teacher_gap_mean
        )
        self.updates += 1
        self.trajectories_seen += trajectory_count
        return {
            "stage_before": stage_before,
            "stage_after": self.stage,
            "ema_decay": decay,
            "ema_teacher_gap_mean": float(self.ema_teacher_gap_mean),
            "updates": int(self.updates),
            "trajectories_seen": int(self.trajectories_seen),
        }

    def state_dict(self) -> dict[str, float | int]:
        return {
            "initial_teacher_gap_mean": float(self.initial_teacher_gap_mean),
            "ema_teacher_gap_mean": float(self.ema_teacher_gap_mean),
            "ema_half_life_trajectories": float(self.ema_half_life_trajectories),
            "progress_power": float(self.progress_power),
            "updates": int(self.updates),
            "trajectories_seen": int(self.trajectories_seen),
        }

    def load_state_dict(self, state: dict[str, float | int]) -> None:
        for key in (
            "initial_teacher_gap_mean",
            "ema_half_life_trajectories",
            "progress_power",
        ):
            if not math.isclose(
                float(state[key]), float(getattr(self, key)), rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    f"Curriculum resume state changed {key}: "
                    f"checkpoint={state[key]} config={getattr(self, key)}."
                )
        self.ema_teacher_gap_mean = float(state["ema_teacher_gap_mean"])
        self.updates = int(state["updates"])
        self.trajectories_seen = int(state["trajectories_seen"])
        self.__post_init__()


def teacher_gap_mass_robustness(
    teacher_gap: torch.Tensor,
    budget_sensitivity: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Calibrate adjacent-budget KL by tokenwise full-teacher correction mass."""

    teacher_gap = teacher_gap.detach().float().flatten()
    budget_sensitivity = budget_sensitivity.detach().float().flatten()
    valid_mask = valid_mask.detach().bool().flatten()
    if teacher_gap.shape != budget_sensitivity.shape or teacher_gap.shape != valid_mask.shape:
        raise ValueError("teacher_gap, budget_sensitivity, and valid_mask must have identical shapes.")
    if not bool(valid_mask.any()):
        raise ValueError("At least one generated-token position must be valid.")
    if not torch.isfinite(teacher_gap).all() or not torch.isfinite(budget_sensitivity).all():
        raise FloatingPointError("Robustness inputs must be finite.")
    numerical_tolerance = 1e-6
    if bool((teacher_gap < -numerical_tolerance).any()) or bool(
        (budget_sensitivity < -numerical_tolerance).any()
    ):
        raise ValueError("KL inputs must be nonnegative up to FP32 roundoff.")
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be positive; got {eps}.")

    with torch.no_grad():
        h = teacher_gap[valid_mask].clamp_min(0.0)
        d = budget_sensitivity[valid_mask].clamp_min(0.0)
        local_relative_sensitivity = d / (h + d + float(eps))
        robustness = 1.0 - (h * local_relative_sensitivity).sum() / h.sum().clamp_min(
            float(eps)
        )
    return robustness.clamp(0.0, 1.0).detach()


def residualized_budget_sensitivity(
    sensitivity: torch.Tensor,
    teacher_gap: torch.Tensor,
    ratios: torch.Tensor,
    *,
    ratio_intercepts: dict[str, float],
    ratio_log_teacher_gap_coefficients: dict[str, float],
    ratio_scales: dict[str, float],
    eps: float = 1e-8,
) -> torch.Tensor:
    """Remove teacher-gap scale and retention-ratio effects from log sensitivity."""

    sensitivity = sensitivity.detach().float().flatten()
    teacher_gap = teacher_gap.detach().float().flatten()
    ratios = ratios.detach().float().flatten()
    if sensitivity.shape != teacher_gap.shape or sensitivity.shape != ratios.shape:
        raise ValueError("sensitivity, teacher_gap, and ratios must have identical shapes.")
    if not torch.isfinite(sensitivity).all() or not torch.isfinite(teacher_gap).all():
        raise FloatingPointError("Residualized sensitivity inputs must be finite.")
    numerical_tolerance = 1e-6
    if bool((sensitivity < -numerical_tolerance).any()) or bool(
        (teacher_gap < -numerical_tolerance).any()
    ):
        raise ValueError("Sensitivity and teacher gap must be nonnegative up to FP32 tolerance.")
    sensitivity = sensitivity.clamp_min(0.0)
    teacher_gap = teacher_gap.clamp_min(0.0)

    output = torch.empty_like(sensitivity)
    for index, ratio in enumerate(ratios.tolist()):
        key = f"{float(ratio):.2f}"
        if (
            key not in ratio_intercepts
            or key not in ratio_log_teacher_gap_coefficients
            or key not in ratio_scales
        ):
            raise KeyError(f"Missing residual calibration for retention ratio {key}.")
        scale = float(ratio_scales[key])
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"Residual scale for ratio {key} must be positive; got {scale}.")
        expected = float(ratio_intercepts[key]) + float(
            ratio_log_teacher_gap_coefficients[key]
        ) * torch.log(teacher_gap[index].clamp_min(float(eps)))
        output[index] = (
            torch.log(sensitivity[index].clamp_min(float(eps))) - expected
        ) / scale
    return output.detach()


def average_rank_priority(values: torch.Tensor, *, higher_is_better: bool) -> torch.Tensor:
    """Map a finite one-dimensional signal to average ranks in ``[0, 1]``."""

    values = values.detach().float().flatten()
    if values.numel() == 0:
        raise ValueError("Trajectory ranking requires at least one value.")
    if not torch.isfinite(values).all():
        raise FloatingPointError("Trajectory ranking received a non-finite signal.")
    if values.numel() == 1:
        return torch.full_like(values, 0.5)

    sorted_values, order = torch.sort(values, stable=True)
    sorted_ranks = torch.empty_like(sorted_values)
    start = 0
    while start < sorted_values.numel():
        end = start + 1
        while end < sorted_values.numel() and bool(sorted_values[end] == sorted_values[start]):
            end += 1
        average_rank = 0.5 * float(start + end - 1)
        sorted_ranks[start:end] = average_rank
        start = end
    ranks = torch.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    priority = ranks / float(values.numel() - 1)
    return priority if higher_is_better else 1.0 - priority


def trajectory_rank_downweights(
    losses: torch.Tensor,
    signals: torch.Tensor,
    *,
    downweight_strength: float,
    higher_is_better: bool,
    eps: float = 1e-8,
) -> TrajectoryRankWeights:
    """Rank trajectories and preserve the detached aggregate OPSD loss mass.

    ``raw_weight`` only downweights: the highest-priority trajectory receives
    one and the lowest receives ``1 - downweight_strength``. A single detached
    scalar then restores the exact aggregate unweighted KL mass. This keeps the
    optimizer scale fixed while changing only relative trajectory gradients.
    """

    priority = average_rank_priority(signals, higher_is_better=higher_is_better)
    return trajectory_priority_downweights(
        losses,
        priority,
        downweight_strength=downweight_strength,
        eps=eps,
    )


def trajectory_sigmoid_downweights(
    losses: torch.Tensor,
    standardized_residuals: torch.Tensor,
    *,
    downweight_strength: float,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> TrajectoryRankWeights:
    """Map calibrated residuals to continuous robustness priorities."""

    standardized_residuals = standardized_residuals.detach().float().flatten()
    if not torch.isfinite(standardized_residuals).all():
        raise FloatingPointError("Residual robustness values must be finite.")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError(f"temperature must be positive; got {temperature}.")
    priority = torch.sigmoid(-standardized_residuals / float(temperature))
    return trajectory_priority_downweights(
        losses,
        priority,
        downweight_strength=downweight_strength,
        eps=eps,
    )


def robustness_gated_curriculum_weights(
    losses: torch.Tensor,
    robustness: torch.Tensor,
    teacher_gap: torch.Tensor,
    *,
    curriculum_stage: float,
    log_teacher_gap_center: float,
    log_teacher_gap_scale: float,
    weight_floor: float = 0.1,
    eps: float = 1e-8,
) -> RobustnessGatedCurriculumWeights:
    """Build detached easy-to-hard trajectory weights with exact KL-mass control."""

    losses = losses.detach().float().flatten()
    robustness = robustness.detach().float().flatten()
    teacher_gap = teacher_gap.detach().float().flatten()
    if losses.shape != robustness.shape or losses.shape != teacher_gap.shape:
        raise ValueError("losses, robustness, and teacher_gap must have identical shapes.")
    if losses.numel() == 0:
        raise ValueError("Curriculum weighting requires at least one trajectory.")
    if not torch.isfinite(losses).all() or bool((losses < 0.0).any()):
        raise FloatingPointError("Trajectory losses must be finite and nonnegative.")
    if not torch.isfinite(teacher_gap).all() or bool((teacher_gap < 0.0).any()):
        raise FloatingPointError("Teacher gaps must be finite and nonnegative.")
    if (
        not torch.isfinite(robustness).all()
        or bool((robustness < 0.0).any())
        or bool((robustness > 1.0).any())
    ):
        raise FloatingPointError("Trajectory robustness must be finite and in [0, 1].")
    if not 0.0 <= float(curriculum_stage) <= 1.0:
        raise ValueError(f"curriculum_stage must be in [0, 1]; got {curriculum_stage}.")
    if not math.isfinite(float(log_teacher_gap_center)):
        raise ValueError("log_teacher_gap_center must be finite.")
    if (
        not math.isfinite(float(log_teacher_gap_scale))
        or float(log_teacher_gap_scale) <= 0.0
    ):
        raise ValueError("log_teacher_gap_scale must be finite and positive.")
    if not 0.0 < float(weight_floor) <= 1.0:
        raise ValueError(f"weight_floor must be in (0, 1]; got {weight_floor}.")
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be positive; got {eps}.")

    with torch.no_grad():
        difficulty = torch.sigmoid(
            (torch.log(teacher_gap.clamp_min(float(eps))) - float(log_teacher_gap_center))
            / float(log_teacher_gap_scale)
        )
        stage = float(curriculum_stage)
        focus = (1.0 - stage) * (1.0 - difficulty) + stage * difficulty
        priority = (robustness * focus).clamp(0.0, 1.0)
        result = trajectory_priority_downweights(
            losses,
            priority,
            downweight_strength=1.0 - float(weight_floor),
            eps=eps,
        )
        mean_normalized_weight = result.raw_weight / result.raw_weight.mean().clamp_min(
            float(eps)
        )

    return RobustnessGatedCurriculumWeights(
        difficulty=difficulty.detach(),
        focus=focus.detach(),
        priority=result.priority.detach(),
        raw_weight=result.raw_weight.detach(),
        mean_normalized_weight=mean_normalized_weight.detach(),
        weight=result.weight.detach(),
        loss_mass_scale=result.loss_mass_scale.detach(),
        unweighted_loss_mass=result.unweighted_loss_mass.detach(),
        weighted_loss_mass=result.weighted_loss_mass.detach(),
    )


def sensitivity_frontier_weights(
    losses: torch.Tensor,
    sensitivities: torch.Tensor,
    ratios: torch.Tensor,
    state: SensitivityFrontierState,
    *,
    weight_floor: float = 0.1,
    eps: float = 1e-8,
) -> SensitivityFrontierWeights:
    """Weight robust trajectories near the current high-to-low ratio frontier."""

    losses = losses.detach().float().flatten()
    sensitivities = sensitivities.detach().float().flatten()
    ratios = ratios.detach().float().flatten()
    if losses.shape != sensitivities.shape or losses.shape != ratios.shape:
        raise ValueError("Losses, sensitivities, and ratios must have identical shapes.")
    if losses.numel() == 0:
        raise ValueError("Sensitivity frontier weighting requires at least one trajectory.")
    if not state.ready:
        raise RuntimeError("Sensitivity frontier must finish calibration before weighting.")
    if not torch.isfinite(losses).all() or bool((losses < 0.0).any()):
        raise FloatingPointError("Trajectory losses must be finite and nonnegative.")
    if not torch.isfinite(sensitivities).all() or bool((sensitivities < 0.0).any()):
        raise FloatingPointError("Trajectory sensitivities must be finite and nonnegative.")
    if not 0.0 < float(weight_floor) <= 1.0:
        raise ValueError(f"weight_floor must be in (0, 1]; got {weight_floor}.")
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be positive; got {eps}.")

    with torch.no_grad():
        ratio_gate = torch.tensor(
            [state.ratio_gate(float(value)) for value in ratios.cpu().tolist()],
            dtype=torch.float32,
            device=losses.device,
        )
        local_robustness = torch.tensor(
            [
                state.local_robustness(float(ratio), float(sensitivity), eps=eps)
                for ratio, sensitivity in zip(ratios.cpu().tolist(), sensitivities.cpu().tolist())
            ],
            dtype=torch.float32,
            device=losses.device,
        )
        priority = (ratio_gate * local_robustness).clamp(0.0, 1.0)
        result = trajectory_priority_downweights(
            losses,
            priority,
            downweight_strength=1.0 - float(weight_floor),
            eps=eps,
        )
        mean_normalized_weight = result.raw_weight / result.raw_weight.mean().clamp_min(
            float(eps)
        )

    return SensitivityFrontierWeights(
        ratio_gate=ratio_gate.detach(),
        local_robustness=local_robustness.detach(),
        priority=result.priority.detach(),
        raw_weight=result.raw_weight.detach(),
        mean_normalized_weight=mean_normalized_weight.detach(),
        weight=result.weight.detach(),
        loss_mass_scale=result.loss_mass_scale.detach(),
        unweighted_loss_mass=result.unweighted_loss_mass.detach(),
        weighted_loss_mass=result.weighted_loss_mass.detach(),
    )


def trajectory_priority_downweights(
    losses: torch.Tensor,
    priority: torch.Tensor,
    *,
    downweight_strength: float,
    eps: float = 1e-8,
) -> TrajectoryRankWeights:
    """Apply detached priorities while preserving aggregate OPSD loss mass."""

    if not 0.0 <= float(downweight_strength) < 1.0:
        raise ValueError(
            "downweight_strength must be in [0, 1); "
            f"got {downweight_strength}."
        )
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be positive; got {eps}.")

    losses = losses.detach().float().flatten()
    priority = priority.detach().float().flatten()
    if losses.shape != priority.shape:
        raise ValueError(
            f"losses and priorities must have the same shape; got {losses.shape} and {priority.shape}."
        )
    if losses.numel() == 0:
        raise ValueError("Trajectory weighting requires at least one loss.")
    if not torch.isfinite(losses).all() or bool((losses < 0).any()):
        raise FloatingPointError("Trajectory losses must be finite and nonnegative.")
    if not torch.isfinite(priority).all() or bool((priority < 0).any()) or bool((priority > 1).any()):
        raise FloatingPointError("Trajectory priorities must be finite and in [0, 1].")

    with torch.no_grad():
        raw_weight = 1.0 - float(downweight_strength) * (1.0 - priority)
        unweighted_mass = losses.sum()
        raw_weighted_mass = (raw_weight * losses).sum()
        if float(unweighted_mass) <= float(eps):
            scale = torch.ones((), dtype=torch.float32, device=losses.device)
            weight = torch.ones_like(losses)
        else:
            if not torch.isfinite(raw_weighted_mass) or float(raw_weighted_mass) <= 0.0:
                raise FloatingPointError(
                    f"Invalid raw weighted trajectory loss mass: {float(raw_weighted_mass)}."
                )
            scale = unweighted_mass / raw_weighted_mass
            weight = raw_weight * scale
        weighted_mass = (weight * losses).sum()
        if not torch.allclose(weighted_mass, unweighted_mass, rtol=2e-6, atol=2e-7):
            raise FloatingPointError(
                "Trajectory KL-mass normalization failed: "
                f"weighted={float(weighted_mass)}, unweighted={float(unweighted_mass)}."
            )

    return TrajectoryRankWeights(
        priority=priority.detach(),
        raw_weight=raw_weight.detach(),
        weight=weight.detach(),
        loss_mass_scale=scale.detach(),
        unweighted_loss_mass=unweighted_mass.detach(),
        weighted_loss_mass=weighted_mass.detach(),
    )
