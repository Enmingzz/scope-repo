from __future__ import annotations

from collections.abc import Mapping
import math

import torch
import torch.nn.functional as F


def _flatten_token_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 2:
        return logits
    if logits.ndim == 3:
        return logits.reshape(-1, logits.shape[-1])
    raise ValueError("Logits must be [T, vocab] or [B, T, vocab].")


def compute_forward_kl(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    temperature: float = 1.0,
    chunk_size: int = 32,
) -> torch.Tensor:
    if teacher_logits.shape != student_logits.shape:
        raise ValueError(f"Teacher/student logits must align by token index: {teacher_logits.shape} vs {student_logits.shape}")
    teacher_logits = _flatten_token_logits(teacher_logits)
    student_logits = _flatten_token_logits(student_logits)
    if int(teacher_logits.shape[0]) <= 0:
        raise ValueError("KL requires at least one token position.")
    temperature = float(temperature)
    chunk_size = max(1, int(chunk_size))
    losses = []
    counts = []
    for start in range(0, int(teacher_logits.shape[0]), chunk_size):
        end = min(start + chunk_size, int(teacher_logits.shape[0]))
        teacher_probs = F.softmax(teacher_logits[start:end].float() / temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits[start:end].float() / temperature, dim=-1)
        losses.append(F.kl_div(student_log_probs, teacher_probs, reduction="sum") * (temperature**2))
        counts.append(end - start)
    return torch.stack(losses).sum() / float(sum(counts))


def compute_per_token_kl(
    source_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    temperature: float = 1.0,
    chunk_size: int = 32,
) -> torch.Tensor:
    """Return KL(source || reference) independently at each token position.

    The vocabulary calculation is performed in FP32 and chunked over token
    positions.  Passing full-token teacher logits as ``source_logits`` and
    pruned-student logits as ``reference_logits`` matches the existing OPSD
    direction used by :func:`compute_forward_kl`.
    """

    if source_logits.shape != reference_logits.shape:
        raise ValueError(
            "Source/reference logits must align by token index: "
            f"{source_logits.shape} vs {reference_logits.shape}"
        )
    source_logits = _flatten_token_logits(source_logits)
    reference_logits = _flatten_token_logits(reference_logits)
    if int(source_logits.shape[0]) <= 0:
        raise ValueError("Per-token KL requires at least one token position.")
    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError(f"KL temperature must be positive, got {temperature}.")
    chunk_size = max(1, int(chunk_size))
    token_losses: list[torch.Tensor] = []
    for start in range(0, int(source_logits.shape[0]), chunk_size):
        end = min(start + chunk_size, int(source_logits.shape[0]))
        source_probs = F.softmax(source_logits[start:end].float() / temperature, dim=-1)
        reference_log_probs = F.log_softmax(reference_logits[start:end].float() / temperature, dim=-1)
        chunk = F.kl_div(reference_log_probs, source_probs, reduction="none").sum(dim=-1)
        token_losses.append(chunk * (temperature**2))
    return torch.cat(token_losses, dim=0)


def compute_per_token_bhattacharyya_affinity(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    temperature: float = 1.0,
    chunk_size: int = 32,
) -> torch.Tensor:
    """Return per-token teacher/student probability-support affinity.

    The Bhattacharyya coefficient is ``sum_v sqrt(q_v * p_v)`` and lies in
    ``[0, 1]``. Calculating it in log space avoids underflow when the teacher
    and student concentrate probability mass on nearly disjoint vocabulary
    regions.
    """

    if teacher_logits.shape != student_logits.shape:
        raise ValueError(
            "Teacher/student logits must align by token index: "
            f"{teacher_logits.shape} vs {student_logits.shape}"
        )
    teacher_logits = _flatten_token_logits(teacher_logits)
    student_logits = _flatten_token_logits(student_logits)
    if int(teacher_logits.shape[0]) <= 0:
        raise ValueError("Bhattacharyya affinity requires at least one token position.")
    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError(f"Affinity temperature must be positive, got {temperature}.")
    chunk_size = max(1, int(chunk_size))
    affinities: list[torch.Tensor] = []
    for start in range(0, int(teacher_logits.shape[0]), chunk_size):
        end = min(start + chunk_size, int(teacher_logits.shape[0]))
        teacher_log_probs = F.log_softmax(
            teacher_logits[start:end].float() / temperature, dim=-1
        )
        student_log_probs = F.log_softmax(
            student_logits[start:end].float() / temperature, dim=-1
        )
        log_affinity = torch.logsumexp(
            0.5 * (teacher_log_probs + student_log_probs), dim=-1
        )
        affinities.append(log_affinity.exp().clamp(0.0, 1.0))
    return torch.cat(affinities, dim=0)


def keep_mask_after_topk_exclusion(
    ranking_kl: torch.Tensor,
    valid_mask: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exclude the largest detached token KL values while retaining one token.

    Ranking is deterministic for ties: lower original token positions are
    removed first. The returned keep mask and removed indices are detached.
    """

    with torch.no_grad():
        values = ranking_kl.detach().float().reshape(-1)
        valid = valid_mask.detach().to(device=values.device, dtype=torch.bool).reshape(-1)
        if values.shape != valid.shape:
            raise ValueError(f"Ranking KL and valid mask must align: {values.shape} vs {valid.shape}.")
        if not valid.any():
            raise ValueError("Top-k token exclusion requires at least one valid token.")
        if not torch.isfinite(values[valid]).all():
            raise ValueError("Ranking KL must be finite on all valid tokens.")
        requested = int(top_k)
        if requested < 0:
            raise ValueError(f"top_k must be nonnegative, got {top_k}.")

        valid_indices = torch.nonzero(valid, as_tuple=False).reshape(-1)
        effective_k = min(requested, max(0, int(valid_indices.numel()) - 1))
        keep = valid.clone()
        if effective_k == 0:
            removed = valid_indices[:0]
        else:
            ranked_local = torch.argsort(
                values[valid_indices],
                descending=True,
                stable=True,
            )
            removed = valid_indices[ranked_local[:effective_k]]
            keep[removed] = False
        if not keep.any():
            raise AssertionError("Top-k token exclusion removed every valid token.")
    return keep.detach(), removed.detach()


def keep_mask_above_kl_floor(
    ranking_kl: torch.Tensor,
    valid_mask: torch.Tensor,
    min_kl: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep valid token positions whose detached KL is at least ``min_kl``.

    The comparison is inclusive so a threshold of ``1e-5`` removes exactly
    the positions with KL < 1e-5. Unlike top-k exclusion, an empty keep set is
    allowed and must be handled by the caller with a graph-connected zero.
    """

    with torch.no_grad():
        values = ranking_kl.detach().float().reshape(-1)
        valid = valid_mask.detach().to(device=values.device, dtype=torch.bool).reshape(-1)
        if values.shape != valid.shape:
            raise ValueError(f"Ranking KL and valid mask must align: {values.shape} vs {valid.shape}.")
        if not valid.any():
            raise ValueError("Token KL floor filtering requires at least one valid token.")
        if not torch.isfinite(values[valid]).all():
            raise ValueError("Ranking KL must be finite on all valid tokens.")
        threshold = float(min_kl)
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError(f"min_kl must be finite and nonnegative, got {min_kl}.")

        keep = valid & (values >= threshold)
        removed = torch.nonzero(valid & ~keep, as_tuple=False).reshape(-1)
    return keep.detach(), removed.detach()


def resolve_token_outlier_top_k(
    default_top_k: int,
    top_k_by_ratio: Mapping[object, object] | None,
    retention_ratio: float,
) -> int:
    """Resolve a nonnegative token-exclusion count for one retention ratio.

    A ratio map takes precedence over the legacy scalar value. Ratio keys may
    be YAML strings or numbers, but they must identify the requested ratio
    unambiguously.
    """

    default = int(default_top_k)
    if default < 0:
        raise ValueError(f"Default top_k must be nonnegative, got {default_top_k}.")
    if top_k_by_ratio is None:
        return default
    if not isinstance(top_k_by_ratio, Mapping) or not top_k_by_ratio:
        raise ValueError("top_k_by_ratio must be a nonempty mapping when provided.")

    target = float(retention_ratio)
    if not torch.isfinite(torch.tensor(target)) or not 0.0 < target <= 1.0:
        raise ValueError(f"retention_ratio must be in (0, 1], got {retention_ratio}.")

    matches: list[int] = []
    for raw_ratio, raw_top_k in top_k_by_ratio.items():
        try:
            ratio = float(raw_ratio)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid top_k_by_ratio key {raw_ratio!r}.") from error
        if not 0.0 < ratio <= 1.0:
            raise ValueError(f"top_k_by_ratio key must be in (0, 1], got {raw_ratio!r}.")
        if isinstance(raw_top_k, bool):
            raise ValueError(f"top_k_by_ratio value must be an integer, got {raw_top_k!r}.")
        try:
            top_k = int(raw_top_k)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid top_k_by_ratio value {raw_top_k!r}.") from error
        if float(top_k) != float(raw_top_k) or top_k < 0:
            raise ValueError(
                f"top_k_by_ratio value must be a nonnegative integer, got {raw_top_k!r}."
            )
        if abs(ratio - target) <= 1e-8:
            matches.append(top_k)

    if len(matches) != 1:
        raise ValueError(
            "top_k_by_ratio must contain exactly one entry for retention ratio "
            f"{target:.8f}; found {len(matches)}."
        )
    return matches[0]


def compute_budget_gradient_alignment(
    teacher_logits: torch.Tensor,
    b_plus_logits: torch.Tensor,
    student_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    chunk_size: int = 16,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Cosine alignment of full-teacher and adjacent-budget logit gradients.

    For forward KL, the gradient with respect to the deployed student's
    logits is proportional to ``p_b - q``.  The adjacent-budget bridge has
    gradient ``p_b - p_b_plus``.  Positive cosine means that descending the
    bridge objective is locally aligned with descending the original OPSD
    objective.  The returned diagnostic is fully detached.
    """

    if teacher_logits.shape != b_plus_logits.shape or teacher_logits.shape != student_logits.shape:
        raise ValueError(
            "Teacher, b_plus, and student logits must align: "
            f"{teacher_logits.shape}, {b_plus_logits.shape}, {student_logits.shape}."
        )
    teacher_logits = _flatten_token_logits(teacher_logits)
    b_plus_logits = _flatten_token_logits(b_plus_logits)
    student_logits = _flatten_token_logits(student_logits)
    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError(f"Alignment temperature must be positive, got {temperature}.")
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}.")

    alignments: list[torch.Tensor] = []
    chunk_size = max(1, int(chunk_size))
    with torch.no_grad():
        for start in range(0, int(teacher_logits.shape[0]), chunk_size):
            end = min(start + chunk_size, int(teacher_logits.shape[0]))
            teacher_probs = F.softmax(teacher_logits[start:end].float() / temperature, dim=-1)
            plus_probs = F.softmax(b_plus_logits[start:end].float() / temperature, dim=-1)
            student_probs = F.softmax(student_logits[start:end].detach().float() / temperature, dim=-1)
            teacher_gradient = student_probs - teacher_probs
            bridge_gradient = student_probs - plus_probs
            numerator = (teacher_gradient * bridge_gradient).sum(dim=-1)
            denominator = (
                teacher_gradient.square().sum(dim=-1).sqrt()
                * bridge_gradient.square().sum(dim=-1).sqrt()
            )
            alignment = torch.where(
                denominator > float(eps),
                numerator / denominator.clamp_min(float(eps)),
                torch.zeros_like(numerator),
            )
            alignments.append(alignment.clamp(-1.0, 1.0))
    return torch.cat(alignments, dim=0).detach()


def compute_budget_gradient_geometry(
    teacher_logits: torch.Tensor,
    b_plus_logits: torch.Tensor,
    student_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    chunk_size: int = 16,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Measure how much an adjacent-budget step explains the teacher gradient.

    For forward KL, let ``g_full = p_b - q`` and
    ``g_budget = p_b - p_b_plus``. The explained fraction is the positive
    finite-step projection ``clip(<g_full, g_budget> / ||g_full||^2, 0, 1)``.
    It is one when the native ``b_plus`` distribution closes the local
    teacher residual, and zero when the budget direction is null, orthogonal,
    or conflicting. The residual fraction is its complement. All outputs are
    detached diagnostics used to construct detached token weights.
    """

    if teacher_logits.shape != b_plus_logits.shape or teacher_logits.shape != student_logits.shape:
        raise ValueError(
            "Teacher, b_plus, and student logits must align: "
            f"{teacher_logits.shape}, {b_plus_logits.shape}, {student_logits.shape}."
        )
    teacher_logits = _flatten_token_logits(teacher_logits)
    b_plus_logits = _flatten_token_logits(b_plus_logits)
    student_logits = _flatten_token_logits(student_logits)
    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError(f"Geometry temperature must be positive, got {temperature}.")
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}.")

    alignments: list[torch.Tensor] = []
    explained_fractions: list[torch.Tensor] = []
    residual_fractions: list[torch.Tensor] = []
    chunk_size = max(1, int(chunk_size))
    with torch.no_grad():
        for start in range(0, int(teacher_logits.shape[0]), chunk_size):
            end = min(start + chunk_size, int(teacher_logits.shape[0]))
            teacher_probs = F.softmax(teacher_logits[start:end].float() / temperature, dim=-1)
            plus_probs = F.softmax(b_plus_logits[start:end].float() / temperature, dim=-1)
            student_probs = F.softmax(student_logits[start:end].detach().float() / temperature, dim=-1)
            teacher_gradient = student_probs - teacher_probs
            budget_gradient = student_probs - plus_probs
            dot = (teacher_gradient * budget_gradient).sum(dim=-1)
            teacher_norm_sq = teacher_gradient.square().sum(dim=-1)
            budget_norm_sq = budget_gradient.square().sum(dim=-1)
            cosine_denom = (teacher_norm_sq * budget_norm_sq).sqrt()
            alignment = torch.where(
                cosine_denom > float(eps),
                dot / cosine_denom.clamp_min(float(eps)),
                torch.zeros_like(dot),
            ).clamp(-1.0, 1.0)
            explained = torch.where(
                teacher_norm_sq > float(eps),
                dot / teacher_norm_sq.clamp_min(float(eps)),
                torch.zeros_like(dot),
            ).clamp(0.0, 1.0)
            alignments.append(alignment)
            explained_fractions.append(explained)
            residual_fractions.append(1.0 - explained)
    return (
        torch.cat(alignments, dim=0).detach(),
        torch.cat(explained_fractions, dim=0).detach(),
        torch.cat(residual_fractions, dim=0).detach(),
    )


def compute_budget_gradient_projection_geometry(
    teacher_logits: torch.Tensor,
    b_plus_logits: torch.Tensor,
    student_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    chunk_size: int = 16,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return detached geometry for projecting out the adjacent-budget direction.

    For forward KL, ``g_full = p_b - q`` and
    ``g_budget = p_b - p_b_plus`` are the logit-space gradients of the full
    teacher and adjacent-budget bridge objectives. The unbounded projection
    coefficient ``<g_full, g_budget> / ||g_budget||^2`` is the coefficient
    required to remove the component of ``g_full`` parallel to
    ``g_budget``. Clipping and causal gating are intentionally left to the
    caller so this function remains a pure diagnostic.
    """

    if teacher_logits.shape != b_plus_logits.shape or teacher_logits.shape != student_logits.shape:
        raise ValueError(
            "Teacher, b_plus, and student logits must align: "
            f"{teacher_logits.shape}, {b_plus_logits.shape}, {student_logits.shape}."
        )
    teacher_logits = _flatten_token_logits(teacher_logits)
    b_plus_logits = _flatten_token_logits(b_plus_logits)
    student_logits = _flatten_token_logits(student_logits)
    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError(f"Projection temperature must be positive, got {temperature}.")
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}.")

    alignments: list[torch.Tensor] = []
    projection_coefficients: list[torch.Tensor] = []
    teacher_norm_squares: list[torch.Tensor] = []
    budget_norm_squares: list[torch.Tensor] = []
    dot_products: list[torch.Tensor] = []
    chunk_size = max(1, int(chunk_size))
    with torch.no_grad():
        for start in range(0, int(teacher_logits.shape[0]), chunk_size):
            end = min(start + chunk_size, int(teacher_logits.shape[0]))
            teacher_probs = F.softmax(teacher_logits[start:end].float() / temperature, dim=-1)
            plus_probs = F.softmax(b_plus_logits[start:end].float() / temperature, dim=-1)
            student_probs = F.softmax(student_logits[start:end].detach().float() / temperature, dim=-1)
            teacher_gradient = student_probs - teacher_probs
            budget_gradient = student_probs - plus_probs
            dot = (teacher_gradient * budget_gradient).sum(dim=-1)
            teacher_norm_sq = teacher_gradient.square().sum(dim=-1)
            budget_norm_sq = budget_gradient.square().sum(dim=-1)
            cosine_denom = (teacher_norm_sq * budget_norm_sq).sqrt()
            alignment = torch.where(
                cosine_denom > float(eps),
                dot / cosine_denom.clamp_min(float(eps)),
                torch.zeros_like(dot),
            ).clamp(-1.0, 1.0)
            projection = torch.where(
                budget_norm_sq > float(eps),
                dot / budget_norm_sq.clamp_min(float(eps)),
                torch.zeros_like(dot),
            )
            alignments.append(alignment)
            projection_coefficients.append(projection)
            teacher_norm_squares.append(teacher_norm_sq)
            budget_norm_squares.append(budget_norm_sq)
            dot_products.append(dot)
    return (
        torch.cat(alignments, dim=0).detach(),
        torch.cat(projection_coefficients, dim=0).detach(),
        torch.cat(teacher_norm_squares, dim=0).detach(),
        torch.cat(budget_norm_squares, dim=0).detach(),
        torch.cat(dot_products, dim=0).detach(),
    )


def compute_teacher_gradient_budget_consensus(
    teacher_logits: torch.Tensor,
    b_plus_logits: torch.Tensor,
    student_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    chunk_size: int = 16,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Measure invariance of the full-teacher correction across two budgets.

    The deployed and probe correction directions are ``p_b - q`` and
    ``p_b_plus - q``. Their cosine tests directional agreement, while the
    norm ratio penalizes a correction whose magnitude collapses under the
    adjacent native VisionZip intervention. Both diagnostics are detached.
    """

    if teacher_logits.shape != b_plus_logits.shape or teacher_logits.shape != student_logits.shape:
        raise ValueError(
            "Teacher, b_plus, and student logits must align: "
            f"{teacher_logits.shape}, {b_plus_logits.shape}, {student_logits.shape}."
        )
    teacher_logits = _flatten_token_logits(teacher_logits)
    b_plus_logits = _flatten_token_logits(b_plus_logits)
    student_logits = _flatten_token_logits(student_logits)
    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError(f"Consensus temperature must be positive, got {temperature}.")
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}.")

    cosines: list[torch.Tensor] = []
    norm_ratios: list[torch.Tensor] = []
    chunk_size = max(1, int(chunk_size))
    with torch.no_grad():
        for start in range(0, int(teacher_logits.shape[0]), chunk_size):
            end = min(start + chunk_size, int(teacher_logits.shape[0]))
            teacher_probs = F.softmax(teacher_logits[start:end].float() / temperature, dim=-1)
            plus_probs = F.softmax(b_plus_logits[start:end].float() / temperature, dim=-1)
            student_probs = F.softmax(student_logits[start:end].detach().float() / temperature, dim=-1)
            deployed_gradient = student_probs - teacher_probs
            probe_gradient = plus_probs - teacher_probs
            deployed_norm = deployed_gradient.square().sum(dim=-1).sqrt()
            probe_norm = probe_gradient.square().sum(dim=-1).sqrt()
            denominator = deployed_norm * probe_norm
            cosine = torch.where(
                denominator > float(eps),
                (deployed_gradient * probe_gradient).sum(dim=-1)
                / denominator.clamp_min(float(eps)),
                torch.zeros_like(denominator),
            ).clamp(-1.0, 1.0)
            norm_ratio = torch.where(
                torch.maximum(deployed_norm, probe_norm) > float(eps),
                torch.minimum(deployed_norm, probe_norm)
                / torch.maximum(deployed_norm, probe_norm).clamp_min(float(eps)),
                torch.ones_like(deployed_norm),
            ).clamp(0.0, 1.0)
            cosines.append(cosine)
            norm_ratios.append(norm_ratio)
    return torch.cat(cosines).detach(), torch.cat(norm_ratios).detach()


def compute_teacher_mass_on_student_support(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    *,
    top_k: int = 32,
    temperature: float = 1.0,
    chunk_size: int = 16,
) -> torch.Tensor:
    """Teacher probability mass covered by the student's local top-k support."""

    if teacher_logits.shape != student_logits.shape:
        raise ValueError(
            "Teacher/student logits must align: "
            f"{teacher_logits.shape} vs {student_logits.shape}."
        )
    teacher_logits = _flatten_token_logits(teacher_logits)
    student_logits = _flatten_token_logits(student_logits)
    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError(f"Support temperature must be positive, got {temperature}.")
    top_k = int(top_k)
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}.")
    top_k = min(top_k, int(student_logits.shape[-1]))
    chunk_size = max(1, int(chunk_size))
    coverage: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, int(teacher_logits.shape[0]), chunk_size):
            end = min(start + chunk_size, int(teacher_logits.shape[0]))
            student_indices = torch.topk(
                student_logits[start:end].detach().float(),
                k=top_k,
                dim=-1,
            ).indices
            teacher_probs = F.softmax(
                teacher_logits[start:end].float() / temperature,
                dim=-1,
            )
            coverage.append(torch.gather(teacher_probs, dim=-1, index=student_indices).sum(dim=-1))
    return torch.cat(coverage, dim=0).clamp(0.0, 1.0).detach()


def compute_budget_contrastive_per_token_kl(
    teacher_logits: torch.Tensor,
    b_plus_logits: torch.Tensor,
    student_logits: torch.Tensor,
    shaping_strength: torch.Tensor,
    *,
    advantage_clip: float = 2.0,
    temperature: float = 1.0,
    chunk_size: int = 16,
    return_target_shift: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Distill a full teacher shaped by an adjacent-budget counterfactual.

    The detached vocabulary advantage ``log p_b_plus - log p_b`` identifies
    directions supported by adding native visual tokens. It is centered under
    the full-teacher distribution, clipped, and applied only through the
    detached per-position ``shaping_strength`` gate. Gradients flow solely
    through ``student_logits``.
    """

    if teacher_logits.shape != b_plus_logits.shape or teacher_logits.shape != student_logits.shape:
        raise ValueError(
            "Teacher, b_plus, and student logits must align: "
            f"{teacher_logits.shape}, {b_plus_logits.shape}, {student_logits.shape}."
        )
    teacher_logits = _flatten_token_logits(teacher_logits)
    b_plus_logits = _flatten_token_logits(b_plus_logits)
    student_logits = _flatten_token_logits(student_logits)
    strength = shaping_strength.detach().float().reshape(-1)
    if strength.shape[0] != teacher_logits.shape[0]:
        raise ValueError(
            f"Shaping strength must have one value per token: {strength.shape} vs {teacher_logits.shape}."
        )
    if not torch.isfinite(strength).all() or (strength < 0.0).any():
        raise ValueError("Shaping strength must be finite and nonnegative.")
    if float(advantage_clip) <= 0.0:
        raise ValueError(f"advantage_clip must be positive, got {advantage_clip}.")
    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError(f"KL temperature must be positive, got {temperature}.")

    token_losses: list[torch.Tensor] = []
    target_shifts: list[torch.Tensor] = []
    chunk_size = max(1, int(chunk_size))
    for start in range(0, int(teacher_logits.shape[0]), chunk_size):
        end = min(start + chunk_size, int(teacher_logits.shape[0]))
        with torch.no_grad():
            teacher_log_probs = F.log_softmax(teacher_logits[start:end].float() / temperature, dim=-1)
            teacher_probs = teacher_log_probs.exp()
            plus_log_probs = F.log_softmax(b_plus_logits[start:end].float() / temperature, dim=-1)
            base_log_probs = F.log_softmax(student_logits[start:end].detach().float() / temperature, dim=-1)
            advantage = plus_log_probs - base_log_probs
            advantage = advantage - (teacher_probs * advantage).sum(dim=-1, keepdim=True)
            advantage = advantage.clamp(min=-float(advantage_clip), max=float(advantage_clip))
            shaped_logits = teacher_log_probs + strength[start:end, None] * advantage
            shaped_probs = F.softmax(shaped_logits, dim=-1).detach()
            if return_target_shift:
                target_shifts.append(
                    F.kl_div(
                        teacher_log_probs,
                        shaped_probs,
                        reduction="none",
                    ).sum(dim=-1)
                    * (temperature**2)
                )
        student_log_probs = F.log_softmax(student_logits[start:end].float() / temperature, dim=-1)
        token_losses.append(
            F.kl_div(student_log_probs, shaped_probs, reduction="none").sum(dim=-1) * (temperature**2)
        )
    losses = torch.cat(token_losses, dim=0)
    if not return_target_shift:
        return losses
    return losses, torch.cat(target_shifts, dim=0).detach()


def _generalized_jsd_token_chunk(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    beta: float,
    temperature: float,
    top_k: int | None,
    token_clip: float | None,
    clip_mode: str,
) -> torch.Tensor:
    teacher_logits = teacher_logits.float() / temperature
    student_logits = student_logits.float() / temperature

    if top_k is not None and int(top_k) > 0:
        k = min(int(top_k), int(teacher_logits.shape[-1]))
        _, top_k_indices = torch.topk(teacher_logits, k=k, dim=-1)
        teacher_logits = torch.gather(teacher_logits, dim=-1, index=top_k_indices)
        student_logits = torch.gather(student_logits, dim=-1, index=top_k_indices)

    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    student_log_probs = F.log_softmax(student_logits, dim=-1)

    if beta == 0.0:
        jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
    elif beta == 1.0:
        jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
    else:
        beta_t = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
        mixture_log_probs = torch.logsumexp(
            torch.stack(
                [
                    student_log_probs + torch.log1p(-beta_t),
                    teacher_log_probs + torch.log(beta_t),
                ]
            ),
            dim=0,
        )
        kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
        kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
        jsd = beta_t * kl_teacher + (1.0 - beta_t) * kl_student

    normalized_clip_mode = str(clip_mode).lower()
    if token_clip is not None and float(token_clip) > 0.0 and normalized_clip_mode in {"official", "vocab", "vocab_element"}:
        jsd = jsd.clamp(max=float(token_clip))
        return jsd.sum(dim=-1)

    token_loss = jsd.sum(dim=-1).clamp_min(0.0)
    if token_clip is not None and float(token_clip) > 0.0 and normalized_clip_mode in {"token", "token_sum"}:
        token_loss = token_loss.clamp(max=float(token_clip))
    elif normalized_clip_mode not in {"token", "token_sum", "official", "vocab", "vocab_element"}:
        raise ValueError(f"Unsupported JSD clip_mode={clip_mode!r}. Use 'token' or 'official'.")
    return token_loss


def _generalized_jsd_chunk(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    beta: float,
    temperature: float,
    top_k: int | None,
    token_clip: float | None,
    clip_mode: str,
) -> torch.Tensor:
    return _generalized_jsd_token_chunk(
        teacher_logits,
        student_logits,
        beta=beta,
        temperature=temperature,
        top_k=top_k,
        token_clip=token_clip,
        clip_mode=clip_mode,
    ).mean()


def compute_generalized_jsd(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    beta: float = 0.0,
    temperature: float = 1.0,
    top_k: int | None = None,
    token_clip: float | None = None,
    clip_mode: str = "token",
    chunk_size: int = 32,
) -> torch.Tensor:
    """Official OPSD-style generalized JSD over generated token positions.

    With beta=0 this is forward KL from teacher to student, matching the main
    OPSD setting.  With beta=1 it becomes reverse KL.  Intermediate beta values
    use the generalized Jensen-Shannon mixture from the official trainer.
    """

    if teacher_logits.shape != student_logits.shape:
        raise ValueError(f"Teacher/student logits must align: {teacher_logits.shape} vs {student_logits.shape}")
    if teacher_logits.ndim not in {2, 3}:
        raise ValueError("Generalized JSD logits must be [T, vocab] or [B, T, vocab].")
    if int(teacher_logits.shape[-2]) <= 0:
        raise ValueError("Generalized JSD requires at least one token position.")

    temperature = float(temperature)
    beta = float(beta)
    if beta < 0.0 or beta > 1.0:
        raise ValueError(f"OPSD beta must be in [0, 1], got {beta}.")

    teacher_logits = _flatten_token_logits(teacher_logits)
    student_logits = _flatten_token_logits(student_logits)
    chunk_size = max(1, int(chunk_size))
    losses = []
    counts = []
    for start in range(0, int(teacher_logits.shape[0]), chunk_size):
        end = min(start + chunk_size, int(teacher_logits.shape[0]))
        losses.append(
            _generalized_jsd_chunk(
                teacher_logits[start:end],
                student_logits[start:end],
                beta=beta,
                temperature=temperature,
                top_k=top_k,
                token_clip=token_clip,
                clip_mode=clip_mode,
            )
        )
        counts.append(end - start)
    return torch.stack([loss * count for loss, count in zip(losses, counts)]).sum() / float(sum(counts))


def compute_per_token_generalized_jsd(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    beta: float = 0.5,
    temperature: float = 1.0,
    top_k: int | None = None,
    token_clip: float | None = None,
    clip_mode: str = "token",
    chunk_size: int = 32,
) -> torch.Tensor:
    """Return generalized JSD independently at each generated-token position."""

    if teacher_logits.shape != student_logits.shape:
        raise ValueError(
            f"Teacher/student logits must align: {teacher_logits.shape} vs {student_logits.shape}"
        )
    if teacher_logits.ndim not in {2, 3}:
        raise ValueError("Generalized JSD logits must be [T, vocab] or [B, T, vocab].")
    if int(teacher_logits.shape[-2]) <= 0:
        raise ValueError("Generalized JSD requires at least one token position.")

    temperature = float(temperature)
    beta = float(beta)
    if temperature <= 0.0:
        raise ValueError(f"JSD temperature must be positive, got {temperature}.")
    if beta < 0.0 or beta > 1.0:
        raise ValueError(f"OPSD beta must be in [0, 1], got {beta}.")

    teacher_logits = _flatten_token_logits(teacher_logits)
    student_logits = _flatten_token_logits(student_logits)
    chunk_size = max(1, int(chunk_size))
    token_losses: list[torch.Tensor] = []
    for start in range(0, int(teacher_logits.shape[0]), chunk_size):
        end = min(start + chunk_size, int(teacher_logits.shape[0]))
        token_losses.append(
            _generalized_jsd_token_chunk(
                teacher_logits[start:end],
                student_logits[start:end],
                beta=beta,
                temperature=temperature,
                top_k=top_k,
                token_clip=token_clip,
                clip_mode=clip_mode,
            )
        )
    return torch.cat(token_losses, dim=0)


def compute_token_ce(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError("CE logits must be [T, vocab].")
    target_ids = target_ids.to(device=logits.device, dtype=torch.long).reshape(-1)
    if int(target_ids.numel()) != int(logits.shape[0]):
        raise ValueError(f"CE target length {int(target_ids.numel())} != logits length {int(logits.shape[0])}.")
    return F.cross_entropy(logits.float(), target_ids)


def compute_sequence_logprob(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    token_ids = token_ids.to(device=logits.device, dtype=torch.long).reshape(-1)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return log_probs.gather(dim=-1, index=token_ids[:, None]).squeeze(-1).sum()


def grpo_policy_loss(sequence_logprobs: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
    if sequence_logprobs.shape != advantages.shape:
        raise ValueError("GRPO sequence_logprobs and advantages must have matching shape.")
    return -(sequence_logprobs.float() * advantages.float().detach()).mean()
