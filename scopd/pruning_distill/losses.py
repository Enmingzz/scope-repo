from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _validate_answer_logits(teacher_logits_answer: torch.Tensor, student_logits_answer: torch.Tensor) -> None:
    if teacher_logits_answer.ndim != 2 or student_logits_answer.ndim != 2:
        raise ValueError("Answer logits must have shape [T, vocab].")
    if teacher_logits_answer.shape != student_logits_answer.shape:
        raise ValueError(
            f"Teacher/student answer logits must align by answer-token index, got "
            f"{tuple(teacher_logits_answer.shape)} vs {tuple(student_logits_answer.shape)}."
        )
    if int(teacher_logits_answer.shape[0]) == 0:
        raise ValueError("Answer logits must contain at least one token.")


def compute_kd_loss(
    teacher_logits_answer: torch.Tensor,
    student_logits_answer: torch.Tensor,
    temperature: float = 2.0,
    topk: int = 0,
    direction: str = "teacher_forward",
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute answer-token distillation loss.

    Default direction is KL(p_teacher || p_student). If topk > 0, this computes
    a normalized top-k teacher cross-entropy approximation against the student
    log probabilities and does not materialize cached full-vocab logits.
    """

    _validate_answer_logits(teacher_logits_answer, student_logits_answer)
    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0.")
    direction = str(direction).strip().lower()
    vocab_size = int(teacher_logits_answer.shape[-1])

    teacher_scaled = teacher_logits_answer.float() / temperature
    student_scaled = student_logits_answer.float() / temperature
    if weights is not None:
        weights = weights.to(device=student_logits_answer.device, dtype=torch.float32).reshape(-1)
        if int(weights.numel()) != int(student_logits_answer.shape[0]):
            raise ValueError(f"weights must have one value per answer token, got {int(weights.numel())}.")
        denom = weights.sum().clamp_min(1e-6)
    else:
        denom = torch.tensor(float(student_logits_answer.shape[0]), device=student_logits_answer.device)

    if int(topk) > 0:
        k = min(int(topk), vocab_size)
        top_values, top_indices = torch.topk(teacher_scaled, k=k, dim=-1)
        teacher_top_probs = F.softmax(top_values, dim=-1)
        student_log_probs = F.log_softmax(student_scaled, dim=-1)
        student_top_log_probs = student_log_probs.gather(dim=-1, index=top_indices)
        per_token = -(teacher_top_probs * student_top_log_probs).sum(dim=-1) * (temperature**2)
        if weights is not None:
            return (per_token * weights).sum() / denom
        return per_token.mean()

    if direction == "teacher_forward":
        teacher_probs = F.softmax(teacher_scaled, dim=-1)
        student_log_probs = F.log_softmax(student_scaled, dim=-1)
        per_token = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1) * (temperature**2)
    elif direction == "student_forward":
        student_probs = F.softmax(student_scaled, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_scaled, dim=-1)
        per_token = F.kl_div(teacher_log_probs, student_probs, reduction="none").sum(dim=-1) * (temperature**2)
    elif direction == "symmetric":
        forward = compute_kd_loss(
            teacher_logits_answer,
            student_logits_answer,
            temperature=temperature,
            topk=0,
            direction="teacher_forward",
            weights=None,
        )
        reverse = compute_kd_loss(
            teacher_logits_answer,
            student_logits_answer,
            temperature=temperature,
            topk=0,
            direction="student_forward",
            weights=None,
        )
        loss = 0.5 * (forward + reverse)
        if weights is not None:
            raise ValueError("direction='symmetric' does not support weights in this helper.")
        return loss
    else:
        raise ValueError("direction must be 'teacher_forward', 'student_forward', or 'symmetric'.")

    if weights is not None:
        return (per_token * weights).sum() / denom
    return per_token.mean()


def compute_ce_loss(student_logits_answer: torch.Tensor, answer_token_ids: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over answer tokens aligned by answer-token index."""

    if student_logits_answer.ndim != 2:
        raise ValueError("student_logits_answer must have shape [T, vocab].")
    answer_token_ids = answer_token_ids.to(device=student_logits_answer.device, dtype=torch.long).reshape(-1)
    if int(answer_token_ids.numel()) != int(student_logits_answer.shape[0]):
        raise ValueError(
            f"answer_token_ids length {int(answer_token_ids.numel())} does not match logits "
            f"length {int(student_logits_answer.shape[0])}."
        )
    return F.cross_entropy(student_logits_answer.float(), answer_token_ids)


def teacher_entropy(teacher_logits_answer: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(teacher_logits_answer.float(), dim=-1)
    log_probs = torch.log(probs.clamp_min(1e-30))
    return -(probs * log_probs).sum(dim=-1)


def teacher_confidence_weights(
    teacher_logits_answer: torch.Tensor,
    min_weight: float = 0.1,
    max_weight: float = 1.0,
) -> torch.Tensor:
    """Return clamp(1 - entropy/log(vocab), min_weight, max_weight)."""

    vocab_size = int(teacher_logits_answer.shape[-1])
    entropy = teacher_entropy(teacher_logits_answer)
    return torch.clamp(1.0 - entropy / math.log(max(vocab_size, 2)), min=float(min_weight), max=float(max_weight))
