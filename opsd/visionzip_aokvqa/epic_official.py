"""Official EPIC TCD curriculum mapped from reduction to retention ratios.

The equations in this module are ported from ZichenWen1/EPIC commit
b2ed9cdfda546bd10c0100e92624d30ccf58af59. The upstream implementation uses
``reduction_ratio`` while the Qwen/VisionZip implementation uses visual-token
``retention_ratio``; therefore retention is exactly ``1 - reduction``.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any

import torch


UPSTREAM_REPOSITORY = "https://github.com/ZichenWen1/EPIC.git"
UPSTREAM_COMMIT = "b2ed9cdfda546bd10c0100e92624d30ccf58af59"
UPSTREAM_TRAINER_PATH = "llava/train/llava_trainer_KD_from_pretrain_dart.py"
UPSTREAM_TRAINER_SHA256 = "76e53ff196eb5e05ad498d30b52562a1deeb68af1936197e1bf1441d8317a813"
UPSTREAM_LAUNCHER_PATH = "scripts/v1_5/finetune_TCD.sh"
UPSTREAM_LAUNCHER_SHA256 = "7d032e8b5d358783ace3f83d4fae2e887126a0f9ba6c8522b6a3d60112ea0549"


def enable_visual_checkpoint_input_grads(model: Any) -> str:
    """Keep visual LoRA gradients alive with reentrant checkpointing.

    Upstream EPIC fine-tunes the patch embedding, so its visual hidden states
    already require gradients. In the controlled LoRA adaptation that base
    module is frozen; marking only its output as requiring gradients preserves
    the same checkpointed graph without unfreezing another parameter.
    """

    matches = [
        (name, module)
        for name, module in model.named_modules()
        if name == "visual.patch_embed" or name.endswith(".visual.patch_embed")
    ]
    if len(matches) != 1:
        names = [name for name, _ in matches]
        raise RuntimeError(f"Expected one Qwen visual.patch_embed module, found {names}.")
    name, module = matches[0]

    def require_output_grad(_module: Any, _inputs: Any, output: Any) -> None:
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"visual.patch_embed returned {type(output)!r}, expected torch.Tensor.")
        output.requires_grad_(True)

    prior = getattr(model, "_opsd_visual_checkpoint_input_grad_hook", None)
    if prior is not None:
        prior.remove()
    handle = module.register_forward_hook(require_output_grad)
    setattr(model, "_opsd_visual_checkpoint_input_grad_hook", handle)
    return name


@dataclass(frozen=True)
class OfficialEpicCurriculumSample:
    optimizer_step: int
    total_optimizer_steps: int
    progress: float
    current_min_reduction_ratio: float
    current_max_reduction_ratio: float
    student_reduction_ratio: float
    teacher_gap: float
    teacher_reduction_ratio: float
    student_retention_ratio: float
    teacher_retention_ratio: float

    def metrics(self) -> dict[str, float | int | str]:
        return {
            **asdict(self),
            "epic_curriculum": "official_progressive_continuous",
            "epic_upstream_commit": UPSTREAM_COMMIT,
        }


def official_epic_curriculum_bounds(progress: float) -> tuple[float, float, float]:
    """Return min reduction, max reduction, and teacher gap from upstream EPIC."""

    progress = min(max(float(progress), 0.0), 1.0)
    initial_max_ratio = 0.1
    final_max_ratio = 0.9
    initial_min_ratio = 0.1
    initial_teacher_gap = 0.1

    current_max_ratio = initial_max_ratio + (final_max_ratio - initial_max_ratio) * progress * 0.8
    current_max_ratio = min(current_max_ratio, 1.0)
    current_min_ratio = initial_min_ratio + (final_max_ratio - initial_max_ratio) * progress * 0.2
    teacher_gap = min(initial_teacher_gap + progress * 0.5, 0.3)
    return current_min_ratio, current_max_ratio, teacher_gap


def extract_official_epic_response_logits(
    logits: torch.Tensor,
    *,
    response_start: int,
    response_count: int,
) -> torch.Tensor:
    """Select the unshifted response-label positions used by upstream EPIC KL.

    The upstream trainer applies its KL mask directly to ``outputs.labels`` and
    ``outputs.logits`` without the causal-LM one-token shift used by CE. Thus an
    answer label at absolute position ``i`` selects logits at position ``i``.
    """

    if logits.ndim != 3 or int(logits.shape[0]) != 1:
        raise ValueError(f"Expected logits [1, seq, vocab], got {tuple(logits.shape)}.")
    response_start = int(response_start)
    response_count = int(response_count)
    if response_start < 0:
        raise ValueError(f"response_start must be non-negative; got {response_start}.")
    if response_count <= 0:
        raise ValueError(f"response_count must be positive; got {response_count}.")
    end = response_start + response_count
    if end > int(logits.shape[1]):
        raise ValueError(
            f"Requested official EPIC logits slice [{response_start}:{end}] beyond "
            f"sequence length {int(logits.shape[1])}."
        )
    return logits[0, response_start:end, :]


def sample_official_epic_curriculum(
    rng: random.Random,
    *,
    optimizer_step: int,
    total_optimizer_steps: int,
) -> OfficialEpicCurriculumSample:
    """Sample one EPIC student/teacher token budget exactly in upstream order."""

    optimizer_step = int(optimizer_step)
    total_optimizer_steps = int(total_optimizer_steps)
    if optimizer_step < 0:
        raise ValueError(f"optimizer_step must be non-negative; got {optimizer_step}.")
    if total_optimizer_steps <= 0:
        raise ValueError(f"total_optimizer_steps must be positive; got {total_optimizer_steps}.")
    progress = min(float(optimizer_step) / float(total_optimizer_steps), 1.0)
    current_min_ratio, current_max_ratio, teacher_gap = official_epic_curriculum_bounds(progress)
    student_reduction_ratio = rng.uniform(current_min_ratio, current_max_ratio)
    teacher_reduction_ratio = max(student_reduction_ratio - teacher_gap, 0.0)
    student_retention_ratio = 1.0 - student_reduction_ratio
    teacher_retention_ratio = 1.0 - teacher_reduction_ratio
    return OfficialEpicCurriculumSample(
        optimizer_step=optimizer_step,
        total_optimizer_steps=total_optimizer_steps,
        progress=progress,
        current_min_reduction_ratio=current_min_ratio,
        current_max_reduction_ratio=current_max_ratio,
        student_reduction_ratio=student_reduction_ratio,
        teacher_gap=teacher_gap,
        teacher_reduction_ratio=teacher_reduction_ratio,
        student_retention_ratio=student_retention_ratio,
        teacher_retention_ratio=teacher_retention_ratio,
    )
