from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class PhaseRatioScale:
    phase_index: int
    phase_number: int
    num_phases: int
    phase_start_fraction: float
    phase_end_fraction: float
    retention_ratio: float
    scale: float

    def metrics(self) -> dict[str, Any]:
        return {
            "phase_ratio_scaling_enabled": True,
            "phase_ratio_scaling_normalization": "none",
            "phase_ratio_phase_index": self.phase_index,
            "phase_ratio_phase_number": self.phase_number,
            "phase_ratio_num_phases": self.num_phases,
            "phase_ratio_phase_start_fraction": self.phase_start_fraction,
            "phase_ratio_phase_end_fraction": self.phase_end_fraction,
            "phase_ratio_retention_ratio": self.retention_ratio,
            "phase_ratio_scale": self.scale,
        }


def _ratio_weights(phase: Mapping[str, Any]) -> dict[float, float]:
    raw = phase.get("weights_by_ratio")
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("Every phase requires a non-empty weights_by_ratio mapping.")
    parsed: dict[float, float] = {}
    for ratio_raw, weight_raw in raw.items():
        ratio = float(ratio_raw)
        weight = float(weight_raw)
        if not math.isfinite(ratio) or not 0.0 < ratio <= 1.0:
            raise ValueError(f"Invalid retention ratio in phase weights: {ratio_raw!r}.")
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError(f"Phase-ratio scales must be finite and positive; got {weight_raw!r}.")
        if any(math.isclose(ratio, existing, rel_tol=0.0, abs_tol=1e-9) for existing in parsed):
            raise ValueError(f"Duplicate retention-ratio key after float conversion: {ratio_raw!r}.")
        parsed[ratio] = weight
    return parsed


def validate_phase_ratio_scaling_config(
    config: Mapping[str, Any] | None,
    *,
    method: str,
    train_retention_ratios: Sequence[float],
) -> None:
    if not config or not bool(config.get("enabled", False)):
        return
    if method != "opsd_nogt":
        raise ValueError("Direct phase-ratio scaling is supported only for training.method=opsd_nogt.")
    normalization = str(config.get("normalization", "none")).strip().lower()
    if normalization != "none":
        raise ValueError("Direct phase-ratio scaling requires normalization=none.")
    phases = config.get("phases")
    if not isinstance(phases, list) or not phases:
        raise ValueError("Direct phase-ratio scaling requires a non-empty phases list.")
    expected = sorted(float(value) for value in train_retention_ratios)
    for phase_index, phase in enumerate(phases):
        if not isinstance(phase, Mapping):
            raise ValueError(f"Phase {phase_index} must be a mapping.")
        actual = sorted(_ratio_weights(phase))
        if len(actual) != len(expected) or any(
            not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
            for left, right in zip(actual, expected)
        ):
            raise ValueError(
                f"Phase {phase_index} ratios {actual} do not match training ratios {expected}."
            )


def resolve_phase_ratio_scale(
    config: Mapping[str, Any],
    *,
    retention_ratio: float,
    progress_step: int,
    total_steps: int,
) -> PhaseRatioScale:
    if not bool(config.get("enabled", False)):
        raise ValueError("Cannot resolve a disabled phase-ratio scaling configuration.")
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive; got {total_steps}.")
    if progress_step < 0 or progress_step >= total_steps:
        raise ValueError(
            f"progress_step must be in [0, total_steps); got {progress_step}/{total_steps}."
        )
    phases = config.get("phases")
    if not isinstance(phases, list) or not phases:
        raise ValueError("Direct phase-ratio scaling requires a non-empty phases list.")
    num_phases = len(phases)
    phase_index = min(num_phases - 1, (int(progress_step) * num_phases) // int(total_steps))
    weights = _ratio_weights(phases[phase_index])
    ratio = float(retention_ratio)
    matches = [weight for key, weight in weights.items() if math.isclose(key, ratio, rel_tol=0.0, abs_tol=1e-9)]
    if len(matches) != 1:
        raise KeyError(f"No unique phase {phase_index} scale for retention ratio {ratio}.")
    return PhaseRatioScale(
        phase_index=phase_index,
        phase_number=phase_index + 1,
        num_phases=num_phases,
        phase_start_fraction=phase_index / num_phases,
        phase_end_fraction=(phase_index + 1) / num_phases,
        retention_ratio=ratio,
        scale=matches[0],
    )
