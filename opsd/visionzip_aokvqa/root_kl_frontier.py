"""Robust online frontier state for adjacent-budget root-KL sensitivity."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import statistics
from typing import Iterable


RATIOS = (0.1, 0.2, 0.3, 0.4)


def root_kl_sensitivity(teacher_gap_b: float, teacher_gap_b_plus: float) -> float:
    """Return |sqrt(K_b) - sqrt(K_b_plus)| for nonnegative KL scalars."""

    gap = float(teacher_gap_b)
    gap_plus = float(teacher_gap_b_plus)
    if not math.isfinite(gap) or not math.isfinite(gap_plus):
        raise ValueError("Teacher gaps must be finite.")
    if gap < 0.0 or gap_plus < 0.0:
        raise ValueError("Teacher gaps must be nonnegative.")
    return abs(math.sqrt(gap) - math.sqrt(gap_plus))


@dataclass
class WinsorizedRootKLFrontierState:
    """Track model progress without allowing rare hard samples to dominate.

    Each ratio is calibrated independently. A trajectory's root-KL
    sensitivity is normalized by the calibration median and capped before it
    enters the progress EMA. The uncapped value remains available in logs.
    """

    retention_ratios: tuple[float, ...] = RATIOS
    calibration_target_per_ratio: int = 64
    ema_half_life_trajectories: float = 256.0
    progress_drop_scale: float = 0.35
    progress_power: float = 2.0
    winsor_cap: float = 4.0
    calibration_values: dict[str, list[float]] = field(default_factory=dict)
    calibration_median: dict[str, float] = field(default_factory=dict)
    initial_robust_signal: dict[str, float] = field(default_factory=dict)
    ema_robust_signal: dict[str, float] = field(default_factory=dict)
    updates: int = 0
    trajectories_seen: int = 0
    calibration_complete_at_trajectory: int | None = None

    def __post_init__(self) -> None:
        self.retention_ratios = tuple(float(value) for value in self.retention_ratios)
        if self.retention_ratios != RATIOS:
            raise ValueError(f"Expected retention ratios {RATIOS}.")
        if int(self.calibration_target_per_ratio) <= 0:
            raise ValueError("calibration_target_per_ratio must be positive.")
        for name in (
            "ema_half_life_trajectories",
            "progress_drop_scale",
            "progress_power",
            "winsor_cap",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.progress_drop_scale > 1.0:
            raise ValueError("progress_drop_scale must be at most one.")
        if self.winsor_cap < 1.0:
            raise ValueError("winsor_cap must be at least one.")
        keys = {self._key(ratio) for ratio in RATIOS}
        if not self.calibration_values:
            self.calibration_values = {key: [] for key in keys}
        if set(self.calibration_values) != keys:
            raise ValueError("Calibration values have inconsistent ratio keys.")
        target = int(self.calibration_target_per_ratio)
        for key, values in self.calibration_values.items():
            if len(values) > target:
                raise ValueError(f"Too many calibration values for ratio {key}.")
            converted = [float(value) for value in values]
            if any(not math.isfinite(value) or value < 0.0 for value in converted):
                raise ValueError(f"Invalid calibration value for ratio {key}.")
            self.calibration_values[key] = converted
        initialized = bool(
            self.calibration_median
            or self.initial_robust_signal
            or self.ema_robust_signal
        )
        if initialized:
            for mapping_name in (
                "calibration_median",
                "initial_robust_signal",
                "ema_robust_signal",
            ):
                mapping = getattr(self, mapping_name)
                if set(mapping) != keys:
                    raise ValueError(f"{mapping_name} has inconsistent ratio keys.")
                for key, value in mapping.items():
                    value = float(value)
                    if not math.isfinite(value) or value <= 0.0:
                        raise ValueError(f"{mapping_name}[{key}] must be positive.")
                    mapping[key] = value
        if self.ready != initialized:
            raise ValueError("Calibration readiness and initialized state disagree.")

    @staticmethod
    def _key(ratio: float) -> str:
        return f"{float(ratio):.2f}"

    def _validated_key(self, ratio: float) -> str:
        for expected in RATIOS:
            if math.isclose(float(ratio), expected, rel_tol=0.0, abs_tol=1e-6):
                return self._key(expected)
        raise ValueError(f"Unsupported retention ratio {ratio}.")

    @property
    def ready(self) -> bool:
        target = int(self.calibration_target_per_ratio)
        return all(len(values) == target for values in self.calibration_values.values())

    @property
    def calibration_counts(self) -> dict[str, int]:
        """Compatibility view used by the shared frontier diagnostics."""

        return {key: len(values) for key, values in self.calibration_values.items()}

    @property
    def initial_sensitivity(self) -> dict[str, float]:
        """Compatibility alias for the robust calibration aggregate."""

        return self.initial_robust_signal

    @property
    def ema_sensitivity(self) -> dict[str, float]:
        """Compatibility alias for the robust online aggregate."""

        return self.ema_robust_signal

    def normalized_signal(self, ratio: float, sensitivity: float) -> float:
        if not self.ready:
            raise RuntimeError("Root-KL frontier is not calibrated.")
        key = self._validated_key(ratio)
        value = float(sensitivity)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("Sensitivity must be finite and nonnegative.")
        return value / self.calibration_median[key]

    def robust_signal(self, ratio: float, sensitivity: float) -> float:
        return min(self.normalized_signal(ratio, sensitivity), self.winsor_cap)

    @property
    def unresolved_sensitivity(self) -> float:
        if not self.ready:
            return 1.0
        values = [
            self.ema_robust_signal[key] / self.initial_robust_signal[key]
            for key in sorted(self.initial_robust_signal)
        ]
        return sum(values) / len(values)

    @property
    def progress(self) -> float:
        contraction = max(0.0, 1.0 - self.unresolved_sensitivity)
        normalized = min(contraction / self.progress_drop_scale, 1.0)
        return normalized**self.progress_power

    @property
    def frontier_index(self) -> float:
        return 3.0 * self.progress

    def ratio_gate(self, ratio: float) -> float:
        key = self._validated_key(ratio)
        descending = tuple(reversed(RATIOS))
        index = [self._key(value) for value in descending].index(key)
        return max(0.0, 1.0 - abs(float(index) - self.frontier_index))

    def local_robustness(self, ratio: float, sensitivity: float, eps: float = 1e-8) -> float:
        robust = self.robust_signal(ratio, sensitivity)
        return 1.0 / (1.0 + robust + float(eps))

    def _finish_calibration(self, position: int) -> None:
        self.calibration_median = {}
        self.initial_robust_signal = {}
        for key, values in sorted(self.calibration_values.items()):
            median = max(float(statistics.median(values)), 1e-12)
            robust = [min(value / median, self.winsor_cap) for value in values]
            self.calibration_median[key] = median
            self.initial_robust_signal[key] = sum(robust) / len(robust)
        self.ema_robust_signal = dict(self.initial_robust_signal)
        self.calibration_complete_at_trajectory = self.trajectories_seen + position + 1

    def update(
        self,
        ratios: Iterable[float],
        sensitivities: Iterable[float],
    ) -> dict[str, object]:
        ratio_values = list(ratios)
        signal_values = list(sensitivities)
        if not ratio_values or len(ratio_values) != len(signal_values):
            raise ValueError("Ratios and sensitivities must be paired and non-empty.")
        pairs: list[tuple[str, float, float]] = []
        for ratio, sensitivity in zip(ratio_values, signal_values):
            key = self._validated_key(ratio)
            value = float(sensitivity)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("Sensitivity must be finite and nonnegative.")
            pairs.append((key, float(ratio), value))

        ready_before = self.ready
        progress_before = self.progress
        frontier_before = self.frontier_index
        target = int(self.calibration_target_per_ratio)
        if not ready_before:
            for position, (key, _, value) in enumerate(pairs):
                if len(self.calibration_values[key]) < target:
                    self.calibration_values[key].append(value)
                if self.ready and not self.calibration_median:
                    self._finish_calibration(position)
        else:
            decay = math.exp(-math.log(2.0) / self.ema_half_life_trajectories)
            for key, ratio, value in pairs:
                robust = self.robust_signal(ratio, value)
                self.ema_robust_signal[key] = (
                    decay * self.ema_robust_signal[key] + (1.0 - decay) * robust
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
            "unresolved_after": self.unresolved_sensitivity,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "retention_ratios": list(self.retention_ratios),
            "calibration_target_per_ratio": int(self.calibration_target_per_ratio),
            "ema_half_life_trajectories": float(self.ema_half_life_trajectories),
            "progress_drop_scale": float(self.progress_drop_scale),
            "progress_power": float(self.progress_power),
            "winsor_cap": float(self.winsor_cap),
            "calibration_values": self.calibration_values,
            "calibration_median": self.calibration_median,
            "initial_robust_signal": self.initial_robust_signal,
            "ema_robust_signal": self.ema_robust_signal,
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
            "winsor_cap",
        ):
            current = list(self.retention_ratios) if key == "retention_ratios" else getattr(self, key)
            if state[key] != current:
                raise ValueError(
                    f"Root-KL frontier resume changed {key}: checkpoint={state[key]} config={current}."
                )
        self.calibration_values = {
            str(key): [float(value) for value in values]
            for key, values in dict(state["calibration_values"]).items()
        }
        for mapping_name in (
            "calibration_median",
            "initial_robust_signal",
            "ema_robust_signal",
        ):
            setattr(
                self,
                mapping_name,
                {
                    str(key): float(value)
                    for key, value in dict(state[mapping_name]).items()
                },
            )
        self.updates = int(state["updates"])
        self.trajectories_seen = int(state["trajectories_seen"])
        value = state.get("calibration_complete_at_trajectory")
        self.calibration_complete_at_trajectory = int(value) if value is not None else None
        self.__post_init__()
