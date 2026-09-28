from __future__ import annotations

import hashlib
from contextlib import contextmanager, nullcontext
from typing import Iterator, Sequence

import torch


def stable_seed(namespace: str, seed: int, global_index: int, sample_id: str) -> int:
    payload = f"{namespace}\0{int(seed)}\0{int(global_index)}\0{sample_id}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big", signed=False)
    return value % (2**63 - 1)


def paired_retention_ratio(
    ratios: Sequence[float],
    seed: int,
    global_index: int,
    sample_id: str,
    namespace: str,
) -> float:
    if not ratios:
        raise ValueError("At least one retention ratio is required.")
    index = stable_seed(f"{namespace}:retention", seed, global_index, sample_id) % len(ratios)
    return float(ratios[index])


def paired_rollout_seed(seed: int, global_index: int, sample_id: str, namespace: str) -> int:
    return stable_seed(f"{namespace}:rollout", seed, global_index, sample_id)


@contextmanager
def torch_seed_scope(seed: int | None, device: torch.device) -> Iterator[None]:
    """Apply a sample-local torch seed without perturbing the training RNG stream."""

    if seed is None:
        with nullcontext():
            yield
        return
    devices: list[int] = []
    if device.type == "cuda":
        devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(int(seed))
        if devices:
            torch.cuda.manual_seed(int(seed))
        yield
