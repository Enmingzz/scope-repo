from __future__ import annotations

import hashlib
import sys
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


class BaseVisualTokenPruner:
    def select(
        self,
        vision_embeds: torch.Tensor,
        grid_thw: torch.Tensor | None,
        keep_ratio: float,
        question: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def _target_count(num_tokens: int, keep_ratio: float) -> int:
        if num_tokens <= 0:
            raise ValueError("vision_embeds must contain at least one visual token.")
        ratio = min(max(float(keep_ratio), 0.0), 1.0)
        return min(num_tokens, max(1, int(round(num_tokens * ratio))))

    @staticmethod
    def _sorted_unique(indices: torch.Tensor, num_tokens: int) -> torch.Tensor:
        indices = indices.to(dtype=torch.long).flatten().unique(sorted=True)
        if indices.numel() == 0:
            raise ValueError("A pruner must keep at least one token.")
        if int(indices.min().item()) < 0 or int(indices.max().item()) >= num_tokens:
            raise IndexError("Pruner returned out-of-range keep indices.")
        return indices


class KeepAllPruner(BaseVisualTokenPruner):
    def select(
        self,
        vision_embeds: torch.Tensor,
        grid_thw: torch.Tensor | None,
        keep_ratio: float,
        question: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        del grid_thw, keep_ratio, question, metadata
        return torch.arange(int(vision_embeds.shape[0]), device=vision_embeds.device, dtype=torch.long)


class RandomPruner(BaseVisualTokenPruner):
    def __init__(self, seed: int = 42) -> None:
        self.seed = int(seed)

    def _seed_for_sample(self, metadata: dict[str, Any] | None) -> int:
        if not metadata:
            return self.seed
        sample_id = str(metadata.get("sample_id", metadata.get("id", "")))
        if not sample_id:
            return self.seed
        digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()
        return (self.seed + int(digest[:8], 16)) % (2**31 - 1)

    def select(
        self,
        vision_embeds: torch.Tensor,
        grid_thw: torch.Tensor | None,
        keep_ratio: float,
        question: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        del grid_thw, question
        num_tokens = int(vision_embeds.shape[0])
        target = self._target_count(num_tokens, keep_ratio)
        if target >= num_tokens:
            return torch.arange(num_tokens, device=vision_embeds.device, dtype=torch.long)
        generator = torch.Generator(device=vision_embeds.device)
        generator.manual_seed(self._seed_for_sample(metadata))
        indices = torch.randperm(num_tokens, generator=generator, device=vision_embeds.device)[:target]
        return self._sorted_unique(indices, num_tokens)


class GridPruner(BaseVisualTokenPruner):
    """Spatially uniform selector using Qwen image_grid_thw when possible."""

    def select(
        self,
        vision_embeds: torch.Tensor,
        grid_thw: torch.Tensor | None,
        keep_ratio: float,
        question: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        del question, metadata
        num_tokens = int(vision_embeds.shape[0])
        target = self._target_count(num_tokens, keep_ratio)
        if target >= num_tokens:
            return torch.arange(num_tokens, device=vision_embeds.device, dtype=torch.long)

        shape = self._infer_visual_grid(num_tokens, grid_thw)
        if shape is None:
            return self._linspace_indices(num_tokens, target, vision_embeds.device)

        t, h, w = shape
        coords = []
        for tt in range(t):
            yy, xx = torch.meshgrid(
                torch.arange(h, device=vision_embeds.device, dtype=torch.float32),
                torch.arange(w, device=vision_embeds.device, dtype=torch.float32),
                indexing="ij",
            )
            flat = torch.stack(
                [
                    torch.full((h * w,), float(tt), device=vision_embeds.device),
                    yy.flatten(),
                    xx.flatten(),
                ],
                dim=-1,
            )
            coords.append(flat)
        coord = torch.cat(coords, dim=0)
        if int(coord.shape[0]) != num_tokens:
            return self._linspace_indices(num_tokens, target, vision_embeds.device)

        selected = [0]
        min_dist = torch.cdist(coord[:1], coord).flatten()
        available = torch.ones(num_tokens, dtype=torch.bool, device=vision_embeds.device)
        available[0] = False
        for _ in range(1, target):
            scores = min_dist.clone()
            scores[~available] = -1.0
            idx = int(torch.argmax(scores).item())
            selected.append(idx)
            available[idx] = False
            min_dist = torch.minimum(min_dist, torch.cdist(coord[idx : idx + 1], coord).flatten())
        return self._sorted_unique(torch.tensor(selected, device=vision_embeds.device), num_tokens)

    @staticmethod
    def _linspace_indices(num_tokens: int, target: int, device: torch.device) -> torch.Tensor:
        idx = torch.linspace(0, num_tokens - 1, steps=target, device=device).round().long().unique(sorted=True)
        if int(idx.numel()) < target:
            fill = torch.arange(num_tokens, device=device, dtype=torch.long)
            mask = torch.ones(num_tokens, device=device, dtype=torch.bool)
            mask[idx] = False
            idx = torch.cat([idx, fill[mask][: target - int(idx.numel())]])
        return idx.sort().values[:target]

    @staticmethod
    def _infer_visual_grid(num_tokens: int, grid_thw: torch.Tensor | None) -> tuple[int, int, int] | None:
        if grid_thw is None:
            return None
        row = grid_thw.reshape(-1, 3)[0].detach().cpu().tolist()
        t, h, w = [max(1, int(x)) for x in row]
        if t * h * w == num_tokens:
            return t, h, w
        if t > 0 and num_tokens % t == 0:
            per_t = num_tokens // t
            for merge in (2, 4, 1, 8):
                hh, ww = h // merge, w // merge
                if hh > 0 and ww > 0 and hh * ww == per_t:
                    return t, hh, ww
            side = int(round(per_t**0.5))
            if side * side == per_t:
                return t, side, side
        return None


def _linspace_indices(num_tokens: int, target: int, device: torch.device) -> torch.Tensor:
    return GridPruner._linspace_indices(num_tokens, target, device)


def _infer_visual_grid(num_tokens: int, grid_thw: torch.Tensor | None) -> tuple[int, int, int] | None:
    return GridPruner._infer_visual_grid(num_tokens, grid_thw)


def _coarse_cell_ids(
    num_tokens: int,
    grid_thw: torch.Tensor | None,
    grid_size: int,
    device: torch.device,
) -> torch.Tensor | None:
    shape = _infer_visual_grid(num_tokens, grid_thw)
    if shape is None:
        return None
    t, h, w = shape
    if t * h * w != num_tokens:
        return None
    grid_size = max(1, int(grid_size))
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )
    cell_h = torch.clamp((yy * grid_size) // max(h, 1), max=grid_size - 1)
    cell_w = torch.clamp((xx * grid_size) // max(w, 1), max=grid_size - 1)
    per_frame = (cell_h * grid_size + cell_w).flatten().long()
    offsets = torch.arange(t, device=device, dtype=torch.long).repeat_interleave(h * w) * (grid_size * grid_size)
    return per_frame.repeat(t) + offsets


def _normalize_vision_embeds(vision_embeds: torch.Tensor) -> torch.Tensor:
    if vision_embeds.ndim != 2:
        raise ValueError(f"vision_embeds must have shape [N, D], got {tuple(vision_embeds.shape)}.")
    return F.normalize(vision_embeds.detach().float(), dim=-1, eps=1e-6)


class DivPruneLitePruner(BaseVisualTokenPruner):
    """Diversity-first pre-LLM visual token selector.

    This is Qwen2.5-VL compatible because it returns only visual-token indices.
    The caller must rebuild the LLM sequence and preserve MRoPE position ids.
    """

    def __init__(
        self,
        grid_floor: bool = False,
        grid_size: int = 4,
        chunk_size: int = 8192,
    ) -> None:
        self.grid_floor = bool(grid_floor)
        self.grid_size = max(1, int(grid_size))
        self.chunk_size = max(1, int(chunk_size))

    def select(
        self,
        vision_embeds: torch.Tensor,
        grid_thw: torch.Tensor | None,
        keep_ratio: float,
        question: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        del question, metadata
        num_tokens = int(vision_embeds.shape[0])
        target = self._target_count(num_tokens, keep_ratio)
        if target >= num_tokens:
            return torch.arange(num_tokens, device=vision_embeds.device, dtype=torch.long)

        x = _normalize_vision_embeds(vision_embeds)
        selected = self._grid_floor_seeds(x, grid_thw, target) if self.grid_floor else []
        if not selected:
            selected.append(self._closest_to_mean(x))

        max_sim = self._max_similarity_to_selected(x, selected)
        available = torch.ones(num_tokens, dtype=torch.bool, device=x.device)
        available[torch.tensor(selected, device=x.device, dtype=torch.long)] = False

        while len(selected) < target:
            score = 1.0 - max_sim
            score = score.masked_fill(~available, -1.0)
            next_idx = int(torch.argmax(score).item())
            selected.append(next_idx)
            available[next_idx] = False
            self._update_max_similarity_(x, max_sim, next_idx)

        return self._sorted_unique(torch.tensor(selected, device=vision_embeds.device), num_tokens)

    @staticmethod
    def _closest_to_mean(x: torch.Tensor) -> int:
        mean = F.normalize(x.mean(dim=0, keepdim=True), dim=-1, eps=1e-6)
        return int(torch.argmax((x @ mean.T).flatten()).item())

    def _grid_floor_seeds(self, x: torch.Tensor, grid_thw: torch.Tensor | None, target: int) -> list[int]:
        cell_ids = _coarse_cell_ids(int(x.shape[0]), grid_thw, self.grid_size, x.device)
        if cell_ids is None:
            return []
        unique_cells = cell_ids.unique(sorted=True)
        if target < int(unique_cells.numel()):
            # The floor is only possible when the target can afford every cell.
            return []

        seeds: list[int] = []
        global_mean = F.normalize(x.mean(dim=0, keepdim=True), dim=-1, eps=1e-6)
        global_scores = (x @ global_mean.T).flatten()
        for cell in unique_cells.tolist():
            members = torch.where(cell_ids == int(cell))[0]
            if int(members.numel()) == 0:
                continue
            cell_x = x.index_select(0, members)
            cell_mean = F.normalize(cell_x.mean(dim=0, keepdim=True), dim=-1, eps=1e-6)
            local_scores = (cell_x @ cell_mean.T).flatten()
            # Tie-break toward globally representative tokens.
            scores = local_scores + 1e-4 * global_scores.index_select(0, members)
            seeds.append(int(members[int(torch.argmax(scores).item())].item()))
        return seeds[:target]

    def _max_similarity_to_selected(self, x: torch.Tensor, selected: list[int]) -> torch.Tensor:
        max_sim = torch.full((int(x.shape[0]),), -float("inf"), device=x.device, dtype=torch.float32)
        for start in range(0, len(selected), self.chunk_size):
            idx = torch.tensor(selected[start : start + self.chunk_size], device=x.device, dtype=torch.long)
            selected_x = x.index_select(0, idx)
            for token_start in range(0, int(x.shape[0]), self.chunk_size):
                token_end = min(token_start + self.chunk_size, int(x.shape[0]))
                sim = x[token_start:token_end] @ selected_x.T
                max_sim[token_start:token_end] = torch.maximum(max_sim[token_start:token_end], sim.max(dim=-1).values)
        return max_sim

    def _update_max_similarity_(self, x: torch.Tensor, max_sim: torch.Tensor, selected_idx: int) -> None:
        selected_x = x[selected_idx : selected_idx + 1]
        for start in range(0, int(x.shape[0]), self.chunk_size):
            end = min(start + self.chunk_size, int(x.shape[0]))
            sim = (x[start:end] @ selected_x.T).flatten()
            max_sim[start:end] = torch.maximum(max_sim[start:end], sim)


class VScanStage1Pruner(BaseVisualTokenPruner):
    """Global-local selector inspired by VScan stage-1 token filtering."""

    def __init__(
        self,
        grid_size: int = 4,
        score_mode: str = "cosine_mean",
        global_fraction: float = 0.5,
        merge_dropped: bool = False,
    ) -> None:
        self.grid_size = max(1, int(grid_size))
        self.score_mode = str(score_mode).strip().lower()
        self.global_fraction = min(max(float(global_fraction), 0.0), 1.0)
        self.merge_dropped = bool(merge_dropped)
        self.last_merge_assignments: torch.Tensor | None = None

    def select(
        self,
        vision_embeds: torch.Tensor,
        grid_thw: torch.Tensor | None,
        keep_ratio: float,
        question: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        del question, metadata
        num_tokens = int(vision_embeds.shape[0])
        target = self._target_count(num_tokens, keep_ratio)
        if target >= num_tokens:
            return torch.arange(num_tokens, device=vision_embeds.device, dtype=torch.long)

        scores = self._global_scores(vision_embeds)
        global_target = max(1, min(target, int(round(target * self.global_fraction))))
        selected = torch.topk(scores, k=global_target, largest=True).indices.tolist()

        local_budget = target - len(set(selected))
        if local_budget > 0:
            local = self._local_coverage_indices(scores, num_tokens, grid_thw, local_budget, vision_embeds.device)
            selected.extend(local.tolist())

        selected_tensor = torch.tensor(selected, device=vision_embeds.device, dtype=torch.long).unique(sorted=False)
        if int(selected_tensor.numel()) < target:
            mask = torch.ones(num_tokens, dtype=torch.bool, device=vision_embeds.device)
            mask[selected_tensor] = False
            fill = torch.argsort(scores.masked_fill(~mask, -float("inf")), descending=True)
            selected_tensor = torch.cat([selected_tensor, fill[: target - int(selected_tensor.numel())]])
        selected_tensor = selected_tensor[:target]

        if self.merge_dropped:
            self.last_merge_assignments = self._merge_dropped_into_kept_(vision_embeds, selected_tensor)
        else:
            self.last_merge_assignments = None

        return self._sorted_unique(selected_tensor, num_tokens)

    def _global_scores(self, vision_embeds: torch.Tensor) -> torch.Tensor:
        if self.score_mode in {"norm", "l2"}:
            return vision_embeds.detach().float().norm(dim=-1)
        if self.score_mode not in {"cosine_mean", "cosine", "mean"}:
            raise ValueError("vscan score_mode must be 'cosine_mean' or 'norm'.")
        x = _normalize_vision_embeds(vision_embeds)
        mean = F.normalize(x.mean(dim=0, keepdim=True), dim=-1, eps=1e-6)
        return (x @ mean.T).flatten()

    def _local_coverage_indices(
        self,
        scores: torch.Tensor,
        num_tokens: int,
        grid_thw: torch.Tensor | None,
        budget: int,
        device: torch.device,
    ) -> torch.Tensor:
        cell_ids = _coarse_cell_ids(num_tokens, grid_thw, self.grid_size, device)
        if cell_ids is None:
            return _linspace_indices(num_tokens, budget, device)

        candidates = []
        for cell in cell_ids.unique(sorted=True).tolist():
            members = torch.where(cell_ids == int(cell))[0]
            if int(members.numel()) == 0:
                continue
            member_scores = scores.index_select(0, members)
            candidates.append(int(members[int(torch.argmax(member_scores).item())].item()))
        if not candidates:
            return _linspace_indices(num_tokens, budget, device)
        candidate_tensor = torch.tensor(candidates, device=device, dtype=torch.long)
        candidate_scores = scores.index_select(0, candidate_tensor)
        order = torch.argsort(candidate_scores, descending=True)
        return candidate_tensor.index_select(0, order)[:budget]

    @staticmethod
    def _merge_dropped_into_kept_(vision_embeds: torch.Tensor, keep_indices: torch.Tensor) -> torch.Tensor:
        original = vision_embeds.detach().clone()
        x = _normalize_vision_embeds(original)
        kept = x.index_select(0, keep_indices.to(device=x.device, dtype=torch.long))
        assignments = torch.argmax(x @ kept.T, dim=-1)
        with torch.no_grad():
            for slot, keep_idx in enumerate(keep_indices.to(device=vision_embeds.device, dtype=torch.long).tolist()):
                members = torch.where(assignments == int(slot))[0].to(device=original.device)
                if int(members.numel()) == 0:
                    continue
                merged = original.index_select(0, members).mean(dim=0).to(
                    device=vision_embeds.device,
                    dtype=vision_embeds.dtype,
                )
                vision_embeds[int(keep_idx)].copy_(merged)
        return assignments


class ExistingMethodPruner(BaseVisualTokenPruner):
    """Placeholder adapter for existing repo pruning methods.

    For single-image distillation this currently tries the repo's DivPrune
    selector with T=1 if it is importable. It returns the selected existing
    method indices, sorted in original visual-token order.
    """

    def __init__(self, method: str = "divprune") -> None:
        self.method = str(method).strip().lower()

    def select(
        self,
        vision_embeds: torch.Tensor,
        grid_thw: torch.Tensor | None,
        keep_ratio: float,
        question: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        del question, metadata
        if self.method not in {"divprune", "existing", ""}:
            warnings.warn(f"ExistingMethodPruner only wires divprune for now, got {self.method!r}; using divprune.")
        try:
            repo_root = Path(__file__).resolve().parents[2]
            lmms_root = repo_root / "vlm" / "official_thinking_in_space"
            if lmms_root.exists() and str(lmms_root) not in sys.path:
                sys.path.insert(0, str(lmms_root))
            from lmms_eval.models.pruning.divprune import prune_video_tokens_with_divprune

            output = prune_video_tokens_with_divprune(
                vision_embeds.unsqueeze(0),
                retention_ratio=float(keep_ratio),
            )
            return self._sorted_unique(output.global_indices.to(device=vision_embeds.device), int(vision_embeds.shape[0]))
        except Exception as exc:
            warnings.warn(f"Could not call existing DivPrune selector ({type(exc).__name__}: {exc}); using GridPruner.")
            return GridPruner().select(vision_embeds, grid_thw, keep_ratio)


def build_pruner(
    name: str,
    seed: int = 42,
    divprune_grid_floor: bool = False,
    divprune_grid_size: int = 4,
    divprune_chunk_size: int = 8192,
    vscan_grid_size: int = 4,
    vscan_score_mode: str = "cosine_mean",
    vscan_global_fraction: float = 0.5,
    vscan_merge_dropped: bool = False,
) -> BaseVisualTokenPruner:
    name = str(name).strip().lower()
    if name in {"keep_all", "all", "none"}:
        return KeepAllPruner()
    if name in {"random", "rand"}:
        return RandomPruner(seed=seed)
    if name in {"grid", "spatial", "uniform", "grid_uniform"}:
        return GridPruner()
    if name in {"divprune_lite", "divprune-lite", "divlite"}:
        return DivPruneLitePruner(
            grid_floor=divprune_grid_floor,
            grid_size=divprune_grid_size,
            chunk_size=divprune_chunk_size,
        )
    if name in {"vscan_stage1", "vscan-stage1", "vscan"}:
        return VScanStage1Pruner(
            grid_size=vscan_grid_size,
            score_mode=vscan_score_mode,
            global_fraction=vscan_global_fraction,
            merge_dropped=vscan_merge_dropped,
        )
    if name in {"existing", "divprune"}:
        return ExistingMethodPruner(method="divprune")
    raise ValueError("pruner must be one of: random, grid, divprune_lite, vscan_stage1, existing, keep_all.")
