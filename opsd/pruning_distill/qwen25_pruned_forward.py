from __future__ import annotations

import inspect
from contextlib import nullcontext
from typing import Any

import torch


def _unwrap_qwen_model(model: Any) -> Any:
    """Return the underlying Qwen2.5-VL module when wrapped by PEFT."""

    candidates = [model]
    seen: set[int] = set()
    for candidate in candidates:
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        config = getattr(candidate, "config", None)
        has_image_token = config is not None and hasattr(config, "image_token_id")
        has_visual = hasattr(candidate, "visual") or hasattr(getattr(candidate, "model", None), "visual")
        if has_image_token and (has_visual or hasattr(candidate, "get_rope_index")):
            return candidate
        if hasattr(candidate, "get_base_model"):
            try:
                child = candidate.get_base_model()
                if child is not None:
                    candidates.append(child)
            except TypeError:
                pass
        for attr in ("module", "base_model", "model"):
            child = getattr(candidate, attr, None)
            if child is not None:
                candidates.append(child)
    return model


def _get_visual_module(qwen: Any) -> Any:
    if hasattr(qwen, "visual"):
        return qwen.visual
    inner = getattr(qwen, "model", None)
    if inner is not None and hasattr(inner, "visual"):
        return inner.visual
    raise AttributeError("Qwen2.5-VL model must expose a visual encoder as .visual or .model.visual.")


def _get_image_token_id(model: Any) -> int:
    qwen = _unwrap_qwen_model(model)
    if not hasattr(qwen, "config") or not hasattr(qwen.config, "image_token_id"):
        raise AttributeError("Qwen2.5-VL model config must expose image_token_id.")
    return int(qwen.config.image_token_id)


def validate_single_image_qwen_inputs(inputs: dict[str, Any], model: Any | None = None) -> None:
    """Reject unsupported multi-image/video cases before building pruned inputs."""

    input_ids = inputs.get("input_ids")
    if input_ids is None:
        raise ValueError("Qwen inputs must contain input_ids.")
    if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
        raise NotImplementedError("Pruned Qwen2.5-VL distillation currently supports batch size 1 only.")

    video_grid_thw = inputs.get("video_grid_thw")
    pixel_values_videos = inputs.get("pixel_values_videos")
    if video_grid_thw is not None or pixel_values_videos is not None:
        raise NotImplementedError("Video pruning-aware distillation is not implemented yet; use single-image samples.")

    image_grid_thw = inputs.get("image_grid_thw")
    pixel_values = inputs.get("pixel_values")
    if image_grid_thw is None or pixel_values is None:
        raise ValueError("Single-image Qwen inputs must contain pixel_values and image_grid_thw.")
    if image_grid_thw.ndim != 2 or int(image_grid_thw.shape[0]) != 1 or int(image_grid_thw.shape[1]) != 3:
        raise NotImplementedError(
            f"Expected exactly one image_grid_thw row [1, 3], got shape {tuple(image_grid_thw.shape)}."
        )

    if model is not None:
        image_token_id = _get_image_token_id(model)
        image_token_count = int((input_ids == image_token_id).sum().item())
        if image_token_count <= 0:
            raise ValueError("input_ids contain no Qwen image token placeholders.")


def _extract_visual_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output"):
        pooler_output = output.pooler_output
        if isinstance(pooler_output, (tuple, list)):
            if len(pooler_output) != 1:
                raise NotImplementedError("Only one image output is supported.")
            return pooler_output[0]
        return pooler_output
    if isinstance(output, (tuple, list)):
        if len(output) == 1:
            return _extract_visual_tensor(output[0])
        first = output[0]
        if isinstance(first, torch.Tensor):
            return first
    raise TypeError(f"Could not extract visual embeddings from model.visual output of type {type(output)!r}.")


def get_qwen25_visual_embeds(model: Any, inputs: dict[str, Any]) -> torch.Tensor:
    """Run Qwen2.5-VL's visual encoder and return image placeholder embeddings.

    The returned tensor is [num_image_tokens, hidden_size] and is checked against
    the number of image token placeholders in input_ids.
    """

    validate_single_image_qwen_inputs(inputs, model=model)
    qwen = _unwrap_qwen_model(model)
    visual = _get_visual_module(qwen)

    input_ids = inputs["input_ids"]
    image_token_id = _get_image_token_id(qwen)
    n_image_tokens = int((input_ids == image_token_id).sum().item())
    try:
        visual_device = next(visual.parameters()).device
    except StopIteration:
        visual_device = qwen.device if hasattr(qwen, "device") else input_ids.device
    pixel_values = inputs["pixel_values"].to(device=visual_device)
    image_grid_thw = inputs["image_grid_thw"].to(device=pixel_values.device)

    visual_dtype = getattr(visual, "dtype", None)
    if visual_dtype is not None and pixel_values.dtype != visual_dtype:
        pixel_values = pixel_values.to(dtype=visual_dtype)

    with torch.no_grad():
        try:
            image_embeds = _extract_visual_tensor(visual(pixel_values, grid_thw=image_grid_thw))
        except TypeError:
            image_embeds = _extract_visual_tensor(visual(pixel_values, image_grid_thw=image_grid_thw))

    if image_embeds.ndim == 3 and int(image_embeds.shape[0]) == 1:
        image_embeds = image_embeds[0]
    if image_embeds.ndim != 2:
        raise ValueError(f"Expected image_embeds [num_tokens, hidden], got {tuple(image_embeds.shape)}.")

    n_image_features = int(image_embeds.shape[0])
    if n_image_features != n_image_tokens:
        raise ValueError(
            "Image features and image placeholders do not match: "
            f"tokens={n_image_tokens}, features={n_image_features}."
        )
    return image_embeds


def compute_full_position_ids(
    model: Any,
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor | None,
    video_grid_thw: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    second_per_grid_ts: torch.Tensor | None = None,
    mm_token_type_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute full-token Qwen2.5-VL MRoPE position ids."""

    qwen = _unwrap_qwen_model(model)
    get_rope_index = getattr(qwen, "get_rope_index", None)
    if get_rope_index is None:
        get_rope_index = getattr(getattr(qwen, "model", None), "get_rope_index", None)
    if get_rope_index is None:
        if attention_mask is None:
            batch, seq_len = input_ids.shape
            return torch.arange(seq_len, device=input_ids.device).view(1, 1, -1).expand(3, batch, -1)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        return position_ids.unsqueeze(0).expand(3, -1, -1)

    signature = inspect.signature(get_rope_index)
    if "mm_token_type_ids" in signature.parameters:
        if mm_token_type_ids is None:
            mm_token_type_ids = torch.zeros_like(input_ids)
            image_token_id = getattr(getattr(qwen, "config", None), "image_token_id", None)
            video_token_id = getattr(getattr(qwen, "config", None), "video_token_id", None)
            if image_token_id is not None:
                mm_token_type_ids = mm_token_type_ids.masked_fill(input_ids == int(image_token_id), 1)
            if video_token_id is not None:
                mm_token_type_ids = mm_token_type_ids.masked_fill(input_ids == int(video_token_id), 2)
        position_ids, _ = get_rope_index(
            input_ids=input_ids,
            mm_token_type_ids=mm_token_type_ids.to(device=input_ids.device),
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            attention_mask=attention_mask,
        )
        return position_ids

    try:
        position_ids, _ = get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            attention_mask=attention_mask,
        )
    except TypeError:
        try:
            position_ids, _ = get_rope_index(input_ids, image_grid_thw, video_grid_thw, second_per_grid_ts, attention_mask)
        except TypeError:
            position_ids, _ = get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask)
    return position_ids


def _normalize_keep_indices(keep_indices: torch.Tensor, num_visual_tokens: int, device: torch.device) -> torch.Tensor:
    keep_indices = keep_indices.to(device=device, dtype=torch.long).flatten()
    if keep_indices.numel() == 0:
        raise ValueError("keep_indices must keep at least one visual token.")
    if int(keep_indices.min().item()) < 0 or int(keep_indices.max().item()) >= int(num_visual_tokens):
        raise IndexError(
            f"keep_indices must be in [0, {num_visual_tokens}), got min={int(keep_indices.min())} "
            f"max={int(keep_indices.max())}."
        )
    keep_indices = keep_indices.unique(sorted=True)
    return keep_indices


def build_pruned_inputs_embeds(
    model: Any,
    full_input_ids: torch.Tensor,
    full_attention_mask: torch.Tensor,
    full_position_ids: torch.Tensor,
    image_embeds: torch.Tensor,
    keep_indices: torch.Tensor,
    mode: str = "drop_tokens",
    prompt_len: int | None = None,
    mask_vector: torch.Tensor | None = None,
    full_mm_token_type_ids: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Build Qwen student inputs after pruning image placeholder positions.

    In drop_tokens mode this physically removes dropped image token positions and
    selects the corresponding full-token 3D MRoPE position ids. In
    mask_fill_debug mode it keeps sequence length unchanged and replaces dropped
    image embeddings with zeros or a caller-provided mask vector.
    """

    qwen = _unwrap_qwen_model(model)
    if full_input_ids.ndim != 2 or int(full_input_ids.shape[0]) != 1:
        raise NotImplementedError("build_pruned_inputs_embeds currently supports batch size 1 only.")
    if full_attention_mask.ndim != 2 or full_attention_mask.shape != full_input_ids.shape:
        raise ValueError("full_attention_mask must have shape [1, seq_len] matching full_input_ids.")
    if full_position_ids.ndim == 2:
        full_position_ids = full_position_ids.unsqueeze(0).expand(3, -1, -1)
    if full_position_ids.ndim != 3 or full_position_ids.shape[1:] != full_input_ids.shape:
        raise ValueError(
            "full_position_ids must have shape [3, 1, seq_len] or [1, seq_len], "
            f"got {tuple(full_position_ids.shape)} for input_ids {tuple(full_input_ids.shape)}."
        )

    mode = str(mode).strip().lower()
    if mode not in {"drop_tokens", "mask_fill_debug"}:
        raise ValueError("mode must be 'drop_tokens' or 'mask_fill_debug'.")

    device = full_input_ids.device
    image_token_id = _get_image_token_id(qwen)
    image_positions = torch.where(full_input_ids[0] == image_token_id)[0]
    num_visual_tokens = int(image_positions.numel())
    if num_visual_tokens != int(image_embeds.shape[0]):
        raise ValueError(
            "image_embeds do not align with image placeholders: "
            f"placeholders={num_visual_tokens}, embeds={int(image_embeds.shape[0])}."
        )
    keep_indices = _normalize_keep_indices(keep_indices, num_visual_tokens, device)

    embed_tokens = qwen.get_input_embeddings()
    inputs_embeds = embed_tokens(full_input_ids.to(device=embed_tokens.weight.device)).to(device)
    image_embeds = image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
    inputs_embeds = inputs_embeds.clone()
    inputs_embeds[0, image_positions] = image_embeds

    visual_keep_mask = torch.zeros(num_visual_tokens, dtype=torch.bool, device=device)
    visual_keep_mask[keep_indices] = True
    dropped_visual_positions = image_positions[~visual_keep_mask]
    kept_visual_positions = image_positions[visual_keep_mask]

    seq_len = int(full_input_ids.shape[1])
    if prompt_len is not None and (prompt_len < 1 or prompt_len > seq_len):
        raise ValueError(f"prompt_len must be in [1, {seq_len}], got {prompt_len}.")

    if mode == "mask_fill_debug":
        if dropped_visual_positions.numel() > 0:
            if mask_vector is None:
                inputs_embeds[0, dropped_visual_positions] = 0
            else:
                mask_vector = mask_vector.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype).reshape(-1)
                if int(mask_vector.numel()) != int(inputs_embeds.shape[-1]):
                    raise ValueError(
                        f"mask_vector hidden size mismatch: expected {int(inputs_embeds.shape[-1])}, "
                        f"got {int(mask_vector.numel())}."
                    )
                inputs_embeds[0, dropped_visual_positions] = mask_vector
        attention_mask = full_attention_mask.to(device=inputs_embeds.device)
        position_ids = full_position_ids.to(device=inputs_embeds.device)
        input_ids = full_input_ids.to(device=inputs_embeds.device)
        mm_token_type_ids = (
            full_mm_token_type_ids.to(device=inputs_embeds.device) if full_mm_token_type_ids is not None else None
        )
        seq_keep_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
    else:
        seq_keep_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
        seq_keep_mask[image_positions] = False
        seq_keep_mask[kept_visual_positions] = True
        inputs_embeds = inputs_embeds[:, seq_keep_mask, :]
        attention_mask = full_attention_mask[:, seq_keep_mask].to(device=inputs_embeds.device)
        position_ids = full_position_ids[:, :, seq_keep_mask].to(device=inputs_embeds.device)
        input_ids = full_input_ids[:, seq_keep_mask].to(device=inputs_embeds.device)
        mm_token_type_ids = (
            full_mm_token_type_ids[:, seq_keep_mask].to(device=inputs_embeds.device)
            if full_mm_token_type_ids is not None
            else None
        )

    full_to_student = torch.full((seq_len,), -1, dtype=torch.long, device=device)
    full_to_student[seq_keep_mask] = torch.arange(int(seq_keep_mask.sum().item()), dtype=torch.long, device=device)
    student_prompt_len = None
    if prompt_len is not None:
        student_prompt_len = int(seq_keep_mask[:prompt_len].sum().item())

    metadata = {
        "student_prompt_len": student_prompt_len,
        "num_full_visual_tokens": num_visual_tokens,
        "num_kept_visual_tokens": int(keep_indices.numel()),
        "kept_indices": keep_indices.detach().clone(),
        "kept_visual_positions": kept_visual_positions.detach().clone(),
        "dropped_visual_positions": dropped_visual_positions.detach().clone(),
        "seq_keep_mask": seq_keep_mask.detach().clone(),
        "full_to_student": full_to_student.detach().clone(),
    }
    result = {
        "input_ids": input_ids,
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "metadata": metadata,
    }
    if mm_token_type_ids is not None:
        result["mm_token_type_ids"] = mm_token_type_ids
    return result


def extract_next_token_logits(logits: torch.Tensor, prefix_len: int, token_count: int) -> torch.Tensor:
    """Return logits aligned to token indices after a prompt/prefix.

    For target token j at absolute position prefix_len + j, causal LM logits
    come from prefix_len + j - 1.
    """

    if logits.ndim != 3 or int(logits.shape[0]) != 1:
        raise ValueError(f"Expected logits [1, seq, vocab], got {tuple(logits.shape)}.")
    if prefix_len < 1:
        raise ValueError("prefix_len must be >= 1 for next-token extraction.")
    start = int(prefix_len) - 1
    end = start + int(token_count)
    if end > int(logits.shape[1]):
        raise ValueError(f"Requested logits slice [{start}:{end}] beyond sequence length {int(logits.shape[1])}.")
    return logits[0, start:end, :]


def maybe_disable_adapter(model: Any):
    """Context manager that disables PEFT adapters if supported."""

    if hasattr(model, "disable_adapter"):
        return model.disable_adapter()
    module = getattr(model, "module", None)
    if module is not None and hasattr(module, "disable_adapter"):
        return module.disable_adapter()
    return nullcontext()
