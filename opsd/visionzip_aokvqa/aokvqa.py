from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterable

_DISALLOWED_QWEN25_BOOTSTRAP = "/__disabled_legacy_runtime__"
sys.path = [path for path in sys.path if not path or not path.startswith(_DISALLOWED_QWEN25_BOOTSTRAP)]

from PIL import Image

from .prompting import (
    FormattedAOKVQASample,
    build_reasoning_prompt,
    build_target,
    convert_target_reasoning_tags,
    is_thinking_prompt_mode,
    option_index_to_letter,
    reasoning_instruction,
    strip_image_tokens,
)


def _first_present(record: dict[str, Any], names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if name in record and record[name] is not None:
            return record[name]
    return default


def _as_dict(record: Any) -> dict[str, Any]:
    if isinstance(record, dict):
        return record
    if hasattr(record, "to_dict"):
        return record.to_dict()
    raise TypeError(f"Expected dict-like A-OKVQA sample, got {type(record)!r}.")


def _extract_options(record: dict[str, Any]) -> list[str]:
    options = _first_present(record, ("choices", "options", "answer_choices", "multiple_choice_answers"))
    if isinstance(options, dict):
        for key in ("text", "choices", "options", "answers"):
            if key in options:
                options = options[key]
                break
    if options is None:
        raise KeyError(f"Could not find A-OKVQA options field in keys: {sorted(record)}")
    values = [str(x).strip() for x in list(options)]
    if len(values) < 4:
        raise ValueError(f"A-OKVQA sample must contain at least four options, got {len(values)}.")
    return values[:4]


def _extract_correct_index(record: dict[str, Any], options: list[str]) -> int:
    value = _first_present(
        record,
        (
            "correct_choice_idx",
            "correct_option_index",
            "correct_option_idx",
            "correct_choice_index",
            "label",
            "answer_idx",
        ),
    )
    if value is not None:
        return int(value)

    answer = _first_present(record, ("multiple_choice_answer", "answer", "correct_answer", "correct_choice"))
    if isinstance(answer, str):
        clean = answer.strip()
        if len(clean) == 1 and clean.upper() in {"A", "B", "C", "D"}:
            return ord(clean.upper()) - ord("A")
        lowered = [option.lower() for option in options]
        if clean.lower() in lowered:
            return lowered.index(clean.lower())
    raise KeyError(f"Could not infer A-OKVQA correct option from keys: {sorted(record)}")


def _extract_reasoning(record: dict[str, Any]) -> str:
    rationale = _first_present(record, ("rationales", "reasoning", "rationale", "rationales_text"), "")
    if isinstance(rationale, (list, tuple)):
        return str(rationale[0]).strip() if rationale else ""
    return str(rationale or "").strip()


def _extract_question(record: dict[str, Any]) -> str:
    question = _first_present(record, ("question", "query", "prompt"))
    if question is None:
        raise KeyError(f"Could not find A-OKVQA question field in keys: {sorted(record)}")
    question = strip_image_tokens(str(question))
    if not question:
        raise ValueError("A-OKVQA question is empty after removing <image> tokens.")
    return question


def _extract_image(record: dict[str, Any]) -> Any:
    image = _first_present(record, ("image", "image_path", "img", "image_id", "file_name"))
    if image is None:
        raise KeyError(f"Could not find A-OKVQA image field in keys: {sorted(record)}")
    return image


def _extract_answer_body(answer: str) -> str:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", str(answer or ""), flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else str(answer or "").strip()


def _extract_optional_correct_letter(answer: str) -> str:
    body = _extract_answer_body(answer)
    match = re.search(r"\b([ABCD])\s*[\.)]", body, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.fullmatch(r"\s*([ABCD])\s*", body, flags=re.IGNORECASE)
    return match.group(1).upper() if match else ""


def _generic_prompt(question: str, prompt_mode: str | bool | None = None) -> str:
    question = strip_image_tokens(question)
    if is_thinking_prompt_mode(prompt_mode):
        return f"<image>\n\n{question}\n\n{reasoning_instruction(prompt_mode)}"
    return f"<image>\n\n{question}"


def normalize_reasoning_answer_jsonl_sample(
    record: Any,
    index: int = 0,
    prompt_mode: str | bool | None = None,
) -> FormattedAOKVQASample:
    raw = _as_dict(record)
    question = strip_image_tokens(str(_first_present(raw, ("question", "raw_question", "prompt"), "") or ""))
    if not question:
        raise ValueError(f"JSONL sample is missing a non-empty question field: keys={sorted(raw)}")
    target = str(_first_present(raw, ("answer", "ground_truth", "target", "processed_target"), "") or "").strip()
    if not target:
        raise ValueError(f"JSONL sample is missing a non-empty answer/target field: keys={sorted(raw)}")
    target = convert_target_reasoning_tags(target, prompt_mode=prompt_mode)
    sample_id = str(_first_present(raw, ("sample_id", "id", "question_id"), index))
    correct_letter = _extract_optional_correct_letter(target)
    correct_index = ord(correct_letter) - ord("A") if correct_letter else -1
    reasoning = str(_first_present(raw, ("reasoning", "rationale"), "") or "").strip()
    return FormattedAOKVQASample(
        sample_id=sample_id,
        image=_extract_image(raw),
        question=question,
        options=list(raw.get("choices") or raw.get("options") or []),
        correct_index=correct_index,
        correct_letter=correct_letter,
        reasoning=reasoning,
        prompt=_generic_prompt(question, prompt_mode=prompt_mode),
        target=target,
        raw=raw,
    )


def normalize_aokvqa_sample(
    record: Any,
    index: int = 0,
    prompt_mode: str | bool | None = None,
) -> FormattedAOKVQASample:
    raw = _as_dict(record)
    question = _extract_question(raw)
    options = _extract_options(raw)
    correct_index = _extract_correct_index(raw, options)
    correct_letter = option_index_to_letter(correct_index)
    reasoning = _extract_reasoning(raw)
    sample_id = str(_first_present(raw, ("sample_id", "question_id", "id"), index))
    prompt = build_reasoning_prompt(question, options, prompt_mode=prompt_mode)
    target = build_target(reasoning, correct_letter, prompt_mode=prompt_mode)
    return FormattedAOKVQASample(
        sample_id=sample_id,
        image=_extract_image(raw),
        question=question,
        options=options,
        correct_index=correct_index,
        correct_letter=correct_letter,
        reasoning=reasoning,
        prompt=prompt,
        target=target,
        raw=raw,
    )


def load_aokvqa_dataset(
    dataset_name: str = "HuggingFaceM4/A-OKVQA",
    splits: list[str] | None = None,
    limit: int = 0,
    seed: int = 42,
    prompt_mode: str | bool | None = None,
    shuffle: bool = True,
) -> list[FormattedAOKVQASample]:
    dataset_path = Path(str(dataset_name))
    if dataset_path.exists() and dataset_path.suffix.lower() == ".jsonl":
        normalized: list[FormattedAOKVQASample] = []
        with dataset_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                if not line.strip():
                    continue
                normalized.append(
                    normalize_reasoning_answer_jsonl_sample(
                        json.loads(line),
                        index=line_no,
                        prompt_mode=prompt_mode,
                    )
                )
        if shuffle:
            rng = random.Random(int(seed))
            rng.shuffle(normalized)
        if limit and limit > 0:
            normalized = normalized[: int(limit)]
        return normalized

    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("Loading A-OKVQA requires the `datasets` package.") from exc

    splits = splits or ["train", "validation"]
    normalized: list[FormattedAOKVQASample] = []
    for split in splits:
        ds = load_dataset(dataset_name, split=split)
        for idx, record in enumerate(ds):
            normalized.append(normalize_aokvqa_sample(record, index=len(normalized), prompt_mode=prompt_mode))
    if shuffle:
        rng = random.Random(int(seed))
        rng.shuffle(normalized)
    if limit and limit > 0:
        normalized = normalized[: int(limit)]
    return normalized


def resolve_image(image_value: Any, image_root: str | Path = "") -> Image.Image:
    """Return a RGB PIL image from a HF image object, PIL image, or path-like value."""

    if isinstance(image_value, Image.Image):
        return image_value.convert("RGB")
    if isinstance(image_value, dict):
        if "path" in image_value and image_value["path"]:
            image_value = image_value["path"]
        elif "bytes" in image_value and image_value["bytes"]:
            import io

            return Image.open(io.BytesIO(image_value["bytes"])).convert("RGB")
    path = Path(str(image_value))
    if not path.is_absolute() and image_root:
        path = Path(image_root) / path
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")
