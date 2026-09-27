from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


OPTION_LETTERS = ("A", "B", "C", "D")
STANDARD_PROMPT_MODES = {"", "standard", "default", "direct", "direct_mode", "none", "false", "0"}
THINKING_PROMPT_MODES = {"thinking", "think", "reasoning", "reasoning_mode", "reasoning_tags", "enable_thinking", "true", "1"}
ANALYSIS_PROMPT_MODES = {"analysis", "analysis_mode", "analysis_tags"}
THINKING_INSTRUCTION = (
    "First output the thinking process in <think> </think> tags and then output the final answer "
    "in <answer> </answer> tags."
)
ANALYSIS_INSTRUCTION = (
    "First output the reasoning process in <analysis> </analysis> tags and then output the final answer "
    "in <answer> </answer> tags."
)


PROMPT_TEMPLATE = """<image>

You are given an image and a multiple-choice visual reasoning question.

Question:
{question}

Options:
A. {option_0}
B. {option_1}
C. {option_2}
D. {option_3}

Answer with brief reasoning, then provide the final option.
Your response must follow exactly this format:

Reasoning: ...
Final answer: ..."""


THINKING_PROMPT_TEMPLATE = """<image>

You are given an image and a multiple-choice visual reasoning question.

Question:
{question}

Options:
A. {option_0}
B. {option_1}
C. {option_2}
D. {option_3}

{thinking_instruction}"""


TARGET_TEMPLATE = """Reasoning: {brief_reasoning}
Final answer: {correct_option_letter}"""


THINKING_TARGET_TEMPLATE = """<think>
{brief_reasoning}
</think>
<answer>{correct_option_letter}</answer>"""


ANALYSIS_TARGET_TEMPLATE = """<analysis>
{brief_reasoning}
</analysis>
<answer>{correct_option_letter}</answer>"""


OPSD_TEACHER_PROMPT_TEMPLATE = """<image>

You are given an image and a multiple-choice visual reasoning question.

Question:
{question}

Options:
A. {option_0}
B. {option_1}
C. {option_2}
D. {option_3}

Here is a reference solution to this problem:
=== Reference Solution Begin ===
{reference_solution}
=== Reference Solution End ===

After reading the reference solution above, make sure you truly understand the reasoning behind each step - do not copy or paraphrase it. Now, using your own words and independent reasoning, derive the same final answer to the problem above. Think step by step, explore different approaches, and do not be afraid to backtrack or reconsider if something does not work out.

Answer with brief reasoning, then provide the final option.
Your response must follow exactly this format:

Reasoning: ...
Final answer: ..."""


OPSD_TEACHER_THINKING_PROMPT_TEMPLATE = """<image>

You are given an image and a multiple-choice visual reasoning question.

Question:
{question}

Options:
A. {option_0}
B. {option_1}
C. {option_2}
D. {option_3}

Here is a reference solution to this problem:
=== Reference Solution Begin ===
{reference_solution}
=== Reference Solution End ===

After reading the reference solution above, make sure you truly understand the reasoning behind each step - do not copy or paraphrase it. Now, using your own words and independent reasoning, derive the same final answer to the problem above. Think step by step, explore different approaches, and do not be afraid to backtrack or reconsider if something does not work out.

{thinking_instruction}"""


OPSD_TEACHER_OPEN_PROMPT_TEMPLATE = """<image>

{question}

Here is a reference solution to this problem:
=== Reference Solution Begin ===
{reference_solution}
=== Reference Solution End ===

After reading the reference solution above, make sure you truly understand the reasoning behind each step - do not copy or paraphrase it. Now, using your own words and independent reasoning, derive the same final answer to the problem above. Think step by step, explore different approaches, and do not be afraid to backtrack or reconsider if something does not work out."""


OPSD_TEACHER_OPEN_THINKING_PROMPT_TEMPLATE = """<image>

{question}

Here is a reference solution to this problem:
=== Reference Solution Begin ===
{reference_solution}
=== Reference Solution End ===

After reading the reference solution above, make sure you truly understand the reasoning behind each step - do not copy or paraphrase it. Now, using your own words and independent reasoning, derive the same final answer to the problem above. Think step by step, explore different approaches, and do not be afraid to backtrack or reconsider if something does not work out.

{thinking_instruction}"""


@dataclass
class FormattedAOKVQASample:
    sample_id: str
    image: Any
    question: str
    options: list[str]
    correct_index: int
    correct_letter: str
    reasoning: str
    prompt: str
    target: str
    raw: dict[str, Any]


def option_index_to_letter(index: int) -> str:
    index = int(index)
    if index < 0 or index >= len(OPTION_LETTERS):
        raise ValueError(f"A-OKVQA correct option index must be in [0, 3], got {index}.")
    return OPTION_LETTERS[index]


def strip_image_tokens(text: str) -> str:
    return re.sub(r"<\s*image\s*>", "", str(text), flags=re.IGNORECASE).strip()


def normalize_prompt_mode(prompt_mode: str | bool | None = None, enable_thinking: bool | None = None) -> str:
    if enable_thinking:
        return "thinking"
    if isinstance(prompt_mode, bool):
        return "thinking" if prompt_mode else "standard"
    value = str(prompt_mode or "").strip().lower().replace("-", "_")
    if value in STANDARD_PROMPT_MODES:
        return "standard"
    if value in THINKING_PROMPT_MODES:
        return "thinking"
    if value in ANALYSIS_PROMPT_MODES:
        return "analysis"
    raise ValueError(
        f"Unsupported prompt_mode={prompt_mode!r}. "
        "Use direct/standard, thinking/reasoning, or analysis tags."
    )


def is_thinking_prompt_mode(prompt_mode: str | bool | None = None, enable_thinking: bool | None = None) -> bool:
    return normalize_prompt_mode(prompt_mode, enable_thinking=enable_thinking) in {"thinking", "analysis"}


def reasoning_instruction(
    prompt_mode: str | bool | None = None,
    enable_thinking: bool | None = None,
) -> str:
    mode = normalize_prompt_mode(prompt_mode, enable_thinking=enable_thinking)
    if mode == "analysis":
        return ANALYSIS_INSTRUCTION
    if mode == "thinking":
        return THINKING_INSTRUCTION
    return ""


def convert_target_reasoning_tags(
    target: str,
    prompt_mode: str | bool | None = None,
    enable_thinking: bool | None = None,
) -> str:
    mode = normalize_prompt_mode(prompt_mode, enable_thinking=enable_thinking)
    if mode != "analysis":
        return str(target)
    converted = re.sub(r"<\s*think\s*>", "<analysis>", str(target), flags=re.IGNORECASE)
    return re.sub(r"<\s*/\s*think\s*>", "</analysis>", converted, flags=re.IGNORECASE)


def build_reasoning_prompt(
    question: str,
    options: list[str],
    prompt_mode: str | bool | None = None,
    enable_thinking: bool | None = None,
) -> str:
    if len(options) != 4:
        raise ValueError(f"A-OKVQA prompt requires exactly four options, got {len(options)}.")
    clean_options = [str(option).strip() for option in options]
    mode = normalize_prompt_mode(prompt_mode, enable_thinking=enable_thinking)
    template = THINKING_PROMPT_TEMPLATE if mode in {"thinking", "analysis"} else PROMPT_TEMPLATE
    return template.format(
        question=strip_image_tokens(question),
        option_0=clean_options[0],
        option_1=clean_options[1],
        option_2=clean_options[2],
        option_3=clean_options[3],
        thinking_instruction=reasoning_instruction(mode),
    )


def build_target(
    reasoning: str | None,
    correct_option_letter: str,
    prompt_mode: str | bool | None = None,
    enable_thinking: bool | None = None,
) -> str:
    brief_reasoning = str(reasoning or "").strip()
    if not brief_reasoning:
        brief_reasoning = "The correct option is supported by the image and question."
    correct = str(correct_option_letter).strip().upper()
    mode = normalize_prompt_mode(prompt_mode, enable_thinking=enable_thinking)
    if mode == "analysis":
        template = ANALYSIS_TARGET_TEMPLATE
    elif mode == "thinking":
        template = THINKING_TARGET_TEMPLATE
    else:
        template = TARGET_TEMPLATE
    return template.format(
        brief_reasoning=brief_reasoning,
        correct_option_letter=correct,
    )


def build_opsd_teacher_prompt(
    question: str,
    options: list[str],
    reference_solution: str,
    prompt_mode: str | bool | None = None,
    enable_thinking: bool | None = None,
) -> str:
    """Build the privileged teacher prompt used by the official OPSD setup."""

    if len(options) not in {0, 4}:
        raise ValueError(f"OPSD teacher prompt requires either zero or four options, got {len(options)}.")
    clean_options = [str(option).strip() for option in options]
    reference = str(reference_solution or "").strip()
    if not reference:
        reference = "Reasoning: The correct option is supported by the image and question."
    mode = normalize_prompt_mode(prompt_mode, enable_thinking=enable_thinking)
    if clean_options:
        template = (
            OPSD_TEACHER_THINKING_PROMPT_TEMPLATE
            if mode in {"thinking", "analysis"}
            else OPSD_TEACHER_PROMPT_TEMPLATE
        )
    else:
        template = (
            OPSD_TEACHER_OPEN_THINKING_PROMPT_TEMPLATE
            if mode in {"thinking", "analysis"}
            else OPSD_TEACHER_OPEN_PROMPT_TEMPLATE
        )
    return template.format(
        question=strip_image_tokens(question),
        option_0=clean_options[0] if clean_options else "",
        option_1=clean_options[1] if clean_options else "",
        option_2=clean_options[2] if clean_options else "",
        option_3=clean_options[3] if clean_options else "",
        reference_solution=reference,
        thinking_instruction=reasoning_instruction(mode),
    )


ANSWER_TAG_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
OPEN_ANSWER_TAG_PATTERN = re.compile(r"<answer>\s*([^<\n\r]+)", re.IGNORECASE)
FINAL_ANSWER_PATTERNS = (
    re.compile(r"final\s+answer\s*[:\-]\s*([ABCD])\b", re.IGNORECASE),
    re.compile(r"the\s+final\s+answer\s+is\s+([ABCD])\b", re.IGNORECASE),
    re.compile(r"answer\s*[:\-]\s*([ABCD])\b", re.IGNORECASE),
)


def _extract_option_letter(value: str) -> str | None:
    for pattern in FINAL_ANSWER_PATTERNS:
        match = pattern.search(value)
        if match:
            return match.group(1).upper()
    match = re.search(r"\b([ABCD])\b", value, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def parse_final_answer(text: str) -> str | None:
    """Extract A/B/C/D from the requested final-answer format and close variants."""

    value = str(text or "")
    answer_tag = ANSWER_TAG_PATTERN.search(value)
    if answer_tag:
        parsed = _extract_option_letter(answer_tag.group(1))
        if parsed:
            return parsed
    open_answer_tag = OPEN_ANSWER_TAG_PATTERN.search(value)
    if open_answer_tag:
        parsed = _extract_option_letter(open_answer_tag.group(1))
        if parsed:
            return parsed
    for pattern in FINAL_ANSWER_PATTERNS:
        match = pattern.search(value)
        if match:
            return match.group(1).upper()
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if lines:
        tail = lines[-1]
        match = re.search(r"\b([ABCD])\b", tail, flags=re.IGNORECASE)
        if match and "final" in tail.lower():
            return match.group(1).upper()
    return None


def has_parseable_final_answer(text: str) -> bool:
    return parse_final_answer(text) is not None


def format_chat_messages(prompt: str) -> list[dict[str, Any]]:
    """Build Qwen-style multimodal chat messages for a prompt containing <image>."""

    text = strip_image_tokens(prompt)
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text},
            ],
        }
    ]


def format_chat_with_assistant(prompt: str, assistant_response: str) -> list[dict[str, Any]]:
    messages = format_chat_messages(prompt)
    messages.append({"role": "assistant", "content": str(assistant_response)})
    return messages
