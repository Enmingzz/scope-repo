from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opsd.pruning_distill.qwen25_pruned_forward import build_pruned_inputs_embeds, extract_next_token_logits
from opsd.pruning_distill.losses import compute_kd_loss
from opsd.pruning_distill.pruners import DivPruneLitePruner, GridPruner, RandomPruner, VScanStage1Pruner
from opsd.visionzip_aokvqa.losses import compute_generalized_jsd
from opsd.visionzip_aokvqa.prompting import build_opsd_teacher_prompt


class FakeConfig:
    image_token_id = 99


class FakeQwen(torch.nn.Module):
    def __init__(self, vocab_size: int = 128, hidden_size: int = 8) -> None:
        super().__init__()
        self.config = FakeConfig()
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)

    def get_input_embeddings(self):
        return self.embed


def make_case():
    model = FakeQwen()
    input_ids = torch.tensor([[1, 2, 99, 99, 99, 3, 4, 5, 6]])
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.stack(
        [
            torch.arange(input_ids.shape[1]),
            torch.arange(input_ids.shape[1]) + 100,
            torch.arange(input_ids.shape[1]) + 200,
        ],
        dim=0,
    ).unsqueeze(1)
    image_embeds = torch.randn(3, model.embed.embedding_dim)
    return model, input_ids, attention_mask, position_ids, image_embeds


def test_mask_fill_debug_keeps_length_and_zeros_dropped_tokens():
    model, input_ids, attention_mask, position_ids, image_embeds = make_case()
    out = build_pruned_inputs_embeds(
        model,
        input_ids,
        attention_mask,
        position_ids,
        image_embeds,
        torch.tensor([0, 2]),
        mode="mask_fill_debug",
        prompt_len=7,
    )
    assert out["inputs_embeds"].shape[1] == input_ids.shape[1]
    assert torch.equal(out["attention_mask"], attention_mask)
    assert torch.equal(out["position_ids"], position_ids)
    dropped_position = 3
    assert torch.allclose(out["inputs_embeds"][0, dropped_position], torch.zeros_like(out["inputs_embeds"][0, dropped_position]))
    assert out["metadata"]["student_prompt_len"] == 7


def test_drop_tokens_removes_only_unselected_image_positions_and_preserves_3d_positions():
    model, input_ids, attention_mask, position_ids, image_embeds = make_case()
    out = build_pruned_inputs_embeds(
        model,
        input_ids,
        attention_mask,
        position_ids,
        image_embeds,
        torch.tensor([0, 2]),
        mode="drop_tokens",
        prompt_len=7,
    )
    assert out["input_ids"].tolist() == [[1, 2, 99, 99, 3, 4, 5, 6]]
    expected_keep = torch.tensor([True, True, True, False, True, True, True, True, True])
    assert torch.equal(out["metadata"]["seq_keep_mask"].cpu(), expected_keep)
    assert torch.equal(out["position_ids"], position_ids[:, :, expected_keep])
    assert out["metadata"]["student_prompt_len"] == 6


def test_keep_all_drop_tokens_matches_full_embedding_path():
    model, input_ids, attention_mask, position_ids, image_embeds = make_case()
    out = build_pruned_inputs_embeds(
        model,
        input_ids,
        attention_mask,
        position_ids,
        image_embeds,
        torch.arange(3),
        mode="drop_tokens",
        prompt_len=7,
    )
    full_embeds = model.get_input_embeddings()(input_ids).clone()
    full_embeds[0, torch.where(input_ids[0] == model.config.image_token_id)[0]] = image_embeds
    assert torch.equal(out["input_ids"], input_ids)
    assert torch.equal(out["attention_mask"], attention_mask)
    assert torch.equal(out["position_ids"], position_ids)
    assert torch.allclose(out["inputs_embeds"], full_embeds)


def test_next_token_logits_align_by_answer_index_not_absolute_position():
    logits = torch.randn(1, 12, 20)
    teacher = extract_next_token_logits(logits, prefix_len=7, token_count=3)
    student = extract_next_token_logits(logits[:, 2:], prefix_len=5, token_count=3)
    assert teacher.shape == student.shape == (3, 20)
    assert torch.allclose(teacher, student)


def test_generalized_jsd_beta_zero_matches_forward_kl_without_clip():
    torch.manual_seed(0)
    teacher_logits = torch.randn(4, 11)
    student_logits = torch.randn(4, 11)
    teacher_log_probs = torch.nn.functional.log_softmax(teacher_logits, dim=-1)
    student_log_probs = torch.nn.functional.log_softmax(student_logits, dim=-1)
    expected = torch.nn.functional.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True).sum() / 4
    actual = compute_generalized_jsd(teacher_logits, student_logits, beta=0.0, token_clip=None)
    assert torch.allclose(actual, expected)


def test_opsd_teacher_prompt_contains_reference_solution():
    prompt = build_opsd_teacher_prompt(
        "What color is the object?",
        ["red", "blue", "green", "yellow"],
        "Reasoning: The object is blue.\nFinal answer: B",
    )
    assert "Reference Solution Begin" in prompt
    assert "Final answer: B" in prompt
    assert "do not copy or paraphrase it" in prompt.lower()


def test_opsd_open_teacher_prompt_contains_reference_solution_without_options():
    target = "<think>Count the three visible objects.</think>\n<answer>3</answer>"
    prompt = build_opsd_teacher_prompt(
        "How many objects are visible?",
        [],
        target,
        prompt_mode="thinking",
    )
    assert target in prompt
    assert "Options:" not in prompt
    assert "First output the thinking process" in prompt
    assert prompt.count("<image>") == 1


def test_divprune_lite_synthetic_tokens_are_sorted_and_diverse():
    vision_embeds = torch.eye(8, dtype=torch.float32)
    vision_embeds = torch.cat([vision_embeds, vision_embeds[:4] * 0.95], dim=0)
    pruner = DivPruneLitePruner(chunk_size=3)
    keep = pruner.select(vision_embeds, grid_thw=None, keep_ratio=0.5)
    assert keep.numel() == 6
    assert torch.equal(keep, keep.sort().values)
    kept = torch.nn.functional.normalize(vision_embeds[keep], dim=-1)
    mean_pairwise_cos = (kept @ kept.T).triu(diagonal=1).mean()
    assert float(mean_pairwise_cos) < 0.35


def test_divprune_lite_grid_floor_covers_coarse_cells():
    torch.manual_seed(0)
    vision_embeds = torch.randn(16, 8)
    grid_thw = torch.tensor([[1, 4, 4]])
    pruner = DivPruneLitePruner(grid_floor=True, grid_size=2, chunk_size=5)
    keep = pruner.select(vision_embeds, grid_thw=grid_thw, keep_ratio=0.25)
    assert keep.numel() == 4
    y = keep // 4
    x = keep % 4
    cells = (y // 2) * 2 + (x // 2)
    assert cells.unique().numel() == 4


def test_random_grid_divprune_vscan_counts_and_synthetic_kl():
    torch.manual_seed(1)
    vision_embeds = torch.randn(32, 16)
    grid_thw = torch.tensor([[1, 8, 8]])
    pruners = {
        "random": RandomPruner(seed=7),
        "grid": GridPruner(),
        "divprune_lite": DivPruneLitePruner(chunk_size=7),
        "vscan_stage1": VScanStage1Pruner(grid_size=4),
    }
    teacher_logits = torch.randn(5, 64)
    results = {}
    for name, pruner in pruners.items():
        keep = pruner.select(vision_embeds, grid_thw=grid_thw, keep_ratio=0.25)
        assert keep.numel() == 8
        assert torch.equal(keep, keep.sort().values)
        offset = keep.float().mean() / 1000.0
        student_logits = teacher_logits - offset
        results[name] = float(compute_kd_loss(teacher_logits, student_logits, temperature=2.0))
    assert set(results) == {"random", "grid", "divprune_lite", "vscan_stage1"}


def test_vscan_optional_merge_updates_kept_embedding_slots():
    vision_embeds = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [-1.0, 0.0],
            [-0.9, -0.1],
        ]
    )
    original = vision_embeds.clone()
    pruner = VScanStage1Pruner(grid_size=2, merge_dropped=True)
    keep = pruner.select(vision_embeds, grid_thw=torch.tensor([[1, 2, 2]]), keep_ratio=0.5)
    assert keep.numel() == 2
    assert pruner.last_merge_assignments is not None
    assert not torch.allclose(vision_embeds[keep], original[keep])


if __name__ == "__main__":
    test_mask_fill_debug_keeps_length_and_zeros_dropped_tokens()
    test_drop_tokens_removes_only_unselected_image_positions_and_preserves_3d_positions()
    test_keep_all_drop_tokens_matches_full_embedding_path()
    test_next_token_logits_align_by_answer_index_not_absolute_position()
    test_generalized_jsd_beta_zero_matches_forward_kl_without_clip()
    test_opsd_teacher_prompt_contains_reference_solution()
    test_divprune_lite_synthetic_tokens_are_sorted_and_diverse()
    test_divprune_lite_grid_floor_covers_coarse_cells()
    test_random_grid_divprune_vscan_counts_and_synthetic_kl()
    test_vscan_optional_merge_updates_kept_embedding_slots()
    print("qwen25 pruned forward smoke tests passed")
