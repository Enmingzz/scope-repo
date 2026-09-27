from __future__ import annotations

import pytest
import torch

from opsd.visionzip_aokvqa.native_budget_weighting import (
    symmetric_teacher_gap_stability_weights,
)


def test_equal_teacher_gaps_are_maximally_robust() -> None:
    gap = torch.tensor([0.01, 0.1, 1.0])
    result = symmetric_teacher_gap_stability_weights(
        gap,
        gap.clone(),
        torch.ones(3, dtype=torch.bool),
        alpha=0.25,
    )
    assert torch.equal(result.normalized_budget_change, torch.zeros_like(gap))
    assert torch.equal(result.robustness, torch.ones_like(gap))
    assert torch.equal(result.weight, torch.ones_like(gap))


def test_budget_change_is_symmetric_and_scale_invariant() -> None:
    gap_b = torch.tensor([0.09, 0.01, 0.04])
    gap_plus = torch.tensor([0.01, 0.09, 0.04])
    valid = torch.ones(3, dtype=torch.bool)
    result = symmetric_teacher_gap_stability_weights(gap_b, gap_plus, valid, alpha=0.25)
    scaled = symmetric_teacher_gap_stability_weights(
        100.0 * gap_b,
        100.0 * gap_plus,
        valid,
        alpha=0.25,
    )
    assert result.signed_budget_change[0] == pytest.approx(-result.signed_budget_change[1])
    assert result.normalized_budget_change[0] == pytest.approx(
        result.normalized_budget_change[1]
    )
    assert torch.allclose(
        result.normalized_budget_change,
        scaled.normalized_budget_change,
        atol=1e-6,
    )
    assert result.raw_weight[2] > result.raw_weight[0]
    assert result.raw_weight[2] > result.raw_weight[1]


def test_token_weighting_preserves_kl_mass_and_is_detached() -> None:
    gap_b = torch.tensor([0.01, 0.04, 0.02, 0.08], requires_grad=True)
    gap_plus = torch.tensor([0.01, 0.01, 0.04, 0.075], requires_grad=True)
    valid = torch.tensor([True, True, True, True])
    result = symmetric_teacher_gap_stability_weights(gap_b, gap_plus, valid, alpha=0.25)
    assert result.raw_weight.min().item() >= 0.75
    assert result.raw_weight.max().item() <= 1.0
    assert torch.allclose(
        (result.weight * gap_b.detach()).sum(),
        gap_b.detach().sum(),
        rtol=2e-6,
        atol=2e-7,
    )
    assert not result.weight.requires_grad
    weighted = (result.weight * gap_b).mean()
    weighted.backward()
    assert torch.allclose(gap_b.grad, result.weight / gap_b.numel())
    assert gap_plus.grad is None


def test_alpha_zero_is_exact_vanilla_weighting() -> None:
    result = symmetric_teacher_gap_stability_weights(
        torch.tensor([0.01, 0.04]),
        torch.tensor([0.04, 0.01]),
        torch.tensor([True, True]),
        alpha=0.0,
    )
    assert torch.equal(result.raw_weight, torch.ones(2))
    assert torch.equal(result.weight, torch.ones(2))
    assert result.loss_mass_scale.item() == pytest.approx(1.0)


def test_invalid_token_stability_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        symmetric_teacher_gap_stability_weights(
            torch.tensor([0.1]),
            torch.tensor([0.1]),
            torch.tensor([True]),
            alpha=1.0,
        )
    with pytest.raises(ValueError):
        symmetric_teacher_gap_stability_weights(
            torch.tensor([-1e-4]),
            torch.tensor([0.1]),
            torch.tensor([True]),
        )
