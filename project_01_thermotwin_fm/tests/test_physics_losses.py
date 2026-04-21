"""Differentiable physics penalty tests."""
from __future__ import annotations

import torch

from thermotwin.models.physics_losses import (
    combine_losses,
    control_lag_penalty,
    monotonicity_penalty,
    pinball_loss,
    smoothness_penalty,
)


def test_monotonicity_zero_on_compliant_sequence():
    # Rising load, flat cooling: temperatures rising → zero violation.
    pred = torch.tensor([[[20.0], [21.0], [22.0], [23.0]]])
    load = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    cooling = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    assert monotonicity_penalty(pred, load, cooling).item() == 0.0


def test_monotonicity_positive_on_violation():
    # Rising load, flat cooling: temperatures dropping → must penalize.
    pred = torch.tensor([[[25.0], [24.0], [23.0], [22.0]]])
    load = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    cooling = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    assert monotonicity_penalty(pred, load, cooling).item() > 0.0


def test_smoothness_small_on_smooth_sequence():
    pred = torch.tensor([[[20.0], [20.1], [20.2], [20.3]]])
    assert smoothness_penalty(pred).item() < 0.02


def test_smoothness_large_on_jumpy_sequence():
    pred = torch.tensor([[[20.0], [25.0], [20.0], [25.0]]])
    assert smoothness_penalty(pred).item() > 1.0


def test_control_lag_zero_when_preds_follow_controls_slowly():
    pred = torch.tensor([[[20.0], [20.1], [20.2]]])
    ctrl = torch.tensor([[[0.0, 0.0], [10.0, 5.0], [12.0, 6.0]]])
    # Control changes a lot, prediction changes little → no penalty.
    assert control_lag_penalty(pred, ctrl).item() == 0.0


def test_pinball_loss_shape_and_nonnegative():
    q = torch.tensor([0.1, 0.5, 0.9])
    pred = torch.randn(2, 4, 3, 3)  # (B, H, T, Q)
    target = torch.randn(2, 4, 3)
    loss = pinball_loss(pred, target, tuple(q.tolist()))
    assert loss.dim() == 0
    assert loss.item() >= 0.0


def test_combine_losses_weights_applied():
    base = torch.tensor(1.0)
    pens = {"a": torch.tensor(2.0), "b": torch.tensor(0.5)}
    total = combine_losses(base, pens, {"a": 0.5, "b": 2.0})
    assert abs(total.item() - (1.0 + 0.5 * 2.0 + 2.0 * 0.5)) < 1e-6
