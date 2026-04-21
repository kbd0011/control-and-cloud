"""Physics-inspired differentiable penalties for the thermal forecaster.

These are soft inductive biases encoding domain knowledge. They do not replace
a CFD simulation; they nudge the model away from physically implausible
outputs. All penalties are non-negative scalars that return 0 on a hand-crafted
plausible sequence.

Conventions:
- `pred` is (B, H, T) point forecast (usually median quantile).
- `load` is a (B, H) proxy for cooling demand (e.g., CPU+GPU load or power).
- `cooling` is a (B, H) proxy for cooling effort (e.g., fan speed or inverse coolant temp).
- `controls` is (B, H, K) future control inputs.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def monotonicity_penalty(pred: torch.Tensor, load: torch.Tensor, cooling: torch.Tensor) -> torch.Tensor:
    """Penalize cases where load rises and cooling stays flat but temperature drops.

    Formal statement: if d_load/dt > 0 and d_cooling/dt <= 0, then d_pred/dt should
    be >= 0 on average. We penalize violations via hinge loss.
    """
    d_pred = pred[:, 1:, :] - pred[:, :-1, :]
    d_load = (load[:, 1:] - load[:, :-1]).unsqueeze(-1)
    d_cool = (cooling[:, 1:] - cooling[:, :-1]).unsqueeze(-1)
    # Rising-load, non-increasing-cooling mask.
    rising = (d_load > 0) & (d_cool <= 0)
    violations = F.relu(-d_pred) * rising.to(d_pred.dtype)
    return violations.mean()


def smoothness_penalty(pred: torch.Tensor) -> torch.Tensor:
    """Penalize unrealistically abrupt temperature swings between adjacent steps."""
    diffs = pred[:, 1:, :] - pred[:, :-1, :]
    return (diffs.pow(2)).mean()


def control_lag_penalty(pred: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
    """Penalize instantaneous full response to control changes.

    If the control changes at step t, the predicted temperature at step t should
    not change by more than a bounded fraction of that control delta. We approximate
    this by encouraging |d_pred[t]| <= 0.5 * ||d_controls[t]||.
    """
    if controls.shape[-1] == 0:
        return pred.new_zeros(())
    d_pred = (pred[:, 1:, :] - pred[:, :-1, :]).abs().mean(dim=-1)        # (B, H-1)
    d_ctrl = (controls[:, 1:, :] - controls[:, :-1, :]).pow(2).sum(-1).sqrt()  # (B, H-1)
    violation = F.relu(d_pred - 0.5 * d_ctrl)
    return violation.mean()


def pinball_loss(pred_quantiles: torch.Tensor, target: torch.Tensor, quantiles: tuple[float, ...]) -> torch.Tensor:
    """Quantile (pinball) loss averaged over batch, horizon, targets, quantiles.

    Args:
        pred_quantiles: (B, H, T, Q)
        target:         (B, H, T)
        quantiles:      tuple of floats in (0, 1), length Q.
    """
    q = torch.tensor(quantiles, device=pred_quantiles.device, dtype=pred_quantiles.dtype)
    tgt = target.unsqueeze(-1)                              # (B, H, T, 1)
    diff = tgt - pred_quantiles                             # (B, H, T, Q)
    loss = torch.maximum(q * diff, (q - 1) * diff)
    return loss.mean()


def combine_losses(base_loss: torch.Tensor, penalties: dict[str, torch.Tensor], weights: dict[str, float]) -> torch.Tensor:
    """Sum the base loss with weighted penalty terms."""
    total = base_loss
    for name, value in penalties.items():
        w = float(weights.get(name, 0.0))
        if w != 0.0:
            total = total + w * value
    return total
