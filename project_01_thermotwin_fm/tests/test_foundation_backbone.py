"""Forecasting backbone behavior tests."""
from __future__ import annotations

import torch

from thermotwin.models.foundation_backbone import BackboneConfig, FoundationBackbone


def _mk_cfg() -> BackboneConfig:
    return BackboneConfig(
        n_features=5,
        n_targets=2,
        n_controls=2,
        horizon=6,
        hidden_dim=16,
        num_tcn_blocks=2,
        tcn_kernel=3,
        gru_layers=1,
        dropout=0.0,
        quantiles=(0.1, 0.5, 0.9),
    )


def test_forward_shapes_match():
    cfg = _mk_cfg()
    model = FoundationBackbone(cfg)
    ctx = torch.randn(3, 12, cfg.n_features)
    ctrl = torch.randn(3, cfg.horizon, cfg.n_controls)
    out = model(ctx, ctrl)
    assert out["quantiles"].shape == (3, cfg.horizon, cfg.n_targets, len(cfg.quantiles))
    assert out["latent"].shape == (3, 12, cfg.hidden_dim)


def test_quantiles_are_monotone():
    cfg = _mk_cfg()
    model = FoundationBackbone(cfg)
    ctx = torch.randn(2, 12, cfg.n_features)
    ctrl = torch.randn(2, cfg.horizon, cfg.n_controls)
    out = model(ctx, ctrl)
    q = out["quantiles"]  # (B, H, T, Q)
    diffs = q[..., 1:] - q[..., :-1]
    assert (diffs >= -1e-5).all()


def test_backbone_backprops():
    cfg = _mk_cfg()
    model = FoundationBackbone(cfg)
    ctx = torch.randn(4, 12, cfg.n_features, requires_grad=False)
    ctrl = torch.randn(4, cfg.horizon, cfg.n_controls, requires_grad=False)
    target = torch.randn(4, cfg.horizon, cfg.n_targets)
    out = model(ctx, ctrl)
    median = out["quantiles"][..., len(cfg.quantiles) // 2]
    loss = (median - target).pow(2).mean()
    loss.backward()
    grad_magnitudes = [p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None]
    assert any(g > 0 for g in grad_magnitudes)
