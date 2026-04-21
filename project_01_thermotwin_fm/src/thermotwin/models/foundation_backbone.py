"""Local sequence backbone for thermal forecasting.

Honest naming note: the scaffold called this file `foundation_backbone.py`
to keep structural compatibility with the TSFM-inspired design. We do NOT
wrap a pretrained TimesFM / Chronos-2 model here — those would require
external weights and network access. Instead we implement a trainable local
backbone that is:

- a stack of causal dilated 1D convolutions (TCN) for long-range context,
- followed by a small GRU for temporal smoothing,
- with quantile forecasting heads (pinball-loss compatible),
- pluggable so the same interface can later host a TSFM if desired.

This is interpretable, small enough to train on a laptop, and the right
baseline for a portfolio-scale project on a synthetic dataset.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BackboneConfig:
    n_features: int
    n_targets: int
    n_controls: int
    horizon: int
    hidden_dim: int = 64
    num_tcn_blocks: int = 3
    tcn_kernel: int = 3
    gru_layers: int = 1
    dropout: float = 0.1
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    # Optional: positions in the feature vector that correspond to targets.
    # When provided, the backbone decodes as a residual from the last observed
    # target value, which is a strong inductive bias for short-horizon forecasts.
    target_feature_indices: tuple[int, ...] | None = None


class _CausalConv1d(nn.Module):
    """Causal 1D conv: pads on the left so no future timestep leaks."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class _TCNBlock(nn.Module):
    def __init__(self, dim: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = _CausalConv1d(dim, dim, kernel, dilation)
        self.conv2 = _CausalConv1d(dim, dim, kernel, dilation)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.conv1(x)
        y = self.norm1(y.transpose(1, 2)).transpose(1, 2)
        y = F.gelu(y)
        y = self.drop(y)
        y = self.conv2(y)
        y = self.norm2(y.transpose(1, 2)).transpose(1, 2)
        y = F.gelu(y)
        return y + residual


class FoundationBackbone(nn.Module):
    """TCN → GRU encoder with quantile decoding over a fixed horizon.

    Inputs:
        context: (B, C, F)      historical features
        controls: (B, H, K)     future control inputs known at decision time
    Output dict:
        quantiles: (B, H, T, Q) forecast quantiles
        latent:    (B, C, D)    final encoder hidden sequence (for graph SSM)
    """

    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.hidden_dim
        self.input_proj = nn.Linear(cfg.n_features, D)
        self.blocks = nn.ModuleList(
            [
                _TCNBlock(D, cfg.tcn_kernel, dilation=2 ** i, dropout=cfg.dropout)
                for i in range(cfg.num_tcn_blocks)
            ]
        )
        self.gru = nn.GRU(D, D, num_layers=cfg.gru_layers, batch_first=True)
        # Decoder: concatenate encoder summary and future controls, produce horizon × targets × quantiles.
        self.ctrl_proj = nn.Linear(cfg.n_controls, D) if cfg.n_controls > 0 else None
        self.head = nn.Sequential(
            nn.Linear(D * 2 if self.ctrl_proj else D, D),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(D, cfg.n_targets * len(cfg.quantiles)),
        )
        # Init last head layer with small weights + zero bias so the initial
        # prediction is close to the residual baseline (naive-last). Without
        # this, the head outputs random values that dominate the residual
        # and make the model start much worse than naive.
        final_linear = self.head[-1]
        nn.init.normal_(final_linear.weight, std=0.01)
        nn.init.zeros_(final_linear.bias)

    def encode(self, context: torch.Tensor) -> torch.Tensor:
        # context: (B, C, F) → (B, D, C) for conv blocks
        x = self.input_proj(context).transpose(1, 2)
        for blk in self.blocks:
            x = blk(x)
        x = x.transpose(1, 2)  # (B, C, D)
        x, _ = self.gru(x)
        return x

    def forward(self, context: torch.Tensor, controls: torch.Tensor | None = None) -> dict:
        B = context.shape[0]
        H = self.cfg.horizon
        T = self.cfg.n_targets
        Q = len(self.cfg.quantiles)

        latent = self.encode(context)                  # (B, C, D)
        summary = latent[:, -1, :]                     # (B, D)

        if self.ctrl_proj is not None and controls is not None:
            ctrl_feat = self.ctrl_proj(controls)       # (B, H, D)
            summary_rep = summary.unsqueeze(1).expand(-1, H, -1)
            dec_in = torch.cat([summary_rep, ctrl_feat], dim=-1)  # (B, H, 2D)
        else:
            dec_in = summary.unsqueeze(1).expand(-1, H, -1)       # (B, H, D)

        out = self.head(dec_in)                                   # (B, H, T*Q)
        out = out.view(B, H, T, Q)

        # Enforce monotone quantiles via cumulative softplus on differences.
        # A small floor keeps intervals from collapsing to zero during training.
        base = out[..., :1]
        deltas = F.softplus(out[..., 1:]) + 0.1
        quantiles = torch.cat([base, base + torch.cumsum(deltas, dim=-1)], dim=-1)

        # If target positions in the feature vector are known, shift ALL
        # quantiles by the last observed target value. This is a strong
        # inductive bias (residual from naive-last) without distorting widths.
        if self.cfg.target_feature_indices is not None:
            idx = torch.tensor(self.cfg.target_feature_indices, dtype=torch.long, device=context.device)
            last_target = context.index_select(-1, idx)[:, -1, :]   # (B, T)
            quantiles = quantiles + last_target.view(B, 1, T, 1)

        return {"quantiles": quantiles, "latent": latent}


def load_backbone(cfg: BackboneConfig, ckpt: str | None = None) -> FoundationBackbone:
    model = FoundationBackbone(cfg)
    if ckpt:
        state = torch.load(ckpt, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
    return model
