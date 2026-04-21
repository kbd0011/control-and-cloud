"""Self-supervised masked encoder for multivariate telemetry.

Honest framing: this is **not** a wrapper around a pretrained time-series
foundation model (TimesFM / Chronos-2 / etc.). It is a small, trainable local
TCN with masked reconstruction. The module is kept here because:

1. The encoder embedding is a useful drift signal for the detector (one of the
   three ensemble streams is the embedding distance across adjacent windows).
2. It demonstrates masked-reconstruction pretraining on multivariate telemetry.

The interface is pluggable so a pretrained TSFM could replace the backbone
later without touching the detector or RCA layers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EncoderConfig:
    n_features: int
    hidden_dim: int = 48
    num_tcn_blocks: int = 2
    tcn_kernel: int = 3
    dropout: float = 0.1


class _CausalConv1d(nn.Module):
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


class MaskedEncoder(nn.Module):
    """TCN backbone with a decoder head for masked reconstruction."""

    def __init__(self, cfg: EncoderConfig):
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
        self.decode = nn.Linear(D, cfg.n_features)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, F) → (B, D, C)
        y = self.input_proj(x).transpose(1, 2)
        for blk in self.blocks:
            y = blk(y)
        return y.transpose(1, 2)  # (B, C, D)

    def forward(self, x: torch.Tensor) -> dict:
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        summary = latent[:, -1, :]
        return {"latent": latent, "reconstruction": reconstruction, "summary": summary}


def make_masks(
    x: torch.Tensor,
    *,
    mask_ratio: float = 0.2,
    policy: str = "span",
    span_length: int = 4,
    seed: int | None = None,
) -> torch.Tensor:
    """Generate a boolean mask of shape (B, C) marking masked-out steps.

    Policies:
    - "random": independent per-step masking.
    - "span":   contiguous spans of `span_length`, covering ~mask_ratio fraction.
    """
    if x.dim() != 3:
        raise ValueError(f"Expected (B, C, F), got {tuple(x.shape)}")
    B, C, _ = x.shape
    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(seed)

    if policy == "random":
        probs = torch.rand(B, C, generator=g)
        return probs < mask_ratio
    if policy == "span":
        if span_length < 1:
            raise ValueError("span_length must be >= 1")
        mask = torch.zeros(B, C, dtype=torch.bool)
        n_spans = max(1, int(round((C * mask_ratio) / span_length)))
        for b in range(B):
            for _ in range(n_spans):
                if C - span_length <= 0:
                    break
                s = int(torch.randint(0, C - span_length, (1,), generator=g).item())
                mask[b, s : s + span_length] = True
        return mask
    raise ValueError(f"Unknown mask policy: {policy}")


def masked_reconstruction_loss(
    model: MaskedEncoder, x: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Mean squared reconstruction error over masked positions only."""
    if mask.shape != x.shape[:2]:
        raise ValueError(f"Mask shape {tuple(mask.shape)} must match x[:2] = {tuple(x.shape[:2])}")
    x_input = x.clone()
    # Replace masked steps with zeros so the model must reconstruct from context.
    x_input[mask] = 0.0
    out = model(x_input)
    recon = out["reconstruction"]
    # Loss only on masked positions (broadcast over feature dim).
    m = mask.unsqueeze(-1).to(recon.dtype)
    diff = (recon - x) * m
    denom = m.sum() + 1e-8
    return (diff.pow(2).sum()) / denom


def pretrain_step(
    model: MaskedEncoder,
    batch: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    mask_ratio: float = 0.2,
    policy: str = "span",
    span_length: int = 4,
) -> dict:
    """One optimizer step of masked reconstruction."""
    model.train()
    mask = make_masks(batch, mask_ratio=mask_ratio, policy=policy, span_length=span_length)
    loss = masked_reconstruction_loss(model, batch, mask)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {"loss": float(loss.item()), "mask_fraction": float(mask.float().mean().item())}


def embed_windows(model: MaskedEncoder, windows: np.ndarray, device: str = "cpu", batch_size: int = 128) -> np.ndarray:
    """Compute per-window summary embeddings for downstream drift scoring."""
    model.eval()
    out = []
    x = torch.from_numpy(windows.astype(np.float32))
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            chunk = x[i : i + batch_size].to(device)
            emb = model(chunk)["summary"].cpu().numpy()
            out.append(emb)
    if not out:
        return np.zeros((0, model.cfg.hidden_dim), dtype=np.float32)
    return np.concatenate(out, axis=0)
