"""Merge facility and workload streams, build forecasting windows, and enforce leakage-safe splits.

Core contracts:

- `build_windows` produces numpy tensors with shapes:
    context: (N, C, F)    where C = context_steps, F = number of features
    target:  (N, H, T)    where H = horizon_steps, T = number of targets
    entity:  (N,)         integer entity (rack) id
    controls:(N, H, K)    future control inputs known at decision time
    mask:    (N, C, F)    1 where observed, 0 where imputed/missing
- `time_block_split` guarantees train/val/test windows are time-disjoint.
- `fit_scaler` computes robust (median / IQR) stats on training data only.

These interfaces are reused by CloudSentinel (anomaly detection windows) and
SafeCool (offline RL observation windows) so keep them small and stable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ScalerStats:
    """Robust per-feature center and scale fit on training data only."""

    center: dict[str, float]
    scale: dict[str, float]
    feature_order: list[str]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for f in self.feature_order:
            if f in out.columns:
                out[f] = (out[f].astype(float) - self.center[f]) / (self.scale[f] + 1e-8)
        return out

    def inverse_transform_array(self, arr: np.ndarray, feature_order: Sequence[str]) -> np.ndarray:
        """Inverse-scale a numpy array whose last dim corresponds to `feature_order`."""
        out = arr.astype(float).copy()
        for i, f in enumerate(feature_order):
            if f in self.center:
                out[..., i] = out[..., i] * (self.scale[f] + 1e-8) + self.center[f]
        return out

    def to_dict(self) -> dict:
        return {"center": self.center, "scale": self.scale, "feature_order": self.feature_order}

    @classmethod
    def from_dict(cls, d: dict) -> "ScalerStats":
        return cls(center=d["center"], scale=d["scale"], feature_order=d["feature_order"])


@dataclass
class WindowArrays:
    context: np.ndarray
    target: np.ndarray
    controls: np.ndarray
    mask: np.ndarray
    entity: np.ndarray
    start_times: np.ndarray
    feature_order: list[str] = field(default_factory=list)
    target_order: list[str] = field(default_factory=list)
    control_order: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return self.context.shape[0]


def merge_streams(frames: Iterable[pd.DataFrame], on: Sequence[str] = ("timestamp", "rack_id")) -> pd.DataFrame:
    """Outer-join multiple streams on shared keys."""
    frames = list(frames)
    if not frames:
        raise ValueError("merge_streams requires at least one DataFrame")
    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on=list(on), how="outer")
    return merged.sort_values(list(on)).reset_index(drop=True)


def fit_scaler(train_df: pd.DataFrame, features: Sequence[str]) -> ScalerStats:
    """Fit robust (median / IQR) statistics on the training split only."""
    center: dict[str, float] = {}
    scale: dict[str, float] = {}
    for f in features:
        if f not in train_df.columns:
            raise KeyError(f"Feature '{f}' missing from training frame")
        vals = train_df[f].astype(float).dropna().values
        if len(vals) == 0:
            center[f] = 0.0
            scale[f] = 1.0
            continue
        med = float(np.median(vals))
        q75, q25 = np.percentile(vals, [75, 25])
        iqr = float(q75 - q25)
        center[f] = med
        scale[f] = iqr if iqr > 1e-6 else float(np.std(vals) + 1e-6)
    return ScalerStats(center=center, scale=scale, feature_order=list(features))


def time_block_split(
    df: pd.DataFrame,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    *,
    time_col: str = "timestamp",
) -> dict[str, pd.DataFrame]:
    """Split by time block so train/val/test are time-disjoint."""
    if not 0 < val_fraction + test_fraction < 1:
        raise ValueError("val_fraction + test_fraction must be in (0, 1)")
    sorted_times = np.sort(df[time_col].unique())
    n = len(sorted_times)
    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)
    train_end = sorted_times[n - n_val - n_test - 1]
    val_end = sorted_times[n - n_test - 1]
    train = df[df[time_col] <= train_end].copy()
    val = df[(df[time_col] > train_end) & (df[time_col] <= val_end)].copy()
    test = df[df[time_col] > val_end].copy()
    return {"train": train, "val": val, "test": test}


def build_windows(
    df: pd.DataFrame,
    *,
    features: Sequence[str],
    targets: Sequence[str],
    controls: Sequence[str],
    context_steps: int,
    horizon_steps: int,
    stride: int = 1,
    entity_col: str = "rack_id",
    time_col: str = "timestamp",
) -> WindowArrays:
    """Build sliding windows per entity, guaranteeing no target timestamp appears inside its input window."""
    if context_steps <= 0 or horizon_steps <= 0:
        raise ValueError("context_steps and horizon_steps must be positive")
    all_needed = list(dict.fromkeys(list(features) + list(targets) + list(controls)))
    for c in all_needed:
        if c not in df.columns:
            raise KeyError(f"Column '{c}' missing from windowing frame")

    ctx_list, tgt_list, ctrl_list, mask_list, ent_list, time_list = [], [], [], [], [], []
    entities = sorted(df[entity_col].unique())
    ent_to_idx = {e: i for i, e in enumerate(entities)}

    for ent, sub in df.groupby(entity_col, sort=False):
        sub = sub.sort_values(time_col).reset_index(drop=True)
        n = len(sub)
        feat = sub[list(features)].to_numpy(dtype=np.float32)
        tgt = sub[list(targets)].to_numpy(dtype=np.float32)
        ctrl = sub[list(controls)].to_numpy(dtype=np.float32)
        obs_mask = (~sub[list(features)].isna()).to_numpy(dtype=np.float32)
        # Replace NaNs with 0 in the model input; mask carries the signal.
        feat = np.nan_to_num(feat, nan=0.0)
        tgt_nan_mask = np.isnan(tgt).any(axis=1)

        for start in range(0, n - context_steps - horizon_steps + 1, stride):
            ctx_end = start + context_steps
            tgt_end = ctx_end + horizon_steps
            # Drop windows whose target region contains any NaN target value.
            if tgt_nan_mask[ctx_end:tgt_end].any():
                continue
            ctx_list.append(feat[start:ctx_end])
            tgt_list.append(tgt[ctx_end:tgt_end])
            ctrl_list.append(ctrl[ctx_end:tgt_end])
            mask_list.append(obs_mask[start:ctx_end])
            ent_list.append(ent_to_idx[ent])
            time_list.append(sub[time_col].iloc[ctx_end])

    if not ctx_list:
        raise ValueError("No valid windows produced; check data cadence and target NaNs")

    return WindowArrays(
        context=np.stack(ctx_list),
        target=np.stack(tgt_list),
        controls=np.stack(ctrl_list),
        mask=np.stack(mask_list),
        entity=np.asarray(ent_list, dtype=np.int64),
        start_times=np.asarray(time_list),
        feature_order=list(features),
        target_order=list(targets),
        control_order=list(controls),
    )
