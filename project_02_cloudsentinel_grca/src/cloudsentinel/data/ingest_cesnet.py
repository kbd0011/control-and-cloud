"""Prepare CESNET-TimeSeries24 style data for out-of-domain evaluation.

The real CESNET dataset carries 40 weeks of traffic from ~275k IPs. We expose a
simple long-form ingest that is compatible with the CloudSentinel pipeline:
`timestamp, entity_id, metric, value`. The caller is responsible for providing
a subset (e.g., institution-level aggregates) in that shape. A small fixture
is produced by the synthetic generator under `cesnet_external.csv`.

Out-of-domain flag is stamped into the manifest so evaluate.py can report
in-domain and transfer performance separately per your requirements.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from cloudsentinel.data.ingest_ibm_cloud import canonicalize_ibm_schema

logger = logging.getLogger(__name__)


def read_cesnet_subset(
    path: str | Path,
    level: Literal["entity", "institution"] = "entity",
) -> pd.DataFrame:
    """Load a CESNET-shaped subset into long form and canonicalize."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CESNET subset not found: {p}")
    df = pd.read_csv(p, parse_dates=["timestamp"])
    if level == "institution":
        # Aggregate to institution granularity if a `institution` column is available.
        if "institution" in df.columns:
            df = (
                df.groupby(["timestamp", "institution", "metric"], as_index=False)["value"]
                .sum()
                .rename(columns={"institution": "entity_id"})
            )
    df = canonicalize_ibm_schema(df)
    return df


def build_external_eval_windows(
    df: pd.DataFrame,
    *,
    context: int,
    horizon: int,
    stride: int = 1,
) -> dict:
    """Build simple eval windows for out-of-domain forecasting checks.

    We don't run the full train pipeline here; we just return per-entity numpy
    windows the forecast-residual detector can score. The manifest records
    the 'ood' flag for clean separation in reports.
    """
    if context <= 0 or horizon <= 0:
        raise ValueError("context and horizon must be positive")
    out_windows: list[dict] = []
    pivot = df.pivot_table(
        index="timestamp", columns=["entity_id", "metric"], values="value", aggfunc="mean"
    ).sort_index()
    arr = pivot.to_numpy(dtype=np.float32)
    cols = list(pivot.columns)
    timestamps = pivot.index.to_numpy()
    for start in range(0, arr.shape[0] - context - horizon + 1, stride):
        ctx = arr[start : start + context]
        tgt = arr[start + context : start + context + horizon]
        if np.isnan(ctx).any() or np.isnan(tgt).any():
            continue
        out_windows.append(
            {"context": ctx, "target": tgt, "start_time": str(timestamps[start + context])}
        )

    manifest = {
        "source": "cesnet_timeseries24_subset",
        "ood": True,
        "n_windows": len(out_windows),
        "n_columns": len(cols),
        "context": int(context),
        "horizon": int(horizon),
    }
    return {"windows": out_windows, "manifest": manifest}
