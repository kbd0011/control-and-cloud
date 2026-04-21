"""Ingest and normalize M100-like telemetry into a research-ready table.

The real M100 ExaData uses a Parquet partition layout and non-uniform metric
names. In this repo we emulate that behavior on CSV files produced by the
synthetic generator (or by the user). The public functions:

- read_m100_partition: load a partition file (CSV or Parquet) into a DataFrame.
- canonicalize_metrics: rename source-specific columns to the canonical schema.
- resample_metric_family: resample per-rack series to a uniform cadence and
  keep a missingness mask.
- build_manifest: build a small JSON manifest describing schema and coverage.
- load_and_prepare: high-level convenience used throughout the repo.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Canonical schema expected downstream. Any dataset ingested must produce these.
CANONICAL_COLUMNS = [
    "timestamp",
    "rack_id",
    "cooling_unit_id",
    "cpu_load",
    "gpu_load",
    "power_draw_kw",
    "fan_speed_pct",
    "coolant_supply_temp",
    "ambient_temp",
    "inlet_temp",
    "outlet_temp",
]

NUMERIC_COLUMNS = [c for c in CANONICAL_COLUMNS if c not in ("timestamp", "rack_id", "cooling_unit_id")]


def read_m100_partition(path: str | Path) -> pd.DataFrame:
    """Read a CSV or Parquet partition into a DataFrame.

    Raises:
        FileNotFoundError: if `path` is missing.
        ValueError: if the file has an unsupported extension.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Telemetry partition not found: {p}")
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p, parse_dates=["timestamp"])
    elif p.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(p)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        raise ValueError(f"Unsupported extension for {p}: expected .csv or .parquet")
    return df


def canonicalize_metrics(df: pd.DataFrame, registry: dict[str, str] | None = None) -> pd.DataFrame:
    """Rename columns using a registry and validate canonical schema.

    Args:
        df: raw DataFrame.
        registry: optional mapping `{source_col -> canonical_col}`. Unknown
            canonical names are rejected loudly instead of silently dropped.
    """
    registry = registry or {}
    unknown_canonical = [v for v in registry.values() if v not in CANONICAL_COLUMNS]
    if unknown_canonical:
        raise ValueError(f"Registry maps to unknown canonical fields: {unknown_canonical}")
    df = df.rename(columns=registry).copy()

    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing canonical columns after rename: {missing}")

    # Type coercion.
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["rack_id"] = df["rack_id"].astype(str)
    df["cooling_unit_id"] = df["cooling_unit_id"].astype(str)
    return df[CANONICAL_COLUMNS].sort_values(["rack_id", "timestamp"]).reset_index(drop=True)


def resample_metric_family(
    df: pd.DataFrame,
    freq: str,
    *,
    group_keys: Iterable[str] = ("rack_id", "cooling_unit_id"),
    keep_missing_mask: bool = True,
) -> pd.DataFrame:
    """Resample each (rack, cooling_unit) group to a uniform cadence.

    Numeric columns are averaged within each bin. Gaps shorter than 3 steps are
    linearly interpolated; longer gaps are left NaN and optionally recorded in a
    `*_missing` mask column.
    """
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must contain 'timestamp' column")
    group_keys = list(group_keys)
    numeric = [c for c in NUMERIC_COLUMNS if c in df.columns]

    out_frames = []
    for keys, sub in df.groupby(group_keys, sort=False):
        sub = sub.set_index("timestamp").sort_index()
        resampled = sub[numeric].resample(freq).mean()
        # Interpolate short gaps only.
        filled = resampled.interpolate(method="linear", limit=2, limit_area="inside")
        if keep_missing_mask:
            mask = resampled.isna().astype(np.int8)
            mask.columns = [f"{c}_missing" for c in mask.columns]
            filled = pd.concat([filled, mask], axis=1)
        filled = filled.reset_index()
        if isinstance(keys, tuple):
            for k, v in zip(group_keys, keys):
                filled[k] = v
        else:
            filled[group_keys[0]] = keys
        out_frames.append(filled)

    out = pd.concat(out_frames, ignore_index=True)
    cols_front = ["timestamp", *group_keys]
    other = [c for c in out.columns if c not in cols_front]
    return out[cols_front + other].sort_values(cols_front).reset_index(drop=True)


def build_manifest(df: pd.DataFrame) -> dict:
    """Summarize schema and coverage for downstream reproducibility."""
    return {
        "n_rows": int(len(df)),
        "n_racks": int(df["rack_id"].nunique()) if "rack_id" in df.columns else 0,
        "n_cooling_units": int(df["cooling_unit_id"].nunique())
        if "cooling_unit_id" in df.columns
        else 0,
        "columns": list(df.columns),
        "time_min": str(df["timestamp"].min()) if "timestamp" in df.columns else None,
        "time_max": str(df["timestamp"].max()) if "timestamp" in df.columns else None,
    }


def load_and_prepare(
    telemetry_path: str | Path,
    cadence_minutes: int = 5,
    registry: dict[str, str] | None = None,
) -> pd.DataFrame:
    """High-level ingestion pipeline used by train/evaluate/serve."""
    raw = read_m100_partition(telemetry_path)
    canon = canonicalize_metrics(raw, registry=registry)
    resampled = resample_metric_family(canon, freq=f"{cadence_minutes}min")
    logger.info(
        "Loaded %d rows across %d racks, %d CUs",
        len(resampled),
        resampled["rack_id"].nunique(),
        resampled["cooling_unit_id"].nunique(),
    )
    return resampled
