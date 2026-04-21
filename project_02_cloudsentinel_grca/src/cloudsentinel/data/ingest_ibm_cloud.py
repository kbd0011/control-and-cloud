"""Load and normalize IBM-Cloud-style high-dimensional telemetry.

The real IBM Cloud anomaly dataset is a wide matrix with ~117k columns. This
repo does not ship it. Instead we support two input shapes that match how a
researcher would pre-process it before modeling:

- **Long-form**: columns `timestamp, entity_id, metric, value`. This is the
  canonical representation used throughout the pipeline and what our synthetic
  generator produces.
- **Wide-form**: columns `timestamp, <entity>__<metric>, ...`. Useful for
  ingesting the raw IBM matrix; `melt_wide_to_long` converts it to long form.

The canonical schema (below) is what every downstream module depends on.
Anything that doesn't map cleanly into it should be rejected loudly.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CANONICAL_COLUMNS = ("timestamp", "entity_id", "metric", "value")


def read_ibm_matrix(path: str | Path) -> pd.DataFrame:
    """Load a telemetry partition (CSV or Parquet) without schema normalization.

    Raises FileNotFoundError / ValueError on bad input.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Telemetry partition not found: {p}")
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p, parse_dates=["timestamp"] if "timestamp" in _peek_columns(p) else False)
    elif p.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(p)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        raise ValueError(f"Unsupported extension for {p}: expected .csv or .parquet")
    return df


def _peek_columns(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        header = f.readline().strip()
    return header.split(",")


def melt_wide_to_long(
    df: pd.DataFrame,
    *,
    sep: str = "__",
    time_col: str = "timestamp",
) -> pd.DataFrame:
    """Convert wide-form (columns like `entity__metric`) to long form."""
    if time_col not in df.columns:
        raise ValueError(f"Missing time column '{time_col}'")
    value_cols = [c for c in df.columns if c != time_col]
    if not value_cols:
        raise ValueError("No non-time columns to melt")
    melted = df.melt(id_vars=[time_col], value_vars=value_cols, var_name="entity_metric", value_name="value")
    parts = melted["entity_metric"].str.split(sep, n=1, expand=True)
    if parts.shape[1] < 2 or parts[1].isna().any():
        raise ValueError(
            f"Wide column names must contain '{sep}' between entity and metric; "
            f"bad examples: {melted.loc[parts.isna().any(axis=1), 'entity_metric'].head().tolist()}"
        )
    melted["entity_id"] = parts[0]
    melted["metric"] = parts[1]
    melted = melted.rename(columns={time_col: "timestamp"})
    return melted[list(CANONICAL_COLUMNS)].copy()


def canonicalize_ibm_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and coerce a long-form telemetry DataFrame to the canonical schema."""
    missing = set(CANONICAL_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Long-form telemetry missing columns: {sorted(missing)}")
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out["entity_id"] = out["entity_id"].astype(str)
    out["metric"] = out["metric"].astype(str)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return (
        out[list(CANONICAL_COLUMNS)]
        .sort_values(["entity_id", "metric", "timestamp"])
        .reset_index(drop=True)
    )


def split_dense_sparse(
    df: pd.DataFrame, threshold: float = 0.3
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split telemetry into dense vs sparse groups based on per-(entity, metric) NaN rate.

    Returns (dense_df, sparse_df). A (entity, metric) pair is "sparse" if its NaN
    rate exceeds `threshold`. Operators typically deprioritize sparse streams
    for real-time detection because their residual distributions are unreliable.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    nan_rates = (
        df.assign(is_nan=df["value"].isna().astype(float))
        .groupby(["entity_id", "metric"])["is_nan"]
        .mean()
    )
    dense_keys = set(nan_rates[nan_rates <= threshold].index)
    key = list(zip(df["entity_id"], df["metric"]))
    in_dense = np.array([k in dense_keys for k in key])
    return df[in_dense].reset_index(drop=True), df[~in_dense].reset_index(drop=True)


def pivot_to_matrix(df: pd.DataFrame, fillna: float | None = 0.0) -> pd.DataFrame:
    """Pivot long-form telemetry to a wide (timestamp × entity__metric) matrix."""
    df = canonicalize_ibm_schema(df)
    df = df.assign(col=df["entity_id"] + "__" + df["metric"])
    mat = df.pivot_table(index="timestamp", columns="col", values="value", aggfunc="mean")
    if fillna is not None:
        mat = mat.fillna(fillna)
    return mat.sort_index()


def load_and_prepare(
    telemetry_path: str | Path,
    *,
    wide: bool = False,
) -> pd.DataFrame:
    """High-level ingestion path used by train / evaluate / serve."""
    raw = read_ibm_matrix(telemetry_path)
    long_df = melt_wide_to_long(raw) if wide else raw
    canon = canonicalize_ibm_schema(long_df)
    logger.info(
        "Loaded telemetry: %d rows, %d entities, %d metrics",
        len(canon),
        canon["entity_id"].nunique(),
        canon["metric"].nunique(),
    )
    return canon
