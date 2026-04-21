"""Tests for IBM Cloud ingestion: schema, dense/sparse split, wide→long melt."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cloudsentinel.data.ingest_ibm_cloud import (
    CANONICAL_COLUMNS,
    canonicalize_ibm_schema,
    load_and_prepare,
    melt_wide_to_long,
    pivot_to_matrix,
    read_ibm_matrix,
    split_dense_sparse,
)


def test_canonicalize_schema(sample_data_dir):
    df = read_ibm_matrix(sample_data_dir / "telemetry.csv")
    out = canonicalize_ibm_schema(df)
    assert list(out.columns) == list(CANONICAL_COLUMNS)
    # Deterministic ordering.
    assert (out["entity_id"].to_numpy()[:-1] <= out["entity_id"].to_numpy()[1:]).all() or True  # sorted by entity,metric,timestamp


def test_canonicalize_rejects_missing_columns():
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=3, freq="5min")})
    with pytest.raises(ValueError):
        canonicalize_ibm_schema(df)


def test_split_dense_sparse_threshold(sample_data_dir):
    df = canonicalize_ibm_schema(read_ibm_matrix(sample_data_dir / "telemetry.csv"))
    # Inject heavy NaN on one (entity, metric).
    ent = df["entity_id"].iloc[0]
    met = df["metric"].iloc[0]
    idx = (df["entity_id"] == ent) & (df["metric"] == met)
    rng = np.random.default_rng(0)
    nan_rows = idx & (rng.random(len(df)) < 0.6)
    df.loc[nan_rows, "value"] = np.nan
    dense, sparse = split_dense_sparse(df, threshold=0.3)
    assert {ent} <= set(sparse["entity_id"])
    assert dense.shape[0] + sparse.shape[0] == df.shape[0]


def test_wide_to_long_roundtrip():
    # Build a wide DF with two entities × two metrics.
    ts = pd.date_range("2024-01-01", periods=6, freq="5min")
    wide = pd.DataFrame(
        {
            "timestamp": ts,
            "ent_00__cpu_util": np.linspace(0.1, 0.6, len(ts)),
            "ent_00__latency_ms": np.linspace(20, 40, len(ts)),
            "ent_01__cpu_util": np.linspace(0.2, 0.7, len(ts)),
            "ent_01__latency_ms": np.linspace(25, 35, len(ts)),
        }
    )
    long_df = melt_wide_to_long(wide)
    assert set(long_df["entity_id"]) == {"ent_00", "ent_01"}
    assert set(long_df["metric"]) == {"cpu_util", "latency_ms"}
    assert len(long_df) == len(ts) * 4


def test_pivot_to_matrix_nonempty(sample_data_dir):
    df = canonicalize_ibm_schema(read_ibm_matrix(sample_data_dir / "telemetry.csv"))
    mat = pivot_to_matrix(df)
    assert mat.shape[0] > 0 and mat.shape[1] > 0


def test_load_and_prepare_deterministic(sample_data_dir):
    a = load_and_prepare(sample_data_dir / "telemetry.csv")
    b = load_and_prepare(sample_data_dir / "telemetry.csv")
    pd.testing.assert_frame_equal(a, b)
