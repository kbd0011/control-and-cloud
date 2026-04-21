"""Ingestion schema validation and determinism."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thermotwin.data.ingest_m100 import (
    CANONICAL_COLUMNS,
    build_manifest,
    canonicalize_metrics,
    load_and_prepare,
    read_m100_partition,
    resample_metric_family,
)


def test_read_partition_returns_canonical_columns(sample_telemetry_path):
    df = read_m100_partition(sample_telemetry_path)
    for col in ["timestamp", "rack_id", "cooling_unit_id", "inlet_temp", "outlet_temp"]:
        assert col in df.columns


def test_canonicalize_rejects_unknown_mapping(sample_telemetry_path):
    df = read_m100_partition(sample_telemetry_path)
    with pytest.raises(ValueError):
        canonicalize_metrics(df, registry={"inlet_temp": "not_a_real_column"})


def test_canonicalize_requires_all_fields():
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=3, freq="5min")})
    with pytest.raises(ValueError):
        canonicalize_metrics(df)


def test_resample_preserves_order_and_has_mask(sample_telemetry_path):
    df = canonicalize_metrics(read_m100_partition(sample_telemetry_path))
    r = resample_metric_family(df, freq="5min", keep_missing_mask=True)
    assert any(col.endswith("_missing") for col in r.columns)
    # Monotone non-decreasing per rack.
    for _, sub in r.groupby("rack_id"):
        ts = sub["timestamp"].to_numpy()
        assert np.all(ts[:-1] <= ts[1:])


def test_resample_is_deterministic(sample_telemetry_path):
    df = canonicalize_metrics(read_m100_partition(sample_telemetry_path))
    a = resample_metric_family(df, freq="5min")
    b = resample_metric_family(df, freq="5min")
    pd.testing.assert_frame_equal(a, b)


def test_load_and_prepare_end_to_end(sample_telemetry_path):
    df = load_and_prepare(sample_telemetry_path, cadence_minutes=5)
    assert "inlet_temp" in df.columns
    assert df["rack_id"].nunique() >= 2
    manifest = build_manifest(df)
    assert manifest["n_rows"] == len(df)
    assert manifest["n_racks"] >= 2
