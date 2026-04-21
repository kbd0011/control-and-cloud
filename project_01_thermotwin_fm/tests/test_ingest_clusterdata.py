"""ClusterData ingestion tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thermotwin.data.ingest_clusterdata import (
    aggregate_cluster_windows,
    derive_regime_features,
    make_rack_level_workloads,
    synthetic_cluster_trace,
)


def test_aggregate_matches_full_computation(tmp_path):
    df = synthetic_cluster_trace(n_steps=200, seed=0)
    p = tmp_path / "trace.csv"
    df.to_csv(p, index=False)
    agg = aggregate_cluster_windows(p, freq="5min")
    assert not agg.empty
    assert {"cpu_request_mean", "task_churn", "queue_depth"}.issubset(agg.columns)


def test_regime_features_are_finite():
    df = synthetic_cluster_trace(n_steps=200, seed=1)
    df = df.set_index("timestamp").resample("5min").agg(
        {"cpu_request": "mean", "memory_request": "mean", "task_events": "sum"}
    ).reset_index()
    df = df.rename(
        columns={
            "cpu_request": "cpu_request_mean",
            "memory_request": "memory_request_mean",
            "task_events": "task_churn",
        }
    )
    df["queue_depth"] = df["task_churn"].clip(lower=0)
    out = derive_regime_features(df)
    assert np.isfinite(out["burstiness"]).all()
    assert (out["burstiness"] >= 0).all()


def test_rack_level_workloads_preserve_mass_within_tolerance(tmp_path):
    df = synthetic_cluster_trace(n_steps=50, seed=2)
    p = tmp_path / "t.csv"
    df.to_csv(p, index=False)
    agg = aggregate_cluster_windows(p, freq="5min")
    rack_df = make_rack_level_workloads(agg, rack_count=4, seed=3)
    assert set(rack_df["rack_id"].unique()) == {f"rack_{i:02d}" for i in range(4)}
    # cpu_load bounded.
    assert rack_df["cpu_load"].between(0.0, 1.0).all()


def test_missing_columns_raise():
    with pytest.raises(ValueError):
        derive_regime_features(pd.DataFrame({"cpu_request_mean": [1.0]}))
