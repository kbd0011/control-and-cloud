"""Tests for CESNET out-of-domain ingestion."""
from __future__ import annotations

import numpy as np
import pandas as pd

from cloudsentinel.data.ingest_cesnet import build_external_eval_windows, read_cesnet_subset


def test_read_cesnet_subset_returns_canonical_schema(sample_data_dir):
    df = read_cesnet_subset(sample_data_dir / "cesnet_external.csv")
    assert {"timestamp", "entity_id", "metric", "value"}.issubset(df.columns)
    assert df["metric"].nunique() >= 1
    assert df["entity_id"].nunique() >= 2


def test_build_external_eval_windows_manifest_marks_ood(sample_data_dir):
    df = read_cesnet_subset(sample_data_dir / "cesnet_external.csv")
    out = build_external_eval_windows(df, context=24, horizon=6, stride=6)
    assert "manifest" in out
    assert out["manifest"]["ood"] is True
    if out["windows"]:
        ctx = out["windows"][0]["context"]
        tgt = out["windows"][0]["target"]
        assert ctx.shape[0] == 24
        assert tgt.shape[0] == 6


def test_build_external_eval_drops_nan_windows():
    rng = np.random.default_rng(0)
    ts = pd.date_range("2024-01-01", periods=60, freq="5min")
    df = pd.DataFrame(
        {
            "timestamp": list(ts) * 2,
            "entity_id": ["ent_a"] * 60 + ["ent_b"] * 60,
            "metric": ["bytes_per_s"] * 120,
            "value": np.concatenate([rng.standard_normal(60), np.nan * np.ones(60)]),
        }
    )
    out = build_external_eval_windows(df, context=24, horizon=6)
    # ent_b is fully NaN; drop. ent_a may contribute windows.
    for w in out["windows"]:
        assert not np.isnan(w["context"]).any()
        assert not np.isnan(w["target"]).any()
