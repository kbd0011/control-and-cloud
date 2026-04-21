"""Tests for synthetic incident injection."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cloudsentinel.features.synthetic_incident_augment import (
    PATTERNS,
    generate_synthetic_batch,
    inject_pattern,
)


def _make_flat_df() -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=200, freq="5min")
    rows = []
    for ent in ("ent_a", "ent_b"):
        for met in ("latency_ms", "cpu_util"):
            rows.append(
                pd.DataFrame(
                    {
                        "timestamp": ts,
                        "entity_id": ent,
                        "metric": met,
                        "value": np.linspace(10, 10, len(ts)) + 0.1 * np.random.randn(len(ts)),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def test_inject_pattern_changes_series_in_window():
    rng = np.random.default_rng(0)
    series = 10.0 + 0.1 * rng.standard_normal(200)
    for p in PATTERNS:
        out = inject_pattern(series, p, start=50, width=10, strength=1.5)
        assert out.shape == series.shape
        # Something must change in the window.
        assert not np.allclose(out[50:60], series[50:60])


def test_inject_pattern_rejects_unknown():
    with pytest.raises(ValueError):
        inject_pattern(np.zeros(50), "nope", start=0, width=5)


def test_generate_batch_returns_events_df():
    df = _make_flat_df()
    aug, events = generate_synthetic_batch(df, n_events=3, seed=0)
    assert len(events) <= 3
    # Each event references a real (entity, metric) pair from df.
    pairs = set(map(tuple, df[["entity_id", "metric"]].drop_duplicates().values))
    for _, ev in events.iterrows():
        assert (ev["entity_id"], ev["metric"]) in pairs


def test_generate_batch_preserves_length():
    df = _make_flat_df()
    aug, _ = generate_synthetic_batch(df, n_events=2, seed=0)
    assert len(aug) == len(df)
    assert set(aug.columns) == set(df.columns)


def test_generate_batch_is_deterministic():
    df = _make_flat_df()
    a, _ = generate_synthetic_batch(df, n_events=3, seed=42)
    b, _ = generate_synthetic_batch(df, n_events=3, seed=42)
    pd.testing.assert_series_equal(a["value"], b["value"])
