"""Forecast-detector behavior tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from cloudsentinel.models.forecast_detector import (
    FusedDetector,
    apply_alert_budget,
    changepoint_score,
    compute_detector_streams,
    forecast_residual_score,
    rolling_zscore,
)


def test_zscore_flat_series_is_zero():
    vals = np.ones(100)
    z = rolling_zscore(vals, window=24)
    assert np.allclose(z, 0.0)


def test_zscore_fires_on_spike():
    rng = np.random.default_rng(0)
    vals = rng.standard_normal(200)
    vals[120:125] += 8.0
    z = rolling_zscore(vals, window=48)
    assert z[120:125].max() > 2.0
    assert z[0:50].max() < z[120:125].max()


def test_residual_score_fires_on_jump():
    vals = np.concatenate([np.linspace(0, 1, 100), np.linspace(1, 1, 100) + 5.0])
    r = forecast_residual_score(vals, window=24, order=3)
    assert r[100:110].max() > r[20:80].max()


def test_changepoint_fires_on_level_shift():
    vals = np.concatenate([np.zeros(60), np.ones(60) * 3.0])
    c = changepoint_score(vals, window=20)
    # Peak should be around the shift boundary.
    assert c[55:70].max() > c[0:40].max()


def test_compute_streams_returns_nonneg():
    rng = np.random.default_rng(1)
    vals = rng.standard_normal(150)
    s = compute_detector_streams(vals, rolling_window=24, residual_window=16, changepoint_window=16)
    for name, arr in s.items():
        assert arr.shape == vals.shape
        assert (arr >= -1e-9).all(), name


def test_fused_detector_fit_and_roundtrip():
    rng = np.random.default_rng(0)
    vals = rng.standard_normal(400)
    vals[200:210] += 6.0
    labels = np.zeros(400, dtype=int)
    labels[198:215] = 1

    det = FusedDetector()
    streams = {
        "zscore": rolling_zscore(vals, window=48),
        "residual": forecast_residual_score(vals, window=24),
        "changepoint": changepoint_score(vals, window=24),
    }
    det.fit_fusion(streams, labels)
    scored = det.score_series(vals)
    assert "prob" in scored
    assert (scored["prob"] >= 0).all() and (scored["prob"] <= 1).all()
    # Probability should be higher inside the labeled window.
    assert scored["prob"][198:215].mean() > scored["prob"][0:150].mean()

    # Serialize + reload.
    d = det.to_dict()
    det2 = FusedDetector.from_dict(d)
    s2 = det2.score_series(vals)
    # Probabilities should match closely after round-trip.
    np.testing.assert_allclose(s2["prob"], scored["prob"], atol=1e-6)


def test_fused_detector_fallback_weights_when_single_class():
    det = FusedDetector()
    det.fit_fusion(
        {"zscore": np.zeros(10), "residual": np.zeros(10), "changepoint": np.zeros(10)},
        labels=np.zeros(10, dtype=int),
    )
    assert det.logistic is None  # did not fit due to single class


def test_alert_budget_caps_events_per_day():
    ts = pd.date_range("2024-01-01", periods=100, freq="30min")
    events = pd.DataFrame(
        {
            "timestamp": ts,
            "entity_id": "e",
            "metric": "m",
            "prob": np.linspace(0.5, 0.95, 100),
        }
    )
    kept = apply_alert_budget(events, daily_budget=3, probability_threshold=0.5)
    # At most 3 events per day.
    per_day = kept.groupby(kept["timestamp"].dt.date).size()
    assert (per_day <= 3).all()
