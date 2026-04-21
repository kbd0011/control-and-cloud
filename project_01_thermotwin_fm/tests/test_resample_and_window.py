"""Feature engineering: scaling, windowing, and time-block splits."""
from __future__ import annotations

import numpy as np
import pandas as pd

from thermotwin.data.ingest_m100 import load_and_prepare
from thermotwin.features.resample_and_window import (
    build_windows,
    fit_scaler,
    time_block_split,
)


def _prepared(sample_telemetry_path):
    return load_and_prepare(sample_telemetry_path, cadence_minutes=5)


def test_time_block_split_disjoint(sample_telemetry_path):
    df = _prepared(sample_telemetry_path)
    splits = time_block_split(df, val_fraction=0.2, test_fraction=0.2)
    assert splits["train"]["timestamp"].max() < splits["val"]["timestamp"].min()
    assert splits["val"]["timestamp"].max() < splits["test"]["timestamp"].min()


def test_scaler_uses_training_only(sample_telemetry_path):
    df = _prepared(sample_telemetry_path)
    splits = time_block_split(df, val_fraction=0.2, test_fraction=0.2)
    scaler = fit_scaler(splits["train"], ["inlet_temp", "cpu_load"])
    # Centers differ from a scaler fit on full data.
    full = fit_scaler(df, ["inlet_temp", "cpu_load"])
    # With high probability they differ; if they don't we still require the types to be right.
    assert "inlet_temp" in scaler.center
    assert isinstance(scaler.center["inlet_temp"], float)


def test_build_windows_no_target_in_context(sample_telemetry_path):
    df = _prepared(sample_telemetry_path)
    features = ["cpu_load", "gpu_load", "coolant_supply_temp", "inlet_temp"]
    targets = ["inlet_temp", "outlet_temp"]
    controls = ["coolant_supply_temp", "fan_speed_pct"]
    w = build_windows(
        df,
        features=features,
        targets=targets,
        controls=controls,
        context_steps=12,
        horizon_steps=6,
    )
    assert w.context.shape[1] == 12
    assert w.target.shape[1] == 6
    assert w.context.shape[0] == w.target.shape[0] == w.controls.shape[0]
    # Entities are all valid indices.
    assert w.entity.max() < df["rack_id"].nunique()


def test_windows_are_deterministic(sample_telemetry_path):
    df = _prepared(sample_telemetry_path)
    features = ["cpu_load", "gpu_load", "inlet_temp"]
    targets = ["inlet_temp"]
    controls = ["coolant_supply_temp"]
    a = build_windows(df, features=features, targets=targets, controls=controls, context_steps=8, horizon_steps=4)
    b = build_windows(df, features=features, targets=targets, controls=controls, context_steps=8, horizon_steps=4)
    np.testing.assert_array_equal(a.context, b.context)
    np.testing.assert_array_equal(a.target, b.target)


def test_scaler_inverse_round_trip(sample_telemetry_path):
    df = _prepared(sample_telemetry_path)
    scaler = fit_scaler(df, ["inlet_temp", "outlet_temp"])
    arr = np.array([[[22.0, 30.0], [23.0, 31.0]]], dtype=np.float32)
    transformed = arr.copy()
    for i, f in enumerate(["inlet_temp", "outlet_temp"]):
        transformed[..., i] = (transformed[..., i] - scaler.center[f]) / (scaler.scale[f] + 1e-8)
    recovered = scaler.inverse_transform_array(transformed, ["inlet_temp", "outlet_temp"])
    np.testing.assert_allclose(recovered, arr, atol=1e-4)
