"""API tests using real feature windows (no random tensors)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from thermotwin.api import create_app
from thermotwin.data.ingest_m100 import load_and_prepare
from thermotwin.features.resample_and_window import ScalerStats, build_windows, time_block_split
from thermotwin.train import main as train_main
from thermotwin.utils.io import load_json


def _write_configs(tmp_path: Path, data_dir: Path, artifacts: Path):
    base = tmp_path / "base.yaml"
    data = tmp_path / "data.yaml"
    model = tmp_path / "model.yaml"
    base.write_text(
        f"""
project_name: "ThermoTwin-FM"
seed: 0
artifacts_dir: "{artifacts}"
data_dir: "{data_dir}"
train:
  batch_size: 16
  epochs: 1
  learning_rate: 0.001
  weight_decay: 0.0
  early_stop_patience: 2
  num_workers: 0
  device: "cpu"
  grad_clip: 1.0
  val_fraction: 0.2
  test_fraction: 0.2
eval:
  horizons_minutes: [5]
  hotspot_threshold_c: 25.0
  conformal_alpha: 0.1
logging:
  level: "WARNING"
"""
    )
    data.write_text(
        f"""
dataset:
  telemetry_path: "{data_dir / 'telemetry.csv'}"
  layout_path: "{data_dir / 'layout.csv'}"
  cadence_minutes: 5
  n_racks: 4
  n_cooling_units: 2
  duration_days: 2
  use_missingness_mask: true
windowing:
  context_steps: 8
  horizon_steps: 4
  stride: 2
features:
  targets: ["inlet_temp", "outlet_temp"]
  covariates: ["cpu_load", "gpu_load", "power_draw_kw", "fan_speed_pct", "coolant_supply_temp", "ambient_temp"]
  controls: ["fan_speed_pct", "coolant_supply_temp"]
  hotspot_target: "inlet_temp"
"""
    )
    model.write_text(
        """
model:
  hidden_dim: 16
  num_tcn_blocks: 1
  tcn_kernel: 3
  gru_layers: 1
  dropout: 0.0
  quantiles: [0.1, 0.5, 0.9]
graph:
  enabled: false
  n_message_passing: 0
  soft_edge_corr_threshold: 0.4
losses:
  point_loss: "pinball"
  use_physics_penalties: false
  monotonicity_weight: 0.0
  smoothness_weight: 0.0
  control_lag_weight: 0.0
baselines:
  naive_last: true
  seasonal_naive_lag_steps: 4
  linear_ar_enabled: false
  linear_ar_lags: 3
"""
    )
    return base, data, model


def _trained_app(tmp_path: Path, sample_data_dir: Path):
    artifacts = tmp_path / "artifacts"
    base, data, model = _write_configs(tmp_path, sample_data_dir, artifacts)
    rc = train_main(["--base-config", str(base), "--data-config", str(data), "--model-config", str(model)])
    assert rc == 0
    return create_app(artifacts), artifacts


def test_health_endpoint(tmp_path, sample_data_dir):
    app, _ = _trained_app(tmp_path, sample_data_dir)
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "feature_order" in body
    assert "target_order" in body


def test_forecast_with_real_window(tmp_path, sample_data_dir):
    app, artifacts = _trained_app(tmp_path, sample_data_dir)
    client = TestClient(app)
    manifest = load_json(artifacts / "model_manifest.json")
    scaler = ScalerStats.from_dict(load_json(artifacts / "scaler.json"))

    # Reconstruct a real (unscaled) context window from the test split.
    telemetry = load_and_prepare(sample_data_dir / "telemetry.csv", cadence_minutes=5)
    splits = time_block_split(telemetry, val_fraction=0.2, test_fraction=0.2)
    features = manifest["feature_order"]
    targets = manifest["target_order"]
    controls = manifest["control_order"]
    # Build windows on UNSCALED data for API input (API scales internally).
    win = build_windows(
        splits["test"],
        features=features,
        targets=targets,
        controls=controls,
        context_steps=int(manifest["context_steps"]),
        horizon_steps=int(manifest["horizon_steps"]),
        stride=1,
    )
    rack_ids = sorted(telemetry["rack_id"].unique())
    # Find first window's rack.
    entity_id = int(win.entity[0])
    rack_id = rack_ids[entity_id]
    ctx = win.context[0].tolist()
    ctrl = win.controls[0].tolist()

    r = client.post("/forecast", json={"rack_id": rack_id, "context": ctx, "controls": ctrl})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["rack_id"] == rack_id
    assert body["horizon_steps"] == int(manifest["horizon_steps"])
    assert len(body["median"]) == int(manifest["horizon_steps"])
    assert len(body["median"][0]) == len(targets)
    # q_low <= median <= q_high
    med = np.asarray(body["median"])
    lo = np.asarray(body["q_low"])
    hi = np.asarray(body["q_high"])
    assert (lo <= med + 1e-5).all()
    assert (med <= hi + 1e-5).all()


def test_forecast_rejects_bad_shape(tmp_path, sample_data_dir):
    app, artifacts = _trained_app(tmp_path, sample_data_dir)
    client = TestClient(app)
    manifest = load_json(artifacts / "model_manifest.json")
    # Known rack_id
    from thermotwin.graphs.layout_graph import LayoutGraph
    lg = LayoutGraph.load(artifacts / "layout_graph.json")
    rack_id = lg.node_order[0]
    # Wrong context shape.
    r = client.post(
        "/forecast",
        json={
            "rack_id": rack_id,
            "context": [[0.0] * len(manifest["feature_order"])],   # only 1 row
            "controls": [[0.0] * len(manifest["control_order"])] * int(manifest["horizon_steps"]),
        },
    )
    assert r.status_code == 422


def test_unknown_rack_rejected(tmp_path, sample_data_dir):
    app, artifacts = _trained_app(tmp_path, sample_data_dir)
    client = TestClient(app)
    manifest = load_json(artifacts / "model_manifest.json")
    C = int(manifest["context_steps"])
    H = int(manifest["horizon_steps"])
    F = len(manifest["feature_order"])
    K = len(manifest["control_order"])
    r = client.post(
        "/forecast",
        json={
            "rack_id": "not_a_real_rack",
            "context": [[0.0] * F for _ in range(C)],
            "controls": [[0.0] * K for _ in range(H)],
        },
    )
    assert r.status_code == 404


def test_what_if_changes_output(tmp_path, sample_data_dir):
    app, artifacts = _trained_app(tmp_path, sample_data_dir)
    client = TestClient(app)
    manifest = load_json(artifacts / "model_manifest.json")
    telemetry = load_and_prepare(sample_data_dir / "telemetry.csv", cadence_minutes=5)
    splits = time_block_split(telemetry, val_fraction=0.2, test_fraction=0.2)
    win = build_windows(
        splits["test"],
        features=manifest["feature_order"],
        targets=manifest["target_order"],
        controls=manifest["control_order"],
        context_steps=int(manifest["context_steps"]),
        horizon_steps=int(manifest["horizon_steps"]),
        stride=1,
    )
    rack_id = sorted(telemetry["rack_id"].unique())[int(win.entity[0])]
    payload = {
        "rack_id": rack_id,
        "context": win.context[0].tolist(),
        "controls": win.controls[0].tolist(),
    }
    base = client.post("/forecast", json=payload).json()
    what_if = client.post(
        "/what_if",
        json={**payload, "control_deltas": {"coolant_supply_temp": -3.0}, "load_multiplier": 1.0},
    ).json()
    # A 3°C cooler setpoint should change the forecast.
    assert base["median"] != what_if["median"]
