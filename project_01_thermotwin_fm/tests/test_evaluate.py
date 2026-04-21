"""Evaluation metrics and end-to-end evaluate smoke test."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from thermotwin.evaluate import (
    compute_calibration,
    compute_forecast_metrics,
    compute_hotspot_metrics,
    conformal_quantile_adjustments,
)


def test_forecast_metrics_on_toy_arrays():
    y = np.zeros((4, 3, 2))
    p = np.ones_like(y)
    m = compute_forecast_metrics(y, p)
    assert abs(m["mae"] - 1.0) < 1e-8
    assert abs(m["rmse"] - 1.0) < 1e-8


def test_forecast_metrics_shape_mismatch():
    try:
        compute_forecast_metrics(np.zeros((1, 2, 3)), np.zeros((1, 2, 4)))
    except ValueError:
        return
    raise AssertionError("expected ValueError on shape mismatch")


def test_calibration_on_fully_covered_case():
    y = np.zeros((10, 5, 2))
    lo = np.full_like(y, -1.0)
    hi = np.full_like(y, 1.0)
    r = compute_calibration(y, lo, hi, alpha=0.1)
    assert abs(r["empirical_coverage"] - 1.0) < 1e-8


def test_hotspot_metrics_degenerate_case():
    # All negatives: auroc/auprc shouldn't be emitted, no exceptions.
    y = np.zeros((5, 3))
    p = np.random.rand(5, 3)
    r = compute_hotspot_metrics(y, p, threshold=0.5)
    assert "n_samples" in r
    assert "auroc" not in r


def test_hotspot_metrics_binary_case():
    y = np.array([[30.0, 31.0], [20.0, 21.0]])
    p = np.array([[30.0, 31.0], [20.0, 21.0]])
    r = compute_hotspot_metrics(y, p, threshold=25.0)
    assert r["f1"] == 1.0
    assert r["auroc"] == 1.0


def test_conformal_quantile_adjustment_shrinks_with_more_data():
    rng = np.random.default_rng(0)
    # Simulate quantiles where model already covers perfectly: scores ~ 0.
    val_q = np.stack([np.zeros((200, 5, 2)), np.ones((200, 5, 2))], axis=-1)
    val_tgt = np.full((200, 5, 2), 0.5)
    out = conformal_quantile_adjustments(val_q, val_tgt, alpha=0.1)
    assert out["q_hat"] <= 0.5 + 1e-6
    assert out["n_calibration"] == 200 * 5 * 2


def test_evaluate_end_to_end(tmp_path, sample_data_dir):
    """Run train, then evaluate, and inspect the produced report."""
    from thermotwin.train import main as train_main
    from thermotwin.evaluate import main as evaluate_main

    base = tmp_path / "base.yaml"
    data = tmp_path / "data.yaml"
    model = tmp_path / "model.yaml"
    artifacts = tmp_path / "artifacts"

    base.write_text(
        f"""
project_name: "ThermoTwin-FM"
seed: 0
artifacts_dir: "{artifacts}"
data_dir: "{sample_data_dir}"
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
  telemetry_path: "{sample_data_dir / 'telemetry.csv'}"
  layout_path: "{sample_data_dir / 'layout.csv'}"
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
  encoder: "none"
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
    args = [
        "--base-config", str(base),
        "--data-config", str(data),
        "--model-config", str(model),
    ]
    assert train_main(args) == 0
    assert evaluate_main(args) == 0
    report = json.loads((artifacts / "eval_report.json").read_text())
    assert "point_metrics" in report
    assert "per_horizon" in report
    assert "hotspot" in report
    assert report["point_metrics"]["mae"] >= 0
