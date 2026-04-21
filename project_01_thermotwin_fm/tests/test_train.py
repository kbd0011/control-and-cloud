"""Train loop smoke test: one epoch on synthetic data, end to end."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from thermotwin.train import main as train_main


@pytest.mark.filterwarnings("ignore")
def test_train_smoke_run(tmp_path, sample_data_dir, monkeypatch):
    # Build tiny configs pointed at the fixture data and a temp artifacts dir.
    base = tmp_path / "base.yaml"
    data = tmp_path / "data.yaml"
    model = tmp_path / "model.yaml"

    base.write_text(
        f"""
project_name: "ThermoTwin-FM"
seed: 0
artifacts_dir: "{tmp_path / 'artifacts'}"
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
  name: "fixture"
  source: "parquet"
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
  backbone: "tcn_gru"
  hidden_dim: 16
  num_tcn_blocks: 2
  tcn_kernel: 3
  gru_layers: 1
  dropout: 0.0
  quantiles: [0.1, 0.5, 0.9]
graph:
  enabled: true
  encoder: "gcn"
  n_message_passing: 1
  soft_edge_corr_threshold: 0.4
losses:
  point_loss: "pinball"
  use_physics_penalties: true
  monotonicity_weight: 0.05
  smoothness_weight: 0.02
  control_lag_weight: 0.02
baselines:
  naive_last: true
  seasonal_naive_lag_steps: 12
  linear_ar_enabled: true
  linear_ar_lags: 4
"""
    )

    rc = train_main(
        [
            "--base-config",
            str(base),
            "--data-config",
            str(data),
            "--model-config",
            str(model),
        ]
    )
    assert rc == 0
    artifacts = tmp_path / "artifacts"
    assert (artifacts / "model.pt").exists()
    assert (artifacts / "model_manifest.json").exists()
    manifest = json.loads((artifacts / "model_manifest.json").read_text())
    assert manifest["feature_order"]
    assert manifest["target_order"]
    assert (artifacts / "scaler.json").exists()
    assert (artifacts / "layout_graph.json").exists()
    assert (artifacts / "val_predictions.npz").exists()
