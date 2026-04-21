"""What-if simulation tests."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from thermotwin.simulate import default_scenarios, main as simulate_main
from thermotwin.train import main as train_main


def _write_configs(tmp_path: Path, data_dir: Path, artifacts: Path) -> tuple[Path, Path, Path]:
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


def test_default_scenarios_nonempty():
    scenarios = default_scenarios()
    names = {s.name for s in scenarios}
    assert {"baseline", "lower_setpoint_2C", "cooling_degrade"}.issubset(names)


def test_simulate_produces_deterministic_ranking(tmp_path, sample_data_dir):
    artifacts = tmp_path / "artifacts"
    base, data, model = _write_configs(tmp_path, sample_data_dir, artifacts)
    args = ["--base-config", str(base), "--data-config", str(data), "--model-config", str(model)]
    assert train_main(args) == 0
    assert simulate_main(args) == 0
    report = json.loads((artifacts / "simulation_report.json").read_text())
    assert "ranking" in report
    assert len(report["ranking"]) == len(default_scenarios())
    # Running simulate again should produce the same ranking order for the same checkpoint.
    assert simulate_main(args) == 0
    report2 = json.loads((artifacts / "simulation_report.json").read_text())
    assert [r["scenario"] for r in report["ranking"]] == [r["scenario"] for r in report2["ranking"]]


def test_different_scenarios_produce_different_forecasts(tmp_path, sample_data_dir):
    artifacts = tmp_path / "artifacts"
    base, data, model = _write_configs(tmp_path, sample_data_dir, artifacts)
    args = ["--base-config", str(base), "--data-config", str(data), "--model-config", str(model)]
    assert train_main(args) == 0
    assert simulate_main(args) == 0
    report = json.loads((artifacts / "simulation_report.json").read_text())
    medians = {r["scenario"]: np.asarray(r["median"]) for r in report["scenarios"]}
    # Baseline vs lower setpoint should differ somewhere.
    assert not np.allclose(medians["baseline"], medians["lower_setpoint_2C"])
