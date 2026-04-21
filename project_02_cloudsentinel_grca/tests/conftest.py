"""Shared pytest fixtures for CloudSentinel-GRCA tests.

Putting shared helpers here (rather than importing them between test files)
means `tests/` does not need to be a package, and `pytest` collection works
on a fresh clone with no `tests/__init__.py` and no PYTHONPATH tricks.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from cloudsentinel.data.synthetic import SyntheticConfig, write_sample_dataset


@pytest.fixture(scope="session")
def sample_data_dir(tmp_path_factory) -> Path:
    """A tiny synthetic dataset reused across most tests."""
    d = tmp_path_factory.mktemp("cloudsentinel_data")
    cfg = SyntheticConfig(
        n_entities=8,
        n_services=3,
        duration_days=3,
        cadence_minutes=5,
        incident_count=4,
        seed=7,
    )
    write_sample_dataset(d, cfg)
    return d


def _render_configs(tmp_path: Path, data_dir: Path, artifacts: Path) -> tuple[Path, Path, Path]:
    """Write minimal base/data/model YAML files and return their paths."""
    base = tmp_path / "base.yaml"
    data = tmp_path / "data.yaml"
    model = tmp_path / "model.yaml"
    base.write_text(
        f"""
project_name: "CloudSentinel"
seed: 0
artifacts_dir: "{artifacts}"
data_dir: "{data_dir}"
train:
  batch_size: 32
  encoder_epochs: 1
  encoder_learning_rate: 0.001
  encoder_weight_decay: 0.0
  device: "cpu"
  grad_clip: 1.0
  val_fraction: 0.2
  test_fraction: 0.2
detector:
  rolling_window: 24
  residual_window: 12
  changepoint_window: 12
  zscore_clip: 10.0
  default_fusion_weights:
    zscore: 0.4
    residual: 0.4
    changepoint: 0.2
rca:
  personalization_alpha: 0.15
  damping: 0.85
  max_iterations: 50
  convergence_tol: 0.00001
  top_k: 3
alerts:
  daily_budget: 20
  probability_threshold: 0.5
logging:
  level: "WARNING"
"""
    )
    data.write_text(
        f"""
dataset:
  telemetry_path: "{data_dir / 'telemetry.csv'}"
  entity_path: "{data_dir / 'entities.csv'}"
  dependencies_path: "{data_dir / 'dependencies.csv'}"
  incidents_path: "{data_dir / 'incidents.csv'}"
  external_telemetry_path: "{data_dir / 'cesnet_external.csv'}"
  cadence_minutes: 5
  n_entities: 8
  n_metrics_per_entity: 4
  duration_days: 3
  sparsity_threshold: 0.3
windowing:
  context_steps: 24
  horizon_steps: 6
  stride: 3
augmentation:
  enabled: false
  event_ratio: 0.0
  patterns: []
  seed: 0
"""
    )
    model.write_text(
        """
encoder:
  backbone: "tcn_masked"
  hidden_dim: 16
  num_tcn_blocks: 1
  tcn_kernel: 3
  dropout: 0.0
  mask_ratio: 0.2
  mask_policy: "span"
  span_length: 3
detector_fusion:
  model: "logistic"
  calibration: "isotonic"
graph:
  use_dependency_edges: true
  use_correlation_edges: true
  correlation_threshold: 0.3
  correlation_window: 48
incident_brief:
  template: "markdown"
  include_top_k_entities: 3
  include_top_k_metrics: 5
  include_graph_paths: 2
  confidence_phrasing:
    high: "Evidence is strongly suggestive"
    medium: "Evidence is moderately suggestive"
    low: "Evidence is weakly suggestive"
"""
    )
    return base, data, model


@pytest.fixture()
def write_configs():
    """Factory fixture returning the `_render_configs` helper."""
    return _render_configs
