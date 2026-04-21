"""Shared pytest fixtures."""
from __future__ import annotations

from pathlib import Path

import pytest

from thermotwin.data.synthetic import SyntheticConfig, write_sample_dataset


@pytest.fixture(scope="session")
def sample_data_dir(tmp_path_factory) -> Path:
    """Generate a tiny synthetic dataset used by multiple tests."""
    d = tmp_path_factory.mktemp("thermotwin_data")
    cfg = SyntheticConfig(n_racks=4, n_cooling_units=2, duration_days=2, cadence_minutes=5, seed=7)
    write_sample_dataset(d, cfg)
    return d


@pytest.fixture(scope="session")
def sample_telemetry_path(sample_data_dir) -> Path:
    return sample_data_dir / "telemetry.csv"


@pytest.fixture(scope="session")
def sample_layout_path(sample_data_dir) -> Path:
    return sample_data_dir / "layout.csv"
