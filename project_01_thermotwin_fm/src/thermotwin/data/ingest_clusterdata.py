"""Convert ClusterData2019-style task traces into workload regime features.

In the absence of the full Google Borg trace we support a CSV with minimal
columns: `timestamp`, `cell`, `cpu_request`, `memory_request`, `task_events`.
For the ThermoTwin-FM demo we expose a small synthetic fallback so the
pipeline is runnable end-to-end without external downloads.

This module is deliberately lightweight and produces features that are also
useful for Project 2 (CloudSentinel) and Project 3 (SafeCool) workload
injectors, so the ingest contract is stable across the portfolio.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"timestamp", "cpu_request", "memory_request", "task_events"}


def aggregate_cluster_windows(path: str | Path, freq: str = "5min") -> pd.DataFrame:
    """Aggregate cluster-trace events into uniform time bins.

    Returns a DataFrame with columns:
        timestamp, cpu_request_mean, memory_request_mean, task_churn, queue_depth
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Cluster trace not found: {p}")
    df = pd.read_csv(p, parse_dates=["timestamp"])
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Cluster trace missing required columns: {sorted(missing)}")
    df = df.sort_values("timestamp").set_index("timestamp")

    agg = pd.DataFrame(
        {
            "cpu_request_mean": df["cpu_request"].resample(freq).mean(),
            "memory_request_mean": df["memory_request"].resample(freq).mean(),
            "task_churn": df["task_events"].resample(freq).sum(),
            "queue_depth": df["task_events"].resample(freq).count(),
        }
    )
    agg = agg.fillna(0.0).reset_index()
    return agg


def derive_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add burstiness and heterogeneity features to aggregated workload data."""
    required = {"cpu_request_mean", "memory_request_mean", "task_churn", "queue_depth"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Aggregated trace missing columns: {sorted(missing)}")
    out = df.copy()
    # Burstiness: rolling stdev / mean of task churn, clipped to a finite range.
    mean = out["task_churn"].rolling(12, min_periods=1).mean().clip(lower=1e-6)
    std = out["task_churn"].rolling(12, min_periods=1).std().fillna(0.0)
    out["burstiness"] = (std / mean).clip(0.0, 10.0)
    # Heterogeneity proxy: |z-score difference| between cpu and memory demand.
    cpu_z = (out["cpu_request_mean"] - out["cpu_request_mean"].mean()) / (
        out["cpu_request_mean"].std() + 1e-6
    )
    mem_z = (out["memory_request_mean"] - out["memory_request_mean"].mean()) / (
        out["memory_request_mean"].std() + 1e-6
    )
    out["heterogeneity"] = (cpu_z - mem_z).abs()
    return out


def make_rack_level_workloads(df: pd.DataFrame, rack_count: int, *, seed: int = 0) -> pd.DataFrame:
    """Split cluster-level demand across synthetic racks while preserving mass.

    The returned frame has one row per (timestamp, rack_id) with rack-scaled
    `cpu_load` and `gpu_load` in [0, 1]. The sum of rack-level loads at each
    timestamp is approximately equal to the scaled cluster-level load.
    """
    if rack_count <= 0:
        raise ValueError("rack_count must be positive")
    rng = np.random.default_rng(seed)
    # Dirichlet weights per rack → total mass preserved up to sampling noise.
    weights = rng.dirichlet(alpha=np.ones(rack_count) * 4.0)
    max_cpu = max(df["cpu_request_mean"].max(), 1e-6)
    rows = []
    for _, row in df.iterrows():
        norm_load = float(row["cpu_request_mean"] / max_cpu)
        for rack_id in range(rack_count):
            w = float(weights[rack_id])
            cpu_l = np.clip(norm_load * w * rack_count, 0.0, 1.0)
            gpu_l = np.clip(norm_load * w * rack_count * 0.9, 0.0, 1.0)
            rows.append(
                {
                    "timestamp": row["timestamp"],
                    "rack_id": f"rack_{rack_id:02d}",
                    "cpu_load": cpu_l,
                    "gpu_load": gpu_l,
                }
            )
    return pd.DataFrame(rows)


def synthetic_cluster_trace(n_steps: int = 500, seed: int = 0) -> pd.DataFrame:
    """Produce a tiny ClusterData-shaped fixture for tests and demos."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n_steps, freq="30s")
    base = 0.5 + 0.3 * np.sin(np.linspace(0, 6 * np.pi, n_steps))
    return pd.DataFrame(
        {
            "timestamp": ts,
            "cell": "demo",
            "cpu_request": np.clip(base + 0.05 * rng.standard_normal(n_steps), 0, None),
            "memory_request": np.clip(0.8 * base + 0.05 * rng.standard_normal(n_steps), 0, None),
            "task_events": rng.poisson(lam=5.0 + 10 * base),
        }
    )
