"""Generate rare but plausible anomaly episodes for training and robustness testing.

These patterns mirror the taxonomy used by the synthetic data generator
(`cloudsentinel.data.synthetic`) but are applied on the fly to any long-form
telemetry table. The module is deliberately stateless: given a DataFrame it
returns a labeled copy with injected events and a metadata frame describing
each event.

Pattern contracts (used by tests):

- `spike`: short transient added to a selected entity/metric.
- `drift`: gradual ramp that persists after the incident window.
- `saturation`: metric forced to its upper ceiling.
- `correlated_failure`: oscillating sustained elevation, typically on error rate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

PATTERNS = ("spike", "drift", "saturation", "correlated_failure")


@dataclass
class InjectedEvent:
    incident_id: str
    pattern: str
    entity_id: str
    metric: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    magnitude: float


def _apply_spike(series: np.ndarray, start: int, width: int, magnitude: float) -> np.ndarray:
    end = min(start + width, len(series))
    out = series.copy()
    profile = np.hanning(2 * (end - start))[: (end - start)]
    out[start:end] = out[start:end] + magnitude * profile
    return out


def _apply_drift(series: np.ndarray, start: int, width: int, magnitude: float) -> np.ndarray:
    end = min(start + width, len(series))
    out = series.copy()
    out[start:end] = out[start:end] + magnitude * np.linspace(0, 1, end - start)
    out[end:] = out[end:] + magnitude
    return out


def _apply_saturation(series: np.ndarray, start: int, width: int, cap: float) -> np.ndarray:
    end = min(start + width, len(series))
    out = series.copy()
    out[start:end] = np.minimum(out[start:end] * 3.0 + cap, cap)
    return out


def _apply_correlated(series: np.ndarray, start: int, width: int, magnitude: float) -> np.ndarray:
    end = min(start + width, len(series))
    out = series.copy()
    t = np.arange(end - start)
    out[start:end] = out[start:end] + magnitude * (1.0 + 0.3 * np.sin(t * 0.5))
    return out


def inject_pattern(
    series: np.ndarray,
    pattern: str,
    *,
    start: int,
    width: int,
    strength: float = 1.0,
    cap: float = 1.0,
) -> np.ndarray:
    """Apply a labeled anomaly pattern to a single 1-D series.

    `strength` scales the default magnitude for each pattern.
    """
    if pattern not in PATTERNS:
        raise ValueError(f"Unknown pattern '{pattern}'; expected one of {PATTERNS}")
    if start < 0 or width <= 0 or start >= len(series):
        raise ValueError("start must be in bounds and width positive")
    if pattern == "spike":
        # Magnitude scaled to ~2× series std.
        default = 2.0 * (np.nanstd(series) + 1e-6)
        return _apply_spike(series, start, width, default * strength)
    if pattern == "drift":
        default = 1.0 * (np.nanstd(series) + 1e-6)
        return _apply_drift(series, start, width, default * strength)
    if pattern == "saturation":
        return _apply_saturation(series, start, width, cap)
    if pattern == "correlated_failure":
        default = 1.5 * (np.nanstd(series) + 1e-6)
        return _apply_correlated(series, start, width, default * strength)
    raise AssertionError("unreachable")


def generate_synthetic_batch(
    df: pd.DataFrame,
    *,
    n_events: int,
    patterns: Iterable[str] = PATTERNS,
    seed: int = 0,
    min_width: int = 6,
    max_width: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inject `n_events` anomalies into a long-form telemetry DataFrame.

    Returns (augmented_df, events_df). `events_df` has one row per injected
    event with entity/metric/window/pattern information suitable for labels.
    """
    required = {"timestamp", "entity_id", "metric", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Augmentation input missing columns: {sorted(missing)}")
    patterns = list(patterns)
    if not patterns:
        raise ValueError("At least one pattern must be provided")

    rng = np.random.default_rng(seed)
    out = df.copy().sort_values(["entity_id", "metric", "timestamp"]).reset_index(drop=True)

    groups = list(out.groupby(["entity_id", "metric"], sort=False).indices.items())
    events: list[InjectedEvent] = []

    for i in range(n_events):
        (entity_id, metric), idxs = groups[int(rng.integers(0, len(groups)))]
        idxs = np.asarray(idxs)
        values = out.loc[idxs, "value"].to_numpy(dtype=float)
        if len(values) < max_width + 2 or np.isnan(values).all():
            continue
        width = int(rng.integers(min_width, max_width + 1))
        start_rel = int(rng.integers(len(values) // 8, len(values) - width - 1))
        pattern = patterns[i % len(patterns)]
        strength = float(1.0 + 0.5 * rng.random())
        cap = float(np.nanmax(values)) if pattern == "saturation" else 1.0
        new_values = inject_pattern(
            values, pattern, start=start_rel, width=width, strength=strength, cap=cap
        )
        out.loc[idxs, "value"] = new_values
        start_ts = pd.Timestamp(out.loc[idxs[start_rel], "timestamp"])
        end_ts = pd.Timestamp(out.loc[idxs[min(start_rel + width - 1, len(idxs) - 1)], "timestamp"])
        events.append(
            InjectedEvent(
                incident_id=f"aug_{i:03d}",
                pattern=pattern,
                entity_id=str(entity_id),
                metric=str(metric),
                start_time=start_ts,
                end_time=end_ts,
                magnitude=strength,
            )
        )

    events_df = pd.DataFrame([e.__dict__ for e in events])
    return out, events_df


def make_incident_templates(events: pd.DataFrame) -> pd.DataFrame:
    """Convert a list of injected events into incident-label rows.

    The returned frame aligns with the incident label format used by the
    synthetic generator (`incidents.csv`): incident_id, pattern, root_entity,
    driving_metric, start_time, end_time, affected_entities.
    """
    if events.empty:
        return pd.DataFrame(
            columns=[
                "incident_id",
                "pattern",
                "root_entity",
                "driving_metric",
                "start_time",
                "end_time",
                "affected_entities",
            ]
        )
    return events.rename(
        columns={"entity_id": "root_entity", "metric": "driving_metric"}
    ).assign(affected_entities=events["entity_id"])[
        [
            "incident_id",
            "pattern",
            "root_entity",
            "driving_metric",
            "start_time",
            "end_time",
            "affected_entities",
        ]
    ].reset_index(drop=True)
