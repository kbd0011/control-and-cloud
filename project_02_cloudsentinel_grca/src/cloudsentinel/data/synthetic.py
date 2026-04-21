"""Synthetic multi-entity telemetry generator with dependency structure.

Produces:
- `telemetry.csv`: long-form telemetry with (timestamp, entity_id, metric, value).
- `entities.csv`: per-entity metadata (entity_id, entity_type, service, region, host).
- `dependencies.csv`: directed edges describing upstream → downstream dependencies.
- `incidents.csv`: ground-truth incident episodes with affected entities, root cause, and time window.

The generator models:
- Diurnal + weekly baseline seasonality per metric.
- Realistic metric families (latency, error_rate, cpu_util, memory_util) with different units
  and different noise regimes.
- A *dependency graph* over services so an incident at one service propagates to its downstream
  consumers with realistic lag and attenuation.
- Four labeled incident patterns: spike, drift, saturation, correlated_failure.

The ground truth (root cause entity + time window) is written out so tests and
evaluation can assess detection and RCA without ambiguity.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from cloudsentinel.utils.io import ensure_dir
from cloudsentinel.utils.seed import set_seed

# Metric families with distinct behavior. Keep the set small and interpretable.
METRICS = ("latency_ms", "error_rate", "cpu_util", "memory_util")


@dataclass
class SyntheticConfig:
    n_entities: int = 24
    duration_days: int = 7
    cadence_minutes: int = 5
    n_regions: int = 2
    n_services: int = 6                 # services grouped into a dependency chain
    incident_count: int = 8
    missing_rate: float = 0.005
    seed: int = 42
    metrics: tuple[str, ...] = METRICS


def _baseline_for_metric(metric: str, t_hours: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return a baseline time series for a given metric, without incidents."""
    n = len(t_hours)
    # Diurnal load proxy used by all metrics.
    diurnal = 0.5 + 0.3 * np.sin(2 * np.pi * (t_hours / 24.0) - np.pi / 2)
    weekend = 1.0 - 0.2 * (((t_hours // 24) % 7) >= 5).astype(float)
    load = np.clip(diurnal * weekend, 0.0, 1.0)

    if metric == "latency_ms":
        # Latency depends on load with heavy-tailed noise.
        return 25.0 + 35.0 * load + 6.0 * np.abs(rng.standard_normal(n))
    if metric == "error_rate":
        # Small baseline error rate, mostly near zero.
        return np.clip(0.01 + 0.02 * load + 0.01 * rng.standard_normal(n), 0.0, 1.0)
    if metric == "cpu_util":
        return np.clip(0.25 + 0.4 * load + 0.05 * rng.standard_normal(n), 0.0, 1.0)
    if metric == "memory_util":
        # Memory is slower-moving: cumulative drift within bounds.
        walk = np.cumsum(0.002 * rng.standard_normal(n))
        base = 0.5 + 0.25 * load + 0.05 * rng.standard_normal(n)
        return np.clip(base + 0.15 * np.tanh(walk), 0.0, 1.0)
    raise ValueError(f"Unknown metric: {metric}")


def _build_dependency_graph(
    cfg: SyntheticConfig, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign entities to services and build a realistic dependency DAG.

    Services form a rough chain (gateway → api → db / cache / queue) with some randomness.
    Each service hosts multiple entities (e.g., replicas).
    """
    services = [f"service_{i:02d}" for i in range(cfg.n_services)]
    # Create a DAG: service_i may depend on service_j for j < i, with probability.
    edges = []
    for i in range(1, cfg.n_services):
        # Each service depends on at least one upstream.
        upstream = rng.choice(i)
        edges.append((services[int(upstream)], services[i]))
        # And optionally a second dependency.
        if i >= 2 and rng.random() < 0.4:
            second = rng.choice(i)
            if services[int(second)] != services[int(upstream)]:
                edges.append((services[int(second)], services[i]))
    deps = pd.DataFrame(edges, columns=["upstream_service", "downstream_service"]).drop_duplicates()

    # Assign entities to services, roughly evenly, and to regions.
    entity_rows = []
    for i in range(cfg.n_entities):
        service = services[i % cfg.n_services]
        region = f"region_{i % cfg.n_regions}"
        host = f"host_{i:02d}"
        entity_rows.append(
            {
                "entity_id": f"ent_{i:02d}",
                "entity_type": "host",
                "service": service,
                "region": region,
                "host": host,
            }
        )
    entities = pd.DataFrame(entity_rows)
    return entities, deps


def _downstream_map(deps: pd.DataFrame) -> dict[str, list[str]]:
    d: dict[str, list[str]] = {}
    for _, r in deps.iterrows():
        d.setdefault(r["upstream_service"], []).append(r["downstream_service"])
    return d


def _inject_spike(series: np.ndarray, start: int, width: int, magnitude: float) -> np.ndarray:
    end = min(start + width, len(series))
    out = series.copy()
    profile = np.hanning(2 * (end - start))[: (end - start)]
    out[start:end] = out[start:end] + magnitude * profile
    return out


def _inject_drift(series: np.ndarray, start: int, width: int, magnitude: float) -> np.ndarray:
    end = min(start + width, len(series))
    out = series.copy()
    out[start:end] = out[start:end] + magnitude * np.linspace(0, 1, end - start)
    # Persist the drift level after the incident window.
    out[end:] = out[end:] + magnitude
    return out


def _inject_saturation(series: np.ndarray, start: int, width: int, cap: float) -> np.ndarray:
    end = min(start + width, len(series))
    out = series.copy()
    out[start:end] = np.minimum(out[start:end] * 3.0 + cap, cap)
    return out


def _inject_correlated(
    series: np.ndarray, start: int, width: int, magnitude: float
) -> np.ndarray:
    """Correlated failure: sustained elevated level with oscillation."""
    end = min(start + width, len(series))
    out = series.copy()
    t = np.arange(end - start)
    out[start:end] = out[start:end] + magnitude * (1.0 + 0.3 * np.sin(t * 0.5))
    return out


def _propagate_to_downstream(
    values_by_entity: dict[str, dict[str, np.ndarray]],
    root_entity: str,
    entities: pd.DataFrame,
    downstream: dict[str, list[str]],
    start: int,
    width: int,
    rng: np.random.Generator,
) -> list[str]:
    """Propagate an incident from the root entity to downstream services.

    Returns the list of entities affected (includes the root).
    """
    root_svc = entities.loc[entities["entity_id"] == root_entity, "service"].iloc[0]
    affected_services = {root_svc}
    frontier = [root_svc]
    while frontier:
        svc = frontier.pop()
        for ds in downstream.get(svc, []):
            if ds not in affected_services:
                affected_services.add(ds)
                frontier.append(ds)

    affected_entities: list[str] = []
    for svc in affected_services:
        ents = entities.loc[entities["service"] == svc, "entity_id"].tolist()
        for ent in ents:
            affected_entities.append(ent)
            # Propagated entities see a smaller, delayed bump in latency + error_rate.
            if ent == root_entity:
                continue
            lag = int(rng.integers(1, 4))  # 1–3 step lag
            propagate_start = start + lag
            for metric in ("latency_ms", "error_rate"):
                series = values_by_entity[ent][metric]
                attenuation = 0.35 + 0.2 * rng.random()
                width_ds = max(1, width - lag)
                if metric == "latency_ms":
                    values_by_entity[ent][metric] = _inject_spike(
                        series, propagate_start, width_ds, 25.0 * attenuation
                    )
                else:
                    values_by_entity[ent][metric] = _inject_spike(
                        series, propagate_start, width_ds, 0.15 * attenuation
                    )
    return affected_entities


def generate(cfg: SyntheticConfig) -> dict[str, pd.DataFrame]:
    """Generate telemetry + metadata + labeled incidents."""
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    steps_per_day = int(24 * 60 / cfg.cadence_minutes)
    n_steps = cfg.duration_days * steps_per_day
    timestamps = pd.date_range("2024-01-01", periods=n_steps, freq=f"{cfg.cadence_minutes}min")
    t_hours = np.arange(n_steps) * cfg.cadence_minutes / 60.0

    entities, deps = _build_dependency_graph(cfg, rng)
    downstream = _downstream_map(deps)

    values_by_entity: dict[str, dict[str, np.ndarray]] = {}
    for ent in entities["entity_id"]:
        values_by_entity[ent] = {m: _baseline_for_metric(m, t_hours, rng) for m in cfg.metrics}

    # Inject incidents on a subset of entities.
    incidents_rows = []
    candidates = list(entities["entity_id"])
    patterns = ["spike", "drift", "saturation", "correlated_failure"]
    for inc_id in range(cfg.incident_count):
        root_ent = str(rng.choice(candidates))
        pattern = patterns[inc_id % len(patterns)]
        # Start at least 1 day in and leave enough room before the end.
        start = int(rng.integers(steps_per_day, n_steps - 2 * steps_per_day // 3))
        width = int(rng.integers(6, 24))  # 30 min – 2 h

        # Inject on the driving metric family for the chosen pattern.
        if pattern == "spike":
            metric = "latency_ms"
            magnitude = 80.0 + 40.0 * rng.random()
            values_by_entity[root_ent][metric] = _inject_spike(
                values_by_entity[root_ent][metric], start, width, magnitude
            )
        elif pattern == "drift":
            metric = "memory_util"
            magnitude = 0.2 + 0.15 * rng.random()
            values_by_entity[root_ent][metric] = _inject_drift(
                values_by_entity[root_ent][metric], start, width, magnitude
            )
        elif pattern == "saturation":
            metric = "cpu_util"
            values_by_entity[root_ent][metric] = _inject_saturation(
                values_by_entity[root_ent][metric], start, width, cap=0.99
            )
        elif pattern == "correlated_failure":
            metric = "error_rate"
            magnitude = 0.15 + 0.1 * rng.random()
            values_by_entity[root_ent][metric] = _inject_correlated(
                values_by_entity[root_ent][metric], start, width, magnitude
            )

        affected = _propagate_to_downstream(
            values_by_entity, root_ent, entities, downstream, start, width, rng
        )

        incidents_rows.append(
            {
                "incident_id": f"inc_{inc_id:03d}",
                "pattern": pattern,
                "root_entity": root_ent,
                "driving_metric": metric,
                "start_time": str(timestamps[start]),
                "end_time": str(timestamps[min(start + width - 1, n_steps - 1)]),
                "affected_entities": ",".join(sorted(set(affected))),
            }
        )

    # Assemble long-form telemetry.
    rows = []
    for ent in entities["entity_id"]:
        for metric, series in values_by_entity[ent].items():
            # Sensor dropouts.
            mask = rng.random(len(series)) < cfg.missing_rate
            series = series.copy()
            series[mask] = np.nan
            rows.append(
                pd.DataFrame(
                    {
                        "timestamp": timestamps,
                        "entity_id": ent,
                        "metric": metric,
                        "value": series,
                    }
                )
            )
    telemetry = pd.concat(rows, ignore_index=True).sort_values(
        ["entity_id", "metric", "timestamp"]
    ).reset_index(drop=True)

    incidents = pd.DataFrame(incidents_rows)

    # Small external (out-of-domain) dataset for CESNET-shaped tests: simpler traffic-like series.
    ext_rows = []
    ext_n_entities = 6
    for i in range(ext_n_entities):
        entity = f"cesnet_ent_{i:02d}"
        # Bytes-per-second style series with a diurnal pattern.
        series = 1e6 * (0.4 + 0.5 * np.sin(2 * np.pi * t_hours / 24.0) + 0.05 * rng.standard_normal(n_steps))
        series = np.clip(series, 0, None)
        ext_rows.append(
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "entity_id": entity,
                    "metric": "bytes_per_s",
                    "value": series,
                }
            )
        )
    cesnet_external = pd.concat(ext_rows, ignore_index=True)

    return {
        "telemetry": telemetry,
        "entities": entities,
        "dependencies": deps,
        "incidents": incidents,
        "cesnet_external": cesnet_external,
    }


def write_sample_dataset(out_dir: str | Path, cfg: SyntheticConfig | None = None) -> dict[str, Path]:
    cfg = cfg or SyntheticConfig()
    out_dir = ensure_dir(out_dir)
    tables = generate(cfg)
    paths: dict[str, Path] = {}
    for name, df in tables.items():
        p = Path(out_dir) / f"{name if name != 'telemetry' else 'telemetry'}.csv"
        df.to_csv(p, index=False)
        paths[name] = p
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic multi-entity telemetry.")
    parser.add_argument("--out-dir", default="data/sample")
    parser.add_argument("--entities", type=int, default=24)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--services", type=int, default=6)
    parser.add_argument("--cadence-min", type=int, default=5)
    parser.add_argument("--incidents", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    cfg = SyntheticConfig(
        n_entities=args.entities,
        duration_days=args.days,
        n_services=args.services,
        cadence_minutes=args.cadence_min,
        incident_count=args.incidents,
        seed=args.seed,
    )
    paths = write_sample_dataset(args.out_dir, cfg)
    for name, p in paths.items():
        print(f"Wrote {name}: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
