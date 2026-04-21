"""Synthetic thermal telemetry generator.

Produces an operationally plausible tabular time-series that downstream modules
can treat as a drop-in substitute for M100-like facility telemetry. We do NOT
claim this is physically accurate CFD; it is a lightweight generator based on:

- diurnal + weekly workload cycles,
- first-order thermal lag between load and inlet/outlet temperature,
- coolant supply temperature as a controllable cooling proxy,
- occasional workload spikes and cooling degradation events (sparse anomalies),
- small Gaussian sensor noise.

This gives the forecasting and simulation stack a credible dataset with
multi-rack heterogeneity, missingness, and rare events, without pretending to
be a real facility trace.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from thermotwin.utils.io import ensure_dir
from thermotwin.utils.seed import set_seed


@dataclass
class SyntheticConfig:
    n_racks: int = 8
    n_cooling_units: int = 2
    duration_days: int = 7
    cadence_minutes: int = 5
    missing_rate: float = 0.01
    anomaly_rate: float = 0.003
    seed: int = 42


def _diurnal_load(t_hours: np.ndarray, phase: float = 0.0) -> np.ndarray:
    """Diurnal + weekly workload intensity in [0, 1]."""
    day = 0.5 + 0.35 * np.sin(2 * np.pi * (t_hours / 24.0) + phase)
    week = 1.0 - 0.15 * (((t_hours // 24) % 7) >= 5).astype(float)  # weekend dip
    noise = 0.05 * np.random.randn(len(t_hours))
    return np.clip(day * week + noise, 0.0, 1.0)


def _thermal_response(
    load: np.ndarray,
    coolant_supply: np.ndarray,
    ambient: np.ndarray,
    *,
    inlet_base: float = 22.0,
    outlet_gain: float = 12.0,
    tau_steps: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """First-order lag model of inlet/outlet temperature.

    inlet_temp[t]  = filter(coolant_supply + alpha * ambient, tau)
    outlet_temp[t] = inlet_temp[t] + outlet_gain * load_filtered
    """
    n = len(load)
    inlet = np.zeros(n)
    load_filt = np.zeros(n)
    alpha = 1.0 / max(tau_steps, 1)
    # Effective inlet is dominated by coolant but with small ambient influence.
    effective = 0.85 * coolant_supply + 0.15 * ambient + 1.5  # +1.5C HX loss
    inlet[0] = effective[0] + inlet_base - coolant_supply[0]  # warm start
    for t in range(1, n):
        inlet[t] = inlet[t - 1] + alpha * (effective[t] - inlet[t - 1])
        load_filt[t] = load_filt[t - 1] + alpha * (load[t] - load_filt[t - 1])
    outlet = inlet + outlet_gain * load_filt
    return inlet, outlet


def generate(cfg: SyntheticConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate telemetry + layout dataframes."""
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    steps_per_day = int(24 * 60 / cfg.cadence_minutes)
    n_steps = cfg.duration_days * steps_per_day
    start = pd.Timestamp("2024-01-01 00:00:00")
    timestamps = pd.date_range(start=start, periods=n_steps, freq=f"{cfg.cadence_minutes}min")
    t_hours = np.arange(n_steps) * cfg.cadence_minutes / 60.0

    # Ambient temperature: mild diurnal sinusoid (outdoor air).
    ambient = 18.0 + 6.0 * np.sin(2 * np.pi * (t_hours - 6) / 24.0) + 0.5 * rng.standard_normal(n_steps)

    rows = []
    for rack_id in range(cfg.n_racks):
        phase = rng.uniform(-0.6, 0.6)
        load = _diurnal_load(t_hours, phase=phase)

        # Occasional workload spikes (rare event augmentation).
        spike_mask = rng.random(n_steps) < cfg.anomaly_rate
        load = np.clip(load + spike_mask * rng.uniform(0.3, 0.6, n_steps), 0.0, 1.0)

        cpu_load = np.clip(load + 0.05 * rng.standard_normal(n_steps), 0.0, 1.0)
        gpu_load = np.clip(load * rng.uniform(0.7, 1.1) + 0.05 * rng.standard_normal(n_steps), 0.0, 1.0)
        power_draw_kw = 3.0 + 12.0 * (0.4 * cpu_load + 0.6 * gpu_load)  # 3–15 kW per rack
        fan_speed_pct = np.clip(30 + 70 * load + 3 * rng.standard_normal(n_steps), 20, 100)

        # Coolant supply setpoint varies slowly per cooling unit with drift.
        cu_id = rack_id % cfg.n_cooling_units
        base_setpoint = 18.0 + 1.5 * cu_id
        slow = 0.5 * np.sin(2 * np.pi * t_hours / (24 * 3.0))
        degrade = np.zeros(n_steps)
        # Rare cooling degradation events: setpoint drifts up for ~1 hour.
        deg_starts = np.where(rng.random(n_steps) < cfg.anomaly_rate / 2)[0]
        for s in deg_starts:
            width = min(12, n_steps - s)
            degrade[s : s + width] += np.linspace(0, 2.5, width)
        coolant_supply = base_setpoint + slow + degrade + 0.1 * rng.standard_normal(n_steps)

        inlet, outlet = _thermal_response(load, coolant_supply, ambient)
        # Sensor noise.
        inlet += 0.15 * rng.standard_normal(n_steps)
        outlet += 0.20 * rng.standard_normal(n_steps)

        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "rack_id": f"rack_{rack_id:02d}",
                "cooling_unit_id": f"cu_{cu_id:02d}",
                "cpu_load": cpu_load,
                "gpu_load": gpu_load,
                "power_draw_kw": power_draw_kw,
                "fan_speed_pct": fan_speed_pct,
                "coolant_supply_temp": coolant_supply,
                "ambient_temp": ambient,
                "inlet_temp": inlet,
                "outlet_temp": outlet,
            }
        )
        rows.append(df)

    telemetry = pd.concat(rows, ignore_index=True)

    # Inject missingness (simulating sensor dropouts) on a subset of rows.
    if cfg.missing_rate > 0:
        miss_cols = ["inlet_temp", "outlet_temp", "fan_speed_pct"]
        for col in miss_cols:
            mask = rng.random(len(telemetry)) < cfg.missing_rate
            telemetry.loc[mask, col] = np.nan

    # Layout: rack -> aisle -> cooling unit, with a simple floorplan coordinate.
    layout_rows = []
    for rack_id in range(cfg.n_racks):
        cu_id = rack_id % cfg.n_cooling_units
        aisle = "A" if rack_id < cfg.n_racks / 2 else "B"
        x = rack_id % (cfg.n_racks // 2)
        y = 0 if aisle == "A" else 1
        layout_rows.append(
            {
                "rack_id": f"rack_{rack_id:02d}",
                "aisle": aisle,
                "cooling_unit_id": f"cu_{cu_id:02d}",
                "x": x,
                "y": y,
            }
        )
    layout = pd.DataFrame(layout_rows)

    return telemetry, layout


def write_sample_dataset(out_dir: str | Path, cfg: SyntheticConfig | None = None) -> dict[str, Path]:
    """Generate a synthetic dataset and write to CSV files under `out_dir`."""
    cfg = cfg or SyntheticConfig()
    out_dir = ensure_dir(out_dir)
    telemetry, layout = generate(cfg)
    tpath = Path(out_dir) / "telemetry.csv"
    lpath = Path(out_dir) / "layout.csv"
    telemetry.to_csv(tpath, index=False)
    layout.to_csv(lpath, index=False)
    return {"telemetry": tpath, "layout": lpath}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic M100-like telemetry.")
    parser.add_argument("--out-dir", default="data/sample")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--racks", type=int, default=8)
    parser.add_argument("--cooling-units", type=int, default=2)
    parser.add_argument("--cadence-min", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    cfg = SyntheticConfig(
        n_racks=args.racks,
        n_cooling_units=args.cooling_units,
        duration_days=args.days,
        cadence_minutes=args.cadence_min,
        seed=args.seed,
    )
    paths = write_sample_dataset(args.out_dir, cfg)
    print(f"Wrote {paths['telemetry']} and {paths['layout']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
