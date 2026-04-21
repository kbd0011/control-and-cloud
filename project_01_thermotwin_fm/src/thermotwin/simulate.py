"""What-if simulator for cooling / workload counterfactuals.

The trained twin takes a context window + future controls and produces a
forecast. We use this to answer operator questions like:

- What if the coolant supply setpoint is lowered by 2°C?
- What if rack load increases by 20%?
- What if a cooling unit degrades by 30% (effective cooling supply rises)?

The simulator takes a real recent context window for a given rack, constructs
counterfactual control sequences, runs the model on each, and ranks scenarios
by a risk-adjusted objective (hotspot probability first, then energy proxy).

Usage:
    python -m thermotwin.simulate
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from thermotwin.data.ingest_m100 import load_and_prepare
from thermotwin.features.resample_and_window import (
    ScalerStats,
    build_windows,
    time_block_split,
)
from thermotwin.graphs.layout_graph import LayoutGraph
from thermotwin.evaluate import _load_model
from thermotwin.utils.config import load_configs, setup_logging
from thermotwin.utils.io import ensure_dir, load_json, save_json
from thermotwin.utils.seed import set_seed

logger = logging.getLogger("thermotwin.simulate")


@dataclass
class Scenario:
    name: str
    control_deltas: dict[str, float] = field(default_factory=dict)
    load_multiplier: float = 1.0
    description: str = ""


def default_scenarios() -> list[Scenario]:
    return [
        Scenario("baseline", description="Current controls, unchanged load"),
        Scenario(
            "lower_setpoint_2C",
            control_deltas={"coolant_supply_temp": -2.0},
            description="Lower coolant supply by 2°C",
        ),
        Scenario(
            "raise_setpoint_2C",
            control_deltas={"coolant_supply_temp": 2.0},
            description="Raise coolant supply by 2°C",
        ),
        Scenario(
            "fan_boost",
            control_deltas={"fan_speed_pct": 15.0},
            description="Increase fan speed by 15 percentage points",
        ),
        Scenario(
            "load_spike_20pct",
            control_deltas={},
            load_multiplier=1.2,
            description="20% load spike across the horizon",
        ),
        Scenario(
            "cooling_degrade",
            control_deltas={"coolant_supply_temp": 3.0, "fan_speed_pct": -10.0},
            description="Cooling degradation: warmer supply + weaker fans",
        ),
    ]


def apply_scenario_to_controls(
    controls_scaled: np.ndarray,
    control_order: list[str],
    scenario: Scenario,
    scaler: ScalerStats,
) -> np.ndarray:
    """Apply a scenario's control deltas in original units, then re-scale."""
    controls_inv = scaler.inverse_transform_array(controls_scaled, control_order).copy()
    for name, delta in scenario.control_deltas.items():
        if name in control_order:
            idx = control_order.index(name)
            controls_inv[..., idx] = controls_inv[..., idx] + float(delta)
    # Clamp to plausible ranges.
    for i, name in enumerate(control_order):
        if name == "fan_speed_pct":
            controls_inv[..., i] = np.clip(controls_inv[..., i], 20.0, 100.0)
        if name == "coolant_supply_temp":
            controls_inv[..., i] = np.clip(controls_inv[..., i], 12.0, 30.0)
    # Re-standardize.
    out = controls_inv.copy()
    for i, name in enumerate(control_order):
        if name in scaler.center:
            out[..., i] = (out[..., i] - scaler.center[name]) / (scaler.scale[name] + 1e-8)
    return out.astype(np.float32)


def apply_scenario_to_context(
    context_scaled: np.ndarray,
    feature_order: list[str],
    scenario: Scenario,
    scaler: ScalerStats,
) -> np.ndarray:
    """Apply a load multiplier to the context's load features in original units."""
    if scenario.load_multiplier == 1.0:
        return context_scaled.copy()
    ctx_inv = scaler.inverse_transform_array(context_scaled, feature_order).copy()
    for i, name in enumerate(feature_order):
        if name in ("cpu_load", "gpu_load"):
            ctx_inv[..., i] = np.clip(ctx_inv[..., i] * scenario.load_multiplier, 0.0, 1.0)
        if name == "power_draw_kw":
            ctx_inv[..., i] = ctx_inv[..., i] * scenario.load_multiplier
    out = ctx_inv.copy()
    for i, name in enumerate(feature_order):
        if name in scaler.center:
            out[..., i] = (out[..., i] - scaler.center[name]) / (scaler.scale[name] + 1e-8)
    return out.astype(np.float32)


def simulate_actions(
    model,
    context_scaled: np.ndarray,
    controls_scaled: np.ndarray,
    entity_id: int,
    scenarios: Iterable[Scenario],
    feature_order: list[str],
    control_order: list[str],
    target_order: list[str],
    scaler: ScalerStats,
    device: str,
) -> list[dict]:
    """Run the model on each scenario and return unscaled forecasts and metrics."""
    results = []
    for sc in scenarios:
        ctx_sc = apply_scenario_to_context(context_scaled, feature_order, sc, scaler)
        ctrl_sc = apply_scenario_to_controls(controls_scaled, control_order, sc, scaler)
        x = torch.from_numpy(ctx_sc[None, ...]).to(device)
        c = torch.from_numpy(ctrl_sc[None, ...]).to(device)
        e = torch.tensor([entity_id], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(x, c, e)
        q = out["quantiles"].cpu().numpy()[0]   # (H, T, Q)
        median = q[..., q.shape[-1] // 2]
        q_low = q[..., 0]
        q_high = q[..., -1]
        results.append(
            {
                "scenario": sc.name,
                "description": sc.description,
                "median": scaler.inverse_transform_array(median, target_order),
                "q_low": scaler.inverse_transform_array(q_low, target_order),
                "q_high": scaler.inverse_transform_array(q_high, target_order),
            }
        )
    return results


def rank_actions(
    sim_results: list[dict],
    target_order: list[str],
    hotspot_target: str,
    hotspot_threshold: float,
    risk_aversion: float = 1.0,
) -> list[dict]:
    """Rank scenarios by hotspot risk first, then by median inlet increase as energy proxy."""
    hs_idx = target_order.index(hotspot_target) if hotspot_target in target_order else 0
    scored = []
    for r in sim_results:
        med = r["median"][..., hs_idx]
        high = r["q_high"][..., hs_idx]
        max_med = float(med.max())
        max_high = float(high.max())
        excess = float(np.maximum(high - hotspot_threshold, 0.0).mean())
        risk = risk_aversion * excess + 0.1 * max(max_high - hotspot_threshold, 0.0)
        energy_proxy = float(med.mean())
        scored.append(
            {
                "scenario": r["scenario"],
                "max_predicted_inlet": max_med,
                "max_q_high_inlet": max_high,
                "risk_score": risk,
                "energy_proxy_mean_inlet": energy_proxy,
            }
        )
    scored.sort(key=lambda d: (d["risk_score"], d["energy_proxy_mean_inlet"]))
    for i, s in enumerate(scored):
        s["rank"] = i + 1
    return scored


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--data-config", default="configs/data_m100.yaml")
    parser.add_argument("--model-config", default="configs/model_graphfm.yaml")
    parser.add_argument("--artifacts", default=None)
    parser.add_argument("--rack-id", default=None, help="Rack id to simulate; defaults to the first")
    args = parser.parse_args(argv)

    cfg = load_configs(args.base_config, args.data_config, args.model_config)
    if args.artifacts:
        cfg["artifacts_dir"] = args.artifacts
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    set_seed(int(cfg["seed"]))
    artifacts_dir = ensure_dir(cfg["artifacts_dir"])

    scaler = ScalerStats.from_dict(load_json(artifacts_dir / "scaler.json"))
    manifest = load_json(artifacts_dir / "model_manifest.json")
    layout_graph = LayoutGraph.load(artifacts_dir / "layout_graph.json")

    telemetry = load_and_prepare(
        cfg["dataset"]["telemetry_path"],
        cadence_minutes=int(cfg["dataset"]["cadence_minutes"]),
    )
    splits = time_block_split(
        telemetry,
        val_fraction=float(cfg["train"]["val_fraction"]),
        test_fraction=float(cfg["train"]["test_fraction"]),
    )
    scaled_test = scaler.transform(splits["test"])
    windows = build_windows(
        scaled_test,
        features=manifest["feature_order"],
        targets=manifest["target_order"],
        controls=manifest["control_order"],
        context_steps=int(manifest["context_steps"]),
        horizon_steps=int(manifest["horizon_steps"]),
        stride=int(cfg["windowing"]["stride"]),
    )

    device = str(cfg["train"]["device"])
    model = _load_model(manifest, layout_graph, artifacts_dir / "model.pt", device)

    # Pick the first window for a chosen rack.
    rack_id = args.rack_id
    racks_sorted = sorted(set(telemetry["rack_id"]))
    if rack_id is None:
        rack_id = racks_sorted[0]
    if rack_id not in racks_sorted:
        raise SystemExit(f"rack_id {rack_id} not found. Available: {racks_sorted}")
    entity_id = racks_sorted.index(rack_id)

    # Find the first window belonging to this entity.
    matches = np.where(windows.entity == entity_id)[0]
    if len(matches) == 0:
        raise SystemExit(f"No test windows available for rack {rack_id}")
    pick = int(matches[0])
    ctx = windows.context[pick]
    ctrl = windows.controls[pick]

    scenarios = default_scenarios()
    sim = simulate_actions(
        model,
        ctx,
        ctrl,
        entity_id=entity_id,
        scenarios=scenarios,
        feature_order=manifest["feature_order"],
        control_order=manifest["control_order"],
        target_order=manifest["target_order"],
        scaler=scaler,
        device=device,
    )
    ranked = rank_actions(
        sim,
        target_order=manifest["target_order"],
        hotspot_target=cfg["features"]["hotspot_target"],
        hotspot_threshold=float(cfg["eval"]["hotspot_threshold_c"]),
    )

    out_path = artifacts_dir / "simulation_report.json"
    payload = {
        "rack_id": rack_id,
        "start_time": str(windows.start_times[pick]),
        "scenarios": [
            {
                "scenario": r["scenario"],
                "description": r["description"],
                "median": r["median"].tolist(),
                "q_low": r["q_low"].tolist(),
                "q_high": r["q_high"].tolist(),
            }
            for r in sim
        ],
        "ranking": ranked,
    }
    save_json(payload, out_path)
    # Tabular summary of the ranking for quick reading.
    pd.DataFrame(ranked).to_csv(artifacts_dir / "simulation_ranking.csv", index=False)
    logger.info("Wrote %s and simulation_ranking.csv", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
