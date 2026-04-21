"""Thermal safety constraints for the SafeCool environment.

Exposes:

- `SafetyConfig`: dataclass of thresholds (max inlet temp, rate change, etc.).
- `compute_constraint_cost(state, action, next_state)`: per-step cost dict with
  named contributors (inlet_hi, inlet_lo, outlet_hi, rate_coolant, rate_fan).
- `project_to_safe_action(action, bounds, state)`: box-clip + rate-limit with
  respect to the current state's controls.
- `compute_rci_proxy(state)`: a simple Rack Cooling Index proxy in [0, 1].
- `count_violations(trajectory)`: post-hoc audit helper.

The cost returned is intentionally **hinge**-shaped (zero in safe regions,
positive when thresholds are exceeded) so it plugs directly into a
Lagrangian constrained-MDP formulation downstream.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from safecool.envs.lcopt_env import ACTION_FIELDS, STATE_FIELDS


@dataclass
class SafetyConfig:
    max_inlet_temp_c: float = 27.0
    min_inlet_temp_c: float = 15.0
    max_outlet_temp_c: float = 40.0
    max_rate_change_coolant_c: float = 1.0
    max_rate_change_fan_pct: float = 10.0
    rci_target: float = 0.98
    inlet_warn_c: float = 26.0
    coolant_supply_bounds: tuple[float, float] = (15.0, 24.0)
    fan_speed_bounds: tuple[float, float] = (25.0, 100.0)

    @classmethod
    def from_dict(cls, d: dict) -> "SafetyConfig":
        s = d.get("safety", d)
        a = d.get("action_space", {})
        return cls(
            max_inlet_temp_c=float(s.get("max_inlet_temp_c", 27.0)),
            min_inlet_temp_c=float(s.get("min_inlet_temp_c", 15.0)),
            max_outlet_temp_c=float(s.get("max_outlet_temp_c", 40.0)),
            max_rate_change_coolant_c=float(s.get("max_rate_change_coolant_c", 1.0)),
            max_rate_change_fan_pct=float(s.get("max_rate_change_fan_pct", 10.0)),
            rci_target=float(s.get("rci_target", 0.98)),
            inlet_warn_c=float(s.get("inlet_warn_c", 26.0)),
            coolant_supply_bounds=tuple(a.get("coolant_supply_bounds_c", [15.0, 24.0])),
            fan_speed_bounds=tuple(a.get("fan_speed_bounds_pct", [25.0, 100.0])),
        )


def _unpack_state(state: np.ndarray | dict) -> dict:
    if isinstance(state, dict):
        return state
    arr = np.asarray(state).flatten()
    return {name: float(v) for name, v in zip(STATE_FIELDS, arr)}


def compute_constraint_cost(
    state: np.ndarray | dict,
    action: np.ndarray | dict,
    next_state: np.ndarray | dict,
    cfg: SafetyConfig | None = None,
) -> dict[str, float]:
    """Compute a hinge-shaped per-step safety cost.

    Returns a dict of named cost components. `total` is their sum (the scalar
    CMDP cost). Each component is non-negative, zero in the safe region.
    """
    cfg = cfg or SafetyConfig()
    s = _unpack_state(state)
    ns = _unpack_state(next_state)
    a = np.asarray(action).flatten() if not isinstance(action, dict) else np.array(
        [action.get("coolant_supply_delta_c", 0.0), action.get("fan_speed_delta_pct", 0.0)]
    )

    inlet_hi = max(0.0, ns["inlet_temp_c"] - cfg.max_inlet_temp_c)
    inlet_lo = max(0.0, cfg.min_inlet_temp_c - ns["inlet_temp_c"])
    outlet_hi = max(0.0, ns["outlet_temp_c"] - cfg.max_outlet_temp_c)
    rate_coolant = max(0.0, abs(float(a[0])) - cfg.max_rate_change_coolant_c)
    rate_fan = max(0.0, abs(float(a[1])) - cfg.max_rate_change_fan_pct)

    components = {
        "inlet_hi": float(inlet_hi),
        "inlet_lo": float(inlet_lo),
        "outlet_hi": float(outlet_hi),
        "rate_coolant": float(rate_coolant),
        "rate_fan": float(rate_fan),
    }
    components["total"] = float(sum(components.values()))
    return components


def project_to_safe_action(
    action: np.ndarray,
    state: np.ndarray | dict,
    cfg: SafetyConfig | None = None,
    *,
    coolant_delta_range: tuple[float, float] = (-1.0, 1.0),
    fan_delta_range: tuple[float, float] = (-10.0, 10.0),
) -> np.ndarray:
    """Project an unsafe candidate action into the feasible set.

    The projection:
    1. Clips each action component to its configured delta range.
    2. Clips by rate limits (which may be tighter than the range).
    3. Clips by resulting control bounds (i.e., the applied delta may not push
       the control outside its operating window).
    """
    cfg = cfg or SafetyConfig()
    s = _unpack_state(state)
    a = np.asarray(action, dtype=np.float32).flatten()
    if a.shape != (2,):
        raise ValueError(f"Expected action shape (2,), got {a.shape}")

    # 1. Delta ranges.
    coolant_delta = float(np.clip(a[0], *coolant_delta_range))
    fan_delta = float(np.clip(a[1], *fan_delta_range))
    # 2. Rate limits (may be tighter than delta ranges).
    coolant_delta = float(np.clip(coolant_delta, -cfg.max_rate_change_coolant_c, cfg.max_rate_change_coolant_c))
    fan_delta = float(np.clip(fan_delta, -cfg.max_rate_change_fan_pct, cfg.max_rate_change_fan_pct))
    # 3. Operating-window clips: the applied delta cannot push controls out of bounds.
    cs_low, cs_high = cfg.coolant_supply_bounds
    f_low, f_high = cfg.fan_speed_bounds
    coolant_now = s["coolant_supply_c"]
    fan_now = s["fan_speed_pct"]
    coolant_delta = float(np.clip(coolant_delta, cs_low - coolant_now, cs_high - coolant_now))
    fan_delta = float(np.clip(fan_delta, f_low - fan_now, f_high - fan_now))
    return np.array([coolant_delta, fan_delta], dtype=np.float32)


def compute_rci_proxy(state: np.ndarray | dict, cfg: SafetyConfig | None = None) -> float:
    """Rack Cooling Index proxy in [0, 1].

    The real ASHRAE RCI integrates across a rack population; here we use a
    single-value proxy: 1.0 when inlet is comfortably below `inlet_warn_c`,
    decaying linearly to 0 as inlet approaches or exceeds `max_inlet_temp_c`.
    """
    cfg = cfg or SafetyConfig()
    s = _unpack_state(state)
    span = max(cfg.max_inlet_temp_c - cfg.inlet_warn_c, 1e-3)
    excess = max(0.0, s["inlet_temp_c"] - cfg.inlet_warn_c)
    return float(np.clip(1.0 - excess / span, 0.0, 1.0))


def count_violations(
    states: np.ndarray,
    actions: np.ndarray,
    next_states: np.ndarray,
    cfg: SafetyConfig | None = None,
) -> dict[str, int]:
    """Count per-field violation events across an episode trajectory."""
    cfg = cfg or SafetyConfig()
    counts = {k: 0 for k in ("inlet_hi", "inlet_lo", "outlet_hi", "rate_coolant", "rate_fan")}
    for s, a, ns in zip(states, actions, next_states):
        c = compute_constraint_cost(s, a, ns, cfg)
        for k in counts:
            if c[k] > 0:
                counts[k] += 1
    counts["any"] = sum(1 for c in counts.values() if c > 0)
    return counts
