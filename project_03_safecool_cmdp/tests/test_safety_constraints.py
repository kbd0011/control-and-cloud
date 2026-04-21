"""Safety-constraint tests: cost zero on safe, positive on unsafe, projection respects bounds."""
from __future__ import annotations

import numpy as np

from safecool.envs.lcopt_env import STATE_FIELDS
from safecool.envs.safety_constraints import (
    SafetyConfig,
    compute_constraint_cost,
    compute_rci_proxy,
    count_violations,
    project_to_safe_action,
)


def _state(inlet=22.0, outlet=28.0, fan=60.0, coolant=18.0, load=0.5, ambient=20.0, power=3.0):
    return np.array([fan, coolant, inlet, outlet, load, ambient, power], dtype=np.float32)


def test_safe_state_has_zero_cost():
    cfg = SafetyConfig()
    s = _state(inlet=22.0, outlet=28.0)
    ns = _state(inlet=22.5, outlet=28.5)
    c = compute_constraint_cost(s, np.array([0.0, 0.0]), ns, cfg)
    assert c["total"] == 0.0


def test_unsafe_inlet_state_has_positive_cost():
    cfg = SafetyConfig(max_inlet_temp_c=27.0)
    s = _state(inlet=26.0)
    ns = _state(inlet=29.0)
    c = compute_constraint_cost(s, np.array([0.0, 0.0]), ns, cfg)
    assert c["inlet_hi"] > 0
    assert c["total"] > 0


def test_rate_violation_triggered_by_large_action():
    cfg = SafetyConfig(max_rate_change_coolant_c=1.0, max_rate_change_fan_pct=10.0)
    s = _state()
    ns = _state()
    # action exceeds rate limit on coolant dimension.
    c = compute_constraint_cost(s, np.array([3.0, 0.0]), ns, cfg)
    assert c["rate_coolant"] > 0


def test_projection_clips_to_rate_limits():
    cfg = SafetyConfig(max_rate_change_coolant_c=1.0, max_rate_change_fan_pct=10.0)
    s = _state(fan=60.0, coolant=18.0)
    proj = project_to_safe_action(np.array([5.0, 50.0]), s, cfg)
    assert abs(proj[0]) <= cfg.max_rate_change_coolant_c + 1e-6
    assert abs(proj[1]) <= cfg.max_rate_change_fan_pct + 1e-6


def test_projection_respects_operating_window():
    cfg = SafetyConfig(coolant_supply_bounds=(15.0, 24.0), fan_speed_bounds=(25.0, 100.0))
    # Coolant already at upper bound, try to push even higher.
    s = _state(coolant=24.0, fan=100.0)
    proj = project_to_safe_action(np.array([1.0, 5.0]), s, cfg)
    assert proj[0] <= 0.0 + 1e-6   # no further upward movement allowed
    assert proj[1] <= 0.0 + 1e-6


def test_rci_proxy_high_when_cool():
    cfg = SafetyConfig(inlet_warn_c=26.0, max_inlet_temp_c=27.0)
    assert compute_rci_proxy(_state(inlet=22.0), cfg) == 1.0


def test_rci_proxy_low_when_hot():
    cfg = SafetyConfig(inlet_warn_c=26.0, max_inlet_temp_c=27.0)
    assert compute_rci_proxy(_state(inlet=27.5), cfg) == 0.0


def test_count_violations_aggregates():
    cfg = SafetyConfig()
    states = np.stack([_state(inlet=22), _state(inlet=22)])
    actions = np.array([[0.0, 0.0], [0.0, 0.0]])
    # Next states: one triggers inlet_hi.
    next_states = np.stack([_state(inlet=22), _state(inlet=29)])
    counts = count_violations(states, actions, next_states, cfg)
    assert counts["inlet_hi"] == 1
    assert counts["any"] >= 1
