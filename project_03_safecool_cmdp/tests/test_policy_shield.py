"""Policy-shield tests: safe passes, unsafe projected or fallback, stats increment."""
from __future__ import annotations

import numpy as np

from safecool.envs.lcopt_env import STATE_FIELDS
from safecool.envs.safety_constraints import SafetyConfig
from safecool.rl.policy_shield import PolicyShield, conservative_fallback


def _state(inlet=22.0, outlet=28.0, fan=60.0, coolant=18.0, load=0.5, ambient=20.0, power=3.0):
    return np.array([fan, coolant, inlet, outlet, load, ambient, power], dtype=np.float32)


def test_safe_action_passes_through_unchanged():
    shield = PolicyShield(
        safety_cfg=SafetyConfig(),
        fallback_policy=conservative_fallback,
        use_lookahead=False,
    )
    s = _state(inlet=22.0)
    a = np.array([0.2, 1.0], dtype=np.float32)
    out = shield(a, s)
    assert not out["intervened"]
    assert out["reason"] is None
    np.testing.assert_allclose(out["action"], a, atol=1e-6)
    assert shield.stats.total_calls == 1
    assert shield.stats.projected == 0


def test_rate_violating_action_gets_projected():
    shield = PolicyShield(
        safety_cfg=SafetyConfig(max_rate_change_coolant_c=1.0),
        fallback_policy=conservative_fallback,
        use_lookahead=False,
    )
    s = _state(inlet=22.0)
    a = np.array([5.0, 0.0], dtype=np.float32)    # exceeds rate limit
    out = shield(a, s)
    assert out["intervened"]
    assert out["reason"] == "projected"
    assert abs(out["action"][0]) <= 1.0 + 1e-6
    assert shield.stats.projected == 1


def test_is_safe_rejects_state_already_hot():
    shield = PolicyShield(
        safety_cfg=SafetyConfig(max_inlet_temp_c=27.0),
        use_lookahead=False,
    )
    s = _state(inlet=28.0)   # already above limit
    assert not shield.is_safe(np.array([0.0, 0.0]), s)


def test_is_safe_rejects_operating_window_overflow():
    shield = PolicyShield(
        safety_cfg=SafetyConfig(coolant_supply_bounds=(15.0, 24.0)),
        use_lookahead=False,
    )
    s = _state(coolant=24.0)
    # Any positive coolant delta would exceed the upper bound.
    assert not shield.is_safe(np.array([0.5, 0.0]), s)


def test_fallback_fires_when_projection_insufficient():
    """When the shield cannot rescue the action (state already unsafe), it should use fallback."""
    shield = PolicyShield(
        safety_cfg=SafetyConfig(max_inlet_temp_c=27.0),
        fallback_policy=conservative_fallback,
        use_lookahead=False,
    )
    s = _state(inlet=28.0)    # pre-shield unsafe
    a = np.array([1.0, 5.0], dtype=np.float32)
    out = shield(a, s)
    assert out["intervened"]
    assert out["reason"] in ("fallback", "projected_warn")
    assert shield.stats.fallback_used >= 1 or shield.stats.lookahead_rejected >= 1


def test_stats_increment_with_each_call():
    shield = PolicyShield(safety_cfg=SafetyConfig(), use_lookahead=False)
    s = _state()
    for _ in range(4):
        shield(np.array([0.2, 1.0]), s)
    assert shield.stats.total_calls == 4
