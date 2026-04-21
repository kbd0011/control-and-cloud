"""Deploy tests: rollback logic + shadow-mode loop + report contents."""
from __future__ import annotations

import json

import numpy as np

from safecool.deploy import run_shadow_mode, should_rollback, write_deploy_report
from safecool.envs.safety_constraints import SafetyConfig
from safecool.rl.offline_agent import AgentConfig, DiscreteOfflineAgent
from safecool.rl.policy_shield import PolicyShield, conservative_fallback


def test_should_rollback_false_when_clean():
    trip, reason = should_rollback(
        recent_inlets=[22.0, 22.5, 23.0],
        recent_violations=[0, 0, 0],
        cvar_limit=27.5, violation_limit=5,
    )
    assert not trip
    assert reason is None


def test_should_rollback_triggers_on_high_cvar():
    trip, reason = should_rollback(
        recent_inlets=[22.0, 22.5, 28.0, 28.5, 29.0],
        recent_violations=[0] * 5,
        cvar_limit=27.5, violation_limit=5,
    )
    assert trip
    assert "CVaR" in reason


def test_should_rollback_triggers_on_violation_count():
    trip, reason = should_rollback(
        recent_inlets=[22.0, 22.5],
        recent_violations=[1, 1, 1, 1, 1, 1, 1, 1],
        cvar_limit=30.0, violation_limit=5,
    )
    assert trip
    assert "violation" in reason


def test_run_shadow_mode_smoke(small_env_cfg, safety_cfg):
    ac = AgentConfig(
        obs_dim=7, hidden_dim=16, num_layers=1,
        action_grid_coolant=(-1.0, 0.0, 1.0),
        action_grid_fan=(-5.0, 0.0, 5.0),
    )
    agent = DiscreteOfflineAgent(ac)
    shield = PolicyShield(
        safety_cfg=safety_cfg,
        fallback_policy=conservative_fallback,
        use_lookahead=False,
    )
    report = run_shadow_mode(
        small_env_cfg, safety_cfg, agent, shield,
        scenario_family="nominal", n_steps=20, seed=0,
        rollback_cvar_temp_c=30.0, rollback_max_violations=50, rollback_trip_window=10,
    )
    assert report["n_steps"] <= 20 + 1
    for key in ("records", "max_inlet_c", "cvar95_inlet_c", "n_shield_interventions", "rollback_active"):
        assert key in report
    rec = report["records"][0]
    for key in ("proposed_action", "shielded_action", "baseline_action", "reward", "cost"):
        assert key in rec


def test_write_deploy_report_includes_disclaimer_and_hash(tmp_path):
    report = {"records": [], "scenario_family": "nominal", "n_steps": 0}
    p = tmp_path / "deploy.json"
    write_deploy_report(report, p, agent_hash="abcd1234")
    data = json.loads(p.read_text())
    assert data["agent_checkpoint_sha256"] == "abcd1234"
    assert "disclaimer" in data
    assert "research" in data["disclaimer"].lower() or "not a production" in data["disclaimer"].lower()
