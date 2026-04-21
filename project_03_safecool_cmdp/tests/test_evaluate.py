"""Evaluate tests: rollout metrics + go/no-go rule + CLI main."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from safecool.envs.lcopt_env import EnvConfig
from safecool.envs.safety_constraints import SafetyConfig
from safecool.evaluate import (
    compute_tail_risk,
    make_go_no_go,
    main as evaluate_main,
    rollout_episode,
)
from safecool.envs.lcopt_env import CoolingEnv, make_scenario
from safecool.train import main as train_main


def test_compute_tail_risk_monotone_on_hotter_tail():
    cool = np.array([22.0, 22.5, 23.0, 23.5])
    hot = np.array([22.0, 22.5, 23.0, 28.0])
    assert compute_tail_risk(hot) > compute_tail_risk(cool)


def test_rollout_episode_returns_expected_keys(small_env_cfg, safety_cfg):
    env = CoolingEnv(small_env_cfg)
    scenario = make_scenario("nominal", 0)

    def const_policy(state):
        return np.array([0.0, 0.0], dtype=np.float32)

    r = rollout_episode(env, const_policy, scenario=scenario, safety_cfg=safety_cfg)
    for key in ("n_steps", "mean_reward", "max_inlet_c", "cvar95_inlet_c",
                "total_cost", "violations"):
        assert key in r


def test_go_no_go_fails_on_violations():
    rows = [{
        "policy": "agent_shielded", "scenario_family": "nominal",
        "cvar95_inlet_c": 25.0, "viol_any": 3,
    }]
    df = pd.DataFrame(rows)
    result = make_go_no_go(df, SafetyConfig(max_inlet_temp_c=27.0), policy="agent_shielded")
    assert result["pass"] is False


def test_go_no_go_passes_when_clean():
    rows = [{
        "policy": "agent_shielded", "scenario_family": "nominal",
        "cvar95_inlet_c": 25.0, "viol_any": 0,
    }]
    df = pd.DataFrame(rows)
    result = make_go_no_go(df, SafetyConfig(max_inlet_temp_c=27.0), policy="agent_shielded")
    assert result["pass"] is True


def test_evaluate_cli_runs(tmp_path):
    artifacts = tmp_path / "art"
    # Need a trained agent first.
    rc = train_main([
        "--base-config", "configs/base.yaml",
        "--env-config", "configs/env_lcopt.yaml",
        "--model-config", "configs/model_cmdp.yaml",
        "--artifacts", str(artifacts),
        "--quick",
    ])
    assert rc == 0
    rc = evaluate_main([
        "--base-config", "configs/base.yaml",
        "--env-config", "configs/env_lcopt.yaml",
        "--model-config", "configs/model_cmdp.yaml",
        "--artifacts", str(artifacts),
        "--episodes", "1",
    ])
    assert rc == 0
    for name in ("eval_episodes.csv", "eval_by_policy.csv", "eval_report.json"):
        assert (artifacts / name).exists()
    report = json.loads((artifacts / "eval_report.json").read_text())
    assert "go_no_go_agent_shielded" in report
