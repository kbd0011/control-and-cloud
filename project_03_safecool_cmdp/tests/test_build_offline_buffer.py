"""Offline buffer tests: schema, safety columns, determinism, manifest."""
from __future__ import annotations

import numpy as np

from safecool.data.build_offline_buffer import (
    POLICY_REGISTRY,
    build_buffer,
    load_buffer,
    rollout_policy,
    save_buffer,
)
from safecool.envs.lcopt_env import CoolingEnv, make_scenario


def test_rollout_returns_all_required_keys(small_env_cfg, safety_cfg):
    env = CoolingEnv(small_env_cfg)
    policy = POLICY_REGISTRY["rule_based"]
    chunk = rollout_policy(env, policy, n_steps=20, seed=0,
                           scenario=make_scenario("nominal", 0), safety_cfg=safety_cfg)
    required = {"state", "action", "reward", "next_state", "done", "cost",
                "cost_inlet_hi", "logp_behavior"}
    assert required.issubset(chunk.keys())
    n = chunk["state"].shape[0]
    assert chunk["action"].shape[0] == n
    assert chunk["cost"].shape[0] == n
    assert chunk["reward"].shape[0] == n


def test_buffer_shapes_are_consistent(small_buffer):
    n = small_buffer["state"].shape[0]
    for k in ("action", "reward", "next_state", "done", "cost", "logp_behavior",
              "episode_id", "policy_id", "scenario_family"):
        assert small_buffer[k].shape[0] == n, f"{k} has wrong length"
    assert small_buffer["state"].shape[1] == 7
    assert small_buffer["action"].shape[1] == 2
    assert small_buffer["next_state"].shape[1] == 7


def test_buffer_preserves_policy_and_scenario_labels(small_buffer):
    policies = set(small_buffer["policy_id"].astype(str).tolist())
    families = set(small_buffer["scenario_family"].astype(str).tolist())
    assert policies == {"rule_based", "conservative", "perturbed_safe"}
    assert families.issubset({"nominal", "spike", "degradation", "heatwave"})


def test_buffer_costs_are_nonnegative(small_buffer):
    assert float(small_buffer["cost"].min()) >= 0.0


def test_determinism_across_rebuilds(small_env_cfg, safety_cfg):
    b1, _ = build_buffer(small_env_cfg, safety_cfg, policy_names=["rule_based"],
                          scenario_families=["nominal"], n_episodes_per_policy=2, seed=11)
    b2, _ = build_buffer(small_env_cfg, safety_cfg, policy_names=["rule_based"],
                          scenario_families=["nominal"], n_episodes_per_policy=2, seed=11)
    assert b1["state"].shape == b2["state"].shape
    np.testing.assert_allclose(b1["state"], b2["state"], atol=1e-5)
    np.testing.assert_allclose(b1["reward"], b2["reward"], atol=1e-5)


def test_save_and_load_roundtrip(small_buffer, tmp_path):
    p = tmp_path / "buf.npz"
    save_buffer(small_buffer, p)
    loaded = load_buffer(p)
    np.testing.assert_allclose(loaded["state"], small_buffer["state"])
    np.testing.assert_allclose(loaded["reward"], small_buffer["reward"])
