"""Environment behavior tests."""
from __future__ import annotations

import numpy as np

from safecool.envs.lcopt_env import (
    ACTION_FIELDS,
    STATE_FIELDS,
    CoolingEnv,
    EnvConfig,
    ScenarioSpec,
    make_scenario,
)


def test_reset_returns_valid_initial_state(small_env_cfg):
    env = CoolingEnv(small_env_cfg)
    s = env.reset()
    assert s.shape == (len(STATE_FIELDS),)
    assert np.isfinite(s).all()


def test_step_shapes_and_info_keys(small_env_cfg):
    env = CoolingEnv(small_env_cfg)
    env.reset()
    ns, r, done, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert ns.shape == (len(STATE_FIELDS),)
    assert isinstance(r, float)
    assert isinstance(done, bool)
    for k in ("scenario_family", "step", "coolant_effective_c"):
        assert k in info


def test_heatwave_scenario_raises_ambient(small_env_cfg):
    env = CoolingEnv(small_env_cfg)
    nominal = env.reset(ScenarioSpec(family="nominal", seed=0))
    heatwave = env.reset(make_scenario("heatwave", seed=0))
    amb_idx = STATE_FIELDS.index("ambient_temp_c")
    assert heatwave[amb_idx] > nominal[amb_idx]


def test_episode_terminates_after_configured_steps(small_env_cfg):
    env = CoolingEnv(small_env_cfg)
    env.reset()
    done = False
    n = 0
    while not done and n < small_env_cfg.episode_steps + 5:
        _, _, done, _ = env.step(np.array([0.0, 0.0]))
        n += 1
    assert done
    assert n == small_env_cfg.episode_steps


def test_action_clipping_keeps_controls_in_bounds(small_env_cfg):
    env = CoolingEnv(small_env_cfg)
    env.reset()
    # Try to push way past bounds; env should clip internally.
    ns, _, _, _ = env.step(np.array([100.0, 500.0]))
    fan_idx = STATE_FIELDS.index("fan_speed_pct")
    coolant_idx = STATE_FIELDS.index("coolant_supply_c")
    assert small_env_cfg.fan_speed_bounds[0] <= ns[fan_idx] <= small_env_cfg.fan_speed_bounds[1]
    assert small_env_cfg.coolant_supply_bounds[0] <= ns[coolant_idx] <= small_env_cfg.coolant_supply_bounds[1]


def test_action_fields_correct_length():
    assert len(ACTION_FIELDS) == 2
