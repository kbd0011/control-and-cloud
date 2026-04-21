"""Shared pytest fixtures for SafeCool-CMDP tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from safecool.data.build_offline_buffer import build_buffer, save_buffer, write_manifest
from safecool.envs.lcopt_env import EnvConfig
from safecool.envs.safety_constraints import SafetyConfig


@pytest.fixture(scope="session")
def small_env_cfg() -> EnvConfig:
    # Short episodes to keep tests fast.
    return EnvConfig(episode_steps=32)


@pytest.fixture(scope="session")
def safety_cfg() -> SafetyConfig:
    return SafetyConfig()


@pytest.fixture(scope="session")
def small_buffer(small_env_cfg, safety_cfg, tmp_path_factory) -> dict:
    d = tmp_path_factory.mktemp("safecool_buf")
    buffer, manifest = build_buffer(
        small_env_cfg, safety_cfg,
        policy_names=["rule_based", "conservative", "perturbed_safe"],
        scenario_families=["nominal", "spike"],
        n_episodes_per_policy=2,
        scenarios_per_policy=2,
        seed=7,
    )
    save_buffer(buffer, d / "offline_buffer.npz")
    write_manifest(manifest, d / "offline_buffer_manifest.json")
    return buffer
