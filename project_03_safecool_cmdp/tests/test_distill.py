"""Distill tests: surrogate fitting, fidelity bounds, rule export non-empty."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from safecool.distill import compute_fidelity, export_rules, fit_surrogate
from safecool.rl.offline_agent import AgentConfig, DiscreteOfflineAgent


def test_fit_surrogate_trees_predict_on_held_out_states(small_buffer):
    ac = AgentConfig(
        hidden_dim=16, num_layers=1,
        action_grid_coolant=(-1.0, 0.0, 1.0),
        action_grid_fan=(-5.0, 0.0, 5.0),
    )
    agent = DiscreteOfflineAgent(ac)
    states = small_buffer["state"][:100]
    trees = fit_surrogate(agent, states, max_depth=3, seed=0)
    assert "coolant" in trees and "fan" in trees
    # Can predict on held-out states.
    pred = np.stack([
        trees["coolant"].predict(small_buffer["state"][100:120]),
        trees["fan"].predict(small_buffer["state"][100:120]),
    ], axis=-1)
    assert pred.shape == (20, 2)


def test_fidelity_within_half_grid_is_reasonable(small_buffer):
    ac = AgentConfig(
        hidden_dim=16, num_layers=1,
        action_grid_coolant=(-1.0, 0.0, 1.0),
        action_grid_fan=(-5.0, 0.0, 5.0),
    )
    agent = DiscreteOfflineAgent(ac)
    fit_states = small_buffer["state"][:100]
    holdout = small_buffer["state"][100:150]
    trees = fit_surrogate(agent, fit_states, max_depth=3, seed=0)
    f = compute_fidelity(agent, trees, holdout)
    # Agreement must be bounded in [0, 1]; untrained agent often picks the same action constantly
    # so surrogate should match most of the time.
    assert 0.0 <= f["agreement_within_half_grid"] <= 1.0
    assert f["mae_coolant"] >= 0
    assert f["mae_fan"] >= 0


def test_export_rules_writes_nonempty_file(small_buffer, tmp_path):
    ac = AgentConfig(
        hidden_dim=16, num_layers=1,
        action_grid_coolant=(-1.0, 0.0, 1.0),
        action_grid_fan=(-5.0, 0.0, 5.0),
    )
    agent = DiscreteOfflineAgent(ac)
    trees = fit_surrogate(agent, small_buffer["state"][:100], max_depth=3, seed=0)
    p = tmp_path / "surrogate.txt"
    body = export_rules(trees, p)
    assert p.exists()
    assert len(body) > 0
    assert "Coolant supply delta" in body
    assert "Fan speed delta" in body
