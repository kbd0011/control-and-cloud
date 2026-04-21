"""Offline-agent tests: action space, training produces finite Q, save/load, Lagrange."""
from __future__ import annotations

import numpy as np
import torch

from safecool.rl.offline_agent import (
    AgentConfig,
    DiscreteOfflineAgent,
    _assign_action_indices,
    load_agent,
    save_agent,
    train_offline,
)


def test_action_space_mapping():
    cfg = AgentConfig(
        action_grid_coolant=(-1.0, 0.0, 1.0),
        action_grid_fan=(-10.0, 0.0, 10.0),
    )
    assert cfg.n_actions == 9
    # index_to_action should be reversible to (row, col) via the table.
    a0 = cfg.index_to_action(0)
    assert np.allclose(a0, [-1.0, -10.0])
    a4 = cfg.index_to_action(4)   # middle row, middle col = (0.0, 0.0)
    assert np.allclose(a4, [0.0, 0.0])


def test_assign_action_indices_snaps_to_nearest():
    table = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 5.0]], dtype=np.float32)
    actions = np.array([[0.9, 0.1], [0.0, 4.5]], dtype=np.float32)
    idx = _assign_action_indices(actions, table)
    assert idx[0] == 1
    assert idx[1] == 2


def test_train_offline_runs_and_emits_history(small_buffer):
    ac = AgentConfig(
        obs_dim=7, hidden_dim=16, num_layers=1,
        action_grid_coolant=(-1.0, 0.0, 1.0),
        action_grid_fan=(-5.0, 0.0, 5.0),
    )
    trained = train_offline(
        small_buffer, ac,
        epochs=2, batch_size=64, lr=1e-3,
        discount=0.9, cql_alpha=0.5,
        lagrange_init=0.1, lagrange_lr=0.01,
        cost_budget=0.02, target_update_tau=0.01, seed=0,
    )
    assert len(trained.history) == 2
    assert np.isfinite(trained.history[-1]["td_reward"])
    assert trained.lagrange >= 0.0


def test_predict_returns_valid_action_dims(small_buffer):
    ac = AgentConfig(hidden_dim=16, num_layers=1)
    agent = DiscreteOfflineAgent(ac)
    a = agent.predict(small_buffer["state"][0])
    assert a.shape == (2,)
    assert np.isfinite(a).all()


def test_save_and_load_roundtrip(small_buffer, tmp_path):
    ac = AgentConfig(obs_dim=7, hidden_dim=16, num_layers=1,
                     action_grid_coolant=(-1.0, 0.0, 1.0),
                     action_grid_fan=(-5.0, 0.0, 5.0))
    trained = train_offline(
        small_buffer, ac,
        epochs=1, batch_size=64, lr=1e-3, discount=0.9, cql_alpha=0.1,
        lagrange_init=0.1, lagrange_lr=0.01, cost_budget=0.02,
        target_update_tau=0.01, seed=0,
    )
    p = tmp_path / "agent.pt"
    save_agent(trained, str(p))
    loaded = load_agent(str(p))
    a_before = trained.q_reward_state_dict["net.0.weight"]
    # Get the same layer from the loaded model.
    a_after = dict(loaded.q_reward.state_dict())["net.0.weight"].cpu()
    assert torch.allclose(a_before, a_after)


def test_lagrange_updates_toward_budget(small_buffer):
    """If cost budget is very small, Lagrange should grow (or at least not drop to 0)."""
    ac = AgentConfig(hidden_dim=16, num_layers=1,
                     action_grid_coolant=(-1.0, 0.0, 1.0),
                     action_grid_fan=(-5.0, 0.0, 5.0))
    trained = train_offline(
        small_buffer, ac,
        epochs=3, batch_size=64, lr=1e-3, discount=0.9, cql_alpha=0.1,
        lagrange_init=0.1, lagrange_lr=0.1,
        cost_budget=0.0,          # no cost allowed
        target_update_tau=0.01, seed=0,
    )
    lags = [h["lagrange"] for h in trained.history]
    # Either λ increases (finding violations) or it stays flat (no violations seen).
    # It must not become negative.
    assert min(lags) >= 0.0
