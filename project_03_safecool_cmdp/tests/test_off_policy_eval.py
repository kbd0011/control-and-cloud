"""Off-policy-evaluation tests: finite outputs, reward + cost reported separately."""
from __future__ import annotations

import numpy as np

from safecool.rl.off_policy_eval import (
    OPEResult,
    fitted_q_evaluation,
    weighted_importance_sampling,
)
from safecool.rl.offline_agent import AgentConfig, DiscreteOfflineAgent, train_offline


def _small_agent(small_buffer):
    ac = AgentConfig(
        obs_dim=7, hidden_dim=16, num_layers=1,
        action_grid_coolant=(-1.0, 0.0, 1.0),
        action_grid_fan=(-5.0, 0.0, 5.0),
    )
    trained = train_offline(
        small_buffer, ac,
        epochs=2, batch_size=64, lr=1e-3, discount=0.9, cql_alpha=0.5,
        lagrange_init=0.1, lagrange_lr=0.01, cost_budget=0.02,
        target_update_tau=0.01, seed=0,
    )
    agent = DiscreteOfflineAgent(ac)
    agent.q_reward.load_state_dict(trained.q_reward_state_dict)
    agent.q_cost.load_state_dict(trained.q_cost_state_dict)
    agent.state_mean = trained.state_mean
    agent.state_std = trained.state_std
    agent.lagrange = trained.lagrange
    return agent


def test_wis_returns_finite_reward_and_cost(small_buffer):
    agent = _small_agent(small_buffer)
    res = weighted_importance_sampling(small_buffer, agent, discount=0.9, n_bootstrap=30, seed=0)
    assert isinstance(res, OPEResult)
    assert np.isfinite(res.reward_estimate)
    assert np.isfinite(res.cost_estimate)
    assert res.cost_estimate >= 0


def test_fqe_returns_finite_reward_and_cost(small_buffer):
    agent = _small_agent(small_buffer)
    res = fitted_q_evaluation(small_buffer, agent, discount=0.9, epochs=3, lr=1e-3, batch_size=64, seed=0)
    assert isinstance(res, OPEResult)
    assert np.isfinite(res.reward_estimate)
    assert np.isfinite(res.cost_estimate)
    assert res.cost_estimate >= 0


def test_wis_bootstrap_ci_order(small_buffer):
    agent = _small_agent(small_buffer)
    res = weighted_importance_sampling(small_buffer, agent, discount=0.9, n_bootstrap=50, seed=0)
    assert res.reward_ci_low is not None
    assert res.reward_ci_high is not None
    assert res.reward_ci_low <= res.reward_ci_high


def test_ope_result_as_dict_shape(small_buffer):
    agent = _small_agent(small_buffer)
    res = fitted_q_evaluation(small_buffer, agent, discount=0.9, epochs=1, lr=1e-3, batch_size=64, seed=0)
    d = res.as_dict()
    assert d["method"] == "FQE"
    assert "reward_estimate" in d and "cost_estimate" in d
