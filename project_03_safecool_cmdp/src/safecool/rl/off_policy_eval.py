"""Off-policy evaluation.

Two estimators implemented:

- **Weighted Importance Sampling (WIS)** on whole episodes, using the
  behavior policy log-prob stored in the buffer and an evaluation policy
  expressed as a discrete distribution over the action grid. WIS is
  well-known to be high-variance but low-bias; we use it as a sanity
  check rather than a primary number.

- **Fitted Q Evaluation (FQE)** as the primary estimator. We train a fresh
  Q-network on the buffer using bootstrap targets computed under the
  evaluation policy's greedy action. This avoids the variance explosion of
  importance ratios at the cost of some bias from function approximation.

Both estimators return separate reward and cost numbers so the downstream
deployment gate can enforce the cost budget.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from safecool.rl.offline_agent import DiscreteOfflineAgent, QNet, _assign_action_indices

logger = logging.getLogger(__name__)


@dataclass
class OPEResult:
    method: str
    reward_estimate: float
    cost_estimate: float
    reward_ci_low: float | None = None
    reward_ci_high: float | None = None

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "reward_estimate": float(self.reward_estimate),
            "cost_estimate": float(self.cost_estimate),
            "reward_ci_low": None if self.reward_ci_low is None else float(self.reward_ci_low),
            "reward_ci_high": None if self.reward_ci_high is None else float(self.reward_ci_high),
        }


def _eval_policy_logprobs(
    agent: DiscreteOfflineAgent,
    states: np.ndarray,
    action_idx: np.ndarray,
    *,
    temperature: float = 0.5,
) -> np.ndarray:
    """Softmax-policy log-prob of observed actions under the learned agent.

    We deliberately use a soft target-policy (temperature > 0) so importance
    weights stay bounded. Pure greedy would give log 0 on any action the agent
    wouldn't have chosen.
    """
    with torch.no_grad():
        s = torch.from_numpy(states.astype(np.float32)).to(agent.device)
        s = agent._scale(s)
        q_r = agent.q_reward(s)
        q_c = agent.q_cost(s)
        logits = (q_r - agent.lagrange * q_c) / temperature
        logp = F.log_softmax(logits, dim=-1).cpu().numpy()
    a_idx = torch.from_numpy(action_idx).long()
    return logp[np.arange(len(a_idx)), a_idx.numpy()]


def weighted_importance_sampling(
    buffer: dict[str, np.ndarray],
    agent: DiscreteOfflineAgent,
    *,
    discount: float = 0.97,
    temperature: float = 0.5,
    n_bootstrap: int = 200,
    seed: int = 0,
) -> OPEResult:
    """Per-episode WIS with bootstrap CI.

    Weights are computed on the log-prob ratio between the soft target policy
    and the Gaussian behavior policy log-prob stored in the buffer.
    """
    states = buffer["state"]
    actions = buffer["action"]
    rewards = buffer["reward"]
    costs = buffer["cost"]
    episode_ids = buffer["episode_id"]
    logp_behavior = buffer["logp_behavior"]

    action_idx = _assign_action_indices(actions, agent.cfg.action_table())
    logp_target = _eval_policy_logprobs(agent, states, action_idx, temperature=temperature)

    # Group by episode and compute per-episode log importance ratio and discounted returns.
    unique = np.unique(episode_ids)
    episode_weights = []
    episode_returns_r = []
    episode_returns_c = []
    for ep in unique:
        mask = episode_ids == ep
        log_ratio = float(np.sum(logp_target[mask] - logp_behavior[mask]))
        # Stabilize: clip sums to avoid ±inf.
        log_ratio = float(np.clip(log_ratio, -20.0, 20.0))
        weight = float(np.exp(log_ratio))
        gamma_vec = discount ** np.arange(int(mask.sum()))
        ret_r = float(np.sum(gamma_vec * rewards[mask]))
        ret_c = float(np.sum(gamma_vec * costs[mask]))
        episode_weights.append(weight)
        episode_returns_r.append(ret_r)
        episode_returns_c.append(ret_c)

    w = np.asarray(episode_weights, dtype=float)
    r_arr = np.asarray(episode_returns_r, dtype=float)
    c_arr = np.asarray(episode_returns_c, dtype=float)
    w_sum = float(w.sum())
    if w_sum <= 0:
        return OPEResult(method="WIS", reward_estimate=float("nan"), cost_estimate=float("nan"))
    reward_hat = float(np.sum(w * r_arr) / w_sum)
    cost_hat = float(np.sum(w * c_arr) / w_sum)

    # Bootstrap CI on reward.
    rng = np.random.default_rng(seed)
    n = len(w)
    boot_r = []
    for _ in range(n_bootstrap):
        b = rng.integers(0, n, size=n)
        w_b = w[b]; r_b = r_arr[b]
        s = float(w_b.sum())
        if s > 0:
            boot_r.append(float(np.sum(w_b * r_b) / s))
    ci_low, ci_high = (
        (float(np.percentile(boot_r, 2.5)), float(np.percentile(boot_r, 97.5)))
        if boot_r else (None, None)
    )
    return OPEResult(
        method="WIS",
        reward_estimate=reward_hat,
        cost_estimate=cost_hat,
        reward_ci_low=ci_low,
        reward_ci_high=ci_high,
    )


def fitted_q_evaluation(
    buffer: dict[str, np.ndarray],
    agent: DiscreteOfflineAgent,
    *,
    discount: float = 0.97,
    epochs: int = 30,
    lr: float = 0.001,
    batch_size: int = 256,
    seed: int = 0,
) -> OPEResult:
    """FQE for the target policy's greedy action selection.

    We fit two Q-networks (reward, cost) by repeatedly bootstrapping targets
    where the next-step action is selected by the target policy (greedy on the
    agent's combined reward − λ·cost logits). The estimator returns the mean
    of the fitted Q at each episode's initial state.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    states = buffer["state"].astype(np.float32)
    actions = buffer["action"].astype(np.float32)
    rewards = buffer["reward"].astype(np.float32)
    next_states = buffer["next_state"].astype(np.float32)
    dones = buffer["done"].astype(np.float32)
    costs = buffer["cost"].astype(np.float32)
    episode_ids = buffer["episode_id"]
    action_idx = _assign_action_indices(actions, agent.cfg.action_table())

    # Normalize.
    mean = states.mean(axis=0)
    std = states.std(axis=0).clip(min=1e-3)
    S = torch.from_numpy((states - mean) / std)
    NS = torch.from_numpy((next_states - mean) / std)
    A = torch.from_numpy(action_idx).long()
    R = torch.from_numpy(rewards)
    C = torch.from_numpy(costs)
    D = torch.from_numpy(dones)

    q_r = QNet(agent.cfg)
    q_c = QNet(agent.cfg)
    opt_r = torch.optim.AdamW(q_r.parameters(), lr=lr)
    opt_c = torch.optim.AdamW(q_c.parameters(), lr=lr)

    # Pre-compute target-policy greedy action indices on next_states. We use the TRAINED agent
    # (Note: greedy actions drift as λ updates, but for OPE we freeze at current λ).
    with torch.no_grad():
        s_ns_for_agent = torch.from_numpy(next_states.astype(np.float32))
        s_ns_scaled_for_agent = (s_ns_for_agent - torch.as_tensor(agent.state_mean)) / torch.as_tensor(agent.state_std).clamp_min(1e-3)
        qr = agent.q_reward(s_ns_scaled_for_agent)
        qc = agent.q_cost(s_ns_scaled_for_agent)
        greedy_next = torch.argmax(qr - agent.lagrange * qc, dim=-1)

    n = S.shape[0]
    for ep in range(epochs):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            b = perm[start : start + batch_size]
            s = S[b]; ai = A[b]; r = R[b]; ns = NS[b]; d = D[b]; c = C[b]
            gn = greedy_next[b]
            with torch.no_grad():
                target_r = r + discount * (1 - d) * q_r(ns).gather(1, gn.view(-1, 1)).squeeze(-1)
                target_c = c + discount * (1 - d) * q_c(ns).gather(1, gn.view(-1, 1)).squeeze(-1)
            q_r_out = q_r(s).gather(1, ai.view(-1, 1)).squeeze(-1)
            q_c_out = q_c(s).gather(1, ai.view(-1, 1)).squeeze(-1)
            loss_r = F.mse_loss(q_r_out, target_r)
            loss_c = F.mse_loss(q_c_out, target_c)
            opt_r.zero_grad(); loss_r.backward(); opt_r.step()
            opt_c.zero_grad(); loss_c.backward(); opt_c.step()

    # Estimate return at initial states.
    unique_eps, first_idx = np.unique(episode_ids, return_index=True)
    init_states = S[first_idx]
    with torch.no_grad():
        # Choose greedy action under the target policy at init states (using the trained agent).
        init_scaled_for_agent = torch.from_numpy(states[first_idx].astype(np.float32))
        init_scaled_for_agent = (init_scaled_for_agent - torch.as_tensor(agent.state_mean)) / torch.as_tensor(agent.state_std).clamp_min(1e-3)
        qr_init = agent.q_reward(init_scaled_for_agent)
        qc_init = agent.q_cost(init_scaled_for_agent)
        greedy_init = torch.argmax(qr_init - agent.lagrange * qc_init, dim=-1)
        q_r_values = q_r(init_states).gather(1, greedy_init.view(-1, 1)).squeeze(-1).numpy()
        q_c_values = q_c(init_states).gather(1, greedy_init.view(-1, 1)).squeeze(-1).numpy()
    reward_hat = float(np.mean(q_r_values))
    cost_hat = float(max(0.0, np.mean(q_c_values)))

    return OPEResult(method="FQE", reward_estimate=reward_hat, cost_estimate=cost_hat)
