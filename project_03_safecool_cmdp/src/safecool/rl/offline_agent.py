"""Offline RL agent: discrete-action Q-learning + CQL penalty + Lagrangian safety cost.

**Design notes.**

- We discretize the 2-D continuous action into a small grid configured in
  `model_cmdp.yaml`. This keeps the agent interpretable (a few thousand Q
  values), enables a clean CQL penalty (logsumexp over discrete actions minus
  in-buffer actions), and gives us a trivial greedy policy for FQE.
- The objective combines reward and a Lagrangian-weighted safety cost. The
  multiplier λ adapts on a dual-ascent schedule when the expected per-step
  cost under the current greedy policy exceeds the budget.
- A separate cost Q-network (safety critic) estimates expected undiscounted
  per-step cost for the dual update and for downstream OPE reporting.

Interface
---------
- `DiscreteOfflineAgent.predict(state)` returns an action vector in the
  original continuous space (the chosen grid point).
- `DiscreteOfflineAgent.action_logits(state)` returns Q-values over the grid
  for interpretability and OPE probability ratios.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    obs_dim: int = 7
    hidden_dim: int = 64
    num_layers: int = 2
    action_grid_coolant: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)
    action_grid_fan: tuple[float, ...] = (-10.0, -5.0, 0.0, 5.0, 10.0)

    @property
    def n_actions(self) -> int:
        return len(self.action_grid_coolant) * len(self.action_grid_fan)

    def index_to_action(self, idx: int) -> np.ndarray:
        nf = len(self.action_grid_fan)
        i, j = divmod(idx, nf)
        return np.array(
            [self.action_grid_coolant[i], self.action_grid_fan[j]], dtype=np.float32
        )

    def action_table(self) -> np.ndarray:
        rows = []
        for c in self.action_grid_coolant:
            for f in self.action_grid_fan:
                rows.append((c, f))
        return np.asarray(rows, dtype=np.float32)


class QNet(nn.Module):
    def __init__(self, cfg: AgentConfig):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = cfg.obs_dim
        for i in range(cfg.num_layers):
            layers.extend([nn.Linear(in_dim, cfg.hidden_dim), nn.GELU()])
            in_dim = cfg.hidden_dim
        layers.append(nn.Linear(in_dim, cfg.n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


def _assign_action_indices(
    actions: np.ndarray, action_table: np.ndarray
) -> np.ndarray:
    """Snap each continuous buffer action to the nearest grid index (L2)."""
    a = actions.astype(np.float32)
    diffs = a[:, None, :] - action_table[None, :, :]
    d2 = (diffs ** 2).sum(axis=-1)
    return np.argmin(d2, axis=-1).astype(np.int64)


@dataclass
class TrainedAgent:
    cfg: AgentConfig
    q_reward_state_dict: dict
    q_cost_state_dict: dict
    state_mean: np.ndarray
    state_std: np.ndarray
    lagrange: float
    cost_budget: float
    history: list
    reward_q_temperature: float = 1.0


class DiscreteOfflineAgent:
    """Wrapper around two Q-networks (reward, cost) + Lagrangian multiplier."""

    def __init__(self, cfg: AgentConfig, *, device: str = "cpu"):
        self.cfg = cfg
        self.device = device
        self.action_table = torch.from_numpy(cfg.action_table()).to(device)
        self.q_reward = QNet(cfg).to(device)
        self.q_cost = QNet(cfg).to(device)
        self.q_reward_target = copy.deepcopy(self.q_reward)
        self.q_cost_target = copy.deepcopy(self.q_cost)
        for p in self.q_reward_target.parameters():
            p.requires_grad_(False)
        for p in self.q_cost_target.parameters():
            p.requires_grad_(False)
        self.state_mean = np.zeros(cfg.obs_dim, dtype=np.float32)
        self.state_std = np.ones(cfg.obs_dim, dtype=np.float32)
        self.lagrange = 0.1

    # ------------------------------------------------------------------ scaling

    def _scale(self, s: torch.Tensor) -> torch.Tensor:
        m = torch.as_tensor(self.state_mean, dtype=s.dtype, device=s.device)
        st = torch.as_tensor(self.state_std, dtype=s.dtype, device=s.device).clamp_min(1e-3)
        return (s - m) / st

    # ------------------------------------------------------------------ API used by eval / deploy

    @torch.no_grad()
    def greedy_action_index(self, state: np.ndarray) -> int:
        s = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0).to(self.device)
        s = self._scale(s)
        q_r = self.q_reward(s)
        q_c = self.q_cost(s)
        combined = q_r - self.lagrange * q_c
        return int(torch.argmax(combined, dim=-1).item())

    def predict(self, state: np.ndarray) -> np.ndarray:
        idx = self.greedy_action_index(state)
        return self.cfg.index_to_action(idx)

    @torch.no_grad()
    def action_logits(self, state: np.ndarray) -> np.ndarray:
        """Return the combined reward − λ · cost logits over the grid."""
        s = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0).to(self.device)
        s = self._scale(s)
        q_r = self.q_reward(s).cpu().numpy()[0]
        q_c = self.q_cost(s).cpu().numpy()[0]
        return q_r - self.lagrange * q_c


def train_offline(
    buffer: dict[str, np.ndarray],
    agent_cfg: AgentConfig,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    discount: float,
    cql_alpha: float,
    lagrange_init: float,
    lagrange_lr: float,
    cost_budget: float,
    target_update_tau: float,
    device: str = "cpu",
    seed: int = 0,
    grad_clip: float = 1.0,
) -> TrainedAgent:
    """Train the discrete-action offline agent with a CQL penalty and Lagrangian dual update."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    states = buffer["state"].astype(np.float32)
    actions = buffer["action"].astype(np.float32)
    rewards = buffer["reward"].astype(np.float32)
    next_states = buffer["next_state"].astype(np.float32)
    dones = buffer["done"].astype(np.float32)
    costs = buffer["cost"].astype(np.float32)

    table = agent_cfg.action_table()
    action_idx = _assign_action_indices(actions, table)

    state_mean = states.mean(axis=0)
    state_std = states.std(axis=0).clip(min=1e-3)
    states_n = (states - state_mean) / state_std
    next_n = (next_states - state_mean) / state_std

    agent = DiscreteOfflineAgent(agent_cfg, device=device)
    agent.state_mean = state_mean.astype(np.float32)
    agent.state_std = state_std.astype(np.float32)
    agent.lagrange = float(lagrange_init)

    opt_r = torch.optim.AdamW(agent.q_reward.parameters(), lr=lr)
    opt_c = torch.optim.AdamW(agent.q_cost.parameters(), lr=lr)

    S = torch.from_numpy(states_n).to(device)
    A_idx = torch.from_numpy(action_idx).to(device)
    R = torch.from_numpy(rewards).to(device)
    NS = torch.from_numpy(next_n).to(device)
    D = torch.from_numpy(dones).to(device)
    C = torch.from_numpy(costs).to(device)
    n = S.shape[0]

    history = []
    for ep in range(epochs):
        perm = rng.permutation(n)
        losses_r, losses_c, cql_pens = [], [], []
        for start in range(0, n, batch_size):
            b = perm[start : start + batch_size]
            s = S[b]; ai = A_idx[b]; r = R[b]; ns = NS[b]; d = D[b]; c = C[b]
            # Reward critic update.
            with torch.no_grad():
                target_q_next = agent.q_reward_target(ns).max(dim=-1).values
                target_q = r + discount * (1 - d) * target_q_next
            q_all = agent.q_reward(s)
            q_taken = q_all.gather(1, ai.view(-1, 1)).squeeze(-1)
            td_r = F.mse_loss(q_taken, target_q)
            # CQL penalty: logsumexp over all actions minus Q-value of in-data action.
            cql_pen = (torch.logsumexp(q_all, dim=-1) - q_taken).mean()
            loss_r = td_r + cql_alpha * cql_pen
            opt_r.zero_grad()
            loss_r.backward()
            torch.nn.utils.clip_grad_norm_(agent.q_reward.parameters(), grad_clip)
            opt_r.step()

            # Cost critic update (separate network, non-negative costs).
            with torch.no_grad():
                # Expected future cost under the current reward-greedy policy.
                qr_next = agent.q_reward(ns)
                qc_next = agent.q_cost(ns)
                greedy_next = torch.argmax(qr_next - agent.lagrange * qc_next, dim=-1)
                cost_next = agent.q_cost_target(ns).gather(1, greedy_next.view(-1, 1)).squeeze(-1)
                cost_target = c + discount * (1 - d) * cost_next
            qc_all = agent.q_cost(s)
            qc_taken = qc_all.gather(1, ai.view(-1, 1)).squeeze(-1)
            loss_c = F.mse_loss(qc_taken, cost_target)
            opt_c.zero_grad()
            loss_c.backward()
            torch.nn.utils.clip_grad_norm_(agent.q_cost.parameters(), grad_clip)
            opt_c.step()

            # Soft-update targets.
            with torch.no_grad():
                for p, pt in zip(agent.q_reward.parameters(), agent.q_reward_target.parameters()):
                    pt.mul_(1 - target_update_tau).add_(p * target_update_tau)
                for p, pt in zip(agent.q_cost.parameters(), agent.q_cost_target.parameters()):
                    pt.mul_(1 - target_update_tau).add_(p * target_update_tau)

            losses_r.append(float(td_r.item()))
            losses_c.append(float(loss_c.item()))
            cql_pens.append(float(cql_pen.item()))

        # Lagrange multiplier update: ascend when expected cost > budget on current greedy policy.
        with torch.no_grad():
            qr_all = agent.q_reward(S)
            qc_all = agent.q_cost(S)
            greedy = torch.argmax(qr_all - agent.lagrange * qc_all, dim=-1)
            est_cost = qc_all.gather(1, greedy.view(-1, 1)).mean().item()
        # Convert discounted sum estimate to approximate per-step budget: cost_budget / (1 - γ).
        per_episode_budget = cost_budget / max(1 - discount, 1e-3)
        lag_grad = est_cost - per_episode_budget
        agent.lagrange = float(max(0.0, agent.lagrange + lagrange_lr * lag_grad))
        history.append(
            {
                "epoch": ep,
                "td_reward": float(np.mean(losses_r)),
                "td_cost": float(np.mean(losses_c)),
                "cql_pen": float(np.mean(cql_pens)),
                "lagrange": agent.lagrange,
                "est_cost": float(est_cost),
            }
        )
        logger.info(
            "agent ep=%d td_r=%.4f td_c=%.4f cql=%.3f lag=%.3f est_cost=%.4f",
            ep, history[-1]["td_reward"], history[-1]["td_cost"],
            history[-1]["cql_pen"], agent.lagrange, est_cost,
        )

    return TrainedAgent(
        cfg=agent_cfg,
        q_reward_state_dict={k: v.cpu() for k, v in agent.q_reward.state_dict().items()},
        q_cost_state_dict={k: v.cpu() for k, v in agent.q_cost.state_dict().items()},
        state_mean=state_mean.astype(np.float32),
        state_std=state_std.astype(np.float32),
        lagrange=float(agent.lagrange),
        cost_budget=float(cost_budget),
        history=history,
    )


def save_agent(trained: TrainedAgent, path: str):
    payload = {
        "cfg": {
            "obs_dim": trained.cfg.obs_dim,
            "hidden_dim": trained.cfg.hidden_dim,
            "num_layers": trained.cfg.num_layers,
            "action_grid_coolant": list(trained.cfg.action_grid_coolant),
            "action_grid_fan": list(trained.cfg.action_grid_fan),
        },
        "q_reward": trained.q_reward_state_dict,
        "q_cost": trained.q_cost_state_dict,
        "state_mean": trained.state_mean.tolist(),
        "state_std": trained.state_std.tolist(),
        "lagrange": trained.lagrange,
        "cost_budget": trained.cost_budget,
        "history": trained.history,
    }
    torch.save(payload, path)


def load_agent(path: str, device: str = "cpu") -> DiscreteOfflineAgent:
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg_d = payload["cfg"]
    cfg = AgentConfig(
        obs_dim=int(cfg_d["obs_dim"]),
        hidden_dim=int(cfg_d["hidden_dim"]),
        num_layers=int(cfg_d["num_layers"]),
        action_grid_coolant=tuple(cfg_d["action_grid_coolant"]),
        action_grid_fan=tuple(cfg_d["action_grid_fan"]),
    )
    agent = DiscreteOfflineAgent(cfg, device=device)
    agent.q_reward.load_state_dict(payload["q_reward"])
    agent.q_cost.load_state_dict(payload["q_cost"])
    agent.q_reward_target.load_state_dict(payload["q_reward"])
    agent.q_cost_target.load_state_dict(payload["q_cost"])
    agent.state_mean = np.asarray(payload["state_mean"], dtype=np.float32)
    agent.state_std = np.asarray(payload["state_std"], dtype=np.float32)
    agent.lagrange = float(payload["lagrange"])
    return agent
