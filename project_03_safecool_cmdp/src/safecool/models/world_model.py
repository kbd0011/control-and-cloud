"""World model for SafeCool: a small MLP that predicts next state, reward, and cost.

Interface
---------
- `WorldModel.forward(state, action) -> {next_state, reward, cost, latent}`.
- `encode(obs) -> latent` is exposed for interpretability and possible re-use
  by downstream modules (e.g. the shield uses it to forecast the next inlet
  temperature before letting an action through).

Training minimizes MSE on (Δstate, reward, cost) from the offline buffer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class WorldModelConfig:
    obs_dim: int = 7
    action_dim: int = 2
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.0


class WorldModel(nn.Module):
    """Predicts next_state delta, scalar reward, and scalar safety cost."""

    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.obs_dim + cfg.action_dim
        layers: list[nn.Module] = []
        for i in range(cfg.num_layers):
            in_d = in_dim if i == 0 else cfg.hidden_dim
            layers.extend([nn.Linear(in_d, cfg.hidden_dim), nn.GELU(), nn.Dropout(cfg.dropout)])
        self.trunk = nn.Sequential(*layers)
        self.delta_head = nn.Linear(cfg.hidden_dim, cfg.obs_dim)
        self.reward_head = nn.Linear(cfg.hidden_dim, 1)
        self.cost_head = nn.Linear(cfg.hidden_dim, 1)
        # Small init for stability on short training runs.
        nn.init.normal_(self.delta_head.weight, std=0.01)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.normal_(self.reward_head.weight, std=0.01)
        nn.init.zeros_(self.reward_head.bias)
        nn.init.normal_(self.cost_head.weight, std=0.01)
        nn.init.zeros_(self.cost_head.bias)

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        # The "latent" here is the trunk output when action = 0 padding.
        B = state.shape[0]
        action0 = torch.zeros(B, self.cfg.action_dim, device=state.device, dtype=state.dtype)
        return self.trunk(torch.cat([state, action0], dim=-1))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> dict:
        h = self.trunk(torch.cat([state, action], dim=-1))
        delta = self.delta_head(h)
        reward = self.reward_head(h).squeeze(-1)
        cost = torch.relu(self.cost_head(h)).squeeze(-1)   # cost is non-negative
        next_state = state + delta
        return {"next_state": next_state, "reward": reward, "cost": cost, "latent": h}


def train_world_model(
    model: WorldModel,
    buffer: dict[str, np.ndarray],
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str = "cpu",
    seed: int = 0,
) -> dict:
    """Train the world model on the offline buffer. Returns training history."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    states = torch.from_numpy(buffer["state"].astype(np.float32))
    actions = torch.from_numpy(buffer["action"].astype(np.float32))
    next_states = torch.from_numpy(buffer["next_state"].astype(np.float32))
    rewards = torch.from_numpy(buffer["reward"].astype(np.float32))
    costs = torch.from_numpy(buffer["cost"].astype(np.float32))
    # Standardize state for more stable regression; save stats for inverse use.
    mean = states.mean(dim=0)
    std = states.std(dim=0).clamp_min(1e-3)
    states_n = (states - mean) / std
    next_states_n = (next_states - mean) / std

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []

    n = states.shape[0]
    for ep in range(epochs):
        idx = rng.permutation(n)
        losses = []
        for start in range(0, n, batch_size):
            b = idx[start : start + batch_size]
            s = states_n[b].to(device)
            a = actions[b].to(device)
            ns = next_states_n[b].to(device)
            r = rewards[b].to(device)
            c = costs[b].to(device)
            out = model(s, a)
            loss = (
                (out["next_state"] - ns).pow(2).mean()
                + (out["reward"] - r).pow(2).mean()
                + (out["cost"] - c).pow(2).mean()
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        history.append({"epoch": ep, "loss": float(np.mean(losses))})

    return {
        "history": history,
        "state_mean": mean.cpu().numpy().tolist(),
        "state_std": std.cpu().numpy().tolist(),
    }


def predict_next_inlet_temp(
    model: WorldModel,
    state: np.ndarray,
    action: np.ndarray,
    state_mean: np.ndarray | None = None,
    state_std: np.ndarray | None = None,
    *,
    inlet_idx: int = 2,
) -> float:
    """Convenience helper used by the shield for look-ahead."""
    with torch.no_grad():
        s = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0)
        a = torch.from_numpy(np.asarray(action, dtype=np.float32)).unsqueeze(0)
        if state_mean is not None and state_std is not None:
            mean = torch.as_tensor(state_mean, dtype=s.dtype)
            std = torch.as_tensor(state_std, dtype=s.dtype).clamp_min(1e-3)
            s_n = (s - mean) / std
            out = model(s_n, a)
            next_scaled = out["next_state"]
            next_unscaled = next_scaled * std + mean
        else:
            next_unscaled = model(s, a)["next_state"]
    return float(next_unscaled[0, inlet_idx].item())


def rollout_latent(
    model: WorldModel,
    state: np.ndarray,
    actions: np.ndarray,
    state_mean: np.ndarray | None = None,
    state_std: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Roll the world model forward with a given action sequence. No env call."""
    with torch.no_grad():
        s = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0)
        mean = torch.as_tensor(state_mean, dtype=s.dtype) if state_mean is not None else None
        std = torch.as_tensor(state_std, dtype=s.dtype).clamp_min(1e-3) if state_std is not None else None
        if mean is not None:
            s = (s - mean) / std
        states_traj, rewards_traj, costs_traj = [s.squeeze(0).numpy()], [], []
        for t in range(len(actions)):
            a = torch.from_numpy(np.asarray(actions[t], dtype=np.float32)).unsqueeze(0)
            out = model(s, a)
            s = out["next_state"]
            if mean is not None:
                s_real = s * std + mean
            else:
                s_real = s
            states_traj.append(s_real.squeeze(0).numpy())
            rewards_traj.append(float(out["reward"].item()))
            costs_traj.append(float(out["cost"].item()))
    return {
        "states": np.stack(states_traj),
        "rewards": np.asarray(rewards_traj, dtype=np.float32),
        "costs": np.asarray(costs_traj, dtype=np.float32),
    }
