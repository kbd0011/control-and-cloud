"""World-model tests: shapes, finite outputs, training reduces loss."""
from __future__ import annotations

import numpy as np
import torch

from safecool.models.world_model import (
    WorldModel,
    WorldModelConfig,
    predict_next_inlet_temp,
    rollout_latent,
    train_world_model,
)


def test_forward_shapes():
    cfg = WorldModelConfig(obs_dim=7, action_dim=2, hidden_dim=32, num_layers=2)
    model = WorldModel(cfg)
    s = torch.randn(5, 7)
    a = torch.randn(5, 2)
    out = model(s, a)
    assert out["next_state"].shape == (5, 7)
    assert out["reward"].shape == (5,)
    assert out["cost"].shape == (5,)


def test_outputs_are_finite():
    cfg = WorldModelConfig(obs_dim=7, action_dim=2, hidden_dim=16)
    model = WorldModel(cfg)
    s = torch.randn(3, 7)
    a = torch.randn(3, 2)
    out = model(s, a)
    assert torch.isfinite(out["next_state"]).all()
    assert torch.isfinite(out["reward"]).all()
    assert torch.isfinite(out["cost"]).all()
    # Cost head is ReLU'd, so must be non-negative.
    assert (out["cost"] >= 0).all()


def test_training_reduces_loss(small_buffer):
    cfg = WorldModelConfig(obs_dim=7, action_dim=2, hidden_dim=32, num_layers=2)
    model = WorldModel(cfg)
    stats = train_world_model(model, small_buffer, epochs=5, lr=1e-3, batch_size=64, seed=0)
    losses = [h["loss"] for h in stats["history"]]
    assert losses[-1] < losses[0]
    assert "state_mean" in stats and len(stats["state_mean"]) == 7


def test_predict_next_inlet_temp_is_finite(small_buffer):
    cfg = WorldModelConfig(obs_dim=7, action_dim=2, hidden_dim=16, num_layers=2)
    model = WorldModel(cfg)
    stats = train_world_model(model, small_buffer, epochs=2, lr=1e-3, batch_size=64, seed=0)
    state = small_buffer["state"][0]
    action = small_buffer["action"][0]
    pred = predict_next_inlet_temp(
        model, state, action,
        state_mean=np.asarray(stats["state_mean"]),
        state_std=np.asarray(stats["state_std"]),
    )
    assert np.isfinite(pred)


def test_rollout_latent_lengths(small_buffer):
    cfg = WorldModelConfig(obs_dim=7, action_dim=2, hidden_dim=16)
    model = WorldModel(cfg)
    # Don't need to train - just check shapes.
    actions = small_buffer["action"][:5]
    out = rollout_latent(model, small_buffer["state"][0], actions)
    assert out["states"].shape == (6, 7)      # initial + 5 steps
    assert out["rewards"].shape == (5,)
    assert out["costs"].shape == (5,)
