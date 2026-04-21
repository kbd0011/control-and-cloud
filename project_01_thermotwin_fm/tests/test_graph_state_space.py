"""GraphStateSpace behavior tests."""
from __future__ import annotations

import numpy as np
import torch

from thermotwin.graphs.layout_graph import normalized_adjacency
from thermotwin.models.graph_state_space import GraphStateSpace


def test_graph_mixing_changes_output():
    D, N, C = 8, 4, 6
    gss = GraphStateSpace(hidden_dim=D, n_message_passing=2, n_nodes=N)
    # Fully connected + self loops.
    adj = np.ones((N, N), dtype=np.float32) - np.eye(N, dtype=np.float32)
    gss.set_adjacency(normalized_adjacency(adj))

    latent = torch.randn(3, C, D)
    entity = torch.tensor([0, 1, 2], dtype=torch.long)
    mixed = gss(latent, entity)
    assert mixed.shape == latent.shape
    # With non-trivial mixing, latents should actually change.
    assert (mixed - latent).abs().mean().item() > 1e-4


def test_graph_backprops():
    D, N, C = 4, 3, 5
    gss = GraphStateSpace(hidden_dim=D, n_message_passing=2, n_nodes=N)
    adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
    gss.set_adjacency(normalized_adjacency(adj))
    latent = torch.randn(2, C, D, requires_grad=True)
    entity = torch.tensor([0, 1], dtype=torch.long)
    out = gss(latent, entity)
    out.sum().backward()
    assert latent.grad is not None
    assert latent.grad.abs().sum().item() > 0


def test_bad_adjacency_shape_rejected():
    gss = GraphStateSpace(hidden_dim=4, n_message_passing=1, n_nodes=3)
    try:
        gss.set_adjacency(np.zeros((4, 4), dtype=np.float32))
    except ValueError:
        return
    raise AssertionError("Expected ValueError for mismatched adjacency shape")
