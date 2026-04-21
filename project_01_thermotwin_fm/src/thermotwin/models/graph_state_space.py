"""Graph state-space module: cross-rack message passing over encoder latents.

We implement a simple, explicit GCN layer operating on the latent sequence
produced by the backbone. The backbone processes each rack independently; the
graph layer then shares information across racks at every timestep via a
symmetrically normalized adjacency. This captures the spatial coupling
(shared cooling, aisle proximity) the per-rack backbone cannot learn.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class GraphStateSpace(nn.Module):
    """Stack of GCN layers mixing per-rack latent sequences.

    Args:
        hidden_dim: width of per-node latent.
        n_message_passing: number of GCN layers.
        n_nodes: number of racks (nodes) in the graph.
    """

    def __init__(self, hidden_dim: int, n_message_passing: int, n_nodes: int):
        super().__init__()
        self.n_nodes = n_nodes
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_message_passing)]
        )
        self.act = nn.GELU()
        self.register_buffer("adj_norm", torch.eye(n_nodes))  # overwritten by set_adjacency

    def set_adjacency(self, adjacency_norm: np.ndarray | torch.Tensor) -> None:
        """Install a symmetrically normalized adjacency (n_nodes, n_nodes)."""
        if isinstance(adjacency_norm, np.ndarray):
            adjacency_norm = torch.from_numpy(adjacency_norm.astype(np.float32))
        if adjacency_norm.shape != (self.n_nodes, self.n_nodes):
            raise ValueError(
                f"Expected adjacency of shape ({self.n_nodes}, {self.n_nodes}), got {tuple(adjacency_norm.shape)}"
            )
        self.adj_norm = adjacency_norm

    def forward(self, latent: torch.Tensor, entity: torch.Tensor) -> torch.Tensor:
        """Mix latents across rack neighbors.

        Args:
            latent: (B, C, D)  per-sample encoder latent sequence.
            entity: (B,)       rack index for each sample.
        Returns:
            mixed_latent: (B, C, D)
        """
        B, C, D = latent.shape
        adj = self.adj_norm.to(latent.device)
        N = adj.shape[0]

        # Build a dense (N, C, D) tensor by scattering per-sample latents to rack slots.
        # If a sample maps to rack i, we place it there; missing racks get a detached
        # average of present racks so the graph multiplication is well-defined.
        node_latent = latent.new_zeros((N, C, D))
        count = latent.new_zeros((N,))
        node_latent.index_add_(0, entity, latent)
        count.index_add_(0, entity, torch.ones_like(entity, dtype=latent.dtype))
        present = count > 0
        if present.any():
            fill = node_latent[present].mean(dim=0)
            node_latent[~present] = fill
            node_latent[present] = node_latent[present] / count[present].view(-1, 1, 1)

        # Apply GCN: x <- adj @ x  per-timestep, then linear + activation.
        h = node_latent
        for layer in self.layers:
            h_mix = torch.einsum("ij,jcd->icd", adj, h)
            h = self.act(layer(h_mix)) + h  # residual

        # Gather back to the per-sample ordering.
        mixed = h[entity]
        return mixed
