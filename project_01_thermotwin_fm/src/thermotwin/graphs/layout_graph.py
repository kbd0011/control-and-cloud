"""Construct a typed layout graph over racks, cooling units, and aisles.

The graph is a small undirected networkx graph of racks (nodes). Edges are
derived from two sources:

1. Topology (hard edges):
   - Racks in the same aisle (physical neighbors by x-coordinate) are connected.
   - Racks sharing a cooling unit are connected.
2. Correlation (soft edges, optional):
   - Pairs of racks whose inlet-temperature series correlate above a threshold
     get an additional weighted edge, capturing thermal coupling not captured
     by the floorplan.

We keep a compact adjacency numpy matrix + node ordering for efficient use in
the graph convolution layer. The same utilities feed Project 2's entity graph
with a different node type registry.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class LayoutGraph:
    """Lightweight graph container usable without torch_geometric/dgl."""

    node_order: list[str]
    node_type: dict[str, str]
    adjacency: np.ndarray       # (N, N) weighted, undirected
    attributes: dict[str, dict] = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return len(self.node_order)

    def to_networkx(self) -> nx.Graph:
        g = nx.Graph()
        for i, n in enumerate(self.node_order):
            g.add_node(n, ntype=self.node_type.get(n, "rack"), **self.attributes.get(n, {}))
        N = self.n_nodes
        for i in range(N):
            for j in range(i + 1, N):
                w = float(self.adjacency[i, j])
                if w > 0:
                    g.add_edge(self.node_order[i], self.node_order[j], weight=w)
        return g

    def export(self, path: str | Path) -> Path:
        """Write the graph to a JSON sidecar for reproducibility."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "node_order": self.node_order,
            "node_type": self.node_type,
            "adjacency": self.adjacency.tolist(),
            "attributes": self.attributes,
        }
        with p.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return p

    @classmethod
    def load(cls, path: str | Path) -> "LayoutGraph":
        with Path(path).open("r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(
            node_order=d["node_order"],
            node_type=d["node_type"],
            adjacency=np.asarray(d["adjacency"], dtype=np.float32),
            attributes=d.get("attributes", {}),
        )


def build_graph(layout_df: pd.DataFrame) -> LayoutGraph:
    """Build the rack-level adjacency from a layout metadata DataFrame.

    Required columns: `rack_id`, `aisle`, `cooling_unit_id`, `x`, `y`.
    """
    required = {"rack_id", "aisle", "cooling_unit_id", "x", "y"}
    missing = required - set(layout_df.columns)
    if missing:
        raise ValueError(f"Layout frame missing columns: {sorted(missing)}")

    layout = layout_df.copy().reset_index(drop=True)
    racks = layout["rack_id"].astype(str).tolist()
    N = len(racks)
    adj = np.zeros((N, N), dtype=np.float32)
    rack_idx = {r: i for i, r in enumerate(racks)}

    for i, r_i in enumerate(racks):
        ai = layout.iloc[i]["aisle"]
        ci = layout.iloc[i]["cooling_unit_id"]
        xi, yi = float(layout.iloc[i]["x"]), float(layout.iloc[i]["y"])
        for j in range(i + 1, N):
            r_j = racks[j]
            aj = layout.iloc[j]["aisle"]
            cj = layout.iloc[j]["cooling_unit_id"]
            xj, yj = float(layout.iloc[j]["x"]), float(layout.iloc[j]["y"])
            w = 0.0
            # Same aisle + adjacent x → strong neighbor.
            if ai == aj and abs(xi - xj) <= 1.0:
                w = max(w, 1.0)
            elif ai == aj:
                w = max(w, 0.5)
            # Shared cooling unit → thermal coupling.
            if ci == cj:
                w = max(w, 0.7)
            # Cross-aisle but adjacent coordinates → weak coupling.
            if ai != aj and abs(xi - xj) <= 0.5 and abs(yi - yj) <= 1.0:
                w = max(w, 0.3)
            if w > 0:
                adj[i, j] = w
                adj[j, i] = w

    node_type = {r: "rack" for r in racks}
    attrs = {
        r: {
            "aisle": layout.iloc[i]["aisle"],
            "cooling_unit_id": layout.iloc[i]["cooling_unit_id"],
            "x": float(layout.iloc[i]["x"]),
            "y": float(layout.iloc[i]["y"]),
        }
        for i, r in enumerate(racks)
    }
    _ = rack_idx  # kept for clarity; useful when extending with edge attributes
    return LayoutGraph(node_order=racks, node_type=node_type, adjacency=adj, attributes=attrs)


def infer_soft_edges(
    metric_df: pd.DataFrame,
    *,
    value_col: str = "inlet_temp",
    entity_col: str = "rack_id",
    time_col: str = "timestamp",
    threshold: float = 0.4,
) -> pd.DataFrame:
    """Infer pairwise soft edges from correlated inlet-temperature series."""
    pivot = metric_df.pivot_table(index=time_col, columns=entity_col, values=value_col).sort_index()
    pivot = pivot.dropna(axis=1, how="all")
    if pivot.shape[1] < 2:
        return pd.DataFrame(columns=["src", "dst", "weight"])
    corr = pivot.corr().fillna(0.0).values
    cols = list(pivot.columns)
    rows = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c = float(corr[i, j])
            if abs(c) >= threshold:
                rows.append({"src": cols[i], "dst": cols[j], "weight": float(np.clip(abs(c), 0.0, 1.0))})
    return pd.DataFrame(rows)


def add_soft_edges(graph: LayoutGraph, soft_edges: pd.DataFrame) -> LayoutGraph:
    """Merge correlation-inferred edges into an existing layout graph."""
    if soft_edges.empty:
        return graph
    adj = graph.adjacency.copy()
    rack_to_idx = {r: i for i, r in enumerate(graph.node_order)}
    for _, row in soft_edges.iterrows():
        if row["src"] in rack_to_idx and row["dst"] in rack_to_idx:
            i, j = rack_to_idx[row["src"]], rack_to_idx[row["dst"]]
            w = max(adj[i, j], float(row["weight"]))
            adj[i, j] = w
            adj[j, i] = w
    return LayoutGraph(
        node_order=graph.node_order,
        node_type=graph.node_type,
        adjacency=adj,
        attributes=graph.attributes,
    )


def normalized_adjacency(adj: np.ndarray) -> np.ndarray:
    """Symmetrically normalize adjacency A_hat = D^{-1/2} (A + I) D^{-1/2}."""
    a = adj + np.eye(adj.shape[0], dtype=adj.dtype)
    deg = a.sum(axis=1)
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    return (a * d_inv_sqrt[:, None]) * d_inv_sqrt[None, :]


def export_graph(graph: LayoutGraph, out_path: str | Path) -> Path:
    return graph.export(out_path)
