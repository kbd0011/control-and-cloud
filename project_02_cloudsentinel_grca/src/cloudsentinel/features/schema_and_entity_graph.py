"""Entity graph construction and feature-schema helpers.

We build a typed graph where nodes are entities (hosts by default; services,
regions, and racks would be additional node types in a larger deployment) and
edges come from:

1. **Service dependencies** (hard edges, directed): edges between entities
   whose services have a directed dependency. We add a same-service edge to
   link replicas of the same service.
2. **Correlation edges** (soft, undirected): pairs of entities whose dominant
   metric series correlate above a threshold.
3. **Same-region edges** (medium): entities sharing a region get a weak edge
   because they may share underlying infrastructure.

Output is a typed adjacency + attribute dict compatible with the graph RCA
module. The class is reused across Project 1's LayoutGraph pattern but with
additional node metadata and directed-edge support.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EntityGraph:
    """Typed, directed/undirected entity graph with named node attributes."""

    node_order: list[str]
    node_type: dict[str, str]
    adjacency: np.ndarray           # (N, N) weighted. Can be asymmetric (directed).
    attributes: dict[str, dict] = field(default_factory=dict)
    edge_source: np.ndarray | None = None  # (N, N) categorical ints for edge provenance

    @property
    def n_nodes(self) -> int:
        return len(self.node_order)

    def neighbors(self, node: str) -> list[tuple[str, float]]:
        if node not in self.node_order:
            return []
        i = self.node_order.index(node)
        out = []
        for j, w in enumerate(self.adjacency[i]):
            if w > 0:
                out.append((self.node_order[j], float(w)))
        return out

    def export(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "node_order": self.node_order,
            "node_type": self.node_type,
            "adjacency": self.adjacency.tolist(),
            "attributes": self.attributes,
        }
        if self.edge_source is not None:
            payload["edge_source"] = self.edge_source.tolist()
        with p.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return p

    @classmethod
    def load(cls, path: str | Path) -> "EntityGraph":
        with Path(path).open("r", encoding="utf-8") as f:
            d = json.load(f)
        edge_source = None
        if "edge_source" in d:
            edge_source = np.asarray(d["edge_source"], dtype=np.int8)
        return cls(
            node_order=d["node_order"],
            node_type=d["node_type"],
            adjacency=np.asarray(d["adjacency"], dtype=np.float32),
            attributes=d.get("attributes", {}),
            edge_source=edge_source,
        )


# Edge provenance codes (compact int storage).
EDGE_DEP = 1             # directed service dependency (upstream → downstream)
EDGE_SAME_SERVICE = 2    # undirected, same service (replicas)
EDGE_SAME_REGION = 3     # undirected, same region
EDGE_CORRELATION = 4     # undirected, correlation-inferred


def build_entity_graph(
    entity_df: pd.DataFrame,
    dep_df: pd.DataFrame | None = None,
    *,
    same_service_weight: float = 0.9,
    same_region_weight: float = 0.2,
    dependency_weight: float = 1.0,
) -> EntityGraph:
    """Build an entity graph from metadata and service dependencies.

    `entity_df` must contain `entity_id`, `entity_type`, `service`, `region`, `host`.
    `dep_df` optionally contains `upstream_service`, `downstream_service`.
    """
    required = {"entity_id", "entity_type", "service", "region", "host"}
    missing = required - set(entity_df.columns)
    if missing:
        raise ValueError(f"entity_df missing columns: {sorted(missing)}")

    ents = entity_df.copy().reset_index(drop=True)
    nodes = ents["entity_id"].astype(str).tolist()
    N = len(nodes)
    adj = np.zeros((N, N), dtype=np.float32)
    src = np.zeros((N, N), dtype=np.int8)

    # Same-service edges (undirected) — represent replica coupling.
    by_service: dict[str, list[int]] = {}
    for i, row in ents.iterrows():
        by_service.setdefault(str(row["service"]), []).append(i)
    for svc, idxs in by_service.items():
        for a in idxs:
            for b in idxs:
                if a != b:
                    adj[a, b] = max(adj[a, b], same_service_weight)
                    src[a, b] = EDGE_SAME_SERVICE

    # Same-region edges (undirected, weaker).
    by_region: dict[str, list[int]] = {}
    for i, row in ents.iterrows():
        by_region.setdefault(str(row["region"]), []).append(i)
    for _region, idxs in by_region.items():
        for a in idxs:
            for b in idxs:
                if a != b and adj[a, b] == 0:
                    adj[a, b] = same_region_weight
                    src[a, b] = EDGE_SAME_REGION

    # Directed service-dependency edges.
    if dep_df is not None and not dep_df.empty:
        needed = {"upstream_service", "downstream_service"}
        if not needed.issubset(dep_df.columns):
            raise ValueError(f"dep_df missing columns: {sorted(needed - set(dep_df.columns))}")
        for _, r in dep_df.iterrows():
            up = str(r["upstream_service"])
            dn = str(r["downstream_service"])
            up_idxs = by_service.get(up, [])
            dn_idxs = by_service.get(dn, [])
            for a in up_idxs:
                for b in dn_idxs:
                    if a != b:
                        # Directed: upstream affects downstream.
                        adj[a, b] = max(adj[a, b], dependency_weight)
                        src[a, b] = EDGE_DEP

    node_type = {str(r["entity_id"]): str(r["entity_type"]) for _, r in ents.iterrows()}
    attrs = {
        str(r["entity_id"]): {
            "service": str(r["service"]),
            "region": str(r["region"]),
            "host": str(r["host"]),
        }
        for _, r in ents.iterrows()
    }
    return EntityGraph(
        node_order=nodes, node_type=node_type, adjacency=adj, attributes=attrs, edge_source=src
    )


def infer_temporal_edges(
    metric_df: pd.DataFrame,
    *,
    value_col: str = "value",
    entity_col: str = "entity_id",
    time_col: str = "timestamp",
    metric_filter: str | None = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Infer pairwise soft edges from correlated metric series.

    If `metric_filter` is provided we correlate only that metric across entities.
    Otherwise we first aggregate per (entity, timestamp) by mean across metrics.
    """
    df = metric_df.copy()
    if metric_filter is not None:
        df = df[df["metric"] == metric_filter]
    if df.empty:
        return pd.DataFrame(columns=["src", "dst", "weight"])
    pivot = df.pivot_table(index=time_col, columns=entity_col, values=value_col, aggfunc="mean").sort_index()
    pivot = pivot.dropna(axis=1, how="all").ffill().fillna(0.0)
    if pivot.shape[1] < 2:
        return pd.DataFrame(columns=["src", "dst", "weight"])
    corr = pivot.corr().fillna(0.0).values
    cols = list(pivot.columns)
    rows = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c = float(corr[i, j])
            if abs(c) >= threshold:
                rows.append({"src": cols[i], "dst": cols[j], "weight": float(min(abs(c), 1.0))})
    return pd.DataFrame(rows)


def add_correlation_edges(graph: EntityGraph, edges: pd.DataFrame) -> EntityGraph:
    """Merge correlation-derived edges into an existing entity graph."""
    if edges.empty:
        return graph
    adj = graph.adjacency.copy()
    src = graph.edge_source.copy() if graph.edge_source is not None else np.zeros_like(adj, dtype=np.int8)
    idx_by_node = {n: i for i, n in enumerate(graph.node_order)}
    for _, row in edges.iterrows():
        a, b = idx_by_node.get(row["src"]), idx_by_node.get(row["dst"])
        if a is None or b is None:
            continue
        w = float(row["weight"])
        if adj[a, b] < w:
            adj[a, b] = w
            src[a, b] = EDGE_CORRELATION
        if adj[b, a] < w:
            adj[b, a] = w
            src[b, a] = EDGE_CORRELATION
    return EntityGraph(
        node_order=graph.node_order,
        node_type=graph.node_type,
        adjacency=adj,
        attributes=graph.attributes,
        edge_source=src,
    )


def row_stochastic(adj: np.ndarray) -> np.ndarray:
    """Normalize rows of a weighted adjacency to sum to 1 (with self-loop fallback)."""
    row_sums = adj.sum(axis=1, keepdims=True)
    # Avoid division by zero; isolated nodes get a self-loop so random walks still terminate.
    out = np.where(row_sums > 0, adj / np.maximum(row_sums, 1e-9), np.zeros_like(adj))
    isolated = (row_sums.squeeze() == 0)
    if isolated.any():
        for i in np.where(isolated)[0]:
            out[i, i] = 1.0
    return out
