"""Graph-based root-cause candidate ranking.

We do NOT claim to prove causality. The module computes a ranked shortlist of
*candidate driver* entities given:

- an entity graph (services, hosts, with dependency + correlation + same-service edges), and
- a per-entity anomaly intensity vector measured during an incident window.

Algorithm: **seeded personalized PageRank on the reversed graph**.

Rationale: if the graph adjacency encodes "upstream → downstream" dependencies,
then random walks on the *reversed* graph starting from anomalous downstream
nodes land most often on plausible upstream drivers. Personalization by the
observed anomaly intensity biases the stationary distribution toward nodes
consistent with the observed evidence. This is transparent, inspectable, and
needs no learned parameters.

We expose two post-processing helpers:
- `extract_evidence_paths`: the top simple paths from ranked candidates back
  to the most anomalous downstream entities, used as human-readable rationale.
- `to_ranked_frame`: a tidy DataFrame of candidate drivers with scores and
  ranks for the incident brief.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Sequence

import networkx as nx
import numpy as np
import pandas as pd

from cloudsentinel.features.schema_and_entity_graph import EntityGraph, row_stochastic

logger = logging.getLogger(__name__)


@dataclass
class RCAConfig:
    personalization_alpha: float = 0.15     # teleport probability (1 - damping)
    damping: float = 0.85
    max_iterations: int = 100
    convergence_tol: float = 1e-5
    top_k: int = 5


def _personalized_pagerank(
    adj: np.ndarray,
    personalization: np.ndarray,
    *,
    damping: float = 0.85,
    max_iterations: int = 100,
    tol: float = 1e-5,
) -> np.ndarray:
    """Power-iteration personalized PageRank. All inputs sum-normalized internally."""
    if adj.shape[0] != adj.shape[1]:
        raise ValueError(f"Adjacency must be square, got {adj.shape}")
    if personalization.shape[0] != adj.shape[0]:
        raise ValueError("Personalization vector size must match adjacency")
    P = row_stochastic(adj)
    p = personalization.astype(float)
    if p.sum() <= 0:
        p = np.ones_like(p) / p.shape[0]
    else:
        p = p / p.sum()
    rank = p.copy()
    for i in range(max_iterations):
        new_rank = damping * (rank @ P) + (1 - damping) * p
        if np.abs(new_rank - rank).sum() < tol:
            rank = new_rank
            break
        rank = new_rank
    return rank


def score_root_causes(
    graph: EntityGraph,
    anomaly_intensity: dict[str, float] | pd.Series,
    *,
    cfg: RCAConfig | None = None,
    reverse_edges: bool = True,
) -> pd.DataFrame:
    """Return a ranked DataFrame of candidate driver entities.

    Args:
        graph: the entity graph.
        anomaly_intensity: per-entity anomaly magnitude for the incident window.
            Entities with higher values get more personalization mass.
        cfg: RCA configuration.
        reverse_edges: when True (default), we walk the reversed graph so walks
            originating from downstream anomalous entities concentrate on their
            upstream drivers. Disable for undirected reasoning.
    """
    cfg = cfg or RCAConfig()
    if graph.n_nodes == 0:
        return pd.DataFrame(columns=["entity_id", "score", "rank", "seed_intensity", "service", "region"])

    adj = graph.adjacency.T.copy() if reverse_edges else graph.adjacency.copy()
    personalization = np.zeros(graph.n_nodes, dtype=float)
    # Coerce intensity dict to vector in node_order.
    if isinstance(anomaly_intensity, pd.Series):
        intensity_map = anomaly_intensity.to_dict()
    else:
        intensity_map = dict(anomaly_intensity)
    for i, node in enumerate(graph.node_order):
        personalization[i] = float(intensity_map.get(node, 0.0))

    ranks = _personalized_pagerank(
        adj,
        personalization,
        damping=cfg.damping,
        max_iterations=cfg.max_iterations,
        tol=cfg.convergence_tol,
    )
    # Build ranked frame.
    rows = []
    for i, node in enumerate(graph.node_order):
        attrs = graph.attributes.get(node, {})
        rows.append(
            {
                "entity_id": node,
                "score": float(ranks[i]),
                "seed_intensity": float(personalization[i]),
                "service": attrs.get("service"),
                "region": attrs.get("region"),
            }
        )
    out = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out[["rank", "entity_id", "score", "seed_intensity", "service", "region"]]


def extract_evidence_paths(
    graph: EntityGraph,
    ranked: pd.DataFrame,
    *,
    top_k_candidates: int = 3,
    max_hops: int = 3,
) -> list[dict]:
    """For each top-k candidate driver, find short paths to the most anomalous downstream entities.

    The path is in the *original* (non-reversed) graph, so it reads naturally as
    "candidate driver → affected entity". Returns a list of dicts, each holding
    a candidate entity_id, the path, and a short rationale string.
    """
    if ranked.empty:
        return []
    # Build a directed networkx graph from the adjacency (use positive weights).
    g = nx.DiGraph()
    for i, a in enumerate(graph.node_order):
        g.add_node(a, **graph.attributes.get(a, {}))
    for i in range(graph.n_nodes):
        for j in range(graph.n_nodes):
            w = float(graph.adjacency[i, j])
            if w > 0 and i != j:
                g.add_edge(graph.node_order[i], graph.node_order[j], weight=w)

    # Anomalous downstream targets: top entities by seed_intensity.
    targets = (
        ranked.sort_values("seed_intensity", ascending=False)["entity_id"].head(top_k_candidates).tolist()
    )
    evidence = []
    for _, row in ranked.head(top_k_candidates).iterrows():
        src = row["entity_id"]
        best_path = None
        best_target = None
        for tgt in targets:
            if tgt == src:
                continue
            try:
                path = nx.shortest_path(g, src, tgt)
                if len(path) - 1 <= max_hops and (best_path is None or len(path) < len(best_path)):
                    best_path = path
                    best_target = tgt
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
        rationale = None
        if best_path is not None and best_target is not None:
            rationale = (
                f"Candidate driver {src} (rank {int(row['rank'])}, score {row['score']:.3f}) "
                f"reaches affected entity {best_target} via {' → '.join(best_path)}"
            )
        else:
            rationale = (
                f"Candidate driver {src} (rank {int(row['rank'])}, score {row['score']:.3f}) "
                "is isolated in the dependency graph for this incident window"
            )
        evidence.append(
            {
                "candidate": src,
                "path": best_path or [src],
                "target": best_target,
                "rationale": rationale,
            }
        )
    return evidence


def incident_window_intensity(
    scored_df: pd.DataFrame,
    *,
    start_time,
    end_time,
    entity_col: str = "entity_id",
    time_col: str = "timestamp",
    prob_col: str = "prob",
) -> pd.Series:
    """Aggregate detector probabilities per entity over an incident window.

    Returns a Series indexed by entity_id with the mean of the most anomalous
    metric's probability during the window.
    """
    t0 = pd.Timestamp(start_time)
    t1 = pd.Timestamp(end_time)
    sub = scored_df[(scored_df[time_col] >= t0) & (scored_df[time_col] <= t1)].copy()
    if sub.empty:
        return pd.Series(dtype=float)
    # For each (entity, metric) average prob during the window, then take per-entity max.
    per_pair = sub.groupby([entity_col, "metric"])[prob_col].mean()
    per_entity = per_pair.groupby(level=0).max()
    return per_entity
