"""Entity graph construction invariants."""
from __future__ import annotations

import numpy as np
import pandas as pd

from cloudsentinel.features.schema_and_entity_graph import (
    EDGE_DEP,
    EDGE_SAME_SERVICE,
    EntityGraph,
    add_correlation_edges,
    build_entity_graph,
    infer_temporal_edges,
    row_stochastic,
)


def test_entity_graph_covers_all_nodes(sample_data_dir):
    ents = pd.read_csv(sample_data_dir / "entities.csv")
    deps = pd.read_csv(sample_data_dir / "dependencies.csv")
    g = build_entity_graph(ents, deps)
    assert set(g.node_order) == set(ents["entity_id"])
    assert g.adjacency.shape == (g.n_nodes, g.n_nodes)


def test_same_service_replicas_are_connected(sample_data_dir):
    ents = pd.read_csv(sample_data_dir / "entities.csv")
    deps = pd.read_csv(sample_data_dir / "dependencies.csv")
    g = build_entity_graph(ents, deps)
    # Find a service with >= 2 entities.
    counts = ents.groupby("service").size()
    multi = counts[counts >= 2]
    if multi.empty:
        return  # nothing to assert
    svc = multi.index[0]
    same = ents[ents["service"] == svc]["entity_id"].tolist()
    i, j = g.node_order.index(same[0]), g.node_order.index(same[1])
    assert g.adjacency[i, j] > 0


def test_directed_dependency_is_asymmetric():
    ents = pd.DataFrame(
        [
            {"entity_id": "a", "entity_type": "host", "service": "svc_a", "region": "r", "host": "h_a"},
            {"entity_id": "b", "entity_type": "host", "service": "svc_b", "region": "r2", "host": "h_b"},
        ]
    )
    deps = pd.DataFrame([{"upstream_service": "svc_a", "downstream_service": "svc_b"}])
    g = build_entity_graph(ents, deps)
    i = g.node_order.index("a")
    j = g.node_order.index("b")
    # Upstream edge a→b from dep set; reverse is only same-region (different regions → zero).
    assert g.adjacency[i, j] > 0
    assert g.adjacency[j, i] == 0
    assert g.edge_source[i, j] == EDGE_DEP


def test_row_stochastic_handles_isolated_nodes():
    adj = np.array([[0.0, 1.0], [0.0, 0.0]])
    P = row_stochastic(adj)
    # Row 0 normalized to 1 on column 1. Row 1 self-loop added since isolated.
    assert P[0, 1] == 1.0
    assert P[1, 1] == 1.0


def test_correlation_edges_merge(sample_data_dir):
    ents = pd.read_csv(sample_data_dir / "entities.csv")
    deps = pd.read_csv(sample_data_dir / "dependencies.csv")
    telemetry = pd.read_csv(sample_data_dir / "telemetry.csv", parse_dates=["timestamp"])
    g = build_entity_graph(ents, deps)
    corr = infer_temporal_edges(telemetry, metric_filter="latency_ms", threshold=0.1)
    g2 = add_correlation_edges(g, corr)
    assert g2.n_nodes == g.n_nodes
    # g2 adjacency is at least as strong everywhere.
    assert (g2.adjacency >= g.adjacency - 1e-6).all()


def test_graph_export_roundtrip(tmp_path, sample_data_dir):
    ents = pd.read_csv(sample_data_dir / "entities.csv")
    deps = pd.read_csv(sample_data_dir / "dependencies.csv")
    g = build_entity_graph(ents, deps)
    p = tmp_path / "g.json"
    g.export(p)
    g2 = EntityGraph.load(p)
    np.testing.assert_allclose(g.adjacency, g2.adjacency)
    assert g.node_order == g2.node_order
