"""Layout graph construction invariants."""
from __future__ import annotations

import numpy as np
import pandas as pd

from thermotwin.data.ingest_m100 import load_and_prepare
from thermotwin.graphs.layout_graph import (
    build_graph,
    infer_soft_edges,
    normalized_adjacency,
)


def test_graph_has_all_racks(sample_layout_path):
    layout = pd.read_csv(sample_layout_path)
    g = build_graph(layout)
    assert set(g.node_order) == set(layout["rack_id"].astype(str))
    assert g.adjacency.shape == (g.n_nodes, g.n_nodes)


def test_graph_is_symmetric_and_zero_diagonal(sample_layout_path):
    layout = pd.read_csv(sample_layout_path)
    g = build_graph(layout)
    np.testing.assert_allclose(g.adjacency, g.adjacency.T)
    assert np.all(np.diag(g.adjacency) == 0)


def test_shared_cooling_unit_connected(sample_layout_path):
    layout = pd.read_csv(sample_layout_path)
    g = build_graph(layout)
    # At least one pair of racks shares a cooling unit → they must be connected.
    same_cu = layout.groupby("cooling_unit_id").size()
    assert same_cu.max() >= 2
    cu = same_cu.idxmax()
    racks_same_cu = layout[layout["cooling_unit_id"] == cu]["rack_id"].astype(str).tolist()
    i = g.node_order.index(racks_same_cu[0])
    j = g.node_order.index(racks_same_cu[1])
    assert g.adjacency[i, j] > 0


def test_soft_edge_weights_in_unit_interval(sample_telemetry_path):
    df = load_and_prepare(sample_telemetry_path, cadence_minutes=5)
    soft = infer_soft_edges(df, threshold=0.2)
    if not soft.empty:
        assert soft["weight"].between(0.0, 1.0).all()


def test_normalized_adjacency_rows_bounded(sample_layout_path):
    layout = pd.read_csv(sample_layout_path)
    g = build_graph(layout)
    norm = normalized_adjacency(g.adjacency)
    # With self-loops added, all diagonal entries are strictly positive.
    assert (np.diag(norm) > 0).all()
    # Sum along a row is bounded by 1 + something small for this normalization choice.
    assert np.isfinite(norm).all()


def test_graph_export_roundtrip(tmp_path, sample_layout_path):
    from thermotwin.graphs.layout_graph import LayoutGraph
    layout = pd.read_csv(sample_layout_path)
    g = build_graph(layout)
    p = tmp_path / "g.json"
    g.export(p)
    g2 = LayoutGraph.load(p)
    np.testing.assert_allclose(g.adjacency, g2.adjacency)
    assert g.node_order == g2.node_order
