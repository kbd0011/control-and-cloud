"""Graph-RCA tests: personalized PageRank on a synthetic chain."""
from __future__ import annotations

import numpy as np
import pandas as pd

from cloudsentinel.features.schema_and_entity_graph import EntityGraph, build_entity_graph
from cloudsentinel.models.graph_rca import (
    RCAConfig,
    extract_evidence_paths,
    incident_window_intensity,
    score_root_causes,
)


def _chain_graph() -> EntityGraph:
    """Build a 4-entity chain: a (svc_a) → b (svc_b) → c (svc_c) → d (svc_d)."""
    ents = pd.DataFrame(
        [
            {"entity_id": "a", "entity_type": "host", "service": "svc_a", "region": "r", "host": "h_a"},
            {"entity_id": "b", "entity_type": "host", "service": "svc_b", "region": "r", "host": "h_b"},
            {"entity_id": "c", "entity_type": "host", "service": "svc_c", "region": "r", "host": "h_c"},
            {"entity_id": "d", "entity_type": "host", "service": "svc_d", "region": "r", "host": "h_d"},
        ]
    )
    deps = pd.DataFrame(
        [
            {"upstream_service": "svc_a", "downstream_service": "svc_b"},
            {"upstream_service": "svc_b", "downstream_service": "svc_c"},
            {"upstream_service": "svc_c", "downstream_service": "svc_d"},
        ]
    )
    # Disable same-region edges so the chain structure dominates.
    return build_entity_graph(ents, deps, same_region_weight=0.0)


def test_root_cause_ranks_upstream_above_downstream():
    g = _chain_graph()
    # Anomaly concentrated downstream on c and d (classic "upstream root cause" scenario).
    intensity = {"a": 0.0, "b": 0.0, "c": 1.0, "d": 1.0}
    ranked = score_root_causes(g, intensity, cfg=RCAConfig())
    order = ranked["entity_id"].tolist()
    # The root 'a' should rank above 'd' (purely downstream) because walks on reversed graph
    # concentrate on upstream nodes.
    assert order.index("a") <= order.index("d")


def test_root_score_sums_to_one_after_normalization():
    g = _chain_graph()
    intensity = {"a": 0.0, "b": 0.0, "c": 1.0, "d": 1.0}
    ranked = score_root_causes(g, intensity)
    assert abs(ranked["score"].sum() - 1.0) < 1e-4


def test_extract_evidence_paths_have_rationale():
    g = _chain_graph()
    intensity = {"a": 0.0, "b": 0.0, "c": 1.0, "d": 1.0}
    ranked = score_root_causes(g, intensity)
    evidence = extract_evidence_paths(g, ranked, top_k_candidates=2, max_hops=4)
    assert len(evidence) >= 1
    for e in evidence:
        assert "candidate" in e and "path" in e and "rationale" in e


def test_incident_window_intensity_filters_by_window():
    ts = pd.date_range("2024-01-01", periods=30, freq="5min")
    df = pd.DataFrame(
        {
            "timestamp": list(ts) * 2,
            "entity_id": ["a"] * 30 + ["b"] * 30,
            "metric": ["m"] * 60,
            "value": 0,
            "prob": list(np.linspace(0, 1, 30)) + list(np.linspace(1, 0, 30)),
        }
    )
    intensity = incident_window_intensity(
        df, start_time=ts[5], end_time=ts[15]
    )
    assert "a" in intensity.index
    assert "b" in intensity.index


def test_rca_handles_missing_graph_entities():
    g = _chain_graph()
    intensity = {"a": 0.5, "unknown_entity": 0.7}
    ranked = score_root_causes(g, intensity)
    # Must not crash; unknown keys are simply ignored.
    assert set(ranked["entity_id"]) == set(g.node_order)
