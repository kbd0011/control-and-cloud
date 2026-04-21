"""API tests using real detector streams (no random tensors)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient



def _trained_app(tmp_path, sample_data_dir, write_configs):
    from cloudsentinel.serve import create_app
    from cloudsentinel.train import main as train_main

    artifacts = tmp_path / "artifacts"
    base, data, model = write_configs(tmp_path, sample_data_dir, artifacts)
    rc = train_main(
        [
            "--base-config", str(base),
            "--data-config", str(data),
            "--model-config", str(model),
        ]
    )
    assert rc == 0
    return create_app(artifacts), artifacts


def test_health(tmp_path, sample_data_dir, write_configs):
    app, _ = _trained_app(tmp_path, sample_data_dir, write_configs)
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["n_nodes"] == 8


def test_score_returns_probabilities_in_bounds(tmp_path, sample_data_dir, write_configs):
    app, _ = _trained_app(tmp_path, sample_data_dir, write_configs)
    client = TestClient(app)
    # Use a real metric series from the fixture telemetry.
    df = pd.read_csv(sample_data_dir / "telemetry.csv", parse_dates=["timestamp"])
    sub = df[(df["entity_id"] == df["entity_id"].iloc[0]) & (df["metric"] == "latency_ms")]
    values = sub["value"].dropna().tolist()[:100]
    r = client.post(
        "/score",
        json={"entity_id": "ent_00", "metric": "latency_ms", "values": values},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["prob"]) == len(values)
    probs = np.asarray(body["prob"])
    assert (probs >= 0).all() and (probs <= 1).all()


def test_score_rejects_too_short_series(tmp_path, sample_data_dir, write_configs):
    app, _ = _trained_app(tmp_path, sample_data_dir, write_configs)
    client = TestClient(app)
    # pydantic min_length=8 should reject shorter series.
    r = client.post(
        "/score",
        json={"entity_id": "ent_00", "metric": "latency_ms", "values": [1.0, 2.0, 3.0]},
    )
    assert r.status_code == 422


def test_rca_rejects_unknown_entity(tmp_path, sample_data_dir, write_configs):
    app, _ = _trained_app(tmp_path, sample_data_dir, write_configs)
    client = TestClient(app)
    r = client.post(
        "/rca",
        json={
            "window_start": "2024-01-01 00:00:00",
            "window_end": "2024-01-01 01:00:00",
            "per_entity_intensity": {"not_a_real_entity": 1.0},
        },
    )
    assert r.status_code == 422


def test_rca_ranks_known_entities(tmp_path, sample_data_dir, write_configs):
    app, _ = _trained_app(tmp_path, sample_data_dir, write_configs)
    client = TestClient(app)
    # Use entity IDs that actually exist in the fixture graph.
    df = pd.read_csv(sample_data_dir / "entities.csv")
    ents = df["entity_id"].tolist()
    intensity = {ents[0]: 1.0, ents[1]: 0.5}
    r = client.post(
        "/rca",
        json={
            "window_start": "2024-01-01 00:00:00",
            "window_end": "2024-01-01 01:00:00",
            "per_entity_intensity": intensity,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert "ranked" in body
    assert len(body["ranked"]) == len(ents)


def test_brief_endpoint_returns_markdown(tmp_path, sample_data_dir, write_configs):
    app, _ = _trained_app(tmp_path, sample_data_dir, write_configs)
    client = TestClient(app)
    df = pd.read_csv(sample_data_dir / "entities.csv")
    ents = df["entity_id"].tolist()
    events = [
        {
            "timestamp": "2024-01-01 00:00:00",
            "entity_id": ents[0],
            "metric": "latency_ms",
            "value": 120.0,
            "prob": 0.9,
        },
        {
            "timestamp": "2024-01-01 00:05:00",
            "entity_id": ents[1],
            "metric": "error_rate",
            "value": 0.2,
            "prob": 0.7,
        },
    ]
    r = client.post(
        "/brief",
        json={
            "incident_id": "inc_test",
            "window_start": "2024-01-01 00:00:00",
            "window_end": "2024-01-01 01:00:00",
            "events": events,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert "markdown" in body and "# Incident Brief" in body["markdown"]
    assert "does not prove root cause" in body["markdown"].lower()
