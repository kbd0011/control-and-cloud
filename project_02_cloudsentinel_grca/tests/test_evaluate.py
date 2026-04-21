"""Evaluate pipeline end-to-end smoke test."""
from __future__ import annotations

import json
from pathlib import Path



def test_evaluate_end_to_end(tmp_path, sample_data_dir, write_configs):
    from cloudsentinel.evaluate import main as evaluate_main
    from cloudsentinel.train import main as train_main

    artifacts = tmp_path / "artifacts"
    base, data, model = write_configs(tmp_path, sample_data_dir, artifacts)
    args = [
        "--base-config", str(base),
        "--data-config", str(data),
        "--model-config", str(model),
    ]
    assert train_main(args) == 0
    assert evaluate_main(args) == 0
    report = json.loads((artifacts / "eval_report.json").read_text())
    assert "point_detection" in report
    assert "rca" in report
    assert "lead_time" in report
    assert "alert_budget" in report
    assert "ood" in report
    # Scored events file exists and is non-empty.
    import pandas as pd
    scored = pd.read_csv(artifacts / "scored_events.csv", parse_dates=["timestamp"])
    assert len(scored) > 0
    assert {"prob", "zscore", "residual", "changepoint", "label"}.issubset(scored.columns)


def test_rca_per_incident_csv_shape(tmp_path, sample_data_dir, write_configs):
    from cloudsentinel.evaluate import main as evaluate_main
    from cloudsentinel.train import main as train_main
    import pandas as pd

    artifacts = tmp_path / "artifacts"
    base, data, model = write_configs(tmp_path, sample_data_dir, artifacts)
    args = [
        "--base-config", str(base),
        "--data-config", str(data),
        "--model-config", str(model),
    ]
    assert train_main(args) == 0
    assert evaluate_main(args) == 0
    csv_path = artifacts / "rca_per_incident.csv"
    assert csv_path.exists()
    # Might be empty if no eval-side incidents, but file must parse.
    df = pd.read_csv(csv_path)
    if len(df) > 0:
        assert {"incident_id", "root_entity", "root_rank"}.issubset(df.columns)
