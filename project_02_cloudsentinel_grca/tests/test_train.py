"""End-to-end train smoke test."""
from __future__ import annotations

import json


def test_train_smoke_run(tmp_path, sample_data_dir, write_configs):
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
    assert (artifacts / "detector.json").exists()
    assert (artifacts / "graph.json").exists()
    assert (artifacts / "training_report.json").exists()
    report = json.loads((artifacts / "training_report.json").read_text())
    assert "detector" in report
    assert report["graph"]["n_nodes"] == 8
