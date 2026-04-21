"""Training-pipeline smoke test: CLI main runs end-to-end and writes artifacts."""
from __future__ import annotations

import json
from pathlib import Path

from safecool.train import main as train_main


def test_train_main_produces_all_artifacts(tmp_path):
    # Use the default configs from the repo but redirect artifacts.
    artifacts = tmp_path / "art"
    rc = train_main([
        "--base-config", "configs/base.yaml",
        "--env-config", "configs/env_lcopt.yaml",
        "--model-config", "configs/model_cmdp.yaml",
        "--artifacts", str(artifacts),
        "--quick",
    ])
    assert rc == 0
    # Expected artifacts.
    for name in ("offline_buffer.npz", "offline_buffer_manifest.json",
                 "world_model.pt", "world_model_stats.json",
                 "agent.pt", "training_report.json"):
        assert (artifacts / name).exists(), f"missing artifact: {name}"
    report = json.loads((artifacts / "training_report.json").read_text())
    assert "ope" in report
    assert "fqe" in report["ope"]
    assert "ope_gate_pass" in report
