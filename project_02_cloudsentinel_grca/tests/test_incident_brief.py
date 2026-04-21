"""Incident brief tests: structure, confidence, evidence traceability."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd



def test_incident_brief_structure_and_content(tmp_path, sample_data_dir, write_configs):
    from cloudsentinel.evaluate import main as evaluate_main
    from cloudsentinel.incident_brief import main as brief_main
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
    rc = brief_main(args)
    assert rc == 0
    briefs_dir = artifacts / "briefs"
    index = json.loads((briefs_dir / "index.json").read_text())
    if not index:
        # No eval-side incidents — no briefs to inspect.
        return
    # Inspect the first brief.
    first = index[0]
    md_path = Path(first["markdown"])
    assert md_path.exists()
    md = md_path.read_text()
    # Required sections.
    assert "# Incident Brief" in md
    assert "## Summary" in md
    assert "## Candidate driver entities" in md
    assert "## Graph evidence paths" in md
    assert "## Recommended follow-up checks" in md
    # No remediation commands.
    assert "sudo " not in md
    assert "kubectl delete" not in md
    # Careful language.
    assert "does not prove root cause" in md.lower()


def test_brief_json_references_markdown(tmp_path, sample_data_dir, write_configs):
    from cloudsentinel.evaluate import main as evaluate_main
    from cloudsentinel.incident_brief import main as brief_main
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
    assert brief_main(args) == 0
    briefs_dir = artifacts / "briefs"
    for jpath in briefs_dir.glob("*.json"):
        if jpath.name == "index.json":
            continue
        payload = json.loads(jpath.read_text())
        assert "top_candidate" in payload
        assert "markdown_path" in payload
        md_p = Path(payload["markdown_path"])
        assert md_p.exists()


def test_build_brief_with_empty_evidence():
    import pandas as pd
    from cloudsentinel.incident_brief import BriefConfig, build_brief

    context = {
        "incident_id": "inc_test",
        "window_start": "2024-01-01 00:00:00",
        "window_end": "2024-01-01 01:00:00",
        "scored": pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3, freq="5min"),
                "entity_id": ["ent"] * 3,
                "metric": ["m"] * 3,
                "prob": [0.2, 0.3, 0.4],
                "value": [1, 2, 3],
            }
        ),
        "ranked": pd.DataFrame(columns=["rank", "entity_id", "score", "seed_intensity", "service"]),
        "evidence": [],
        "incident": None,
        "threshold": 0.5,
    }
    brief = build_brief(context, cfg=BriefConfig())
    assert "markdown" in brief
    # Even with empty ranked, structure must exist.
    assert "Candidate driver entities" in brief["markdown"]
    assert brief["top_candidate"] is None
