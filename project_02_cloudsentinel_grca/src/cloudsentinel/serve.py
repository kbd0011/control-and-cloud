"""FastAPI service for CloudSentinel-GRCA.

Endpoints:
    GET  /health                - liveness + version + loaded artifacts summary
    POST /score                 - score a real telemetry series
    POST /rca                   - rank candidate drivers given an incident window
    POST /brief                 - produce an incident brief for a window

All inputs are validated via pydantic. No random tensors are ever produced.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from cloudsentinel.features.schema_and_entity_graph import EntityGraph
from cloudsentinel.incident_brief import (
    BriefConfig,
    build_brief,
    generate_brief_for_incident,
)
from cloudsentinel.models.forecast_detector import FusedDetector
from cloudsentinel.models.graph_rca import (
    RCAConfig,
    extract_evidence_paths,
    incident_window_intensity,
    score_root_causes,
)
from cloudsentinel.utils.io import load_json

logger = logging.getLogger("cloudsentinel.serve")


class ScoreRequest(BaseModel):
    entity_id: str
    metric: str
    values: list[float] = Field(..., min_length=8, description="Time-ordered numeric series")


class ScoreResponse(BaseModel):
    entity_id: str
    metric: str
    prob: list[float]
    zscore: list[float]
    residual: list[float]
    changepoint: list[float]
    model_version: str


class RcaRequest(BaseModel):
    window_start: str
    window_end: str
    per_entity_intensity: dict[str, float]


class BriefRequest(BaseModel):
    incident_id: str
    window_start: str
    window_end: str
    events: list[dict] = Field(
        ...,
        description="Per-row events (timestamp, entity_id, metric, value, prob) "
        "inside the incident window. At least one row is required.",
    )


def _resolve_artifacts_dir() -> Path:
    return Path(os.environ.get("CLOUDSENTINEL_ARTIFACTS", "artifacts"))


def create_app(artifacts_dir: Path | None = None) -> FastAPI:
    artifacts_dir = Path(artifacts_dir) if artifacts_dir is not None else _resolve_artifacts_dir()
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Artifacts dir not found: {artifacts_dir}")

    detector = FusedDetector.from_dict(load_json(artifacts_dir / "detector.json"))
    graph = EntityGraph.load(artifacts_dir / "graph.json")

    app = FastAPI(title="CloudSentinel-GRCA", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "n_nodes": graph.n_nodes,
            "detector_rolling_window": detector.rolling_window,
            "detector_fusion_trained": detector.logistic is not None,
            "model_version": "0.1.0",
        }

    @app.post("/score", response_model=ScoreResponse)
    def score(req: ScoreRequest) -> ScoreResponse:
        values = np.asarray(req.values, dtype=float)
        if not np.isfinite(values).any():
            raise HTTPException(status_code=422, detail="All values are non-finite")
        # Forward-fill any NaNs so the detector produces finite streams.
        values = pd.Series(values).ffill().bfill().to_numpy()
        out = detector.score_series(values)
        return ScoreResponse(
            entity_id=req.entity_id,
            metric=req.metric,
            prob=out["prob"].tolist(),
            zscore=out["zscore"].tolist(),
            residual=out["residual"].tolist(),
            changepoint=out["changepoint"].tolist(),
            model_version="0.1.0",
        )

    @app.post("/rca")
    def rca(req: RcaRequest) -> dict:
        # Validate intensity keys belong to the graph to avoid silent errors.
        unknown = set(req.per_entity_intensity) - set(graph.node_order)
        if unknown:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown entity_ids in intensity map: {sorted(unknown)[:5]}",
            )
        ranked = score_root_causes(graph, req.per_entity_intensity)
        evidence = extract_evidence_paths(graph, ranked, top_k_candidates=3)
        return {
            "window_start": req.window_start,
            "window_end": req.window_end,
            "ranked": ranked.to_dict(orient="records"),
            "evidence": evidence,
        }

    @app.post("/brief")
    def brief(req: BriefRequest) -> dict:
        if not req.events:
            raise HTTPException(status_code=422, detail="At least one event required")
        df = pd.DataFrame(req.events)
        required = {"timestamp", "entity_id", "metric", "value", "prob"}
        missing = required - set(df.columns)
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Events missing columns: {sorted(missing)}",
            )
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        intensity = incident_window_intensity(
            df, start_time=req.window_start, end_time=req.window_end
        )
        ranked = score_root_causes(graph, intensity)
        evidence = extract_evidence_paths(graph, ranked, top_k_candidates=3)
        b = build_brief(
            {
                "incident_id": req.incident_id,
                "window_start": req.window_start,
                "window_end": req.window_end,
                "scored": df,
                "ranked": ranked,
                "evidence": evidence,
                "incident": None,
                "threshold": 0.5,
            },
            cfg=BriefConfig(),
        )
        return {
            "incident_id": b["incident_id"],
            "top_candidate": b["top_candidate"],
            "confidence_label": b["confidence_label"],
            "markdown": b["markdown"],
            "evidence": b["evidence"],
        }

    return app
