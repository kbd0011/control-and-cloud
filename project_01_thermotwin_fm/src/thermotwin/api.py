"""FastAPI service for ThermoTwin-FM.

Endpoints:
    GET /health            - liveness
    POST /forecast         - forecast from a real context window payload
    POST /what_if          - counterfactual forecast given control deltas

The service loads the trained model and scaler on startup. All inputs are
validated via pydantic and run through the same preprocessing path the
training pipeline uses. No random tensors are ever passed to the model.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from thermotwin.features.resample_and_window import ScalerStats
from thermotwin.graphs.layout_graph import LayoutGraph
from thermotwin.simulate import (
    Scenario,
    apply_scenario_to_context,
    apply_scenario_to_controls,
)
from thermotwin.evaluate import _load_model
from thermotwin.utils.io import load_json

logger = logging.getLogger("thermotwin.api")


class ForecastRequest(BaseModel):
    """Forecast request payload.

    Fields:
        rack_id: which rack the context belongs to (must exist in the trained graph)
        context: 2-D array of shape (context_steps, n_features), in feature_order
        controls: 2-D array of shape (horizon_steps, n_controls), in control_order
    """

    rack_id: str
    context: list[list[float]] = Field(..., description="Shape (context_steps, n_features)")
    controls: list[list[float]] = Field(..., description="Shape (horizon_steps, n_controls)")


class WhatIfRequest(ForecastRequest):
    control_deltas: dict[str, float] = Field(default_factory=dict)
    load_multiplier: float = 1.0


class ForecastResponse(BaseModel):
    rack_id: str
    target_order: list[str]
    horizon_steps: int
    median: list[list[float]]
    q_low: list[list[float]]
    q_high: list[list[float]]
    model_version: str
    cadence_minutes: int


def _resolve_artifacts_dir() -> Path:
    return Path(os.environ.get("THERMOTWIN_ARTIFACTS", "artifacts"))


def create_app(artifacts_dir: Path | None = None) -> FastAPI:
    """Factory that loads artifacts once and registers routes."""
    artifacts_dir = Path(artifacts_dir) if artifacts_dir is not None else _resolve_artifacts_dir()
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Artifacts dir not found: {artifacts_dir}. Train a model first.")

    scaler = ScalerStats.from_dict(load_json(artifacts_dir / "scaler.json"))
    manifest = load_json(artifacts_dir / "model_manifest.json")
    layout_graph = LayoutGraph.load(artifacts_dir / "layout_graph.json")
    device = "cpu"
    model = _load_model(manifest, layout_graph, artifacts_dir / "model.pt", device)

    rack_to_idx = {r: i for i, r in enumerate(layout_graph.node_order)}
    feature_order: list[str] = manifest["feature_order"]
    target_order: list[str] = manifest["target_order"]
    control_order: list[str] = manifest["control_order"]
    C = int(manifest["context_steps"])
    H = int(manifest["horizon_steps"])
    F = len(feature_order)
    K = len(control_order)

    app = FastAPI(title="ThermoTwin-FM", version="0.1.0")

    def _validate_shape(x: np.ndarray, shape: tuple[int, int], name: str) -> None:
        if x.shape != shape:
            raise HTTPException(
                status_code=422,
                detail=f"{name} must have shape {shape}, got {x.shape}",
            )

    def _scale_context(raw_context: np.ndarray) -> np.ndarray:
        out = raw_context.astype(np.float32).copy()
        for i, f in enumerate(feature_order):
            if f in scaler.center:
                out[..., i] = (out[..., i] - scaler.center[f]) / (scaler.scale[f] + 1e-8)
        return out

    def _scale_controls(raw_ctrl: np.ndarray) -> np.ndarray:
        out = raw_ctrl.astype(np.float32).copy()
        for i, f in enumerate(control_order):
            if f in scaler.center:
                out[..., i] = (out[..., i] - scaler.center[f]) / (scaler.scale[f] + 1e-8)
        return out

    def _predict(context: np.ndarray, controls: np.ndarray, entity_id: int) -> np.ndarray:
        x = torch.from_numpy(context[None, ...])
        c = torch.from_numpy(controls[None, ...])
        e = torch.tensor([entity_id], dtype=torch.long)
        with torch.no_grad():
            out = model(x, c, e)
        return out["quantiles"].cpu().numpy()[0]

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "n_racks": layout_graph.n_nodes,
            "context_steps": C,
            "horizon_steps": H,
            "feature_order": feature_order,
            "target_order": target_order,
            "control_order": control_order,
        }

    @app.post("/forecast", response_model=ForecastResponse)
    def forecast(req: ForecastRequest) -> ForecastResponse:
        if req.rack_id not in rack_to_idx:
            raise HTTPException(status_code=404, detail=f"Unknown rack_id: {req.rack_id}")
        ctx_raw = np.asarray(req.context, dtype=np.float32)
        ctrl_raw = np.asarray(req.controls, dtype=np.float32)
        _validate_shape(ctx_raw, (C, F), "context")
        _validate_shape(ctrl_raw, (H, K), "controls")
        ctx = _scale_context(ctx_raw)
        ctrl = _scale_controls(ctrl_raw)
        q = _predict(ctx, ctrl, rack_to_idx[req.rack_id])
        mid = q.shape[-1] // 2
        return ForecastResponse(
            rack_id=req.rack_id,
            target_order=target_order,
            horizon_steps=H,
            median=scaler.inverse_transform_array(q[..., mid], target_order).tolist(),
            q_low=scaler.inverse_transform_array(q[..., 0], target_order).tolist(),
            q_high=scaler.inverse_transform_array(q[..., -1], target_order).tolist(),
            model_version="0.1.0",
            cadence_minutes=int(manifest["cadence_minutes"]),
        )

    @app.post("/what_if", response_model=ForecastResponse)
    def what_if(req: WhatIfRequest) -> ForecastResponse:
        if req.rack_id not in rack_to_idx:
            raise HTTPException(status_code=404, detail=f"Unknown rack_id: {req.rack_id}")
        ctx_raw = np.asarray(req.context, dtype=np.float32)
        ctrl_raw = np.asarray(req.controls, dtype=np.float32)
        _validate_shape(ctx_raw, (C, F), "context")
        _validate_shape(ctrl_raw, (H, K), "controls")
        ctx = _scale_context(ctx_raw)
        ctrl = _scale_controls(ctrl_raw)
        sc = Scenario(name="ad_hoc", control_deltas=req.control_deltas, load_multiplier=req.load_multiplier)
        ctx = apply_scenario_to_context(ctx, feature_order, sc, scaler)
        ctrl = apply_scenario_to_controls(ctrl, control_order, sc, scaler)
        q = _predict(ctx, ctrl, rack_to_idx[req.rack_id])
        mid = q.shape[-1] // 2
        return ForecastResponse(
            rack_id=req.rack_id,
            target_order=target_order,
            horizon_steps=H,
            median=scaler.inverse_transform_array(q[..., mid], target_order).tolist(),
            q_low=scaler.inverse_transform_array(q[..., 0], target_order).tolist(),
            q_high=scaler.inverse_transform_array(q[..., -1], target_order).tolist(),
            model_version="0.1.0",
            cadence_minutes=int(manifest["cadence_minutes"]),
        )

    return app
