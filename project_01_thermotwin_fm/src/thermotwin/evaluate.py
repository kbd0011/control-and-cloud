"""Evaluate forecasts: point error, pinball/coverage, conformal bands, hotspot detection.

Usage:
    python -m thermotwin.evaluate
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from thermotwin.data.ingest_m100 import load_and_prepare
from thermotwin.features.resample_and_window import (
    ScalerStats,
    WindowArrays,
    build_windows,
    time_block_split,
)
from thermotwin.graphs.layout_graph import LayoutGraph, normalized_adjacency
from thermotwin.models.baselines import LinearARForecaster, NaiveLastForecaster
from thermotwin.models.foundation_backbone import BackboneConfig, FoundationBackbone
from thermotwin.models.graph_state_space import GraphStateSpace
from thermotwin.train import _CombinedModel, _make_loader
from thermotwin.utils.config import load_configs, setup_logging
from thermotwin.utils.io import ensure_dir, load_json, save_json
from thermotwin.utils.seed import set_seed

logger = logging.getLogger("thermotwin.evaluate")


def compute_forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE and RMSE over (N, H, T). Returns aggregate numbers."""
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return {"mae": mae, "rmse": rmse}


def compute_per_horizon(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    """Compute MAE/RMSE per horizon step."""
    out = []
    for h in range(y_true.shape[1]):
        m = compute_forecast_metrics(y_true[:, h : h + 1, :], y_pred[:, h : h + 1, :])
        out.append({"horizon_step": h, **m})
    return out


def compute_calibration(y_true: np.ndarray, q_low: np.ndarray, q_high: np.ndarray, alpha: float) -> dict:
    """Empirical coverage of a nominal (1 - alpha) interval."""
    within = (y_true >= q_low) & (y_true <= q_high)
    coverage = float(within.mean())
    return {
        "nominal_coverage": 1.0 - alpha,
        "empirical_coverage": coverage,
        "coverage_error": coverage - (1.0 - alpha),
    }


def conformal_quantile_adjustments(
    val_pred_quantiles: np.ndarray, val_target: np.ndarray, alpha: float
) -> dict:
    """Split conformal: compute an additive width adjustment for [q_low, q_high].

    We use non-conformity score s = max(q_low - y, y - q_high). The corrected
    interval at alpha is [q_low - q̂, q_high + q̂] where q̂ is the ceil((n+1)(1-alpha))/n
    empirical quantile of the calibration scores.
    """
    q_low = val_pred_quantiles[..., 0]
    q_high = val_pred_quantiles[..., -1]
    score = np.maximum(q_low - val_target, val_target - q_high).reshape(-1)
    n = score.shape[0]
    if n == 0:
        return {"q_hat": 0.0, "n_calibration": 0}
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    sorted_scores = np.sort(score)
    q_hat = float(sorted_scores[k - 1])
    return {"q_hat": q_hat, "n_calibration": n}


def compute_hotspot_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> dict[str, float]:
    """Hotspot classification: y_true_c = (y_true > threshold), score via y_prob.

    y_true and y_prob must share shape (N, H). We flatten and, for AUROC, need
    at least one positive and one negative sample.
    """
    yt = (y_true > threshold).astype(int).reshape(-1)
    yp = y_prob.reshape(-1)
    out: dict[str, float] = {
        "n_samples": float(len(yt)),
        "positive_rate": float(yt.mean()),
    }
    # Binary classification at the threshold (predicted).
    yhat = (yp > threshold).astype(int)
    tp = int(((yhat == 1) & (yt == 1)).sum())
    fp = int(((yhat == 1) & (yt == 0)).sum())
    fn = int(((yhat == 0) & (yt == 1)).sum())
    tn = int(((yhat == 0) & (yt == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    out.update({"precision": float(prec), "recall": float(rec), "f1": float(f1), "tp": tp, "fp": fp, "fn": fn, "tn": tn})
    if yt.sum() > 0 and yt.sum() < len(yt):
        out["auroc"] = float(roc_auc_score(yt, yp))
        out["auprc"] = float(average_precision_score(yt, yp))
    return out


def _load_model(
    manifest: dict, layout_graph: LayoutGraph, state_path: Path, device: str
) -> _CombinedModel:
    cfg_b = BackboneConfig(**manifest["backbone_config"])
    cfg_b.quantiles = tuple(float(q) for q in cfg_b.quantiles)
    backbone = FoundationBackbone(cfg_b)
    gss = None
    if manifest.get("graph_enabled"):
        gss = GraphStateSpace(
            hidden_dim=cfg_b.hidden_dim,
            n_message_passing=int(manifest["n_message_passing"]),
            n_nodes=layout_graph.n_nodes,
        )
        gss.set_adjacency(normalized_adjacency(layout_graph.adjacency))
    model = _CombinedModel(backbone, gss).to(device)
    state = torch.load(state_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def _predict_all(model: _CombinedModel, windows: WindowArrays, batch_size: int, device: str) -> np.ndarray:
    loader = _make_loader(windows, batch_size, shuffle=False)
    out = []
    with torch.no_grad():
        for ctx, _tgt, ctrl, _mask, ent in loader:
            ctx, ctrl, ent = ctx.to(device), ctrl.to(device), ent.to(device)
            q = model(ctx, ctrl, ent)["quantiles"].cpu().numpy()
            out.append(q)
    return np.concatenate(out, axis=0) if out else np.zeros((0,))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--data-config", default="configs/data_m100.yaml")
    parser.add_argument("--model-config", default="configs/model_graphfm.yaml")
    parser.add_argument("--artifacts", default=None)
    args = parser.parse_args(argv)

    cfg = load_configs(args.base_config, args.data_config, args.model_config)
    if args.artifacts:
        cfg["artifacts_dir"] = args.artifacts
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    set_seed(int(cfg["seed"]))

    artifacts_dir = Path(cfg["artifacts_dir"])
    scaler = ScalerStats.from_dict(load_json(artifacts_dir / "scaler.json"))
    manifest = load_json(artifacts_dir / "model_manifest.json")
    layout_graph = LayoutGraph.load(artifacts_dir / "layout_graph.json")

    # Rebuild test windows deterministically.
    telemetry = load_and_prepare(
        cfg["dataset"]["telemetry_path"],
        cadence_minutes=int(cfg["dataset"]["cadence_minutes"]),
    )
    splits = time_block_split(
        telemetry,
        val_fraction=float(cfg["train"]["val_fraction"]),
        test_fraction=float(cfg["train"]["test_fraction"]),
    )
    features = manifest["feature_order"]
    targets = manifest["target_order"]
    controls = manifest["control_order"]
    scaled_test = scaler.transform(splits["test"])
    test_windows = build_windows(
        scaled_test,
        features=features,
        targets=targets,
        controls=controls,
        context_steps=int(manifest["context_steps"]),
        horizon_steps=int(manifest["horizon_steps"]),
        stride=int(cfg["windowing"]["stride"]),
    )

    device = str(cfg["train"]["device"])
    model = _load_model(manifest, layout_graph, artifacts_dir / "model.pt", device)
    pred_q = _predict_all(model, test_windows, int(cfg["train"]["batch_size"]), device)

    # Median forecast for point metrics, then invert scaling.
    quantiles = list(manifest["quantiles"])
    mid_idx = len(quantiles) // 2
    median_scaled = pred_q[..., mid_idx]
    target_scaled = test_windows.target

    median = scaler.inverse_transform_array(median_scaled, targets)
    target = scaler.inverse_transform_array(target_scaled, targets)

    point_metrics = compute_forecast_metrics(target, median)
    per_horizon = compute_per_horizon(target, median)

    # Coverage using inverse-transformed quantiles (bounds only, symmetric conformal adj).
    q_low_scaled = pred_q[..., 0]
    q_high_scaled = pred_q[..., -1]
    q_low = scaler.inverse_transform_array(q_low_scaled, targets)
    q_high = scaler.inverse_transform_array(q_high_scaled, targets)
    calibration = compute_calibration(target, q_low, q_high, alpha=float(cfg["eval"]["conformal_alpha"]))

    # Conformal correction computed on validation predictions saved by train.py.
    val_path = artifacts_dir / "val_predictions.npz"
    if val_path.exists():
        vp = np.load(val_path)
        conformal = conformal_quantile_adjustments(
            vp["pred_quantiles"], vp["target"], alpha=float(cfg["eval"]["conformal_alpha"])
        )
        q_low_cc = q_low - conformal["q_hat"] * (scaler.scale.get(targets[0], 1.0))
        q_high_cc = q_high + conformal["q_hat"] * (scaler.scale.get(targets[0], 1.0))
        calibration_cc = compute_calibration(target, q_low_cc, q_high_cc, alpha=float(cfg["eval"]["conformal_alpha"]))
    else:
        conformal = {"q_hat": 0.0, "n_calibration": 0}
        calibration_cc = calibration

    # Hotspot metrics on the inlet temperature (first target by convention).
    hotspot_target = cfg["features"]["hotspot_target"]
    if hotspot_target not in targets:
        logger.warning("hotspot_target %s not in targets; defaulting to %s", hotspot_target, targets[0])
        hs_idx = 0
    else:
        hs_idx = targets.index(hotspot_target)
    threshold = float(cfg["eval"]["hotspot_threshold_c"])
    hotspot_truth = target[..., hs_idx]
    hotspot_pred = median[..., hs_idx]
    hotspot = compute_hotspot_metrics(hotspot_truth, hotspot_pred, threshold)

    # Baseline comparison: naive-last on scaled windows, then invert.
    naive = NaiveLastForecaster(horizon=len(per_horizon), n_targets=len(targets)).fit(test_windows).predict(test_windows)
    naive_inv = scaler.inverse_transform_array(naive, targets)
    naive_metrics = compute_forecast_metrics(target, naive_inv)

    report = {
        "point_metrics": point_metrics,
        "per_horizon": per_horizon,
        "calibration_raw": calibration,
        "calibration_conformal": calibration_cc,
        "conformal": conformal,
        "hotspot": hotspot,
        "baseline_naive_last": naive_metrics,
    }
    save_json(report, artifacts_dir / "eval_report.json")
    # Tiny CSV for quick eyeballing.
    pd.DataFrame(per_horizon).to_csv(artifacts_dir / "per_horizon_metrics.csv", index=False)
    logger.info("Evaluation complete: MAE=%.3f RMSE=%.3f (naive MAE=%.3f)", point_metrics["mae"], point_metrics["rmse"], naive_metrics["mae"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
