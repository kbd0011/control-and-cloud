"""Evaluate detection quality, alert budget, and RCA ranking quality.

Produces:
    artifacts/scored_events.csv       Per-(entity, metric, timestamp) scores (val+test).
    artifacts/eval_report.json        Point and ranking metrics (in-domain + OOD).
    artifacts/rca_per_incident.csv    Per-incident top-k candidate ranking tables.

Metrics:
    - Point detection: AUROC, AUPRC, best-F1 threshold, confusion matrix at configured threshold.
    - Lead time: mean time-to-detect from incident start (negative = late).
    - RCA: top-k accuracy over incidents where ground-truth root_entity is known.
    - Alert budget: number of events retained per day under the daily budget cap.
    - OOD detector score distribution compared against in-domain.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from cloudsentinel.data.ingest_cesnet import read_cesnet_subset
from cloudsentinel.data.ingest_ibm_cloud import canonicalize_ibm_schema
from cloudsentinel.features.schema_and_entity_graph import EntityGraph
from cloudsentinel.models.forecast_detector import FusedDetector, apply_alert_budget
from cloudsentinel.models.graph_rca import (
    RCAConfig,
    incident_window_intensity,
    score_root_causes,
)
from cloudsentinel.utils.config import load_configs, setup_logging
from cloudsentinel.utils.io import ensure_dir, load_json, save_json
from cloudsentinel.utils.seed import set_seed

logger = logging.getLogger("cloudsentinel.evaluate")


def score_telemetry(detector: FusedDetector, df: pd.DataFrame, incidents: pd.DataFrame) -> pd.DataFrame:
    """Score every (entity, metric) series and attach binary labels from incidents."""
    from cloudsentinel.train import _label_series

    rows = []
    for (entity, metric), sub in df.groupby(["entity_id", "metric"], sort=False):
        sub = sub.sort_values("timestamp").reset_index(drop=True)
        values = sub["value"].to_numpy(dtype=float)
        if np.isnan(values).all() or len(values) < detector.rolling_window:
            continue
        s = pd.Series(values).ffill().bfill().to_numpy()
        scored = detector.score_series(s)
        y = _label_series(sub["timestamp"], str(entity), str(metric), incidents)
        out = sub[["timestamp", "entity_id", "metric", "value"]].copy()
        out["prob"] = scored["prob"]
        out["zscore"] = scored["zscore"]
        out["residual"] = scored["residual"]
        out["changepoint"] = scored["changepoint"]
        out["label"] = y
        rows.append(out)
    if not rows:
        return pd.DataFrame(
            columns=["timestamp", "entity_id", "metric", "value", "prob", "label"]
        )
    return pd.concat(rows, ignore_index=True)


def point_metrics(scored: pd.DataFrame, threshold: float) -> dict:
    y = scored["label"].to_numpy().astype(int)
    p = scored["prob"].to_numpy()
    if y.sum() == 0 or y.sum() == len(y):
        # Can't compute AUROC/AUPRC; still report positive rate and threshold-based counts.
        yhat = (p >= threshold).astype(int)
        return {
            "n": int(len(y)),
            "positive_rate": float(y.mean()),
            "auroc": None,
            "auprc": None,
            "precision_at_threshold": float(((yhat == 1) & (y == 1)).sum() / max(yhat.sum(), 1)),
            "recall_at_threshold": float(((yhat == 1) & (y == 1)).sum() / max(y.sum(), 1)),
            "threshold": threshold,
        }
    yhat = (p >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
    return {
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "precision_at_threshold": float(prec),
        "recall_at_threshold": float(rec),
        "f1_at_threshold": float(f1),
        "threshold": threshold,
    }


def best_threshold_by_f1(scored: pd.DataFrame, grid: int = 40) -> dict:
    y = scored["label"].to_numpy().astype(int)
    p = scored["prob"].to_numpy()
    if y.sum() == 0 or y.sum() == len(y):
        return {"threshold": 0.5, "f1": 0.0}
    thresholds = np.linspace(0.05, 0.95, grid)
    best = {"threshold": 0.5, "f1": 0.0}
    for t in thresholds:
        yhat = (p >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            y, yhat, average="binary", zero_division=0
        )
        if f1 > best["f1"]:
            best = {"threshold": float(t), "f1": float(f1)}
    return best


def lead_time_summary(scored: pd.DataFrame, incidents: pd.DataFrame, threshold: float) -> dict:
    """Mean lead time (minutes) between first positive detection and incident start."""
    if incidents.empty or scored.empty:
        return {"mean_lead_time_minutes": None, "n_incidents_detected": 0}
    lead_times = []
    incidents = incidents.copy()
    incidents["start_time"] = pd.to_datetime(incidents["start_time"])
    incidents["end_time"] = pd.to_datetime(incidents["end_time"])
    for _, inc in incidents.iterrows():
        root = str(inc["root_entity"])
        driving = str(inc["driving_metric"])
        sub = scored[
            (scored["entity_id"] == root)
            & (scored["metric"] == driving)
            & (scored["timestamp"] <= inc["end_time"])
            & (scored["prob"] >= threshold)
        ]
        if sub.empty:
            continue
        first = sub["timestamp"].min()
        lead = (inc["start_time"] - first).total_seconds() / 60.0
        lead_times.append(lead)
    if not lead_times:
        return {"mean_lead_time_minutes": None, "n_incidents_detected": 0}
    return {
        "mean_lead_time_minutes": float(np.mean(lead_times)),
        "n_incidents_detected": len(lead_times),
    }


def rca_top_k_accuracy(
    graph: EntityGraph,
    scored: pd.DataFrame,
    incidents: pd.DataFrame,
    *,
    k_values: tuple[int, ...] = (1, 3, 5),
    rca_cfg: RCAConfig | None = None,
) -> tuple[dict, pd.DataFrame]:
    """For each incident, compute whether root_entity is in the top-k candidates."""
    results = {f"top_{k}_accuracy": 0 for k in k_values}
    results["n_incidents"] = 0
    per_incident_rows = []
    if incidents.empty:
        return results, pd.DataFrame(columns=["incident_id", "root_entity", "top_entity", "root_rank"])
    incidents = incidents.copy()
    incidents["start_time"] = pd.to_datetime(incidents["start_time"])
    incidents["end_time"] = pd.to_datetime(incidents["end_time"])
    correct_at_k = {k: 0 for k in k_values}
    for _, inc in incidents.iterrows():
        intensity = incident_window_intensity(
            scored, start_time=inc["start_time"], end_time=inc["end_time"]
        )
        if intensity.empty:
            continue
        ranked = score_root_causes(graph, intensity, cfg=rca_cfg)
        root = str(inc["root_entity"])
        if root in set(ranked["entity_id"]):
            root_rank = int(ranked.loc[ranked["entity_id"] == root, "rank"].iloc[0])
        else:
            root_rank = len(ranked) + 1
        per_incident_rows.append(
            {
                "incident_id": inc["incident_id"],
                "pattern": inc["pattern"],
                "root_entity": root,
                "top_entity": ranked.iloc[0]["entity_id"] if not ranked.empty else None,
                "root_rank": root_rank,
                "n_candidates": len(ranked),
            }
        )
        for k in k_values:
            if root_rank <= k:
                correct_at_k[k] += 1
        results["n_incidents"] += 1

    for k in k_values:
        results[f"top_{k}_accuracy"] = (
            correct_at_k[k] / results["n_incidents"] if results["n_incidents"] > 0 else 0.0
        )
    return results, pd.DataFrame(per_incident_rows)


def ood_summary(
    detector: FusedDetector, cesnet_path: Path | None
) -> dict:
    if cesnet_path is None or not cesnet_path.exists():
        return {"available": False}
    df = read_cesnet_subset(cesnet_path)
    all_probs = []
    for (_ent, _met), sub in df.groupby(["entity_id", "metric"], sort=False):
        sub = sub.sort_values("timestamp")
        values = sub["value"].to_numpy(dtype=float)
        if len(values) < detector.rolling_window:
            continue
        all_probs.append(detector.score_series(values)["prob"])
    if not all_probs:
        return {"available": True, "n_series": 0}
    probs = np.concatenate(all_probs)
    return {
        "available": True,
        "n_series": len(all_probs),
        "n_timesteps": int(probs.shape[0]),
        "mean_prob": float(probs.mean()),
        "p95_prob": float(np.percentile(probs, 95)),
        "max_prob": float(probs.max()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--data-config", default="configs/data_ibm.yaml")
    parser.add_argument("--model-config", default="configs/model_grca.yaml")
    parser.add_argument("--artifacts", default=None)
    args = parser.parse_args(argv)

    cfg = load_configs(args.base_config, args.data_config, args.model_config)
    if args.artifacts:
        cfg["artifacts_dir"] = args.artifacts
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    set_seed(int(cfg["seed"]))
    artifacts_dir = ensure_dir(cfg["artifacts_dir"])

    detector = FusedDetector.from_dict(load_json(artifacts_dir / "detector.json"))
    graph = EntityGraph.load(artifacts_dir / "graph.json")

    telemetry = canonicalize_ibm_schema(
        pd.read_csv(cfg["dataset"]["telemetry_path"], parse_dates=["timestamp"])
    )
    incidents = pd.read_csv(cfg["dataset"]["incidents_path"])

    # Build eval split by reusing train_end recorded in training report.
    report = load_json(artifacts_dir / "training_report.json")
    train_end = pd.Timestamp(report["train_end"])
    eval_df = telemetry[telemetry["timestamp"] > train_end].copy()

    scored = score_telemetry(detector, eval_df, incidents)
    if scored.empty:
        raise RuntimeError("No eval windows produced; check eval slice")
    scored.to_csv(artifacts_dir / "scored_events.csv", index=False)

    threshold = float(cfg["alerts"]["probability_threshold"])
    point = point_metrics(scored, threshold)
    best = best_threshold_by_f1(scored)
    lead = lead_time_summary(scored, incidents, threshold)
    rca_metrics, per_incident = rca_top_k_accuracy(
        graph,
        scored,
        incidents[pd.to_datetime(incidents["start_time"]) > train_end],
        rca_cfg=RCAConfig(
            personalization_alpha=float(cfg["rca"]["personalization_alpha"]),
            damping=float(cfg["rca"]["damping"]),
            max_iterations=int(cfg["rca"]["max_iterations"]),
            convergence_tol=float(cfg["rca"]["convergence_tol"]),
            top_k=int(cfg["rca"]["top_k"]),
        ),
    )
    per_incident.to_csv(artifacts_dir / "rca_per_incident.csv", index=False)

    kept = apply_alert_budget(
        scored.rename(columns={"timestamp": "timestamp"}),
        daily_budget=int(cfg["alerts"]["daily_budget"]),
        probability_threshold=threshold,
    )
    alert_budget = {
        "daily_budget": int(cfg["alerts"]["daily_budget"]),
        "probability_threshold": threshold,
        "kept_events": int(len(kept)),
        "total_above_threshold": int((scored["prob"] >= threshold).sum()),
    }

    cesnet_path = Path(cfg["dataset"].get("external_telemetry_path", ""))
    ood = ood_summary(detector, cesnet_path if str(cesnet_path) else None)

    eval_report = {
        "point_detection": point,
        "best_threshold": best,
        "lead_time": lead,
        "rca": rca_metrics,
        "alert_budget": alert_budget,
        "ood": ood,
    }
    save_json(eval_report, artifacts_dir / "eval_report.json")
    logger.info(
        "Eval done. AUROC=%s AUPRC=%s top1=%.3f top3=%.3f",
        point.get("auroc"),
        point.get("auprc"),
        rca_metrics.get("top_1_accuracy", 0.0),
        rca_metrics.get("top_3_accuracy", 0.0),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
