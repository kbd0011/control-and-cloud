"""Training entry point for CloudSentinel-GRCA.

Loads config and data, ensures synthetic telemetry exists, pretrains the
masked encoder (optional but default) via self-supervised reconstruction, and
fits the forecast-detector ensemble:

1. Compute per-(entity, metric) detector streams on the training split.
2. Derive labels from the incidents manifest (point-in-time incident window
   membership per series).
3. Fit a logistic meta-model and optional isotonic calibration.

Artifacts:
    artifacts/detector.json         Serialized FusedDetector (weights + scalers).
    artifacts/encoder.pt            Pretrained masked-encoder state dict.
    artifacts/encoder_config.json   Encoder config + input width.
    artifacts/graph.json            Entity graph (adjacency + metadata).
    artifacts/training_report.json  Train/val label counts, fusion coefficients.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cloudsentinel.data.ingest_ibm_cloud import canonicalize_ibm_schema, pivot_to_matrix
from cloudsentinel.data.synthetic import SyntheticConfig, write_sample_dataset
from cloudsentinel.features.schema_and_entity_graph import (
    EntityGraph,
    add_correlation_edges,
    build_entity_graph,
    infer_temporal_edges,
)
from cloudsentinel.models.forecast_detector import FusedDetector
from cloudsentinel.models.masked_encoder import (
    EncoderConfig,
    MaskedEncoder,
    pretrain_step,
)
from cloudsentinel.utils.config import load_configs, setup_logging
from cloudsentinel.utils.io import ensure_dir, save_json
from cloudsentinel.utils.seed import set_seed

logger = logging.getLogger("cloudsentinel.train")


def _ensure_data(cfg: dict) -> dict[str, Path]:
    ds = cfg["dataset"]
    paths = {
        "telemetry": Path(ds["telemetry_path"]),
        "entities": Path(ds["entity_path"]),
        "dependencies": Path(ds["dependencies_path"]),
        "incidents": Path(ds["incidents_path"]),
    }
    if not all(p.exists() for p in paths.values()):
        logger.info("No dataset found; generating synthetic telemetry at %s", paths["telemetry"].parent)
        syn_cfg = SyntheticConfig(
            n_entities=int(ds["n_entities"]),
            duration_days=int(ds["duration_days"]),
            cadence_minutes=int(ds["cadence_minutes"]),
            seed=int(cfg["seed"]),
        )
        write_sample_dataset(paths["telemetry"].parent, syn_cfg)
    return paths


def _label_series(
    timestamps: pd.Series,
    entity: str,
    metric: str,
    incidents: pd.DataFrame,
) -> np.ndarray:
    """For a single (entity, metric) series, build a binary label per timestamp.

    A timestamp is positive if the incident (a) names this entity in its
    affected_entities list, and (b) the timestamp falls within the incident
    window. The "driving_metric" is weighted higher: label 1 only on the
    driving metric; downstream-propagated entities with a different metric are
    still labeled 1 but only on latency_ms/error_rate (the propagation metrics
    used by the generator).
    """
    n = len(timestamps)
    y = np.zeros(n, dtype=np.int8)
    if incidents.empty:
        return y
    t_arr = pd.to_datetime(timestamps).to_numpy()
    for _, inc in incidents.iterrows():
        affected = set(str(inc["affected_entities"]).split(","))
        if entity not in affected:
            continue
        start = np.datetime64(pd.Timestamp(inc["start_time"]))
        end = np.datetime64(pd.Timestamp(inc["end_time"]))
        in_window = (t_arr >= start) & (t_arr <= end)
        is_root = entity == str(inc["root_entity"])
        driving = str(inc["driving_metric"])
        # Driving metric on root: always labeled in window.
        if is_root and metric == driving:
            y = np.where(in_window, 1, y)
        # Downstream propagation labels: latency / error_rate on non-root affected.
        if (not is_root) and metric in ("latency_ms", "error_rate"):
            y = np.where(in_window, 1, y)
        # Root but not driving metric: label if the metric is latency/error_rate (load-correlated).
        if is_root and metric != driving and metric in ("latency_ms", "error_rate"):
            y = np.where(in_window, 1, y)
    return y


def _pretrain_encoder(
    train_df: pd.DataFrame, cfg: dict, artifacts_dir: Path
) -> tuple[MaskedEncoder, dict]:
    """Pretrain the masked encoder on a wide matrix derived from telemetry."""
    enc_cfg_raw = cfg["encoder"]
    # Build a small wide matrix: (T, entity__metric) and create windows.
    matrix = pivot_to_matrix(train_df).to_numpy(dtype=np.float32)
    if matrix.shape[0] < 32 or matrix.shape[1] == 0:
        logger.warning("Not enough data to pretrain encoder; skipping")
        return None, {"skipped": True, "rows": int(matrix.shape[0]), "cols": int(matrix.shape[1])}
    # Normalize columns by median / IQR for stable reconstruction targets.
    med = np.median(matrix, axis=0)
    iqr = np.subtract(*np.percentile(matrix, [75, 25], axis=0))
    iqr = np.where(iqr < 1e-6, 1.0, iqr)
    matrix = ((matrix - med) / iqr).astype(np.float32)

    # Windowing.
    ctx = int(cfg["windowing"]["context_steps"])
    stride = max(1, ctx // 2)
    windows = []
    for start in range(0, matrix.shape[0] - ctx + 1, stride):
        windows.append(matrix[start : start + ctx])
    if len(windows) < 8:
        logger.warning("Too few windows to pretrain encoder; skipping")
        return None, {"skipped": True, "n_windows": len(windows)}
    X = np.stack(windows, axis=0)

    enc_cfg = EncoderConfig(
        n_features=X.shape[-1],
        hidden_dim=int(enc_cfg_raw["hidden_dim"]),
        num_tcn_blocks=int(enc_cfg_raw["num_tcn_blocks"]),
        tcn_kernel=int(enc_cfg_raw["tcn_kernel"]),
        dropout=float(enc_cfg_raw["dropout"]),
    )
    model = MaskedEncoder(enc_cfg).to(cfg["train"]["device"])
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["encoder_learning_rate"]),
        weight_decay=float(cfg["train"]["encoder_weight_decay"]),
    )

    epochs = int(cfg["train"]["encoder_epochs"])
    bs = int(cfg["train"]["batch_size"])
    losses: list[float] = []
    X_tensor = torch.from_numpy(X)
    for ep in range(epochs):
        perm = torch.randperm(X_tensor.shape[0])
        ep_loss = 0.0
        n = 0
        for i in range(0, X_tensor.shape[0], bs):
            batch = X_tensor[perm[i : i + bs]].to(cfg["train"]["device"])
            out = pretrain_step(
                model,
                batch,
                opt,
                mask_ratio=float(enc_cfg_raw["mask_ratio"]),
                policy=str(enc_cfg_raw["mask_policy"]),
                span_length=int(enc_cfg_raw["span_length"]),
            )
            ep_loss += out["loss"] * batch.shape[0]
            n += batch.shape[0]
        ep_mean = ep_loss / max(n, 1)
        losses.append(ep_mean)
        logger.info("encoder epoch=%d loss=%.4f", ep, ep_mean)

    torch.save(model.state_dict(), artifacts_dir / "encoder.pt")
    save_json(
        {
            "config": asdict(enc_cfg),
            "feature_order": pivot_to_matrix(train_df).columns.tolist(),
            "median": med.tolist(),
            "iqr": iqr.tolist(),
        },
        artifacts_dir / "encoder_config.json",
    )
    return model, {"losses": losses, "n_windows": int(X.shape[0])}


def _fit_detector(
    train_df: pd.DataFrame, incidents: pd.DataFrame, cfg: dict
) -> tuple[FusedDetector, dict]:
    """Compute streams for all (entity, metric), assemble labels, fit fusion."""
    det_cfg = cfg["detector"]
    det = FusedDetector(
        rolling_window=int(det_cfg["rolling_window"]),
        residual_window=int(det_cfg["residual_window"]),
        changepoint_window=int(det_cfg["changepoint_window"]),
        zscore_clip=float(det_cfg["zscore_clip"]),
        default_weights={k: float(v) for k, v in det_cfg["default_fusion_weights"].items()},
    )

    all_streams = {n: [] for n in det.stream_names}
    all_labels: list[np.ndarray] = []
    pair_keys: list[tuple[str, str]] = []

    for (entity, metric), sub in train_df.groupby(["entity_id", "metric"], sort=False):
        sub = sub.sort_values("timestamp").reset_index(drop=True)
        values = sub["value"].to_numpy(dtype=float)
        if np.isnan(values).all() or len(values) < det.rolling_window * 2:
            continue
        # Replace NaNs with forward-filled values for score computation.
        s = pd.Series(values).ffill().bfill().to_numpy()
        streams = det.streams_for_series(s)
        labels = _label_series(sub["timestamp"], str(entity), str(metric), incidents)
        for k, v in streams.items():
            all_streams[k].append(v)
        all_labels.append(labels)
        pair_keys.append((str(entity), str(metric)))

    if not all_labels:
        raise RuntimeError("No series available to fit detector")
    stacked = {k: np.concatenate(v) for k, v in all_streams.items()}
    y = np.concatenate(all_labels)
    det = det.fit_fusion(stacked, y)

    info = {
        "n_series": len(pair_keys),
        "n_timesteps_total": int(y.shape[0]),
        "positive_rate": float(y.mean()),
        "default_weights": det.default_weights,
    }
    if det.logistic is not None:
        info["logistic_coef"] = det.logistic.coef_.tolist()
        info["logistic_intercept"] = float(det.logistic.intercept_[0])
    return det, info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--data-config", default="configs/data_ibm.yaml")
    parser.add_argument("--model-config", default="configs/model_grca.yaml")
    parser.add_argument("--quick", action="store_true", help="Fewer encoder epochs for a smoke run")
    parser.add_argument("--artifacts", default=None)
    parser.add_argument("--skip-encoder", action="store_true", help="Skip encoder pretraining")
    args = parser.parse_args(argv)

    cfg = load_configs(args.base_config, args.data_config, args.model_config)
    if args.artifacts:
        cfg["artifacts_dir"] = args.artifacts
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    set_seed(int(cfg["seed"]))
    if args.quick:
        cfg["train"]["encoder_epochs"] = min(1, int(cfg["train"]["encoder_epochs"]))

    artifacts_dir = ensure_dir(cfg["artifacts_dir"])
    paths = _ensure_data(cfg)

    telemetry = canonicalize_ibm_schema(pd.read_csv(paths["telemetry"], parse_dates=["timestamp"]))
    entities = pd.read_csv(paths["entities"])
    dependencies = pd.read_csv(paths["dependencies"])
    incidents = pd.read_csv(paths["incidents"])

    # Time-based split: train on first 70%, val on next 15%, test on last 15%.
    # Detector is fit on train; evaluate.py scores val/test.
    timestamps = np.sort(telemetry["timestamp"].unique())
    val_frac = float(cfg["train"]["val_fraction"])
    test_frac = float(cfg["train"]["test_fraction"])
    n = len(timestamps)
    train_end = timestamps[n - int(n * (val_frac + test_frac)) - 1]
    train_df = telemetry[telemetry["timestamp"] <= train_end].copy()
    logger.info("Training slice: %d rows up to %s", len(train_df), train_end)

    # Keep only incidents whose start falls inside the training window for fusion labels.
    inc_train = incidents.copy()
    inc_train["start_time"] = pd.to_datetime(inc_train["start_time"])
    inc_train = inc_train[inc_train["start_time"] <= train_end]

    # 1) Fit detector.
    detector, det_info = _fit_detector(train_df, inc_train, cfg)
    save_json(detector.to_dict(), artifacts_dir / "detector.json")
    save_json(det_info, artifacts_dir / "detector_info.json")

    # 2) Pretrain encoder (optional).
    if not args.skip_encoder:
        _, enc_info = _pretrain_encoder(train_df, cfg, artifacts_dir)
    else:
        enc_info = {"skipped": True, "reason": "--skip-encoder"}

    # 3) Build entity graph.
    graph = build_entity_graph(entities, dependencies)
    if bool(cfg["graph"]["use_correlation_edges"]):
        corr = infer_temporal_edges(
            train_df,
            metric_filter="latency_ms",
            threshold=float(cfg["graph"]["correlation_threshold"]),
        )
        graph = add_correlation_edges(graph, corr)
    graph.export(artifacts_dir / "graph.json")

    # 4) Save manifests.
    save_json(
        {
            "detector": det_info,
            "encoder": enc_info,
            "graph": {
                "n_nodes": graph.n_nodes,
                "n_edges": int((graph.adjacency > 0).sum()),
            },
            "train_end": str(train_end),
            "val_fraction": val_frac,
            "test_fraction": test_frac,
        },
        artifacts_dir / "training_report.json",
    )
    logger.info(
        "Training complete. positive_rate=%.4f n_series=%d",
        det_info["positive_rate"],
        det_info["n_series"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
