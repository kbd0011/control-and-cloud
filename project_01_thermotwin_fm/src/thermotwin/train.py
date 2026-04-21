"""Training entry point for ThermoTwin-FM.

Loads config, prepares data, fits the scaler, builds windows, fits baselines
(naive-last, linear AR), trains the neural forecaster with optional graph
mixing and physics penalties, and writes all artifacts to `artifacts/`.

Run:
    python -m thermotwin.train
    python -m thermotwin.train --quick   # fast smoke run on fewer epochs
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from thermotwin.data.ingest_m100 import load_and_prepare
from thermotwin.data.synthetic import SyntheticConfig, write_sample_dataset
from thermotwin.features.resample_and_window import (
    ScalerStats,
    WindowArrays,
    build_windows,
    fit_scaler,
    time_block_split,
)
from thermotwin.graphs.layout_graph import LayoutGraph, build_graph, normalized_adjacency
from thermotwin.models.baselines import LinearARForecaster, NaiveLastForecaster
from thermotwin.models.foundation_backbone import BackboneConfig, FoundationBackbone
from thermotwin.models.graph_state_space import GraphStateSpace
from thermotwin.models.physics_losses import (
    combine_losses,
    control_lag_penalty,
    monotonicity_penalty,
    pinball_loss,
    smoothness_penalty,
)
from thermotwin.utils.config import load_configs, setup_logging
from thermotwin.utils.io import ensure_dir, save_json
from thermotwin.utils.seed import set_seed

logger = logging.getLogger("thermotwin.train")


def _ensure_data(data_cfg: dict, artifacts_dir: Path) -> Path:
    """Ensure telemetry + layout exist; generate synthetic if missing."""
    telemetry_path = Path(data_cfg["dataset"]["telemetry_path"])
    layout_path = Path(data_cfg["dataset"]["layout_path"])
    if not telemetry_path.exists() or not layout_path.exists():
        logger.info("No dataset found; generating synthetic telemetry at %s", telemetry_path.parent)
        syn_cfg = SyntheticConfig(
            n_racks=int(data_cfg["dataset"]["n_racks"]),
            n_cooling_units=int(data_cfg["dataset"]["n_cooling_units"]),
            duration_days=int(data_cfg["dataset"]["duration_days"]),
            cadence_minutes=int(data_cfg["dataset"]["cadence_minutes"]),
        )
        write_sample_dataset(telemetry_path.parent, syn_cfg)
    return telemetry_path


def _prepare_windows(cfg: dict) -> tuple[dict[str, WindowArrays], ScalerStats, pd.DataFrame]:
    data_cfg = cfg
    telemetry_path = _ensure_data(data_cfg, Path(cfg["artifacts_dir"]))
    layout_path = Path(data_cfg["dataset"]["layout_path"])

    telemetry = load_and_prepare(telemetry_path, cadence_minutes=int(data_cfg["dataset"]["cadence_minutes"]))
    layout = pd.read_csv(layout_path)

    splits = time_block_split(
        telemetry,
        val_fraction=float(cfg["train"]["val_fraction"]),
        test_fraction=float(cfg["train"]["test_fraction"]),
    )

    features = list(data_cfg["features"]["covariates"]) + list(data_cfg["features"]["targets"])
    # De-dup while preserving order.
    features = list(dict.fromkeys(features))
    targets = list(data_cfg["features"]["targets"])
    controls = list(data_cfg["features"]["controls"])

    scaler = fit_scaler(splits["train"], features)
    scaled = {k: scaler.transform(v) for k, v in splits.items()}

    windows = {
        name: build_windows(
            df,
            features=features,
            targets=targets,
            controls=controls,
            context_steps=int(data_cfg["windowing"]["context_steps"]),
            horizon_steps=int(data_cfg["windowing"]["horizon_steps"]),
            stride=int(data_cfg["windowing"]["stride"]),
        )
        for name, df in scaled.items()
        if len(df) > 0
    }
    return windows, scaler, layout


def _make_loader(w: WindowArrays, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(w.context),
        torch.from_numpy(w.target),
        torch.from_numpy(w.controls),
        torch.from_numpy(w.mask),
        torch.from_numpy(w.entity),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


class _CombinedModel(nn.Module):
    """Wraps backbone + optional graph mixing + decoder alignment."""

    def __init__(self, backbone: FoundationBackbone, gss: GraphStateSpace | None):
        super().__init__()
        self.backbone = backbone
        self.gss = gss

    def forward(self, context: torch.Tensor, controls: torch.Tensor, entity: torch.Tensor) -> dict:
        latent = self.backbone.encode(context)
        if self.gss is not None:
            latent = self.gss(latent, entity)
        summary = latent[:, -1, :]
        H = self.backbone.cfg.horizon
        T = self.backbone.cfg.n_targets
        Q = len(self.backbone.cfg.quantiles)
        B = context.shape[0]
        if self.backbone.ctrl_proj is not None:
            ctrl_feat = self.backbone.ctrl_proj(controls)
            summary_rep = summary.unsqueeze(1).expand(-1, H, -1)
            dec_in = torch.cat([summary_rep, ctrl_feat], dim=-1)
        else:
            dec_in = summary.unsqueeze(1).expand(-1, H, -1)
        out = self.backbone.head(dec_in).view(B, H, T, Q)
        base = out[..., :1]
        deltas = torch.nn.functional.softplus(out[..., 1:]) + 0.1
        quantiles = torch.cat([base, base + torch.cumsum(deltas, dim=-1)], dim=-1)
        if self.backbone.cfg.target_feature_indices is not None:
            idx = torch.tensor(
                self.backbone.cfg.target_feature_indices, dtype=torch.long, device=context.device
            )
            last_target = context.index_select(-1, idx)[:, -1, :]
            quantiles = quantiles + last_target.view(B, 1, T, 1)
        return {"quantiles": quantiles, "latent": latent}


def _train_epoch(model: _CombinedModel, loader: DataLoader, opt: torch.optim.Optimizer, cfg: dict, device: str) -> dict:
    model.train()
    total, n = 0.0, 0
    use_phys = bool(cfg["losses"]["use_physics_penalties"])
    weights = {
        "monotonicity": float(cfg["losses"]["monotonicity_weight"]),
        "smoothness": float(cfg["losses"]["smoothness_weight"]),
        "control_lag": float(cfg["losses"]["control_lag_weight"]),
    }
    quantiles = tuple(float(q) for q in cfg["model"]["quantiles"])
    for ctx, tgt, ctrl, _mask, ent in loader:
        ctx = ctx.to(device)
        tgt = tgt.to(device)
        ctrl = ctrl.to(device)
        ent = ent.to(device)
        out = model(ctx, ctrl, ent)
        pred_q = out["quantiles"]
        base = pinball_loss(pred_q, tgt, quantiles)
        penalties = {}
        if use_phys:
            median = pred_q[..., len(quantiles) // 2]
            # load proxy: mean of non-target covariates known in context last step
            load_proxy = ctx[:, -1, :].mean(dim=-1, keepdim=True).expand(-1, median.shape[1])
            cool_proxy = -ctrl.mean(dim=-1) if ctrl.shape[-1] > 0 else torch.zeros_like(load_proxy)
            penalties["monotonicity"] = monotonicity_penalty(median, load_proxy, cool_proxy)
            penalties["smoothness"] = smoothness_penalty(median)
            penalties["control_lag"] = control_lag_penalty(median, ctrl)
        loss = combine_losses(base, penalties, weights)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]))
        opt.step()
        total += float(loss.item()) * ctx.shape[0]
        n += ctx.shape[0]
    return {"train_loss": total / max(n, 1)}


def _eval_epoch(model: _CombinedModel, loader: DataLoader, cfg: dict, device: str) -> dict:
    model.eval()
    quantiles = tuple(float(q) for q in cfg["model"]["quantiles"])
    total, n = 0.0, 0
    with torch.no_grad():
        for ctx, tgt, ctrl, _mask, ent in loader:
            ctx, tgt, ctrl, ent = ctx.to(device), tgt.to(device), ctrl.to(device), ent.to(device)
            out = model(ctx, ctrl, ent)
            loss = pinball_loss(out["quantiles"], tgt, quantiles)
            total += float(loss.item()) * ctx.shape[0]
            n += ctx.shape[0]
    return {"val_loss": total / max(n, 1)}


def predict_all(model: _CombinedModel, windows: WindowArrays, cfg: dict, device: str) -> np.ndarray:
    model.eval()
    loader = _make_loader(windows, int(cfg["train"]["batch_size"]), shuffle=False)
    preds = []
    with torch.no_grad():
        for ctx, _tgt, ctrl, _mask, ent in loader:
            ctx, ctrl, ent = ctx.to(device), ctrl.to(device), ent.to(device)
            out = model(ctx, ctrl, ent)
            preds.append(out["quantiles"].cpu().numpy())
    return np.concatenate(preds, axis=0) if preds else np.zeros((0,))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--data-config", default="configs/data_m100.yaml")
    parser.add_argument("--model-config", default="configs/model_graphfm.yaml")
    parser.add_argument("--quick", action="store_true", help="Reduce epochs and windows for a smoke test")
    parser.add_argument("--artifacts", default=None)
    args = parser.parse_args(argv)

    cfg = load_configs(args.base_config, args.data_config, args.model_config)
    if args.artifacts:
        cfg["artifacts_dir"] = args.artifacts
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    set_seed(int(cfg["seed"]))

    if args.quick:
        cfg["train"]["epochs"] = max(1, min(2, int(cfg["train"]["epochs"])))
        cfg["dataset"]["duration_days"] = min(2, int(cfg["dataset"]["duration_days"]))

    artifacts_dir = ensure_dir(cfg["artifacts_dir"])

    logger.info("Preparing data and windows")
    windows, scaler, layout = _prepare_windows(cfg)
    save_json(scaler.to_dict(), artifacts_dir / "scaler.json")

    # Graph
    graph_obj = build_graph(layout)
    graph_obj.export(artifacts_dir / "layout_graph.json")
    adj_norm = normalized_adjacency(graph_obj.adjacency)

    # Baselines (on scaled windows). We fit on train and score on val/test.
    n_targets = windows["train"].target.shape[-1]
    horizon = windows["train"].target.shape[1]
    baselines = {
        "naive_last": NaiveLastForecaster(horizon=horizon, n_targets=n_targets).fit(windows["train"]),
    }
    if bool(cfg["baselines"]["linear_ar_enabled"]):
        baselines["linear_ar"] = LinearARForecaster(
            horizon=horizon,
            n_targets=n_targets,
            lags=int(cfg["baselines"]["linear_ar_lags"]),
        ).fit(windows["train"])

    # Neural model
    device = str(cfg["train"]["device"])
    feature_order = windows["train"].feature_order
    target_order = windows["train"].target_order
    target_feature_indices = tuple(
        feature_order.index(t) for t in target_order if t in feature_order
    )
    if len(target_feature_indices) != len(target_order):
        # Fall back to no residual if not every target is present in features.
        target_feature_indices = None
    cfg_b = BackboneConfig(
        n_features=windows["train"].context.shape[-1],
        n_targets=n_targets,
        n_controls=windows["train"].controls.shape[-1],
        horizon=horizon,
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        num_tcn_blocks=int(cfg["model"]["num_tcn_blocks"]),
        tcn_kernel=int(cfg["model"]["tcn_kernel"]),
        gru_layers=int(cfg["model"]["gru_layers"]),
        dropout=float(cfg["model"]["dropout"]),
        quantiles=tuple(float(q) for q in cfg["model"]["quantiles"]),
        target_feature_indices=target_feature_indices,
    )
    backbone = FoundationBackbone(cfg_b)
    gss = None
    if bool(cfg["graph"]["enabled"]):
        gss = GraphStateSpace(
            hidden_dim=cfg_b.hidden_dim,
            n_message_passing=int(cfg["graph"]["n_message_passing"]),
            n_nodes=graph_obj.n_nodes,
        )
        gss.set_adjacency(adj_norm)
    model = _CombinedModel(backbone, gss).to(device)

    train_loader = _make_loader(windows["train"], int(cfg["train"]["batch_size"]), shuffle=True)
    val_loader = _make_loader(windows["val"], int(cfg["train"]["batch_size"]), shuffle=False)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    history = []
    best_val = math.inf
    best_state = None
    patience = int(cfg["train"]["early_stop_patience"])
    bad_epochs = 0
    for epoch in range(int(cfg["train"]["epochs"])):
        tr = _train_epoch(model, train_loader, opt, cfg, device)
        vl = _eval_epoch(model, val_loader, cfg, device)
        history.append({"epoch": epoch, **tr, **vl})
        logger.info("epoch=%d train_loss=%.4f val_loss=%.4f", epoch, tr["train_loss"], vl["val_loss"])
        if vl["val_loss"] < best_val - 1e-5:
            best_val = vl["val_loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Save artifacts.
    torch.save(model.state_dict(), artifacts_dir / "model.pt")
    save_json(
        {
            "backbone_config": asdict(cfg_b),
            "graph_enabled": bool(cfg["graph"]["enabled"]),
            "n_message_passing": int(cfg["graph"]["n_message_passing"]),
            "feature_order": windows["train"].feature_order,
            "target_order": windows["train"].target_order,
            "control_order": windows["train"].control_order,
            "context_steps": int(cfg["windowing"]["context_steps"]),
            "horizon_steps": int(cfg["windowing"]["horizon_steps"]),
            "cadence_minutes": int(cfg["dataset"]["cadence_minutes"]),
            "hotspot_threshold_c": float(cfg["eval"]["hotspot_threshold_c"]),
            "quantiles": list(cfg_b.quantiles),
        },
        artifacts_dir / "model_manifest.json",
    )
    save_json(history, artifacts_dir / "train_history.json")

    # Baseline predictions on val for conformal calibration.
    val_pred_q = predict_all(model, windows["val"], cfg, device)
    np.savez_compressed(
        artifacts_dir / "val_predictions.npz",
        pred_quantiles=val_pred_q,
        target=windows["val"].target,
        entity=windows["val"].entity,
    )
    # Cache train windows for conformal, if needed later.
    logger.info("Saved model + manifest + history to %s", artifacts_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
