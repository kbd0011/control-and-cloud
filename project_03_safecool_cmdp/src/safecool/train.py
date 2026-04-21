"""Training pipeline: buffer → world model → offline RL agent → OPE gate.

Artifacts written:
    artifacts/offline_buffer.npz           Offline transitions.
    artifacts/offline_buffer_manifest.json Episode-level metadata.
    artifacts/world_model.pt               Trained world model weights.
    artifacts/world_model_stats.json       Training history + normalization stats.
    artifacts/agent.pt                     Trained offline agent (reward + cost Q + λ).
    artifacts/training_report.json         OPE gate result + pipeline summary.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from safecool.data.build_offline_buffer import (
    build_buffer,
    load_buffer,
    save_buffer,
    write_manifest,
)
from safecool.envs.lcopt_env import env_config_from_dict
from safecool.envs.safety_constraints import SafetyConfig
from safecool.models.world_model import (
    WorldModel,
    WorldModelConfig,
    train_world_model,
)
from safecool.rl.off_policy_eval import fitted_q_evaluation, weighted_importance_sampling
from safecool.rl.offline_agent import (
    AgentConfig,
    save_agent,
    train_offline,
)
from safecool.utils.config import load_configs, setup_logging
from safecool.utils.io import ensure_dir, save_json
from safecool.utils.seed import set_seed

logger = logging.getLogger("safecool.train")


def _ensure_buffer(cfg: dict, artifacts_dir: Path) -> dict[str, np.ndarray]:
    buffer_path = artifacts_dir / "offline_buffer.npz"
    if buffer_path.exists():
        logger.info("Loading existing buffer from %s", buffer_path)
        return load_buffer(buffer_path)
    logger.info("No buffer found; building fresh offline dataset")
    env_cfg = env_config_from_dict(cfg)
    safety_cfg = SafetyConfig.from_dict(cfg)
    buf_cfg = cfg.get("buffer", {})
    buffer, manifest = build_buffer(
        env_cfg,
        safety_cfg,
        policy_names=list(buf_cfg.get("policies", ["rule_based", "conservative", "perturbed_safe"])),
        scenario_families=list(cfg.get("environment", {}).get("scenario_families", ["nominal"])),
        n_episodes_per_policy=int(buf_cfg.get("n_episodes_per_policy", 8)),
        scenarios_per_policy=int(buf_cfg.get("scenarios_sampled_per_policy", 4)),
        seed=int(cfg["seed"]),
    )
    save_buffer(buffer, buffer_path)
    write_manifest(manifest, artifacts_dir / "offline_buffer_manifest.json")
    return buffer


def _train_world_model(buffer: dict, cfg: dict, artifacts_dir: Path) -> dict:
    model_cfg = cfg.get("model", {}).get("world_model", {})
    wm_cfg = WorldModelConfig(
        obs_dim=int(model_cfg.get("obs_dim", 7)),
        action_dim=int(model_cfg.get("action_dim", 2)),
        hidden_dim=int(model_cfg.get("hidden_dim", 64)),
        num_layers=int(model_cfg.get("num_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )
    model = WorldModel(wm_cfg)
    stats = train_world_model(
        model,
        buffer,
        epochs=int(cfg["train"]["world_model_epochs"]),
        lr=float(cfg["train"]["world_model_lr"]),
        batch_size=int(cfg["train"]["agent_batch_size"]),
        device=str(cfg["train"]["device"]),
        seed=int(cfg["seed"]),
    )
    torch.save(
        {
            "cfg": wm_cfg.__dict__,
            "state_dict": model.state_dict(),
            "state_mean": stats["state_mean"],
            "state_std": stats["state_std"],
        },
        artifacts_dir / "world_model.pt",
    )
    save_json({"history": stats["history"], "state_mean": stats["state_mean"], "state_std": stats["state_std"]},
              artifacts_dir / "world_model_stats.json")
    return {"final_loss": stats["history"][-1]["loss"] if stats["history"] else None}


def _train_agent(buffer: dict, cfg: dict, artifacts_dir: Path):
    agent_raw = cfg.get("model", {}).get("agent", {})
    grid = agent_raw.get("action_grid", {})
    ac = AgentConfig(
        obs_dim=int(cfg.get("model", {}).get("world_model", {}).get("obs_dim", 7)),
        hidden_dim=int(agent_raw.get("hidden_dim", 64)),
        num_layers=int(agent_raw.get("num_layers", 2)),
        action_grid_coolant=tuple(grid.get("coolant_supply_delta_c", [-1.0, -0.5, 0.0, 0.5, 1.0])),
        action_grid_fan=tuple(grid.get("fan_speed_delta_pct", [-10.0, -5.0, 0.0, 5.0, 10.0])),
    )
    tcfg = cfg["train"]
    trained = train_offline(
        buffer,
        ac,
        epochs=int(tcfg["agent_epochs"]),
        batch_size=int(tcfg["agent_batch_size"]),
        lr=float(tcfg["agent_lr"]),
        discount=float(tcfg["discount"]),
        cql_alpha=float(tcfg["cql_alpha"]),
        lagrange_init=float(tcfg["lagrange_init"]),
        lagrange_lr=float(tcfg["lagrange_lr"]),
        cost_budget=float(tcfg["cost_budget"]),
        target_update_tau=float(tcfg["target_update_tau"]),
        device=str(tcfg["device"]),
        seed=int(cfg["seed"]),
        grad_clip=float(tcfg["grad_clip"]),
    )
    save_agent(trained, str(artifacts_dir / "agent.pt"))
    return trained


def _run_ope(buffer: dict, cfg: dict, artifacts_dir: Path) -> dict:
    from safecool.rl.offline_agent import load_agent

    agent = load_agent(str(artifacts_dir / "agent.pt"))
    ope = cfg.get("ope", {})
    wis = weighted_importance_sampling(
        buffer, agent,
        discount=float(ope.get("discount", 0.97)),
        n_bootstrap=int(ope.get("wis_n_bootstrap", 200)),
        seed=int(cfg["seed"]),
    )
    fqe = fitted_q_evaluation(
        buffer, agent,
        discount=float(ope.get("discount", 0.97)),
        epochs=int(ope.get("fqe_epochs", 30)),
        lr=float(ope.get("fqe_lr", 0.001)),
        seed=int(cfg["seed"]),
    )
    return {"wis": wis.as_dict(), "fqe": fqe.as_dict()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--env-config", default="configs/env_lcopt.yaml")
    parser.add_argument("--model-config", default="configs/model_cmdp.yaml")
    parser.add_argument("--artifacts", default=None)
    parser.add_argument("--quick", action="store_true", help="Few epochs and small buffer.")
    parser.add_argument("--skip-world-model", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_configs(args.base_config, args.env_config, args.model_config)
    if args.artifacts:
        cfg["artifacts_dir"] = args.artifacts
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    set_seed(int(cfg["seed"]))
    if args.quick:
        cfg["train"]["world_model_epochs"] = min(2, int(cfg["train"]["world_model_epochs"]))
        cfg["train"]["agent_epochs"] = min(4, int(cfg["train"]["agent_epochs"]))
        cfg["buffer"]["n_episodes_per_policy"] = min(2, int(cfg["buffer"]["n_episodes_per_policy"]))

    artifacts_dir = ensure_dir(cfg["artifacts_dir"])
    buffer = _ensure_buffer(cfg, artifacts_dir)
    logger.info("Buffer size: %d transitions", buffer["state"].shape[0])

    wm_summary = None
    if not args.skip_world_model:
        wm_summary = _train_world_model(buffer, cfg, artifacts_dir)
    _train_agent(buffer, cfg, artifacts_dir)
    ope = _run_ope(buffer, cfg, artifacts_dir)

    # Go/no-go gate: pass if FQE cost ≤ configured cost_budget × horizon_steps approx.
    budget_threshold = float(cfg["train"]["cost_budget"]) * float(cfg["environment"]["episode_steps"])
    go = ope["fqe"]["cost_estimate"] <= budget_threshold
    report = {
        "n_transitions": int(buffer["state"].shape[0]),
        "world_model": wm_summary,
        "ope": ope,
        "budget_threshold": budget_threshold,
        "ope_gate_pass": bool(go),
    }
    save_json(report, artifacts_dir / "training_report.json")
    logger.info(
        "Training complete. FQE reward=%.3f cost=%.3f gate=%s",
        ope["fqe"]["reward_estimate"], ope["fqe"]["cost_estimate"], "PASS" if go else "FAIL"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
