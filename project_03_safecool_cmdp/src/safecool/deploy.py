"""Shadow-mode deployment harness with automatic rollback.

The deploy loop advances the environment under the agent's shielded actions,
but simultaneously logs what the baseline controller would have done so an
operator can compare. Every step the loop records:

- proposed action (from the trained agent, pre-shield)
- shielded action (what actually executed)
- baseline action (what the rule-based controller would have chosen)
- realized next state, reward, and safety cost
- shield intervention flag + reason

Rollback rule: if any of the following are true over the rolling window, the
loop trips, falls back to the baseline controller for the rest of the
episode, and stamps the rollback event into the audit log:

- CVaR-95 of inlet temperature exceeds `rollback_cvar_temp_c`.
- Count of hard-constraint violations exceeds `rollback_max_violations`.

Nothing is ever sent to a real facility from this module. The phrase
"shadow mode" here strictly means a local simulator comparison run.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from safecool.data.build_offline_buffer import POLICY_REGISTRY
from safecool.envs.lcopt_env import (
    STATE_FIELDS,
    CoolingEnv,
    env_config_from_dict,
    make_scenario,
)
from safecool.envs.safety_constraints import (
    SafetyConfig,
    compute_constraint_cost,
    compute_rci_proxy,
)
from safecool.rl.offline_agent import load_agent
from safecool.rl.policy_shield import PolicyShield, conservative_fallback
from safecool.utils.config import load_configs, setup_logging
from safecool.utils.io import ensure_dir, save_json
from safecool.utils.seed import set_seed

logger = logging.getLogger("safecool.deploy")


def _baseline_act(policy_name: str = "rule_based") -> Callable[[np.ndarray], np.ndarray]:
    policy = POLICY_REGISTRY[policy_name]
    rng = np.random.default_rng(0)
    def _act(state: np.ndarray) -> np.ndarray:
        # Deterministic behavior for audit: use policy mean, no noise.
        return np.asarray(policy.mu_fn(state), dtype=np.float32)
    return _act


def should_rollback(
    recent_inlets: list[float],
    recent_violations: list[int],
    *,
    cvar_limit: float,
    violation_limit: int,
    alpha: float = 0.95,
) -> tuple[bool, str | None]:
    if len(recent_inlets) == 0:
        return False, None
    sorted_vals = np.sort(np.asarray(recent_inlets, dtype=float))
    n_tail = max(1, int(np.ceil((1 - alpha) * len(sorted_vals))))
    cvar = float(np.mean(sorted_vals[-n_tail:]))
    total_viol = int(sum(recent_violations))
    if cvar > cvar_limit:
        return True, f"CVaR-{int(alpha * 100)} inlet {cvar:.2f}C exceeds {cvar_limit:.2f}C"
    if total_viol > violation_limit:
        return True, f"rolling violation count {total_viol} exceeds {violation_limit}"
    return False, None


@dataclass
class DeployRecord:
    step: int
    scenario_family: str
    state: list
    proposed_action: list
    shielded_action: list
    baseline_action: list
    reward: float
    cost: float
    inlet_temp_c: float
    rci: float
    shield_intervened: bool
    shield_reason: str | None
    rollback_active: bool


def run_shadow_mode(
    env_cfg,
    safety_cfg: SafetyConfig,
    agent,
    shield: PolicyShield | None,
    *,
    scenario_family: str = "nominal",
    n_steps: int = 200,
    baseline_policy_name: str = "rule_based",
    rollback_cvar_temp_c: float = 27.5,
    rollback_max_violations: int = 5,
    rollback_trip_window: int = 20,
    seed: int = 0,
) -> dict:
    """Run the shadow loop; return a structured audit report."""
    env = CoolingEnv(env_cfg)
    scenario = make_scenario(scenario_family, seed)
    state = env.reset(scenario)
    baseline_fn = _baseline_act(baseline_policy_name)

    inlet_idx = STATE_FIELDS.index("inlet_temp_c")
    records: list[DeployRecord] = []
    recent_inlets: list[float] = []
    recent_violations: list[int] = []
    rollback_active = False
    rollback_reason: str | None = None
    rollback_step: int | None = None

    for step in range(n_steps):
        proposed = np.asarray(agent.predict(state), dtype=np.float32)
        baseline = baseline_fn(state)
        # Choose the action to actually execute.
        if rollback_active:
            action = baseline.copy()
            shield_intervened = False
            shield_reason = "rollback_baseline"
        elif shield is not None:
            sh = shield(proposed, state)
            action = np.asarray(sh["action"], dtype=np.float32)
            shield_intervened = bool(sh["intervened"])
            shield_reason = sh["reason"]
        else:
            action = proposed.copy()
            shield_intervened = False
            shield_reason = None

        next_state, reward, done, _ = env.step(action)
        cost_dict = compute_constraint_cost(state, action, next_state, safety_cfg)
        record = DeployRecord(
            step=step,
            scenario_family=scenario_family,
            state=state.tolist(),
            proposed_action=proposed.tolist(),
            shielded_action=action.tolist(),
            baseline_action=baseline.tolist(),
            reward=float(reward),
            cost=float(cost_dict["total"]),
            inlet_temp_c=float(next_state[inlet_idx]),
            rci=float(compute_rci_proxy(next_state, safety_cfg)),
            shield_intervened=shield_intervened,
            shield_reason=shield_reason,
            rollback_active=rollback_active,
        )
        records.append(record)

        recent_inlets.append(record.inlet_temp_c)
        recent_violations.append(1 if cost_dict["total"] > 0 else 0)
        if len(recent_inlets) > rollback_trip_window:
            recent_inlets.pop(0)
            recent_violations.pop(0)

        if not rollback_active:
            trip, reason = should_rollback(
                recent_inlets, recent_violations,
                cvar_limit=rollback_cvar_temp_c,
                violation_limit=rollback_max_violations,
            )
            if trip:
                rollback_active = True
                rollback_reason = reason
                rollback_step = step
                logger.warning("Rollback tripped at step %d: %s", step, reason)

        state = next_state
        if done:
            break

    inlet_series = np.asarray([r.inlet_temp_c for r in records], dtype=float)
    cost_series = np.asarray([r.cost for r in records], dtype=float)
    return {
        "scenario_family": scenario_family,
        "n_steps": len(records),
        "rollback_active": rollback_active,
        "rollback_step": rollback_step,
        "rollback_reason": rollback_reason,
        "max_inlet_c": float(inlet_series.max() if inlet_series.size else 0.0),
        "cvar95_inlet_c": float(np.mean(np.sort(inlet_series)[-max(1, int(0.05 * len(inlet_series))):])) if inlet_series.size else 0.0,
        "total_cost": float(cost_series.sum()),
        "n_shield_interventions": int(sum(1 for r in records if r.shield_intervened)),
        "records": [asdict(r) for r in records],
    }


def write_deploy_report(report: dict, path: str | Path, *, agent_hash: str | None = None) -> Path:
    out = dict(report)
    out["agent_checkpoint_sha256"] = agent_hash
    out["disclaimer"] = (
        "Shadow-mode simulation output. Not a production deployment artifact. "
        "Policies require human review and operational validation before live use."
    )
    p = Path(path)
    save_json(out, p)
    return p


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--env-config", default="configs/env_lcopt.yaml")
    parser.add_argument("--model-config", default="configs/model_cmdp.yaml")
    parser.add_argument("--artifacts", default=None)
    parser.add_argument("--scenario", default="nominal")
    args = parser.parse_args(argv)

    cfg = load_configs(args.base_config, args.env_config, args.model_config)
    if args.artifacts:
        cfg["artifacts_dir"] = args.artifacts
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    set_seed(int(cfg["seed"]))
    artifacts_dir = ensure_dir(cfg["artifacts_dir"])

    env_cfg = env_config_from_dict(cfg)
    safety_cfg = SafetyConfig.from_dict(cfg)
    agent_path = artifacts_dir / "agent.pt"
    agent = load_agent(str(agent_path))

    # Optionally hook up world model-backed shield.
    shield = None
    shield_cfg = cfg.get("shield", {})
    if bool(shield_cfg.get("enabled", True)):
        wm = None
        wm_mean = None
        wm_std = None
        wm_path = artifacts_dir / "world_model.pt"
        if wm_path.exists() and bool(shield_cfg.get("use_world_model_lookahead", True)):
            import torch
            payload = torch.load(wm_path, map_location="cpu", weights_only=False)
            from safecool.models.world_model import WorldModel, WorldModelConfig
            wm_cfg = WorldModelConfig(**payload["cfg"])
            wm = WorldModel(wm_cfg)
            wm.load_state_dict(payload["state_dict"])
            wm.eval()
            wm_mean = np.asarray(payload["state_mean"], dtype=np.float32)
            wm_std = np.asarray(payload["state_std"], dtype=np.float32)
        shield = PolicyShield(
            safety_cfg=safety_cfg,
            coolant_delta_range=env_cfg.coolant_supply_delta_range,
            fan_delta_range=env_cfg.fan_speed_delta_range,
            fallback_policy=conservative_fallback,
            world_model=wm,
            wm_state_mean=wm_mean,
            wm_state_std=wm_std,
            use_lookahead=wm is not None,
            lookahead_margin_c=float(shield_cfg.get("rate_limit_safety_margin_c", 0.2)),
        )

    deploy_cfg = cfg.get("deploy", {})
    report = run_shadow_mode(
        env_cfg,
        safety_cfg,
        agent,
        shield,
        scenario_family=args.scenario,
        n_steps=int(deploy_cfg.get("shadow_steps", 200)),
        rollback_cvar_temp_c=float(deploy_cfg.get("rollback_cvar_temp_c", 27.5)),
        rollback_max_violations=int(deploy_cfg.get("rollback_max_violations", 5)),
        rollback_trip_window=int(deploy_cfg.get("rollback_trip_window", 20)),
        seed=int(cfg["seed"]),
    )
    agent_hash = _hash_file(agent_path) if agent_path.exists() else None
    out_path = write_deploy_report(report, artifacts_dir / "deploy_report.json", agent_hash=agent_hash)
    logger.info(
        "Shadow run complete. rollback=%s  max_inlet=%.2fC  interventions=%d  → %s",
        report["rollback_active"], report["max_inlet_c"], report["n_shield_interventions"], out_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
