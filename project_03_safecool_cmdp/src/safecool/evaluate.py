"""Evaluation: roll out candidate and baseline policies across scenario families.

For each scenario family (nominal, spike, degradation, heatwave) we run N
episodes with each candidate policy and record reward, energy proxy, per-step
safety costs, and violation counts. Metrics reported include:

- mean episode reward
- mean total energy (sum of per-step power proxy)
- max inlet temperature per episode and across episodes
- CVaR-95 of per-step max inlet temperature (worst-case tail risk)
- violation counts by component
- RCI compliance

Finally a simple **go/no-go** rule: pass only if every scenario family's
max violation count is 0 and CVaR-95 inlet temp is below the safety limit.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

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
    count_violations,
)
from safecool.rl.offline_agent import load_agent
from safecool.rl.policy_shield import PolicyShield, conservative_fallback
from safecool.utils.config import load_configs, setup_logging
from safecool.utils.io import ensure_dir, save_json
from safecool.utils.seed import set_seed

logger = logging.getLogger("safecool.evaluate")


def rollout_episode(
    env: CoolingEnv,
    act_fn: Callable[[np.ndarray], np.ndarray],
    *,
    scenario,
    shield: PolicyShield | None = None,
    safety_cfg: SafetyConfig | None = None,
) -> dict:
    safety_cfg = safety_cfg or SafetyConfig()
    state = env.reset(scenario)
    states, actions, raw_actions, next_states, rewards, costs = [], [], [], [], [], []
    n_shield_intervened = 0
    while True:
        raw = np.asarray(act_fn(state), dtype=np.float32).flatten()
        if shield is not None:
            shielded = shield(raw, state)
            action = np.asarray(shielded["action"], dtype=np.float32).flatten()
            if shielded["intervened"]:
                n_shield_intervened += 1
        else:
            action = raw
        next_state, reward, done, _ = env.step(action)
        states.append(state.copy())
        actions.append(action.copy())
        raw_actions.append(raw.copy())
        next_states.append(next_state.copy())
        rewards.append(reward)
        costs.append(compute_constraint_cost(state, action, next_state, safety_cfg)["total"])
        state = next_state
        if done:
            break
    states_arr = np.stack(states)
    actions_arr = np.stack(actions)
    next_arr = np.stack(next_states)
    rewards_arr = np.asarray(rewards, dtype=np.float32)
    costs_arr = np.asarray(costs, dtype=np.float32)
    v_counts = count_violations(states_arr, actions_arr, next_arr, safety_cfg)
    inlet_idx = STATE_FIELDS.index("inlet_temp_c")
    power_idx = STATE_FIELDS.index("power_kw")
    rci_values = np.array([compute_rci_proxy(s, safety_cfg) for s in next_arr])
    return {
        "n_steps": int(states_arr.shape[0]),
        "mean_reward": float(np.mean(rewards_arr)),
        "sum_reward": float(np.sum(rewards_arr)),
        "mean_energy_kw": float(np.mean(next_arr[:, power_idx])),
        "sum_energy_kw_steps": float(np.sum(next_arr[:, power_idx])),
        "max_inlet_c": float(np.max(next_arr[:, inlet_idx])),
        "mean_inlet_c": float(np.mean(next_arr[:, inlet_idx])),
        "cvar95_inlet_c": float(np.mean(np.sort(next_arr[:, inlet_idx])[-max(1, int(0.05 * len(next_arr))):])),
        "mean_rci": float(np.mean(rci_values)),
        "total_cost": float(np.sum(costs_arr)),
        "violations": v_counts,
        "n_shield_intervened": int(n_shield_intervened),
        "raw_actions": raw_actions,
        "shielded_actions": actions_arr,
        "inlet_temps": next_arr[:, inlet_idx],
    }


def _agent_policy(agent):
    def _act(state: np.ndarray) -> np.ndarray:
        return agent.predict(state)
    return _act


def _behavior_policy(name: str):
    policy = POLICY_REGISTRY[name]
    rng = np.random.default_rng(0)
    def _act(state: np.ndarray) -> np.ndarray:
        a, _ = policy.sample(state, rng)
        return a
    return _act


def evaluate_policy(
    env_cfg,
    safety_cfg: SafetyConfig,
    scenarios: list[str],
    policies: dict[str, Callable[[np.ndarray], np.ndarray]],
    *,
    n_episodes_per_scenario: int = 3,
    base_seed: int = 0,
    shield: PolicyShield | None = None,
) -> pd.DataFrame:
    """Evaluate all policies across all (scenario × seed) pairs."""
    rows = []
    env = CoolingEnv(env_cfg)
    for pname, act_fn in policies.items():
        for family in scenarios:
            for k in range(n_episodes_per_scenario):
                scen_seed = base_seed + k + 1000 * hash(family) % 10000
                scen = make_scenario(family, scen_seed)
                summary = rollout_episode(
                    env, act_fn, scenario=scen,
                    shield=shield if pname == "agent_shielded" else None,
                    safety_cfg=safety_cfg,
                )
                summary = {k: v for k, v in summary.items() if k not in ("raw_actions", "shielded_actions", "inlet_temps")}
                summary["policy"] = pname
                summary["scenario_family"] = family
                summary["scenario_seed"] = scen_seed
                # Flatten violation dict.
                for vk, vv in summary.pop("violations").items():
                    summary[f"viol_{vk}"] = vv
                rows.append(summary)
    return pd.DataFrame(rows)


def compute_tail_risk(inlet_series: np.ndarray, alpha: float = 0.95) -> float:
    """CVaR-alpha on inlet temperatures — mean of the worst (1-α) tail."""
    if inlet_series.size == 0:
        return float("nan")
    sorted_vals = np.sort(inlet_series)
    n_tail = max(1, int(np.ceil((1 - alpha) * len(sorted_vals))))
    return float(np.mean(sorted_vals[-n_tail:]))


def make_go_no_go(df: pd.DataFrame, safety_cfg: SafetyConfig, *, policy: str = "agent_shielded") -> dict:
    """Deterministic pass/fail rule for one policy across all scenarios."""
    sub = df[df["policy"] == policy]
    if sub.empty:
        return {"pass": False, "reason": f"no rows for policy={policy}"}
    total_viol = int(sub["viol_any"].sum())
    max_cvar = float(sub["cvar95_inlet_c"].max())
    reason = []
    if total_viol > 0:
        reason.append(f"{total_viol} safety violations across scenarios")
    if max_cvar > safety_cfg.max_inlet_temp_c:
        reason.append(f"CVaR-95 inlet {max_cvar:.2f}C > limit {safety_cfg.max_inlet_temp_c}C")
    passed = len(reason) == 0
    return {
        "pass": bool(passed),
        "policy": policy,
        "total_violations": total_viol,
        "max_cvar95_inlet_c": max_cvar,
        "reason": "; ".join(reason) if reason else "all checks passed",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--env-config", default="configs/env_lcopt.yaml")
    parser.add_argument("--model-config", default="configs/model_cmdp.yaml")
    parser.add_argument("--artifacts", default=None)
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args(argv)

    cfg = load_configs(args.base_config, args.env_config, args.model_config)
    if args.artifacts:
        cfg["artifacts_dir"] = args.artifacts
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    set_seed(int(cfg["seed"]))
    artifacts_dir = ensure_dir(cfg["artifacts_dir"])

    env_cfg = env_config_from_dict(cfg)
    safety_cfg = SafetyConfig.from_dict(cfg)
    scen_families = list(cfg.get("environment", {}).get("scenario_families", ["nominal"]))

    agent = load_agent(str(artifacts_dir / "agent.pt"))
    # Build a shield optionally wired to the trained world model for look-ahead.
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

    policies = {
        "rule_based": _behavior_policy("rule_based"),
        "conservative": _behavior_policy("conservative"),
        "agent": _agent_policy(agent),
        "agent_shielded": _agent_policy(agent),   # same policy, wrapped by the shield at rollout
    }

    df = evaluate_policy(env_cfg, safety_cfg, scen_families, policies,
                        n_episodes_per_scenario=args.episodes,
                        base_seed=int(cfg["seed"]), shield=shield)

    # Per-policy aggregate across scenarios.
    metrics_cols = [
        "mean_reward", "sum_reward", "mean_energy_kw", "max_inlet_c",
        "mean_inlet_c", "cvar95_inlet_c", "mean_rci", "total_cost",
        "viol_any", "n_shield_intervened",
    ]
    agg = df.groupby("policy")[metrics_cols].mean().round(4)
    # Per-(policy, scenario) breakdown.
    by_scen = df.groupby(["policy", "scenario_family"])[metrics_cols].mean().round(4)

    df.to_csv(artifacts_dir / "eval_episodes.csv", index=False)
    agg.to_csv(artifacts_dir / "eval_by_policy.csv")
    by_scen.to_csv(artifacts_dir / "eval_by_policy_scenario.csv")

    go_agent = make_go_no_go(df, safety_cfg, policy="agent_shielded")
    go_unshielded = make_go_no_go(df, safety_cfg, policy="agent")

    report = {
        "scenarios": scen_families,
        "episodes_per_scenario": args.episodes,
        "aggregate_by_policy": agg.reset_index().to_dict(orient="records"),
        "by_policy_scenario": by_scen.reset_index().to_dict(orient="records"),
        "go_no_go_agent_shielded": go_agent,
        "go_no_go_agent_unshielded": go_unshielded,
    }
    save_json(report, artifacts_dir / "eval_report.json")
    logger.info(
        "Evaluation complete. shielded gate=%s  unshielded gate=%s",
        "PASS" if go_agent["pass"] else "FAIL",
        "PASS" if go_unshielded["pass"] else "FAIL",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
