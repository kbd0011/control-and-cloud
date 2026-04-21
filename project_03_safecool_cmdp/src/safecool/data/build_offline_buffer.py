"""Offline RL buffer construction.

We roll out three named behavior policies across scenario families and
collect transitions with all the signals needed downstream: state, action,
reward, next_state, done, per-component safety cost, total safety cost,
behavior action log-probability, scenario label, and policy label.

Policies
--------
- ``rule_based``: a simple PID-ish rule with small exploration noise that
  responds to inlet temperature error vs a setpoint.
- ``conservative``: keeps coolant setpoint cold and fans elevated; small
  energy cost but strong safety margin. Often slightly over-cools.
- ``perturbed_safe``: the rule-based policy with larger Gaussian noise on
  both action dimensions; behavior is diverse but stays within safety limits
  most of the time. This broadens the state-action support for offline RL.

Behavior action log-probabilities are computed under the Gaussian model the
behavior uses to sample, which makes WIS-style OPE well-defined.
"""
from __future__ import annotations

import argparse
import itertools
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from safecool.envs.lcopt_env import (
    ACTION_FIELDS,
    STATE_FIELDS,
    CoolingEnv,
    EnvConfig,
    ScenarioSpec,
    env_config_from_dict,
    make_scenario,
)
from safecool.envs.safety_constraints import SafetyConfig, compute_constraint_cost
from safecool.utils.config import load_configs
from safecool.utils.io import ensure_dir, save_json
from safecool.utils.seed import set_seed

logger = logging.getLogger(__name__)


@dataclass
class BehaviorPolicy:
    name: str
    mu_fn: Callable[[np.ndarray], np.ndarray]
    sigma: np.ndarray

    def sample(self, state: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float]:
        mu = np.asarray(self.mu_fn(state), dtype=np.float32)
        eps = rng.standard_normal(size=mu.shape).astype(np.float32)
        action = mu + self.sigma * eps
        # Gaussian log-probability of the sampled action under this policy.
        logp = float(
            -0.5 * np.sum(((action - mu) / (self.sigma + 1e-6)) ** 2)
            - np.sum(np.log(self.sigma + 1e-6))
            - 0.5 * action.size * np.log(2 * np.pi)
        )
        return action, logp

    def log_prob(self, state: np.ndarray, action: np.ndarray) -> float:
        mu = np.asarray(self.mu_fn(state), dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        logp = float(
            -0.5 * np.sum(((action - mu) / (self.sigma + 1e-6)) ** 2)
            - np.sum(np.log(self.sigma + 1e-6))
            - 0.5 * action.size * np.log(2 * np.pi)
        )
        return logp


def _rule_based_mu(
    target_inlet_c: float = 24.0, gain_coolant: float = 0.3, gain_fan: float = 2.0
) -> Callable[[np.ndarray], np.ndarray]:
    def mu(state: np.ndarray) -> np.ndarray:
        inlet = float(state[STATE_FIELDS.index("inlet_temp_c")])
        error = inlet - target_inlet_c  # positive = too warm
        coolant_delta = -gain_coolant * error  # warm → lower setpoint
        fan_delta = gain_fan * error
        return np.array([coolant_delta, fan_delta], dtype=np.float32)

    return mu


def _conservative_mu() -> Callable[[np.ndarray], np.ndarray]:
    def mu(state: np.ndarray) -> np.ndarray:
        # Always push coolant supply a touch cooler and fans a bit higher.
        return np.array([-0.3, 1.0], dtype=np.float32)

    return mu


def _perturbed_safe_mu() -> Callable[[np.ndarray], np.ndarray]:
    # Same mean as rule-based, noisier at sampling time (σ set when building the policy).
    return _rule_based_mu(target_inlet_c=23.5)


POLICY_REGISTRY: dict[str, BehaviorPolicy] = {
    "rule_based": BehaviorPolicy(
        name="rule_based",
        mu_fn=_rule_based_mu(target_inlet_c=24.0),
        sigma=np.array([0.1, 1.0], dtype=np.float32),
    ),
    "conservative": BehaviorPolicy(
        name="conservative",
        mu_fn=_conservative_mu(),
        sigma=np.array([0.05, 0.5], dtype=np.float32),
    ),
    "perturbed_safe": BehaviorPolicy(
        name="perturbed_safe",
        mu_fn=_perturbed_safe_mu(),
        sigma=np.array([0.3, 3.0], dtype=np.float32),
    ),
}


def rollout_policy(
    env: CoolingEnv,
    policy: BehaviorPolicy,
    n_steps: int,
    *,
    seed: int = 0,
    scenario: ScenarioSpec | None = None,
    safety_cfg: SafetyConfig | None = None,
) -> dict[str, np.ndarray]:
    """Run one episode and return column-oriented arrays for the transitions."""
    safety_cfg = safety_cfg or SafetyConfig()
    rng = np.random.default_rng(seed)
    state = env.reset(scenario)
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    next_states: list[np.ndarray] = []
    dones: list[bool] = []
    cost_totals: list[float] = []
    cost_components: list[dict] = []
    logps: list[float] = []
    for _ in range(n_steps):
        action, logp = policy.sample(state, rng)
        next_state, reward, done, _info = env.step(action)
        cost_dict = compute_constraint_cost(state, action, next_state, safety_cfg)
        states.append(state.copy())
        actions.append(action.copy())
        rewards.append(reward)
        next_states.append(next_state.copy())
        dones.append(bool(done))
        cost_totals.append(cost_dict["total"])
        cost_components.append(cost_dict)
        logps.append(logp)
        state = next_state
        if done:
            break
    return {
        "state": np.stack(states),
        "action": np.stack(actions),
        "reward": np.asarray(rewards, dtype=np.float32),
        "next_state": np.stack(next_states),
        "done": np.asarray(dones, dtype=np.bool_),
        "cost": np.asarray(cost_totals, dtype=np.float32),
        "cost_inlet_hi": np.asarray([c["inlet_hi"] for c in cost_components], dtype=np.float32),
        "cost_inlet_lo": np.asarray([c["inlet_lo"] for c in cost_components], dtype=np.float32),
        "cost_outlet_hi": np.asarray([c["outlet_hi"] for c in cost_components], dtype=np.float32),
        "cost_rate_coolant": np.asarray([c["rate_coolant"] for c in cost_components], dtype=np.float32),
        "cost_rate_fan": np.asarray([c["rate_fan"] for c in cost_components], dtype=np.float32),
        "logp_behavior": np.asarray(logps, dtype=np.float32),
    }


def build_buffer(
    env_cfg: EnvConfig,
    safety_cfg: SafetyConfig,
    *,
    policy_names: list[str],
    scenario_families: list[str],
    n_episodes_per_policy: int = 8,
    scenarios_per_policy: int = 4,
    seed: int = 0,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Build a unified buffer by rolling all (policy, scenario) pairs."""
    rng = np.random.default_rng(seed)
    env = CoolingEnv(env_cfg)
    all_chunks: list[dict[str, np.ndarray]] = []
    manifest: list[dict] = []
    ep_id = 0
    for pname in policy_names:
        if pname not in POLICY_REGISTRY:
            raise ValueError(f"Unknown policy: {pname}")
        policy = POLICY_REGISTRY[pname]
        # Cycle through scenario families deterministically.
        families_cycle = list(itertools.islice(itertools.cycle(scenario_families), n_episodes_per_policy))
        for ep, family in enumerate(families_cycle):
            scenario_seed = int(rng.integers(0, 2**31 - 1))
            scenario = make_scenario(family, scenario_seed)
            chunk = rollout_policy(
                env,
                policy,
                env_cfg.episode_steps,
                seed=scenario_seed,
                scenario=scenario,
                safety_cfg=safety_cfg,
            )
            # Stamp episode / policy / scenario columns.
            n = chunk["state"].shape[0]
            chunk["episode_id"] = np.full(n, ep_id, dtype=np.int32)
            chunk["policy_id"] = np.full(n, POLICY_REGISTRY[pname].name, dtype=object)
            chunk["scenario_family"] = np.full(n, family, dtype=object)
            all_chunks.append(chunk)
            manifest.append(
                {
                    "episode_id": ep_id,
                    "policy": pname,
                    "scenario_family": family,
                    "scenario_seed": scenario_seed,
                    "n_steps": int(n),
                    "mean_reward": float(np.mean(chunk["reward"])),
                    "sum_cost": float(np.sum(chunk["cost"])),
                    "max_inlet_c": float(np.max(chunk["next_state"][:, STATE_FIELDS.index("inlet_temp_c")])),
                }
            )
            ep_id += 1

    # Concatenate.
    merged: dict[str, np.ndarray] = {}
    for key in all_chunks[0]:
        if all_chunks[0][key].dtype == object:
            merged[key] = np.concatenate([c[key] for c in all_chunks])
        else:
            merged[key] = np.concatenate([c[key] for c in all_chunks])
    return merged, manifest


def write_manifest(manifest: list[dict], path: str | Path) -> None:
    save_json(manifest, path)


def save_buffer(buffer: dict[str, np.ndarray], path: str | Path) -> Path:
    """Persist the buffer to a compressed npz. Object arrays (policy_id, scenario_family) are stored."""
    p = Path(path)
    ensure_dir(p.parent)
    np.savez_compressed(
        p,
        **{k: v for k, v in buffer.items() if v.dtype != object},
        policy_id=buffer["policy_id"].astype(str),
        scenario_family=buffer["scenario_family"].astype(str),
    )
    return p


def load_buffer(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--env-config", default="configs/env_lcopt.yaml")
    parser.add_argument("--model-config", default="configs/model_cmdp.yaml")
    parser.add_argument("--artifacts", default=None)
    args = parser.parse_args(argv)

    cfg = load_configs(args.base_config, args.env_config, args.model_config)
    set_seed(int(cfg["seed"]))
    if args.artifacts:
        cfg["artifacts_dir"] = args.artifacts
    artifacts_dir = ensure_dir(cfg["artifacts_dir"])

    env_cfg = env_config_from_dict(cfg)
    safety_cfg = SafetyConfig.from_dict(cfg)
    buffer_cfg = cfg.get("buffer", {})
    scen_families = list(cfg.get("environment", {}).get("scenario_families", ["nominal"]))
    buffer, manifest = build_buffer(
        env_cfg,
        safety_cfg,
        policy_names=list(buffer_cfg.get("policies", ["rule_based"])),
        scenario_families=scen_families,
        n_episodes_per_policy=int(buffer_cfg.get("n_episodes_per_policy", 8)),
        scenarios_per_policy=int(buffer_cfg.get("scenarios_sampled_per_policy", 4)),
        seed=int(cfg["seed"]),
    )
    buffer_path = save_buffer(buffer, artifacts_dir / "offline_buffer.npz")
    write_manifest(manifest, artifacts_dir / "offline_buffer_manifest.json")
    print(
        f"Wrote buffer: {buffer_path} (n_transitions={buffer['state'].shape[0]}), "
        f"manifest: {artifacts_dir / 'offline_buffer_manifest.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
