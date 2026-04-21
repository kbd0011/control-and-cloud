"""Policy distillation to interpretable surrogates.

We fit a shallow decision tree per action dimension (coolant delta, fan
delta) that maps the state vector to the agent's greedy action. This gives
operators a readable rule set they can sanity-check.

Outputs:
    artifacts/surrogate.json   Rule list + fidelity on held-out states.
    artifacts/surrogate.txt    Human-readable rules printed via sklearn.tree.export_text.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeRegressor, export_text

from safecool.data.build_offline_buffer import load_buffer
from safecool.envs.lcopt_env import STATE_FIELDS
from safecool.rl.offline_agent import DiscreteOfflineAgent, load_agent
from safecool.utils.config import load_configs, setup_logging
from safecool.utils.io import ensure_dir, save_json
from safecool.utils.seed import set_seed

logger = logging.getLogger("safecool.distill")


def _sample_states_from_buffer(buffer: dict, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.choice(buffer["state"].shape[0], size=min(n, buffer["state"].shape[0]), replace=False)
    return buffer["state"][idx]


def fit_surrogate(
    agent: DiscreteOfflineAgent, states: np.ndarray, *, max_depth: int = 4, seed: int = 0
) -> dict:
    """Fit one shallow regression tree per action dimension."""
    predictions = np.stack([agent.predict(s) for s in states])
    tree_coolant = DecisionTreeRegressor(max_depth=max_depth, random_state=seed).fit(states, predictions[:, 0])
    tree_fan = DecisionTreeRegressor(max_depth=max_depth, random_state=seed).fit(states, predictions[:, 1])
    return {"coolant": tree_coolant, "fan": tree_fan, "targets": predictions}


def compute_fidelity(
    agent: DiscreteOfflineAgent,
    trees: dict,
    states: np.ndarray,
) -> dict:
    """Fidelity = fraction of states where (surrogate action) is within tolerance of (agent action).

    We also report per-dim MAE and R². A held-out sample is recommended; callers
    should pass states the trees did not see during fitting.
    """
    agent_actions = np.stack([agent.predict(s) for s in states])
    surr = np.stack(
        [trees["coolant"].predict(states), trees["fan"].predict(states)],
        axis=-1,
    )
    mae_coolant = float(np.mean(np.abs(surr[:, 0] - agent_actions[:, 0])))
    mae_fan = float(np.mean(np.abs(surr[:, 1] - agent_actions[:, 1])))
    # Agreement within half a grid step (0.25 C coolant, 2.5 % fan).
    agreement = float(
        np.mean(
            (np.abs(surr[:, 0] - agent_actions[:, 0]) < 0.25)
            & (np.abs(surr[:, 1] - agent_actions[:, 1]) < 2.5)
        )
    )
    return {"mae_coolant": mae_coolant, "mae_fan": mae_fan, "agreement_within_half_grid": agreement}


def export_rules(trees: dict, path: str | Path) -> str:
    """Emit a human-readable rule list to `path` and return it as a string."""
    rules_coolant = export_text(trees["coolant"], feature_names=list(STATE_FIELDS))
    rules_fan = export_text(trees["fan"], feature_names=list(STATE_FIELDS))
    body = (
        "# SafeCool-CMDP distilled surrogate\n\n"
        "Each tree approximates the trained agent's greedy action for one control dimension.\n"
        "These rules are an interpretable approximation intended for human review only.\n\n"
        "## Coolant supply delta (°C)\n\n"
        + rules_coolant
        + "\n## Fan speed delta (percentage points)\n\n"
        + rules_fan
    )
    p = Path(path)
    p.write_text(body)
    return body


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--env-config", default="configs/env_lcopt.yaml")
    parser.add_argument("--model-config", default="configs/model_cmdp.yaml")
    parser.add_argument("--artifacts", default=None)
    parser.add_argument("--n-states", type=int, default=1000)
    parser.add_argument("--max-depth", type=int, default=4)
    args = parser.parse_args(argv)

    cfg = load_configs(args.base_config, args.env_config, args.model_config)
    if args.artifacts:
        cfg["artifacts_dir"] = args.artifacts
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    set_seed(int(cfg["seed"]))
    artifacts_dir = ensure_dir(cfg["artifacts_dir"])

    agent = load_agent(str(artifacts_dir / "agent.pt"))
    buffer = load_buffer(artifacts_dir / "offline_buffer.npz")
    states = _sample_states_from_buffer(buffer, args.n_states, seed=int(cfg["seed"]))
    split = int(0.8 * len(states))
    fit_states, eval_states = states[:split], states[split:]

    trees = fit_surrogate(agent, fit_states, max_depth=args.max_depth, seed=int(cfg["seed"]))
    fidelity = compute_fidelity(agent, trees, eval_states)
    export_rules(trees, artifacts_dir / "surrogate.txt")

    save_json(
        {
            "feature_names": list(STATE_FIELDS),
            "fidelity_holdout": fidelity,
            "max_depth": args.max_depth,
            "n_fit_states": int(len(fit_states)),
            "n_eval_states": int(len(eval_states)),
        },
        artifacts_dir / "surrogate.json",
    )
    logger.info("Surrogate fidelity agreement=%.3f  mae_coolant=%.3f  mae_fan=%.3f",
                fidelity["agreement_within_half_grid"],
                fidelity["mae_coolant"], fidelity["mae_fan"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
