# SafeCool-CMDP

**Safety-constrained offline RL baseline for data-centre cooling control.**

A runnable, end-to-end Python project that builds an offline RL dataset from three behavior policies across four cooling scenarios, trains a conservative Q-learning agent with a Lagrangian safety multiplier, runs off-policy evaluation with a go/no-go cost gate, wraps the agent in a deterministic safety shield with world-model look-ahead, distills the learned policy into inspectable decision-tree rules, and packages a shadow-mode deploy harness with audit logging and rollback.

## Table of contents

- [Problem framing](#problem-framing)
- [Causality and safety discipline](#causality-and-safety-discipline)
- [What is actually implemented](#what-is-actually-implemented)
- [Repository structure](#repository-structure)
- [Simulation provenance](#simulation-provenance)
- [Environment design](#environment-design)
- [Safety mechanisms](#safety-mechanisms)
- [Offline learning approach](#offline-learning-approach)
- [Off-policy evaluation](#off-policy-evaluation)
- [Interpretability](#interpretability)
- [Deploy path](#deploy-path)
- [Results](#results)
- [Limitations](#limitations)
- [Install](#install)
- [How to run: buffer](#how-to-run-buffer)
- [How to run: training](#how-to-run-training)
- [How to run: evaluation](#how-to-run-evaluation)
- [How to run: distillation](#how-to-run-distillation)
- [How to run: deploy / decision path](#how-to-run-deploy--decision-path)
- [How to run: tests](#how-to-run-tests)
- [Design notes](#design-notes)

## Problem framing

Cooling is a large energy lever in data centres, but unsafe exploration can overheat equipment and damage SLAs. A usable RL system for this domain needs three things at once: (1) a way to learn without online trial and error, because real facilities cannot tolerate unsafe actions; (2) a hard safety layer that blocks unsafe policy outputs before they reach the plant; and (3) a credible way to evaluate a candidate policy before deploying it, even in shadow. This repository stitches all three together as a self-contained research baseline.

## Causality and safety discipline

Every surface — README, CLI output, deploy report, docstrings — is explicit that this is a **research baseline**. Offline policy recommendations require human review and operational validation. "Shadow mode" here means running the agent against a local simulator next to a baseline controller for comparison; it never touches a real facility. The deploy report embeds the SHA-256 of the agent checkpoint plus a disclaimer string. The shield is deterministic, inspectable, and deliberately separate from the learned agent.

## What is actually implemented

- **Local cooling environment** (`src/safecool/envs/lcopt_env.py`): 7-D state (fan speed, coolant supply, inlet temp, outlet temp, CPU load, ambient temp, power), 2-D delta action (coolant setpoint delta, fan speed delta), first-order thermal lag dynamics, and four configurable scenario families (nominal / spike / degradation / heatwave). **This is not a wrapper around LC-Opt**; the module is named `lcopt_env.py` for continuity with the scaffold but the docstring is explicit that it implements a local approximation, not a Modelica runtime.
- **Safety constraints** (`src/safecool/envs/safety_constraints.py`): hinge-shaped per-step cost (`inlet_hi`, `inlet_lo`, `outlet_hi`, `rate_coolant`, `rate_fan`), action projection to the feasible set, RCI-proxy helper, and violation counter.
- **Offline buffer** (`src/safecool/data/build_offline_buffer.py`): three configured behavior policies (`rule_based`, `conservative`, `perturbed_safe`), Gaussian action sampling with log-prob, rollouts across scenario families, compressed `.npz` buffer plus a JSON manifest.
- **World model** (`src/safecool/models/world_model.py`): small MLP that predicts (next_state, reward, cost), ReLU on the cost head for non-negativity, median/IQR state normalization. Exposes `predict_next_inlet_temp` for the shield.
- **Offline agent** (`src/safecool/rl/offline_agent.py`): discrete-action CQL-lite — Q-learning with a logsumexp conservative penalty plus a **separate cost Q-network and Lagrangian multiplier**. The multiplier adapts online to the gap between estimated expected cost and the configured budget. A 5×5 action grid over (coolant delta, fan delta) keeps the agent interpretable and enables clean FQE.
- **Policy shield** (`src/safecool/rl/policy_shield.py`): `is_safe` checks box bounds + rate limits + operating-window overflow + optional one-step world-model look-ahead. Unsafe candidates get projected into the feasible set; if projection cannot rescue the action, the shield falls back to a conservative rule-based policy. Every intervention is counted.
- **Off-policy evaluation** (`src/safecool/rl/off_policy_eval.py`): Weighted Importance Sampling with bootstrap CI (low-bias, high-variance sanity check) and Fitted Q Evaluation (the primary estimator; reports reward and cost separately). A go/no-go gate in `train.py` refuses to mark the agent deployable if FQE cost exceeds the configured budget.
- **Distillation** (`src/safecool/distill.py`): one sklearn `DecisionTreeRegressor` per action dimension, depth-limited, with fidelity metrics (MAE and agreement-within-half-grid on a held-out sample) and a human-readable exported rule list.
- **Shadow-mode deploy** (`src/safecool/deploy.py`): per-step logging of (proposed, shielded, baseline, realized) tuples, rolling CVaR-95 + violation-count rollback rules, SHA-256 checkpoint hash, explicit disclaimer in the audit log.
- **55 behavioral tests** covering the environment, safety constraints, buffer schema, world model, agent training, shield behavior, OPE estimators, distillation, rollback logic, and end-to-end CLI runs.

## Repository structure

```
project_03_safecool_cmdp/
├── configs/
│   ├── base.yaml                    # seeds, training schedule, OPE, deploy thresholds
│   ├── env_lcopt.yaml               # environment + action space + safety thresholds
│   └── model_cmdp.yaml              # world model, agent, shield, buffer composition
├── src/safecool/
│   ├── data/
│   │   └── build_offline_buffer.py  # CLI: safecool-build-buffer
│   ├── envs/
│   │   ├── lcopt_env.py             # local cooling simulator
│   │   └── safety_constraints.py    # hinge costs + projection + RCI
│   ├── models/
│   │   └── world_model.py           # MLP dynamics / reward / cost
│   ├── rl/
│   │   ├── offline_agent.py         # CQL-lite with Lagrangian safety multiplier
│   │   ├── policy_shield.py         # is_safe + project + fallback + lookahead
│   │   └── off_policy_eval.py       # WIS + FQE
│   ├── utils/                       # seed, config, io (copied verbatim from Project 1)
│   ├── train.py                     # CLI: safecool-train
│   ├── evaluate.py                  # CLI: safecool-evaluate
│   ├── distill.py                   # CLI: safecool-distill
│   └── deploy.py                    # CLI: safecool-deploy
├── tests/                           # 11 test files, 55 tests
├── notebooks/
│   └── demo.ipynb                   # executed end-to-end walk-through
├── artifacts/                       # models, reports, audit logs (created by CLI)
├── pyproject.toml
└── README.md
```

## Simulation provenance

The bundled environment is a **local approximation**, not real telemetry or real LC-Opt. It models:

- Thermal dynamics: first-order lag coupling coolant supply temperature and ambient temperature to rack inlet temperature, with time constant configurable in steps.
- IT heat: outlet temperature = inlet + gain × low-pass-filtered CPU load.
- Power: IT draw proportional to CPU load; fan power quadratic in fan speed; pump power linear in fan speed.
- Diurnal seasonality: sinusoidal ambient temperature and CPU load with configurable amplitudes and phase.
- Scenario disturbances:
  - `nominal`: baseline conditions only.
  - `spike`: short-duration CPU load burst of configurable magnitude.
  - `degradation`: persistent positive offset on effective coolant supply temperature (simulates a degraded CDU / CRAC).
  - `heatwave`: persistent positive offset on ambient temperature.

The environment exposes the same interface a real LC-Opt wrapper would expose — `reset(scenario)` and `step(action)` returning `(next_state, reward, done, info)` — so swapping in the real simulator later is a drop-in change.

## Environment design

| Dimension | Symbol | Units | Range |
|---|---|---|---|
| Fan speed | `fan_speed_pct` | % | 25 – 100 |
| Coolant supply temperature | `coolant_supply_c` | °C | 15 – 24 |
| Rack inlet temperature | `inlet_temp_c` | °C | unbounded (safety gate at 27) |
| Rack outlet temperature | `outlet_temp_c` | °C | unbounded (safety gate at 40) |
| CPU load | `cpu_load` | fraction | 0.05 – 1.0 |
| Ambient temperature | `ambient_temp_c` | °C | driven by scenario |
| Total power | `power_kw` | kW | derived |

Actions are **deltas** (rate-limited by safety config): `(Δ coolant supply °C, Δ fan speed %)`. A discrete 5×5 grid gives 25 candidate actions.

Reward is the negative of total power scaled so episode sums are `O(-40)` to `O(-80)` on the default horizon (96 steps ≈ 8 hours at 5-min cadence).

## Safety mechanisms

Two separate layers:

1. **CMDP-style penalty during learning.** `compute_constraint_cost` emits a hinge-shaped non-negative cost when inlet/outlet temperatures exceed configured limits or when the action violates rate limits. The agent's Lagrangian multiplier applies this cost as an adversarial term during Q updates.
2. **Deterministic shield at decision time.** Before any action reaches the simulator (or the real plant, in a hypothetical production path) the shield runs `is_safe` and, on failure, either projects the action into the feasible set or falls back to the conservative rule-based controller. Projection uses `project_to_safe_action`, which clips to delta ranges, rate limits, and the operating window.

The two layers are intentionally redundant. The CMDP penalty shapes what the agent learns; the shield catches what the agent learned wrong.

## Offline learning approach

- Discrete 25-action grid over `(Δ coolant ∈ {-1, -0.5, 0, 0.5, 1}°C, Δ fan ∈ {-10, -5, 0, 5, 10}%)`.
- Two Q-networks: `q_reward` (maximize returns) and `q_cost` (estimate per-step safety cost).
- Greedy decision: `argmax over actions of q_reward − λ · q_cost`.
- CQL-lite conservative penalty: `logsumexp(q_reward over actions) − q_reward(taken action)` — pushes Q down on actions outside the buffer distribution so the agent doesn't rely on extrapolations.
- Soft target network updates at each step: `target ← (1-τ) · target + τ · online`.
- Lagrangian dual ascent: `λ ← max(0, λ + η (est_cost − per_episode_budget))` where the per-episode budget converts the per-step cost budget through `1/(1-γ)`.

## Off-policy evaluation

- **WIS**. Per-episode log-importance ratios under a softmax target policy (temperature 0.5 to keep ratios finite). Returns reward estimate, cost estimate, and bootstrap CI on the reward.
- **FQE**. A fresh pair of Q-networks trained on the buffer with bootstrap targets computed under the target policy's greedy action. Returns `(reward_est, cost_est)` at the mean initial state. Cost is reported clamped to non-negative (the network can undershoot zero on tiny buffers, but cost is non-negative by construction).
- **Gate**. `training_report.json` reports `ope_gate_pass`: True iff FQE cost ≤ `cost_budget × episode_steps`.

## Interpretability

`safecool-distill` fits one shallow decision-tree regressor per action dimension on states sampled from the offline buffer, labeled by the agent's greedy action. It writes:

- `artifacts/surrogate.json` with fidelity metrics on a held-out split.
- `artifacts/surrogate.txt` with the full exported rule list, ready for operator review.

## Deploy path

`safecool-deploy --scenario <nominal|spike|degradation|heatwave>` advances the environment under the agent's shielded action, while simultaneously logging the raw agent proposal, the baseline rule-based action, the realized reward and cost, and whether the shield intervened. It maintains a rolling CVaR-95 inlet temperature and a rolling violation count; when either exceeds configured thresholds, the loop trips rollback and falls back to the baseline for the remainder of the episode. The output `deploy_report.json` includes the SHA-256 of the agent checkpoint and a disclaimer confirming this is not a production artifact.

## Results

All numbers below come from a single CLI run on the default configs (local CPU, 3,456 transitions, 20 world-model epochs, 30 agent epochs, 3 eval episodes per scenario).

**FQE after training:** reward estimate **-15.56** (sum of discounted episode rewards), cost estimate **0.00** (below the 1.92 budget). Training gate **PASS**. The Lagrangian multiplier λ drops from 0.1 to 0.0 over the run because the conservative-policy-heavy buffer dominates the learned behavior.

**Evaluation across 4 scenarios (3 episodes each):**

| Policy | Mean energy (kW) | Mean inlet (°C) | CVaR-95 inlet (°C) | Violations / ep. | Shield interventions / ep. |
| --- | ---: | ---: | ---: | ---: | ---: |
| rule_based | 3.28 | 23.89 | 24.70 | 0.00 | 0.0 |
| conservative | 7.57 | 17.24 | 21.14 | 0.00 | 0.0 |
| agent | 4.29 | 19.36 | 22.21 | 0.17 | 0.0 |
| **agent_shielded** | **4.30** | **19.32** | **22.16** | **0.08** | **67.6** |

**What to read from this:**

- The shield genuinely helps: it **halves the per-episode violation rate** of the learned agent (0.17 → 0.08) and **drops per-episode total safety cost from 0.58 to 0.005** at essentially zero energy overhead (+0.01 kW).
- The conservative behavior policy dominated the buffer, so the agent learned to imitate its over-cooling tendency (mean inlet 19.3°C instead of the rule_based's 23.9°C). A richer buffer with more diverse reward-seeking trajectories would let the agent operate closer to the energy-optimal setpoint.
- Both agent variants fail the CVaR-95 gate in one heatwave scenario (CVaR-95 inlet reaches 27.5°C > 27.0°C limit without the shield; 26.9°C with shield, still over 27.0°C in one run out of three — a narrow miss). The honest read is: this is a research baseline that demonstrates the full stack, but a production-grade policy needs a larger buffer, longer training, and tighter shield thresholds before any live-facility evaluation.
- Distilled surrogate achieves **95.5% action agreement within half a grid cell** on the held-out split — the tree rules are a faithful interpretable approximation of the agent's greedy policy.
- Shadow-mode deploy runs on all three tested scenarios (nominal, spike, heatwave) complete **without rollback** under the default thresholds, logging ≈90 shield interventions over 200 steps in each case.

## Limitations

Being explicit about what this repo is and is not:

1. **The environment is a local approximation, not real telemetry or real LC-Opt.** Numbers above reflect simulator dynamics, not a real facility. The ingestion interface matches what a Modelica-backed LC-Opt wrapper would expose, but the connection is not made.
2. **The agent under-explores.** Because the offline buffer is dominated by the conservative behavior policy, the CQL penalty pulls the agent toward conservative behavior and the reward-seeking pressure is weak. A larger, more balanced buffer would change this.
3. **OPE is approximate.** WIS is high-variance; FQE inherits bias from function approximation. Both are reported side-by-side so the reader can triangulate.
4. **The shield's world-model look-ahead is only one step.** Longer horizons would catch slower build-ups but at the cost of compounding model error.
5. **The Lagrangian multiplier schedule is simple dual ascent.** More sophisticated approaches (e.g., PID on the constraint, trust-region updates) would likely improve cost tracking.
6. **No Lyapunov formulation.** The scaffold mentions a Lyapunov-based CMDP; implementing that properly needs a certified Lyapunov function over the continuous state, which is out of scope for this pass. The Lagrangian approach used here is the common practical stand-in.
7. **Distillation uses one tree per action dim.** A joint classifier over the 25-action grid would also be reasonable; the separate-regressors choice trades a tiny amount of fidelity for much cleaner operator-facing rules.
8. **Not a production deployment artifact.** Research baseline only. Any live use requires human review, simulator validation against a real plant, and operational guardrails well beyond what this repo provides.

## Install

```bash
pip install -e ".[dev]"         # package + test deps
pip install -e ".[notebook]"    # optional: Jupyter for the demo notebook
```

Dependencies: `numpy`, `pandas`, `scipy`, `scikit-learn`, `torch`, `pyyaml`, `pydantic`, `matplotlib`.

## How to run: buffer

```bash
python -m safecool.data.build_offline_buffer
# or: safecool-build-buffer
```

Writes `artifacts/offline_buffer.npz` + `artifacts/offline_buffer_manifest.json`.

## How to run: training

```bash
python -m safecool.train                   # full run (≈30 s on CPU)
python -m safecool.train --quick           # 2/4 epochs, tiny buffer
python -m safecool.train --skip-world-model
# or: safecool-train
```

Writes `world_model.pt`, `world_model_stats.json`, `agent.pt`, `training_report.json`.

## How to run: evaluation

```bash
python -m safecool.evaluate --episodes 3
# or: safecool-evaluate --episodes 3
```

Writes `eval_episodes.csv`, `eval_by_policy.csv`, `eval_by_policy_scenario.csv`, `eval_report.json` (with go/no-go for both shielded and unshielded agent).

## How to run: distillation

```bash
python -m safecool.distill --max-depth 4
# or: safecool-distill --max-depth 4
```

Writes `surrogate.json` (fidelity + metadata) and `surrogate.txt` (readable rules).

## How to run: deploy / decision path

```bash
python -m safecool.deploy --scenario nominal
python -m safecool.deploy --scenario spike
python -m safecool.deploy --scenario degradation
python -m safecool.deploy --scenario heatwave
# or: safecool-deploy --scenario nominal
```

Each run writes `artifacts/deploy_report.json` with the full audit trail (proposed / shielded / baseline / realized actions per step, shield intervention reasons, rollback status, checkpoint hash, disclaimer).

## How to run: tests

```bash
pytest                                     # 55 tests, ≈20 s on CPU
pytest -k shield                           # just the shield tests
pytest -x -v                               # stop at first failure, verbose
```
