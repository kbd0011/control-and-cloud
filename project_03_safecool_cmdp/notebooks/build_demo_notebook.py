"""Build a Jupyter notebook that runs the full SafeCool-CMDP pipeline."""
from __future__ import annotations

import json
from pathlib import Path


def _as_lines(source: str) -> list[str]:
    s = source.strip("\n")
    parts = s.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def _code(src: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "source": _as_lines(src), "outputs": [],
    }


def _md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _as_lines(src)}


def build() -> dict:
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }

    nb["cells"].append(_md(
        "# SafeCool-CMDP demo\n"
        "\n"
        "End-to-end walk-through of the safety-constrained offline RL baseline:\n"
        "build buffer → train world model + agent → off-policy evaluate → roll out with shield → distill → shadow-mode deploy.\n"
        "\n"
        "**Research baseline only.** Offline policy recommendations require human review and operational validation. Not a production deployment artifact."
    ))

    nb["cells"].append(_code(
        """
# Hop to repo root if kernel started in notebooks/.
import os
from pathlib import Path
if Path.cwd().name == 'notebooks':
    os.chdir(Path.cwd().parent)
print('cwd:', Path.cwd())
"""
    ))

    nb["cells"].append(_md("## 1. Explore the cooling environment"))
    nb["cells"].append(_code(
        """
import numpy as np
from safecool.envs.lcopt_env import CoolingEnv, EnvConfig, ScenarioSpec, STATE_FIELDS, make_scenario

env = CoolingEnv(EnvConfig(episode_steps=48))
state = env.reset(make_scenario('nominal', seed=0))
print('Initial state:')
for name, v in zip(STATE_FIELDS, state):
    print(f'  {name:22s} = {v:7.3f}')

# Run a few steps under a zero action and watch dynamics.
for _ in range(5):
    state, reward, done, info = env.step(np.array([0.0, 0.0]))
print(f'After 5 steps  reward={reward:.3f}  inlet={state[STATE_FIELDS.index(\"inlet_temp_c\")]:.2f}C  power={state[-1]:.2f}kW')
"""
    ))

    nb["cells"].append(_md("## 2. Build the offline buffer"))
    nb["cells"].append(_code(
        """
from safecool.data.build_offline_buffer import main as build_main
rc = build_main(['--base-config', 'configs/base.yaml',
                 '--env-config', 'configs/env_lcopt.yaml',
                 '--model-config', 'configs/model_cmdp.yaml'])
print('rc =', rc)

import numpy as np
with np.load('artifacts/offline_buffer.npz') as data:
    print('Buffer keys:', list(data.keys())[:10], '...')
    print('n_transitions:', data['state'].shape[0])
    print('action shape:', data['action'].shape)
"""
    ))

    nb["cells"].append(_md("## 3. Inspect episode-level composition"))
    nb["cells"].append(_code(
        """
import json, pandas as pd
manifest = json.load(open('artifacts/offline_buffer_manifest.json'))
df = pd.DataFrame(manifest)
print('Episodes:', len(df), '| policies:', df['policy'].nunique(), '| scenarios:', df['scenario_family'].nunique())
df.groupby(['policy', 'scenario_family'])[['mean_reward', 'sum_cost', 'max_inlet_c']].mean().round(3)
"""
    ))

    nb["cells"].append(_md("## 4. Train world model + CQL-lite agent + run OPE"))
    nb["cells"].append(_code(
        """
from safecool.train import main as train_main
rc = train_main(['--base-config', 'configs/base.yaml',
                 '--env-config', 'configs/env_lcopt.yaml',
                 '--model-config', 'configs/model_cmdp.yaml'])
print('rc =', rc)

import json
report = json.load(open('artifacts/training_report.json'))
print(f'Buffer size: {report[\"n_transitions\"]}')
print(f'FQE reward: {report[\"ope\"][\"fqe\"][\"reward_estimate\"]:.3f}')
print(f'FQE cost:   {report[\"ope\"][\"fqe\"][\"cost_estimate\"]:.3f}  (budget {report[\"budget_threshold\"]:.2f})')
print(f'WIS reward: {report[\"ope\"][\"wis\"][\"reward_estimate\"]:.3f}  CI [{report[\"ope\"][\"wis\"][\"reward_ci_low\"]:.2f}, {report[\"ope\"][\"wis\"][\"reward_ci_high\"]:.2f}]')
print(f'Gate: {\"PASS\" if report[\"ope_gate_pass\"] else \"FAIL\"}')
"""
    ))

    nb["cells"].append(_md("## 5. Evaluate all policies across scenarios"))
    nb["cells"].append(_code(
        """
from safecool.evaluate import main as eval_main
rc = eval_main(['--episodes', '3'])
import json, pandas as pd
report = json.load(open('artifacts/eval_report.json'))
agg = pd.DataFrame(report['aggregate_by_policy']).set_index('policy')
agg[['mean_reward', 'mean_energy_kw', 'mean_inlet_c', 'cvar95_inlet_c', 'viol_any', 'n_shield_intervened']].round(3)
"""
    ))

    nb["cells"].append(_md("## 6. Shield effect: compare agent vs agent+shield"))
    nb["cells"].append(_code(
        """
import pandas as pd
ep = pd.read_csv('artifacts/eval_episodes.csv')
comparison = ep.groupby('policy').agg(
    mean_reward=('mean_reward', 'mean'),
    mean_energy=('mean_energy_kw', 'mean'),
    max_inlet=('max_inlet_c', 'max'),
    total_cost=('total_cost', 'mean'),
    violations=('viol_any', 'sum'),
    shield_per_ep=('n_shield_intervened', 'mean'),
).round(3)
comparison
"""
    ))

    nb["cells"].append(_md("## 7. Plot inlet temperature trace for one episode per policy"))
    nb["cells"].append(_code(
        """
import matplotlib.pyplot as plt
import numpy as np
from safecool.envs.lcopt_env import CoolingEnv, env_config_from_dict, STATE_FIELDS, make_scenario
from safecool.envs.safety_constraints import SafetyConfig
from safecool.rl.offline_agent import load_agent
from safecool.rl.policy_shield import PolicyShield, conservative_fallback
from safecool.data.build_offline_buffer import POLICY_REGISTRY
from safecool.utils.config import load_configs

cfg = load_configs('configs/base.yaml', 'configs/env_lcopt.yaml', 'configs/model_cmdp.yaml')
env_cfg = env_config_from_dict(cfg)
env_cfg.episode_steps = 96
safety = SafetyConfig.from_dict(cfg)
agent = load_agent('artifacts/agent.pt')

def rollout(act_fn, scenario_seed=0):
    env = CoolingEnv(env_cfg)
    s = env.reset(make_scenario('spike', seed=scenario_seed))
    trace = [float(s[STATE_FIELDS.index('inlet_temp_c')])]
    while True:
        a = np.asarray(act_fn(s), dtype=np.float32)
        s, r, done, _ = env.step(a)
        trace.append(float(s[STATE_FIELDS.index('inlet_temp_c')]))
        if done: break
    return trace

rng = np.random.default_rng(0)
rb_policy = POLICY_REGISTRY['rule_based']
rule_act = lambda st: rb_policy.sample(st, rng)[0]
agent_act = lambda st: agent.predict(st)

shield = PolicyShield(safety_cfg=safety, fallback_policy=conservative_fallback, use_lookahead=False)
def shielded_act(st):
    return shield(agent.predict(st), st)['action']

traces = {
    'rule_based': rollout(rule_act),
    'agent': rollout(agent_act),
    'agent_shielded': rollout(shielded_act),
}
fig, ax = plt.subplots(figsize=(10, 4))
for name, t in traces.items():
    ax.plot(t, label=name, lw=2)
ax.axhline(safety.max_inlet_temp_c, color='red', ls='--', alpha=0.6, label=f'max limit {safety.max_inlet_temp_c}°C')
ax.axhline(safety.inlet_warn_c, color='orange', ls=':', alpha=0.5, label=f'warn {safety.inlet_warn_c}°C')
ax.set_xlabel('step'); ax.set_ylabel('inlet temp (°C)')
ax.set_title('Spike scenario — inlet temperature trace by policy')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
"""
    ))

    nb["cells"].append(_md("## 8. Distill the agent into interpretable decision-tree rules"))
    nb["cells"].append(_code(
        """
from safecool.distill import main as distill_main
rc = distill_main(['--max-depth', '4'])

import json
info = json.load(open('artifacts/surrogate.json'))
print(f'Fidelity on held-out states: {info[\"fidelity_holdout\"]}')
print('--- first 30 lines of surrogate.txt ---')
print('\\n'.join(open('artifacts/surrogate.txt').read().splitlines()[:30]))
"""
    ))

    nb["cells"].append(_md("## 9. Shadow-mode deploy with audit log and rollback rules"))
    nb["cells"].append(_code(
        """
from safecool.deploy import main as deploy_main
rc = deploy_main(['--scenario', 'heatwave'])

import json
report = json.load(open('artifacts/deploy_report.json'))
print(f'Scenario: {report[\"scenario_family\"]}')
print(f'Steps: {report[\"n_steps\"]}  max_inlet={report[\"max_inlet_c\"]:.2f}C  CVaR-95={report[\"cvar95_inlet_c\"]:.2f}C')
print(f'Shield interventions: {report[\"n_shield_interventions\"]}')
print(f'Rollback active: {report[\"rollback_active\"]}  reason={report[\"rollback_reason\"]}')
print(f'Agent checkpoint SHA-256: {report[\"agent_checkpoint_sha256\"][:32]}...')
print()
print('Disclaimer:', report['disclaimer'])
"""
    ))

    return nb


def main() -> int:
    out = Path("notebooks/demo.ipynb")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build(), indent=1))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
