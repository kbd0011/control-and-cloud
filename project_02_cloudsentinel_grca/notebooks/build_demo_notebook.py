"""Build a Jupyter notebook that runs the full CloudSentinel-GRCA pipeline.

Run this from the repo root to regenerate `notebooks/demo.ipynb` with a
deterministic set of cells. Execute with:
    jupyter nbconvert --to notebook --execute notebooks/demo.ipynb --output demo.ipynb
"""
from __future__ import annotations

import json
from pathlib import Path


def _as_lines(source: str) -> list[str]:
    s = source.strip("\n")
    parts = s.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def _code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": _as_lines(src),
        "outputs": [],
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
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    nb["cells"].append(_md(
        "# CloudSentinel-GRCA demo\n"
        "\n"
        "End-to-end walk-through: synthetic data → fit detector + graph → detect → rank → brief → call the API.\n"
        "\n"
        "Run from the repo root after `pip install -e \".[dev,notebook]\"`."
    ))

    nb["cells"].append(_code(
        """
# Hop up to the repo root if the kernel started in notebooks/.
import os
from pathlib import Path
if Path.cwd().name == 'notebooks':
    os.chdir(Path.cwd().parent)
print('cwd:', Path.cwd())
"""
    ))

    nb["cells"].append(_md("## 1. Generate synthetic multi-entity telemetry"))
    nb["cells"].append(_code(
        """
from cloudsentinel.data.synthetic import SyntheticConfig, write_sample_dataset
paths = write_sample_dataset('data/sample', SyntheticConfig(
    n_entities=24, n_services=6, duration_days=7, cadence_minutes=5,
    incident_count=12, seed=42))
for name, p in paths.items():
    print(f'{name}: {p}')
"""
    ))

    nb["cells"].append(_md("## 2. Inspect the entities and the service dependency graph"))
    nb["cells"].append(_code(
        """
import pandas as pd
entities = pd.read_csv('data/sample/entities.csv')
deps = pd.read_csv('data/sample/dependencies.csv')
incidents = pd.read_csv('data/sample/incidents.csv')
print('Entities:', len(entities), '| Services:', entities['service'].nunique())
print('Dependency edges:', len(deps))
print('Labeled incidents:', len(incidents))
display(incidents.head())
"""
    ))

    nb["cells"].append(_md("## 3. Train the detector + build the entity graph"))
    nb["cells"].append(_code(
        """
from cloudsentinel.train import main as train_main
rc = train_main(['--base-config', 'configs/base.yaml',
                 '--data-config', 'configs/data_ibm.yaml',
                 '--model-config', 'configs/model_grca.yaml'])
print('train rc:', rc)
"""
    ))

    nb["cells"].append(_md("## 4. Evaluate and inspect detection + RCA metrics"))
    nb["cells"].append(_code(
        """
import json
from cloudsentinel.evaluate import main as evaluate_main
evaluate_main(['--base-config', 'configs/base.yaml',
               '--data-config', 'configs/data_ibm.yaml',
               '--model-config', 'configs/model_grca.yaml'])
report = json.load(open('artifacts/eval_report.json'))
print('Detection:')
for k, v in report['point_detection'].items():
    print(f'  {k}: {v}')
print('Best F1 threshold:', report['best_threshold'])
print('RCA:', report['rca'])
print('OOD:', report['ood'])
"""
    ))

    nb["cells"].append(_md("## 5. Plot the detector's three streams for one series"))
    nb["cells"].append(_code(
        """
import pandas as pd, matplotlib.pyplot as plt
scored = pd.read_csv('artifacts/scored_events.csv', parse_dates=['timestamp'])
# Pick the series with the highest mean probability.
series_ranking = (
    scored.groupby(['entity_id', 'metric'])['prob'].mean().sort_values(ascending=False)
)
ent, met = series_ranking.index[0]
print(f'Plotting {ent} / {met}  mean_prob={series_ranking.iloc[0]:.3f}')
s = scored[(scored.entity_id == ent) & (scored.metric == met)].sort_values('timestamp')
fig, axes = plt.subplots(4, 1, figsize=(11, 7), sharex=True)
axes[0].plot(s.timestamp, s.value); axes[0].set_ylabel(met)
axes[1].plot(s.timestamp, s.zscore, color='tab:orange'); axes[1].set_ylabel('z-score')
axes[2].plot(s.timestamp, s.residual, color='tab:green'); axes[2].set_ylabel('residual')
axes[3].plot(s.timestamp, s.prob, color='tab:red'); axes[3].set_ylabel('fused prob')
axes[3].axhline(0.7, linestyle='--', color='gray', alpha=0.5)
plt.tight_layout(); plt.show()
"""
    ))

    nb["cells"].append(_md("## 6. Generate incident briefs"))
    nb["cells"].append(_code(
        """
from cloudsentinel.incident_brief import main as brief_main
brief_main(['--base-config', 'configs/base.yaml',
            '--data-config', 'configs/data_ibm.yaml',
            '--model-config', 'configs/model_grca.yaml'])

from pathlib import Path
md_files = sorted(Path('artifacts/briefs').glob('*.md'))
print(f'Wrote {len(md_files)} briefs')
if md_files:
    print('---- first 35 lines of', md_files[0].name, '----')
    print('\\n'.join(md_files[0].read_text().splitlines()[:35]))
"""
    ))

    nb["cells"].append(_md("## 7. Call the API in-process"))
    nb["cells"].append(_code(
        """
from fastapi.testclient import TestClient
from cloudsentinel.serve import create_app
app = create_app('artifacts')
client = TestClient(app)
print('health:', client.get('/health').json())

# Score a real telemetry series (first entity's latency).
telemetry = pd.read_csv('data/sample/telemetry.csv', parse_dates=['timestamp'])
sub = telemetry[(telemetry.entity_id == 'ent_00') & (telemetry.metric == 'latency_ms')].sort_values('timestamp')
values = sub['value'].dropna().tolist()[:200]
score = client.post('/score', json={'entity_id': 'ent_00', 'metric': 'latency_ms', 'values': values}).json()
print('score[prob] first 5:', [round(p, 3) for p in score['prob'][:5]])
print('score[prob] max:', round(max(score['prob']), 3))
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
