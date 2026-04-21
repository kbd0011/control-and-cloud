"""
Run this from the repo root AFTER training has produced artifacts/model.pt.
"""
from __future__ import annotations

import json
from pathlib import Path


def _as_lines(source: str) -> list[str]:
    """Split into lines while keeping trailing newlines as Jupyter expects."""
    s = source.strip("\n")
    parts = s.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def _code_cell(source: str, outputs: list | None = None):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": _as_lines(source),
        "outputs": outputs or [],
    }


def _md_cell(source: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _as_lines(source),
    }


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

    nb["cells"].append(_md_cell(
        "# ThermoTwin-FM demo\n"
        "\n"
        "End-to-end walk-through: generate synthetic telemetry → train → evaluate → simulate → call the API.\n"
        "\n"
        "Run from the repo root after installing with `pip install -e \".[dev,notebook]\"`."
    ))

    nb["cells"].append(_code_cell(
        """
# If the kernel started in notebooks/, hop up to the repo root so config and
# data paths resolve. No-op when run from the repo root already.
import os
from pathlib import Path
if Path.cwd().name == 'notebooks':
    os.chdir(Path.cwd().parent)
print('cwd:', Path.cwd())
"""
    ))

    nb["cells"].append(_md_cell("## 1. Generate a small synthetic telemetry dataset"))
    nb["cells"].append(_code_cell(
        """
from thermotwin.data.synthetic import SyntheticConfig, write_sample_dataset

paths = write_sample_dataset('data/sample', SyntheticConfig(n_racks=8, n_cooling_units=2, duration_days=10, seed=42))
print(paths)
"""
    ))

    nb["cells"].append(_md_cell("## 2. Inspect the data and build the layout graph"))
    nb["cells"].append(_code_cell(
        """
import pandas as pd
from thermotwin.data.ingest_m100 import load_and_prepare
from thermotwin.graphs.layout_graph import build_graph

telemetry = load_and_prepare('data/sample/telemetry.csv', cadence_minutes=5)
layout = pd.read_csv('data/sample/layout.csv')
print('Telemetry rows:', len(telemetry), '| racks:', telemetry['rack_id'].nunique())
display(telemetry.head(3))
graph = build_graph(layout)
print('Graph nodes:', graph.n_nodes, '| edges>0:', int((graph.adjacency > 0).sum() // 2))
"""
    ))

    nb["cells"].append(_md_cell("## 3. Train the forecaster\n\nWe call the same `train.main` the CLI uses."))
    nb["cells"].append(_code_cell(
        """
from thermotwin.train import main as train_main
rc = train_main(['--base-config', 'configs/base.yaml',
                 '--data-config', 'configs/data_m100.yaml',
                 '--model-config', 'configs/model_graphfm.yaml'])
print('train rc:', rc)
"""
    ))

    nb["cells"].append(_md_cell("## 4. Evaluate and plot per-horizon error"))
    nb["cells"].append(_code_cell(
        """
import json, pandas as pd, matplotlib.pyplot as plt

from thermotwin.evaluate import main as evaluate_main
evaluate_main(['--base-config', 'configs/base.yaml',
               '--data-config', 'configs/data_m100.yaml',
               '--model-config', 'configs/model_graphfm.yaml'])

report = json.load(open('artifacts/eval_report.json'))
print('Model  MAE={:.3f}  RMSE={:.3f}'.format(report['point_metrics']['mae'], report['point_metrics']['rmse']))
print('Naive  MAE={:.3f}  RMSE={:.3f}'.format(report['baseline_naive_last']['mae'], report['baseline_naive_last']['rmse']))
print('Raw 90% coverage:', round(report['calibration_raw']['empirical_coverage'], 3))
print('Conformal 90% coverage:', round(report['calibration_conformal']['empirical_coverage'], 3))

per_h = pd.DataFrame(report['per_horizon'])
per_h.plot(x='horizon_step', y=['mae', 'rmse'], marker='o', title='Forecast error by horizon step')
plt.grid(True); plt.xlabel('horizon step (5 min each)'); plt.ylabel('°C'); plt.show()
"""
    ))

    nb["cells"].append(_md_cell("## 5. What-if simulation"))
    nb["cells"].append(_code_cell(
        """
from thermotwin.simulate import main as simulate_main
simulate_main(['--base-config', 'configs/base.yaml',
               '--data-config', 'configs/data_m100.yaml',
               '--model-config', 'configs/model_graphfm.yaml'])

ranking = pd.read_csv('artifacts/simulation_ranking.csv')
display(ranking)
"""
    ))

    nb["cells"].append(_md_cell("## 6. Call the API in-process"))
    nb["cells"].append(_code_cell(
        """
from fastapi.testclient import TestClient
from thermotwin.api import create_app
from thermotwin.data.ingest_m100 import load_and_prepare
from thermotwin.features.resample_and_window import build_windows, time_block_split
from thermotwin.utils.io import load_json

app = create_app('artifacts')
client = TestClient(app)
health = client.get('/health').json()
print('Service health:', {k: health[k] for k in ('status', 'n_racks', 'context_steps', 'horizon_steps')})

manifest = load_json('artifacts/model_manifest.json')
telemetry = load_and_prepare('data/sample/telemetry.csv', cadence_minutes=5)
splits = time_block_split(telemetry, val_fraction=0.15, test_fraction=0.15)
win = build_windows(splits['test'],
                    features=manifest['feature_order'],
                    targets=manifest['target_order'],
                    controls=manifest['control_order'],
                    context_steps=int(manifest['context_steps']),
                    horizon_steps=int(manifest['horizon_steps']),
                    stride=1)
racks = sorted(telemetry['rack_id'].unique())
rack_id = racks[int(win.entity[0])]

payload = {'rack_id': rack_id,
           'context': win.context[0].tolist(),
           'controls': win.controls[0].tolist()}
r = client.post('/forecast', json=payload).json()
print('Forecast for', rack_id, ': first horizon median °C =', [round(x, 2) for x in r['median'][0]])
"""
    ))

    return nb


def main() -> int:
    out = Path("notebooks/demo.ipynb")
    out.parent.mkdir(parents=True, exist_ok=True)
    nb = build()
    out.write_text(json.dumps(nb, indent=1))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
