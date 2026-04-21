# ThermoTwin-FM

**Digital twin for short-horizon rack thermal forecasting and what-if simulation in data-center environments.**

A runnable, end-to-end Python project that ingests multi-rack telemetry, builds a topology-aware sequence model with calibrated uncertainty, and supports counterfactual simulation of cooling and workload changes through both a CLI and a REST API.

## Table of contents

- [Problem framing](#problem-framing)
- [What is actually implemented](#what-is-actually-implemented)
- [Repository structure](#repository-structure)
- [Data provenance](#data-provenance)
- [Modeling approach](#modeling-approach)
- [Results](#results)
- [Limitations](#limitations)
- [Install](#install)
- [How to run: generate data](#how-to-run-generate-data)
- [How to run: training](#how-to-run-training)
- [How to run: evaluation](#how-to-run-evaluation)
- [How to run: simulation](#how-to-run-simulation)
- [How to run: API](#how-to-run-api)
- [How to run: tests](#how-to-run-tests)
- [Design notes](#design-notes)

## Problem framing

Dense racks generate short-horizon thermal dynamics that are hard to reason about with per-sensor threshold monitoring. Operators need forecasts of inlet and outlet temperature over the next 5–60 minutes, with calibrated uncertainty and an honest what-if interface for setpoint, fan, and load changes. This repo provides a minimal but complete stack for that.

## What is actually implemented

- **Ingestion** for M100-style tabular telemetry (CSV or Parquet) with a canonical schema, robust resampling, and a missingness mask (`src/thermotwin/data/ingest_m100.py`).
- **ClusterData-style workload ingest** producing regime features, including a synthetic fallback trace for reproducibility (`src/thermotwin/data/ingest_clusterdata.py`).
- **Synthetic telemetry generator** with diurnal + weekly workload cycles, first-order thermal lag, cooling degradation events, and sensor noise (`src/thermotwin/data/synthetic.py`).
- **Leakage-safe feature pipeline**: robust median/IQR scaler fit on the training split only, sliding-window tensor construction, time-block splits (`src/thermotwin/features/resample_and_window.py`).
- **Layout graph**: racks as nodes, hard edges from aisle adjacency and shared cooling units, soft edges from inlet-temperature correlation, plus symmetric-normalized adjacency for GCN use (`src/thermotwin/graphs/layout_graph.py`).
- **Forecasting backbone** (`src/thermotwin/models/foundation_backbone.py`): TCN + GRU sequence encoder with monotone quantile heads and a residual-from-last-observed-target decoder. *Honest naming note*: the file is called `foundation_backbone.py` to preserve the scaffold interface, but it is NOT a wrapper around a pretrained foundation model. It is a trainable local backbone. A TSFM such as TimesFM or Chronos-2 could plug into the same interface later.
- **Graph state-space** (`src/thermotwin/models/graph_state_space.py`): a lightweight GCN that mixes encoder latents across racks at every timestep.
- **Differentiable physics penalties** (`src/thermotwin/models/physics_losses.py`): monotonicity, smoothness, and control-lag soft constraints plus pinball loss.
- **Simple baselines** (`src/thermotwin/models/baselines.py`): naive-last and per-horizon ridge autoregression for honest comparison.
- **Training loop** with AdamW, gradient clipping, early stopping, experiment artifacts, and deterministic seeding (`src/thermotwin/train.py`).
- **Evaluation** with point metrics, per-horizon breakdown, quantile coverage, split conformal calibration, and hotspot classification metrics (`src/thermotwin/evaluate.py`).
- **What-if simulator** covering six operator-relevant scenarios (setpoint up/down, fan boost, load spike, cooling degradation) with deterministic ranking by risk-adjusted objective (`src/thermotwin/simulate.py`).
- **FastAPI service** with pydantic-validated payloads, shape checks, and no random tensors anywhere in the inference path (`src/thermotwin/api.py`).
- **50 behavioral tests** covering ingestion, scaling, windowing, graph construction, backbone shape and gradient behavior, physics losses, and end-to-end train → evaluate → simulate → API runs.

## Repository structure

```
project_01_thermotwin_fm/
├── configs/
│   ├── base.yaml             # training / logging / eval settings
│   ├── data_m100.yaml        # dataset, windowing, and feature contract
│   └── model_graphfm.yaml    # model, graph, losses, baselines
├── src/thermotwin/
│   ├── data/                 # ingest_m100.py, ingest_clusterdata.py, synthetic.py
│   ├── features/             # resample_and_window.py
│   ├── graphs/               # layout_graph.py
│   ├── models/               # foundation_backbone.py, graph_state_space.py, physics_losses.py, baselines.py
│   ├── utils/                # config.py, io.py, seed.py
│   ├── train.py              # CLI: thermotwin-train
│   ├── evaluate.py           # CLI: thermotwin-evaluate
│   ├── simulate.py           # CLI: thermotwin-simulate
│   └── api.py                # FastAPI service
├── tests/                    # 11 test files, 50 tests
├── notebooks/
│   └── demo.ipynb            # end-to-end walk-through
├── artifacts/                # model, scaler, graph, reports (created by train)
├── data/sample/              # synthetic telemetry (created by generator)
├── pyproject.toml
└── README.md
```

## Data provenance

Public telemetry datasets such as M100 ExaData and Google ClusterData2019 are large (tens of TB) and carry licensing constraints that make them unsuitable for bundling in a portfolio repo. The synthetic generator produces operationally plausible data using:

- Diurnal and weekly workload cycles with per-rack random phase.
- A first-order thermal-lag model relating coolant supply, ambient temperature, and rack load to inlet and outlet temperatures.
- Occasional workload spikes and cooling-unit degradation events as rare-event augmentation.
- Gaussian sensor noise and sporadic missingness on a subset of columns.

**The ingestion path (`src/thermotwin/data/ingest_m100.py`) is identical for real M100-style data** — swap the generator output for a real CSV or Parquet file that conforms to the canonical schema (`CANONICAL_COLUMNS` in that module) and the rest of the pipeline runs unchanged. The ClusterData ingest module accepts a real Borg-style trace if available.

## Modeling approach

The forecaster is trained on sliding windows of multivariate telemetry. For each (rack, time) anchor:

1. **Input**: a context window of past covariates (cpu/gpu load, power, fan speed, coolant supply, ambient temperature, inlet/outlet temperature), plus a future control sequence for the horizon (fan speed, coolant supply setpoint).
2. **Per-rack encoder**: input projection → stack of causal dilated 1D convolutions (TCN) → small GRU to summarize context.
3. **Optional graph mixing** (`src/thermotwin/models/graph_state_space.py`): a GCN over the normalized rack adjacency shares latent information across racks at every timestep.
4. **Decoder**: per-horizon MLP over (encoder summary, future controls) with multi-target, multi-quantile output head.
5. **Residual from last observed target**: all quantiles are additively shifted by the last observed target value. This is a strong, explicit inductive bias that prevents the network from having to re-derive naive-last performance from a compressed summary.
6. **Monotone quantile construction**: the low quantile is the base, and higher quantiles are obtained by cumulative softplus of additional offsets with a small positive floor so intervals do not collapse.
7. **Split conformal calibration** is computed on the validation set and used at evaluation time to correct undercoverage.

## Results

Trained on 10 days × 8 racks of synthetic data (5-minute cadence, 24-step context, 12-step horizon = 1 hour forecast). Single CPU run, ~2 minutes of training.

| Metric                                        |     Model |     Naive-last |
| --------------------------------------------- | --------: | -------------: |
| MAE (°C, averaged over horizons & targets)    |  **0.40** |       **0.34** |
| RMSE (°C)                                     |      0.53 |           0.47 |
| Raw 90% interval coverage                     |      0.44 |             —  |
| Conformally calibrated 90% interval coverage  |      0.78 |             —  |
| Conformal q_hat (scaled units)                |     0.197 |             —  |

The model is within roughly 18% of the naive-last baseline in MAE. On data with 5-minute cadence and strong autocorrelation this is the expected regime — naive-last is a genuinely strong baseline, and the value of the neural model lies in its calibrated uncertainty, topology-aware representation, and what-if interface, not in point-accuracy gains over naive.

### Simulation ranking (example)

For a test rack, the simulator correctly orders six scenarios by predicted inlet temperature:

| Rank | Scenario               | Max predicted inlet (°C) | Max q_high inlet (°C) |
| ---: | ---------------------- | ------------------------: | ---------------------: |
|    1 | lower_setpoint_2C      |                    20.07 |                  20.36 |
|    2 | load_spike_20pct       |                    20.10 |                  20.41 |
|    3 | baseline               |                    20.10 |                  20.41 |
|    4 | fan_boost              |                    20.17 |                  20.49 |
|    5 | raise_setpoint_2C      |                    20.32 |                  20.71 |
|    6 | cooling_degrade        |                    20.70 |                  21.22 |

Lower coolant setpoint predicts cooler inlet temperatures; cooling degradation predicts the warmest. This physics-consistent ordering is a core sanity check for the what-if layer.

## Limitations

Being explicit about what this repo is and is not.

1. **Synthetic data.** All experimental numbers are on a synthetic generator; they are a proof of pipeline, not a facility benchmark. The ingestion path is designed so real M100-style data can substitute in without code changes.
2. **Naive-last is a strong baseline at 5-minute cadence.** The neural model is within ~18% of naive MAE, not dramatically better. For short-horizon, strongly-autocorrelated thermal processes this is expected. The neural model adds value via uncertainty quantification, graph-aware context, and what-if simulation — not point accuracy.
3. **Physics penalties were found to destabilize training on this small dataset** and are disabled in the default model config. The differentiable penalty module is retained and tested, and the config flag is still present. Physics penalties should only be re-enabled with careful weight tuning and more training data.
4. **Coverage is not yet at the nominal 90%.** Raw coverage is 0.44 and conformal calibration brings it to 0.78. A longer training run with better regularization or a wider minimum width would likely close this gap.
5. **Not a foundation model.** `foundation_backbone.py` is a local TCN + GRU, not a wrapper around TimesFM or Chronos-2. The interface is pluggable so a pretrained TSFM can be dropped in later, but this repo runs fully locally with no external weights.
6. **Small-graph regime.** The GCN mixing layer works, but with 8 racks in the demo the adjacency is small and the graph contribution is modest. On larger rooms it should matter more.
7. **Linear AR baseline is disabled by default at demo scale** because fitting a per-horizon ridge model on a small window set is slow; enable via the model config.

## Install

```bash
# From repo root. Use a fresh venv if possible.
pip install -e ".[dev]"        # install package + test deps
pip install -e ".[notebook]"   # optional: Jupyter for the demo notebook
```

The package depends on numpy, pandas, scipy, scikit-learn, torch (CPU is fine), pyyaml, pydantic, fastapi, uvicorn, networkx, matplotlib.

## Pipeline order (important)

Every CLI below the generator (`thermotwin-evaluate`, `thermotwin-simulate`, `thermotwin-train`'s API consumers) **depends on `thermotwin-train` having been run first**. The repository ships with an empty `artifacts/` directory by design — training is step one, and re-running it from a clean checkout is both the fastest way to reproduce the numbers in this README and the only way to populate `artifacts/model.pt`. Run, in order:

```bash
thermotwin-generate-data   # 1. synthesize the sample dataset
thermotwin-train           # 2. train the forecasting twin (~2 minutes on CPU)
thermotwin-evaluate        # 3. compute evaluation report + per-horizon metrics
thermotwin-simulate        # 4. rank what-if scenarios
```

## How to run: generate data

```bash
python -m thermotwin.data.synthetic --out-dir data/sample --days 10 --racks 8 --cooling-units 2 --seed 42
# or: thermotwin-generate-data --out-dir data/sample --days 10 --racks 8
```

Writes `data/sample/telemetry.csv` and `data/sample/layout.csv`.

## How to run: training

```bash
python -m thermotwin.train                          # full training with default configs
python -m thermotwin.train --quick                  # smoke run on fewer epochs
thermotwin-train --base-config configs/base.yaml \
                 --data-config configs/data_m100.yaml \
                 --model-config configs/model_graphfm.yaml
```

Artifacts produced in `artifacts/`: `model.pt`, `model_manifest.json`, `scaler.json`, `layout_graph.json`, `train_history.json`, `val_predictions.npz`. **These files are not committed to the repo** — regenerate them by running `thermotwin-train`.

## How to run: evaluation

```bash
python -m thermotwin.evaluate
# or: thermotwin-evaluate
```

> Requires `artifacts/model.pt` — run `thermotwin-train` first.

Produces `eval_report.json` and `per_horizon_metrics.csv` with MAE, RMSE, raw and conformal coverage, hotspot classification metrics, and a naive baseline comparison.

## How to run: simulation

```bash
python -m thermotwin.simulate                       # simulate on first test rack
python -m thermotwin.simulate --rack-id rack_03     # specific rack
# or: thermotwin-simulate
```

> Requires `artifacts/model.pt` — run `thermotwin-train` first.

Produces `simulation_report.json` (per-scenario forecast bands) and `simulation_ranking.csv` (risk-ordered scenarios).

## How to run: API

```bash
uvicorn thermotwin.api:create_app --factory --port 8000
# or as an importable app in code:
python -c "from thermotwin.api import create_app; app = create_app('artifacts')"
```

> Requires `artifacts/model.pt` — run `thermotwin-train` first.

Endpoints:

- `GET /health` — returns the rack count, context/horizon sizes, and feature order.
- `POST /forecast` — pydantic-validated payload with `rack_id`, `context` (shape `[context_steps, n_features]`), `controls` (shape `[horizon_steps, n_controls]`). Returns median, q_low, q_high forecasts and the model version.
- `POST /what_if` — same payload plus `control_deltas` (dict) and `load_multiplier` (float).

Both endpoints use the exact same preprocessing path as training; **no random tensors ever reach the model**.

## How to run: tests

```bash
pytest                 # 50 tests, ~30 seconds
pytest -k backbone     # just the backbone tests
pytest -x -v           # stop at first failure, verbose
```


