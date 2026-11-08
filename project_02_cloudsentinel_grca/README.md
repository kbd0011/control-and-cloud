# CloudSentinel-GRCA

**Graph root-cause analysis and retrospective anomaly scoring for multi-entity telemetry.**

A runnable, end-to-end Python project that ingests multi-entity telemetry, scores anomalous episodes in a recorded series with an interpretable detector ensemble, ranks candidate driver entities via seeded graph propagation over a service-dependency graph, and emits human-readable incident briefs with cited evidence — all exposed via a CLI and a REST API. All data in this repo is synthetic, produced by the bundled generator.

## Table of contents

- [Problem framing](#problem-framing)
- [What is actually implemented](#what-is-actually-implemented)
- [Repository structure](#repository-structure)
- [Data provenance](#data-provenance)
- [Anomaly detection approach](#anomaly-detection-approach)
- [Graph RCA approach](#graph-rca-approach)
- [Results](#results)
- [Causality discipline](#causality-discipline)
- [Limitations](#limitations)
- [Install](#install)
- [How to run: generate data](#how-to-run-generate-data)
- [How to run: training / fitting](#how-to-run-training--fitting)
- [How to run: detection / evaluation](#how-to-run-detection--evaluation)
- [How to run: incident brief generation](#how-to-run-incident-brief-generation)
- [How to run: API](#how-to-run-api)
- [How to run: tests](#how-to-run-tests)

## Problem framing

Modern cloud platforms emit tens of thousands of metrics per minute. Threshold-based alerting produces too much noise and fails to localize the failing entity. Operators need three things at once: (1) anomaly scores that combine several signals instead of one static threshold, (2) a way to rank likely driver entities given a dependency graph, and (3) a human-readable incident brief that points to evidence and suggests next checks without claiming to have proved root cause. This repo builds a small end-to-end version of all three and evaluates it on synthetic telemetry.

## What is actually implemented

- **Ingestion** of IBM-Cloud-style long-form and wide-form telemetry with a canonical schema, dense/sparse split based on NaN rates, and wide→long melt (`src/cloudsentinel/data/ingest_ibm_cloud.py`).
- **Out-of-domain evaluation** path for CESNET-TimeSeries24-shaped traffic data with an `ood=True` manifest flag (`src/cloudsentinel/data/ingest_cesnet.py`).
- **Synthetic telemetry generator** producing a multi-entity, multi-metric dataset with diurnal + weekly seasonality, a service dependency DAG, and labeled incidents that propagate to downstream services with lag and attenuation (`src/cloudsentinel/data/synthetic.py`).
- **Entity graph construction** with four typed edge sources — service dependency (directed), same-service replicas, same-region, and correlation-inferred — plus a row-stochastic normalizer for random-walk-based RCA (`src/cloudsentinel/features/schema_and_entity_graph.py`).
- **Synthetic incident augmentation** with four pattern types (spike / drift / saturation / correlated_failure) and deterministic seeding for reproducibility (`src/cloudsentinel/features/synthetic_incident_augment.py`).
- **Masked self-supervised encoder** (local TCN with span/random masking, not a pretrained TSFM) for representation learning and optional drift scoring (`src/cloudsentinel/models/masked_encoder.py`).
- **Detector ensemble** (`src/cloudsentinel/models/forecast_detector.py`): three interpretable streams — robust rolling z-score, AR(k) forecast residual, CUSUM-style changepoint — fused by a logistic meta-model with an isotonic layer fit on the training slice. Falls back to fixed-weight convex combination when labels are unavailable. The changepoint stream compares the 24 points before each step with the 24 points after it, so the ensemble scores a recorded series retrospectively (in batch); it is not a streaming detector.
- **Graph RCA** by seeded personalized PageRank on the reversed dependency graph: walks seeded by observed anomaly intensity concentrate mass on plausible upstream driver entities. Evidence paths are extracted via networkx shortest-path from candidates to the most anomalous downstream entities (`src/cloudsentinel/models/graph_rca.py`).
- **Training pipeline** that fits the detector on the training slice, pretrains the encoder, and exports a full set of artifacts (`src/cloudsentinel/train.py`).
- **Evaluation pipeline** reporting in-domain point detection metrics, lead-time, RCA top-k accuracy, alert-budget retention, and an OOD probability summary (`src/cloudsentinel/evaluate.py`).
- **Template-driven incident briefs** in markdown with a hard causality disclaimer, ranked candidate drivers, graph evidence paths, and follow-up suggestions — no LLM, no remediation commands (`src/cloudsentinel/incident_brief.py`).
- **FastAPI service** exposing `/health`, `/score`, `/rca`, and `/brief` endpoints with pydantic-validated real-feature inputs (`src/cloudsentinel/serve.py`).
- **50 behavioral tests** covering ingestion schemas, graph invariants, detector stream behavior, fused-probability calibration, RCA rank ordering on a synthetic chain graph, end-to-end train→evaluate→brief runs, and API payload validation. They run locally with `pytest`; see [How to run: tests](#how-to-run-tests) on CI.

## Repository structure

```
project_02_cloudsentinel_grca/
├── configs/
│   ├── base.yaml             # seeds, training, detector windows, RCA PageRank, alert budget
│   ├── data_ibm.yaml         # dataset paths, windowing, augmentation
│   └── model_grca.yaml       # encoder, fusion, graph, incident-brief template settings
├── src/cloudsentinel/
│   ├── data/                 # ingest_ibm_cloud.py, ingest_cesnet.py, synthetic.py
│   ├── features/             # schema_and_entity_graph.py, synthetic_incident_augment.py
│   ├── models/               # masked_encoder.py, forecast_detector.py, graph_rca.py
│   ├── utils/                # seed.py, config.py, io.py
│   ├── train.py              # CLI: cloudsentinel-train
│   ├── evaluate.py           # CLI: cloudsentinel-evaluate
│   ├── incident_brief.py     # CLI: cloudsentinel-brief
│   └── serve.py              # FastAPI service
├── tests/                    # 11 test files, 50 tests
├── notebooks/
│   └── demo.ipynb            # executed end-to-end walk-through
├── artifacts/                # model, graph, reports, briefs (created by CLI)
├── data/sample/              # synthetic telemetry (created by generator)
├── pyproject.toml
└── README.md
```

## Data provenance

 The IBM Cloud anomaly dataset (~117k columns, 4.5 months of data) and CESNET-TimeSeries24 are large and carry licensing constraints that make them unsuitable for bundling. The synthetic generator produces operationally plausible data using:

- Four interpretable metric families per entity: `latency_ms`, `error_rate`, `cpu_util`, `memory_util`.
- Diurnal + weekly seasonality baselines scaled per metric.
- A **service dependency DAG**: services assigned to a chain with random second-edge branching; each entity belongs to one service and one region.
- Four labeled incident patterns: `spike` (transient latency), `drift` (memory walk), `saturation` (CPU ceiling), `correlated_failure` (oscillating error rate).
- **Propagation**: an incident at a root entity ripples into downstream services with 1–3 step lag and 35–55 % attenuation on `latency_ms` / `error_rate`, producing cross-entity signatures.
- Gaussian sensor noise plus a small configurable dropout rate.
- An out-of-domain fixture (`cesnet_external.csv`) generated in the shape of CESNET-TimeSeries24 traffic, with no incidents. It is synthetic, not CESNET data.

**The ingestion modules accept real data** that conforms to the canonical long-form schema (`timestamp, entity_id, metric, value`) or wide form `timestamp, entity__metric, ...`. Swap the generator output for a real telemetry CSV / Parquet and the rest of the pipeline runs unchanged.

## Anomaly detection approach

The detector is an **ensemble of three interpretable streams** fused by a calibrated logistic meta-model:

1. **Robust rolling z-score** — `(value − rolling_median) / (1.4826 · rolling_MAD)`. Non-parametric, resistant to short outliers polluting the baseline.
2. **AR(k) forecast residual** — on each step, fit `k`-lag AR coefficients on the prior `residual_window` observations and score the current step's standardized residual. Interpretable and cheap per series.
3. **CUSUM-style changepoint** — absolute difference between the mean of the 24 points before each step and the mean of the 24 points after it, normalized by the rolling MAD. Because it uses the points after each step, scores are retrospective: they describe a completed window, and the most recent steps of a series have little or no changepoint evidence.

Each stream is squashed into `[0, 1]` with `1 − exp(−x)`. A logistic meta-model over the three squashed scores is fit on labels derived from the incident manifest (positive where the series is inside an incident window for either the root entity's driving metric or a downstream-propagated latency / error_rate series). An isotonic layer, fit on the same training slice, maps the fused score to a probability. When labels are absent, the detector falls back to a config-driven fixed-weight convex combination.

An **alert-budget cap** (`apply_alert_budget`) keeps at most `daily_budget` events per UTC day, sorted by probability.

## Graph RCA approach

The RCA ranker is **seeded personalized PageRank on the reversed dependency graph**.

- For each incident window, compute a per-entity anomaly intensity (max of the mean window probability across that entity's metrics).
- Build the personalization vector from those intensities (normalized to sum to 1).
- Walk the *reversed* entity graph adjacency: random walks that start at anomalous downstream entities concentrate mass on plausible upstream driver nodes.
- Rank by stationary probability.
- Extract short-path rationales from the top candidates back to their most anomalous downstream entities using networkx shortest-path.

This is transparent, needs no learned parameters, and matches operator intuition: "what upstream entity is most compatible with this downstream evidence?"

## Results

Trained on the first 70 % (by time) of 7 days × 24 entities × 4 metrics of synthetic data (5-minute cadence, 12 labeled incidents, 135,552 training rows) in a single CPU run. Evaluated on the remaining 30 % time slice: 57,984 scored points and the 2 incidents that start inside it. Values are from `artifacts/eval_report.json` and the executed `notebooks/demo.ipynb`.

| Metric                                                        |              Value |
| ------------------------------------------------------------- | -----------------: |
| Detection AUROC (retrospective, in-domain)                    |          **0.796** |
| Detection AUPRC (in-domain)                                   |          **0.079** |
| Positive rate (class imbalance)                               |           **1.1%** |
| Best-F1 threshold                                             | **0.073** (F1 = 0.156) |
| Alerts at the configured 0.7 threshold                        |              **0** |
| RCA: injected root entity ranked first                        |  **2 of 2 incidents** |
| OOD probability on the synthetic CESNET-shaped fixture (mean / p95 / max) | **0.037 / 0.182 / 0.486** |

What the numbers show:

- **Detection is retrospective.** The AUROC of 0.796 measures how well the ensemble marks anomalous points in a recorded series; the changepoint stream uses the 24 points after each step, so this is not the accuracy of a live detector, and no lead time is claimed.
- **Scoring, not alerting.** At the 0.7 threshold in `base.yaml` no alerts fired on the held-out slice. The best-F1 threshold is 0.073, and AUPRC of 0.079 against a 1.1 % base rate shows how hard the rare-event problem is.
- **RCA ranked the injected root first for both held-out incidents** (both `spike` incidents, 24 candidate entities each; `artifacts/rca_per_incident.csv`). Two incidents show the mechanism works on this generator; they are not an accuracy rate.
- **OOD scores stay low** on the incident-free CESNET-shaped fixture, which is a sanity check rather than an external benchmark.

## Causality discipline

Every surface — incident brief, API response, evaluation report — treats ranked entities as **candidate drivers**, never as proven root causes. The incident brief always begins with:

> **Causality disclaimer.** This brief identifies *candidate driver* entities and *suggestive patterns* based on anomaly signals and graph context. It does not prove root cause. A human operator should review the evidence and run the recommended follow-up checks before taking any action.

Follow-up suggestions are phrased as checks, not commands. There are no shell invocations or "run this" strings in the output.

## Limitations

Scope, stated plainly:

1. **Synthetic data.** All results above are on the bundled generator; they demonstrate the pipeline, not behavior on a real platform. The ingestion modules accept real data in the canonical schema without code changes.
2. **Retrospective scoring.** The changepoint stream looks 24 steps ahead, so the detector scores recorded windows in batch. A live version would need a one-sided changepoint statistic and a fresh evaluation.
3. **Small held-out set.** The held-out slice contains 2 incidents, so the RCA result is a demonstration, not an accuracy estimate. A longer synthetic run or real incident data is needed for a rate.
4. **Low AUPRC at 1 % positives, and no alerts at the default threshold.** AUPRC is better read alongside an alert-budget curve than as a single number, and the 0.7 threshold needs tuning to this score distribution.
5. **Not a foundation model.** `masked_encoder.py` is a trainable local TCN with masked reconstruction, not a wrapper around TimesFM or Chronos-2. The interface is pluggable.
6. **Graph RCA is structural, not causal.** Personalized PageRank on a dependency graph is transparent and operator-interpretable but cannot disentangle common-cause drivers from true upstream faults. The incident brief names this limitation explicitly.
7. **Fusion is label-dependent.** Without labeled incidents, the detector falls back to fixed weights; the logistic path requires at least one positive and one negative during training.
8. **Probabilities are approximate.** The isotonic layer is fit in-sample on imbalanced data and can saturate; read the probabilities with the best-F1 threshold in mind.
9. **The incident brief is template-driven and does not call an LLM.** It never invents remediation commands; it does not paraphrase evidence; it does not fetch external context at runtime.

## Install

```bash
pip install -e ".[dev]"        # package + test deps
pip install -e ".[notebook]"   # optional: Jupyter for the demo notebook
```

Dependencies: numpy, pandas, scipy, scikit-learn, torch (CPU is fine), pyyaml, pydantic, fastapi, uvicorn, networkx, matplotlib.

## How to run: generate data

```bash
python -m cloudsentinel.data.synthetic --out-dir data/sample --days 7 --entities 24 --services 6 --incidents 12 --seed 42
# or: cloudsentinel-generate-data --out-dir data/sample --days 7 --entities 24
```

Writes `telemetry.csv`, `entities.csv`, `dependencies.csv`, `incidents.csv`, and `cesnet_external.csv`.

## How to run: training / fitting

```bash
python -m cloudsentinel.train                 # full training with default configs
python -m cloudsentinel.train --quick         # reduced encoder epochs
python -m cloudsentinel.train --skip-encoder  # skip encoder pretraining entirely
# or: cloudsentinel-train
```

Artifacts written to `artifacts/`: `detector.json`, `encoder.pt`, `encoder_config.json`, `graph.json`, `training_report.json`, `detector_info.json`.

## How to run: detection / evaluation

```bash
python -m cloudsentinel.evaluate
# or: cloudsentinel-evaluate
```

Produces `scored_events.csv`, `eval_report.json`, and `rca_per_incident.csv`.

## How to run: incident brief generation

```bash
python -m cloudsentinel.incident_brief                # brief every eval-side incident
python -m cloudsentinel.incident_brief --incident-id inc_000
# or: cloudsentinel-brief
```

Produces one markdown brief and one JSON metadata file per incident under `artifacts/briefs/`, plus an `index.json` summary.

## How to run: API

```bash
uvicorn cloudsentinel.serve:create_app --factory --port 8000
# or import programmatically:
python -c "from cloudsentinel.serve import create_app; app = create_app('artifacts')"
```

Endpoints:

- `GET /health` — liveness, node count, model version
- `POST /score` — `{entity_id, metric, values: [...]}` → `{prob, zscore, residual, changepoint}` for every step of the submitted series, scored in batch (the same retrospective scoring as evaluation)
- `POST /rca` — `{window_start, window_end, per_entity_intensity: {...}}` → ranked candidate drivers and evidence paths
- `POST /brief` — `{incident_id, window_start, window_end, events: [{timestamp, entity_id, metric, value, prob}, ...]}` → markdown brief with citations

All endpoints validate payloads through pydantic and run the same preprocessing path as training. There are **no random tensors** anywhere in the inference path.

## How to run: tests

```bash
pytest                       # 50 tests, ~30 seconds
pytest -k rca                # just RCA tests
pytest -x -v                 # stop at first failure, verbose
```

The workflow in `.github/workflows/tests.yml` sits inside this project folder, and GitHub Actions only runs workflows from the repository root, so these tests have not run in CI. Run them locally as above.


