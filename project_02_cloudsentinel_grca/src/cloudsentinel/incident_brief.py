"""Human-readable incident brief generation.

Strictly template-driven. There is no LLM call here. The brief is a markdown
document composed from:

- The anomaly window and detector scores that triggered it (cited).
- The ranked candidate drivers from graph RCA (cited with scores).
- Short, inspectable evidence paths from candidate drivers to the most
  anomalous downstream entities.
- Follow-up checks phrased as suggestions, never commands.

We intentionally use hedged language ("candidate driver", "follow-up review
recommended") and explicitly state that root cause is not proven.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cloudsentinel.features.schema_and_entity_graph import EntityGraph
from cloudsentinel.models.forecast_detector import FusedDetector
from cloudsentinel.models.graph_rca import (
    RCAConfig,
    extract_evidence_paths,
    incident_window_intensity,
    score_root_causes,
)
from cloudsentinel.utils.config import load_configs, setup_logging
from cloudsentinel.utils.io import ensure_dir, load_json, save_json
from cloudsentinel.utils.seed import set_seed

logger = logging.getLogger("cloudsentinel.incident_brief")


@dataclass
class BriefConfig:
    top_k_entities: int = 5
    top_k_metrics: int = 8
    graph_path_k: int = 3
    confidence_high_top1: float = 0.5
    confidence_med_top1: float = 0.2
    phrasing: dict | None = None


def _confidence_label(top_score: float, cfg: BriefConfig) -> str:
    phrasing = cfg.phrasing or {
        "high": "Evidence is strongly suggestive",
        "medium": "Evidence is moderately suggestive",
        "low": "Evidence is weakly suggestive",
    }
    if top_score >= cfg.confidence_high_top1:
        return phrasing["high"]
    if top_score >= cfg.confidence_med_top1:
        return phrasing["medium"]
    return phrasing["low"]


def format_evidence(evidence: list[dict]) -> str:
    """Format graph-path evidence as a bulleted markdown section."""
    if not evidence:
        return "- _No graph paths available for this window._\n"
    lines = []
    for e in evidence:
        path_str = " → ".join(e["path"]) if e.get("path") else e["candidate"]
        lines.append(f"- **{e['candidate']}** — path: `{path_str}`. {e['rationale']}")
    return "\n".join(lines) + "\n"


def build_brief(incident_context: dict, *, cfg: BriefConfig | None = None) -> dict:
    """Assemble a markdown incident brief from a ready-made context dict.

    `incident_context` keys:
        incident_id (str), window_start (str), window_end (str),
        scored (pd.DataFrame), ranked (pd.DataFrame), evidence (list[dict]),
        incident (dict | None), threshold (float).
    """
    cfg = cfg or BriefConfig()
    scored: pd.DataFrame = incident_context["scored"]
    ranked: pd.DataFrame = incident_context["ranked"]
    evidence: list[dict] = incident_context["evidence"]
    incident: dict | None = incident_context.get("incident")
    threshold: float = float(incident_context.get("threshold", 0.5))

    # Top entities and metrics by mean probability in the window.
    top_entities = (
        scored.groupby("entity_id")["prob"].mean().sort_values(ascending=False).head(cfg.top_k_entities)
    )
    top_metric_rows = (
        scored.groupby(["entity_id", "metric"])["prob"]
        .max()
        .sort_values(ascending=False)
        .head(cfg.top_k_metrics)
    )

    top_candidate_row = ranked.iloc[0] if not ranked.empty else None
    top_score = float(top_candidate_row["score"]) if top_candidate_row is not None else 0.0
    confidence = _confidence_label(top_score, cfg)

    lines: list[str] = []
    lines.append(f"# Incident Brief — {incident_context['incident_id']}")
    lines.append("")
    lines.append(
        "> **Causality disclaimer.** This brief identifies *candidate driver* entities "
        "and *suggestive patterns* based on anomaly signals and graph context. "
        "It does not prove root cause. A human operator should review the evidence "
        "and run the recommended follow-up checks before taking any action."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Window:** {incident_context['window_start']} → {incident_context['window_end']}")
    lines.append(f"- **Confidence language:** {confidence}")
    lines.append(f"- **Detector threshold applied:** {threshold:.2f}")
    if incident is not None:
        lines.append(f"- **Ground-truth incident** (synthetic label): pattern `{incident.get('pattern')}`")
    lines.append("")

    lines.append("## Candidate driver entities (graph RCA)")
    lines.append("")
    if ranked.empty:
        lines.append("_No candidates ranked._")
    else:
        lines.append("| Rank | Entity | Service | Score | Seed intensity |")
        lines.append("|---:|:---|:---|---:|---:|")
        for _, r in ranked.head(cfg.top_k_entities).iterrows():
            lines.append(
                f"| {int(r['rank'])} | {r['entity_id']} | {r.get('service', '')} | "
                f"{r['score']:.4f} | {r['seed_intensity']:.4f} |"
            )
    lines.append("")

    lines.append("## Graph evidence paths")
    lines.append("")
    lines.append(format_evidence(evidence[: cfg.graph_path_k]))

    lines.append("## Most anomalous entities in window")
    lines.append("")
    if top_entities.empty:
        lines.append("_No entities to display._")
    else:
        lines.append("| Entity | Mean anomaly probability |")
        lines.append("|:---|---:|")
        for ent, p in top_entities.items():
            lines.append(f"| {ent} | {p:.3f} |")
    lines.append("")

    lines.append("## Most anomalous metrics in window")
    lines.append("")
    if top_metric_rows.empty:
        lines.append("_No metric rows to display._")
    else:
        lines.append("| Entity | Metric | Max probability |")
        lines.append("|:---|:---|---:|")
        for (ent, met), p in top_metric_rows.items():
            lines.append(f"| {ent} | {met} | {p:.3f} |")
    lines.append("")

    lines.append("## Recommended follow-up checks")
    lines.append("")
    lines.append(
        "- Review recent deployments and configuration changes affecting the top candidate "
        f"entity **{top_candidate_row['entity_id'] if top_candidate_row is not None else 'N/A'}** "
        "and its immediate dependencies."
    )
    lines.append(
        "- Correlate the detector's z-score, residual, and change-point streams for the "
        "affected metrics over the window to distinguish short spikes from sustained shifts."
    )
    lines.append(
        "- If the top candidate is a downstream service, re-run RCA with correlation edges "
        "disabled to check whether dependency structure alone still points at the same entity."
    )
    lines.append(
        "- Inspect external out-of-domain signals (e.g., CESNET traffic patterns) to rule out "
        "environmental noise."
    )
    lines.append("")
    lines.append("---")
    lines.append(
        "_Generated by CloudSentinel-GRCA incident_brief. "
        "This artifact is advisory; no automated remediation is performed._"
    )

    body = "\n".join(lines)
    return {
        "incident_id": incident_context["incident_id"],
        "window_start": incident_context["window_start"],
        "window_end": incident_context["window_end"],
        "markdown": body,
        "top_candidate": (
            {
                "entity_id": top_candidate_row["entity_id"],
                "service": top_candidate_row.get("service"),
                "score": top_score,
            }
            if top_candidate_row is not None
            else None
        ),
        "confidence_label": confidence,
        "evidence": evidence,
    }


def generate_brief_for_incident(
    *,
    detector: FusedDetector,
    graph: EntityGraph,
    scored: pd.DataFrame,
    incident_row: dict,
    threshold: float,
    cfg: BriefConfig,
    rca_cfg: RCAConfig,
) -> dict:
    """End-to-end: given scored events + an incident, produce a brief dict."""
    start = pd.Timestamp(incident_row["start_time"])
    end = pd.Timestamp(incident_row["end_time"])
    sub = scored[(scored["timestamp"] >= start) & (scored["timestamp"] <= end)].copy()
    intensity = incident_window_intensity(scored, start_time=start, end_time=end)
    ranked = score_root_causes(graph, intensity, cfg=rca_cfg)
    evidence = extract_evidence_paths(graph, ranked, top_k_candidates=cfg.graph_path_k)
    return build_brief(
        {
            "incident_id": incident_row.get("incident_id", "inc_unknown"),
            "window_start": str(start),
            "window_end": str(end),
            "scored": sub,
            "ranked": ranked,
            "evidence": evidence,
            "incident": incident_row,
            "threshold": threshold,
        },
        cfg=cfg,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--data-config", default="configs/data_ibm.yaml")
    parser.add_argument("--model-config", default="configs/model_grca.yaml")
    parser.add_argument("--artifacts", default=None)
    parser.add_argument("--incident-id", default=None, help="If set, only brief this incident")
    args = parser.parse_args(argv)

    cfg = load_configs(args.base_config, args.data_config, args.model_config)
    if args.artifacts:
        cfg["artifacts_dir"] = args.artifacts
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    set_seed(int(cfg["seed"]))
    artifacts_dir = ensure_dir(cfg["artifacts_dir"])

    detector = FusedDetector.from_dict(load_json(artifacts_dir / "detector.json"))
    graph = EntityGraph.load(artifacts_dir / "graph.json")
    scored = pd.read_csv(artifacts_dir / "scored_events.csv", parse_dates=["timestamp"])
    incidents = pd.read_csv(cfg["dataset"]["incidents_path"])

    report = load_json(artifacts_dir / "training_report.json")
    train_end = pd.Timestamp(report["train_end"])
    incidents = incidents[pd.to_datetime(incidents["start_time"]) > train_end].copy()
    if incidents.empty:
        logger.warning("No eval-side incidents to brief")
        return 0

    brief_cfg_raw = cfg["incident_brief"]
    brief_cfg = BriefConfig(
        top_k_entities=int(brief_cfg_raw["include_top_k_entities"]),
        top_k_metrics=int(brief_cfg_raw["include_top_k_metrics"]),
        graph_path_k=int(brief_cfg_raw["include_graph_paths"]),
        phrasing=brief_cfg_raw.get("confidence_phrasing"),
    )
    rca_cfg = RCAConfig(
        personalization_alpha=float(cfg["rca"]["personalization_alpha"]),
        damping=float(cfg["rca"]["damping"]),
        max_iterations=int(cfg["rca"]["max_iterations"]),
        convergence_tol=float(cfg["rca"]["convergence_tol"]),
        top_k=int(cfg["rca"]["top_k"]),
    )
    threshold = float(cfg["alerts"]["probability_threshold"])

    targets = (
        incidents[incidents["incident_id"] == args.incident_id]
        if args.incident_id
        else incidents
    )
    briefs_dir = ensure_dir(artifacts_dir / "briefs")
    brief_index = []
    for _, row in targets.iterrows():
        brief = generate_brief_for_incident(
            detector=detector,
            graph=graph,
            scored=scored,
            incident_row=row.to_dict(),
            threshold=threshold,
            cfg=brief_cfg,
            rca_cfg=rca_cfg,
        )
        md_path = briefs_dir / f"{brief['incident_id']}.md"
        md_path.write_text(brief["markdown"])
        json_path = briefs_dir / f"{brief['incident_id']}.json"
        save_json(
            {
                "incident_id": brief["incident_id"],
                "window_start": brief["window_start"],
                "window_end": brief["window_end"],
                "top_candidate": brief["top_candidate"],
                "confidence_label": brief["confidence_label"],
                "evidence": brief["evidence"],
                "markdown_path": str(md_path),
            },
            json_path,
        )
        brief_index.append(
            {
                "incident_id": brief["incident_id"],
                "top_candidate": brief["top_candidate"],
                "confidence_label": brief["confidence_label"],
                "markdown": str(md_path),
            }
        )
    save_json(brief_index, briefs_dir / "index.json")
    logger.info("Wrote %d briefs to %s", len(brief_index), briefs_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
