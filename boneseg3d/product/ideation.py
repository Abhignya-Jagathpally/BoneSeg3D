"""
boneseg3d/product/ideation.py
==============================
PHASE 1 — IDEATION
Role:  Data Product Manager
Goal:  Lock the problem definition, use-cases, and stakeholder needs
       before any data moves or model trains.

Outputs
───────
  outputs/product/ideation_report.json   — machine-readable artefact
  outputs/product/ideation_report.md     — human-readable summary

Use-case registry
─────────────────
  UC-01  Radiologist decision support    — flag lesions on WBLDCT at point-of-care
  UC-02  Clinical trial endpoint         — automated IMWG response classification
  UC-03  Longitudinal burden tracking    — quantify total lesion volume over serial CTs
  UC-04  Research benchmark              — reproducible TCIA leaderboard entry
  UC-05  Drug efficacy monitoring        — lesion volume trajectory as surrogate endpoint
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from boneseg3d.utils.logging_utils import get_logger

log = get_logger("ideation")

OUTPUT_DIR = Path("outputs/product")


# ── Personas ─────────────────────────────────────────────────────────────────

@dataclass
class Stakeholder:
    role: str                    # e.g. "Radiologist", "Oncologist", "Clinical Trialist"
    primary_need: str
    success_metric: str
    pain_point: str


STAKEHOLDERS: list[Stakeholder] = [
    Stakeholder(
        role="Radiologist",
        primary_need="Reduce manual review time for whole-body low-dose CT",
        success_metric="< 5 min/patient review time with AI pre-read",
        pain_point="200+ axial slices per scan; lesion count varies from 1 to 50+",
    ),
    Stakeholder(
        role="Oncologist",
        primary_need="Objective response assessment without inter-reader variability",
        success_metric="CR/PR/SD/PD classification matching IMWG consensus within ±1 category",
        pain_point="RECIST is diameter-based and misses volumetric burden changes",
    ),
    Stakeholder(
        role="Clinical Trialist",
        primary_need="Automated, reproducible endpoint for MM drug trials",
        success_metric="Intraclass correlation coefficient (ICC) > 0.90 vs. expert reads",
        pain_point="No open benchmark; every trial re-validates its own AI tool",
    ),
    Stakeholder(
        role="Researcher",
        primary_need="Reproducible baseline on TCIA MM CT dataset",
        success_metric="Published Dice, HD95, per-lesion sensitivity vs. prior work",
        pain_point="No open-source 3D CT pipeline for MM bone lesion segmentation",
    ),
]


# ── Use-case registry ─────────────────────────────────────────────────────────

@dataclass
class UseCase:
    id: str
    title: str
    actor: str
    trigger: str
    expected_output: str
    priority: str          # "must-have" | "should-have" | "nice-to-have"
    kpi: str


USE_CASE_REGISTRY: list[UseCase] = [
    UseCase(
        id="UC-01",
        title="Radiologist Decision Support",
        actor="Radiologist",
        trigger="New WBLDCT scan uploaded to PACS",
        expected_output="Overlay mask + lesion count + flagged suspicious regions",
        priority="must-have",
        kpi="Sensitivity ≥ 0.85 at specificity ≥ 0.80 (per-lesion)",
    ),
    UseCase(
        id="UC-02",
        title="IMWG Response Classification",
        actor="Oncologist",
        trigger="Follow-up CT at treatment cycle N",
        expected_output="CR / sCR / PR / VGPR / SD / PD label + volumetric delta",
        priority="must-have",
        kpi="Agreement with expert consensus κ ≥ 0.65",
    ),
    UseCase(
        id="UC-03",
        title="Longitudinal Burden Tracking",
        actor="Oncologist / Trialist",
        trigger="Serial CT scans (baseline + ≥1 follow-up)",
        expected_output="Per-lesion volume trajectory JSON + waterfall plot",
        priority="must-have",
        kpi="Volume estimation error ≤ 15% vs. manual ground truth",
    ),
    UseCase(
        id="UC-04",
        title="Reproducible TCIA Benchmark",
        actor="Researcher",
        trigger="Public dataset release + code publication",
        expected_output="Standardised train/val/test splits + metrics.json leaderboard",
        priority="should-have",
        kpi="Reproducible by independent lab within 5% of reported Dice",
    ),
    UseCase(
        id="UC-05",
        title="Drug Efficacy Monitoring",
        actor="Clinical Trialist",
        trigger="End-of-treatment CTs from Phase II/III trial arm",
        expected_output="Aggregate lesion burden trajectory per treatment arm",
        priority="nice-to-have",
        kpi="AUC of lesion-burden curve correlates with PFS at ρ ≥ 0.6",
    ),
]


# ── Objectives ────────────────────────────────────────────────────────────────

@dataclass
class ClinicalObjective:
    id: str
    statement: str
    measurable_outcome: str
    linked_use_cases: list[str]


CLINICAL_OBJECTIVES: list[ClinicalObjective] = [
    ClinicalObjective(
        id="OBJ-01",
        statement=(
            "Develop a fully automated 3D segmentation model for osteolytic "
            "bone lesions on whole-body low-dose CT in multiple myeloma patients."
        ),
        measurable_outcome="Global Dice ≥ 0.70 on held-out TCIA test set",
        linked_use_cases=["UC-01", "UC-04"],
    ),
    ClinicalObjective(
        id="OBJ-02",
        statement=(
            "Enable longitudinal lesion volume tracking via deformable "
            "CT-to-CT registration across serial scans."
        ),
        measurable_outcome=(
            "Mean registration error (MRE) ≤ 3 mm; volume delta error ≤ 15%"
        ),
        linked_use_cases=["UC-02", "UC-03"],
    ),
    ClinicalObjective(
        id="OBJ-03",
        statement=(
            "Provide a clinician-facing dashboard that translates model outputs "
            "into actionable IMWG response categories."
        ),
        measurable_outcome="System usability scale (SUS) score ≥ 70 in radiologist pilot",
        linked_use_cases=["UC-01", "UC-02"],
    ),
    ClinicalObjective(
        id="OBJ-04",
        statement=(
            "Publish a reproducible open benchmark on TCIA MM CT data "
            "to accelerate community progress."
        ),
        measurable_outcome=(
            "GitHub repo with ≥50 stars within 6 months of MICCAI publication"
        ),
        linked_use_cases=["UC-04"],
    ),
]


# ── Entry point ───────────────────────────────────────────────────────────────

def define_clinical_objectives(cfg: Any) -> None:
    """
    Validate the use-case registry and objectives against config constraints,
    then persist the ideation artefact for downstream traceability.

    Called by: main.py → ideate()
    Role:      Data Product Manager
    """
    log.info("Defining clinical objectives and use-case registry...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "product_name": "BoneSeg3D",
        "version": getattr(cfg, "version", "0.1.0"),
        "stakeholders": [asdict(s) for s in STAKEHOLDERS],
        "use_cases": [asdict(uc) for uc in USE_CASE_REGISTRY],
        "objectives": [asdict(o) for o in CLINICAL_OBJECTIVES],
        "must_have_use_cases": [
            uc.id for uc in USE_CASE_REGISTRY if uc.priority == "must-have"
        ],
    }

    json_path = OUTPUT_DIR / "ideation_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info(f"Ideation report → {json_path}")

    md_path = OUTPUT_DIR / "ideation_report.md"
    _write_markdown_summary(report, md_path)
    log.info(f"Ideation summary → {md_path}")

    # Validate: every must-have UC must map to at least one objective
    must_have_ids = {uc.id for uc in USE_CASE_REGISTRY if uc.priority == "must-have"}
    covered = {uid for obj in CLINICAL_OBJECTIVES for uid in obj.linked_use_cases}
    uncovered = must_have_ids - covered
    if uncovered:
        log.warning(
            f"IDEATION WARNING: Must-have use cases without an objective: {uncovered}"
        )
    else:
        log.info("✓ All must-have use cases are covered by at least one objective.")


def _write_markdown_summary(report: dict, path: Path) -> None:
    lines = [
        f"# BoneSeg3D — Ideation Report",
        f"Generated: {report['generated_at']}",
        "",
        "## Clinical Objectives",
    ]
    for obj in report["objectives"]:
        lines += [
            f"### {obj['id']}: {obj['statement'][:60]}…",
            f"- **Outcome**: {obj['measurable_outcome']}",
            f"- **Use Cases**: {', '.join(obj['linked_use_cases'])}",
            "",
        ]
    lines += ["## Use-Case Registry", "| ID | Title | Priority | KPI |", "|---|---|---|---|"]
    for uc in report["use_cases"]:
        lines.append(f"| {uc['id']} | {uc['title']} | {uc['priority']} | {uc['kpi']} |")

    path.write_text("\n".join(lines), encoding="utf-8")