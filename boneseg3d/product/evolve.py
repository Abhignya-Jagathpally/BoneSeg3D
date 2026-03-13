"""
boneseg3d/product/evolve.py
============================
PHASE 4 — EVOLVE
Role:  Researcher
Goal:  Compare the current model iteration against the registered baseline,
       detect regressions, flag ablation candidates, and sunset deprecated models.

Decisions this module surfaces
───────────────────────────────
  PROMOTE   — new model is statistically better → promote to "production"
  HOLD      — improvement is marginal (< δ) → keep current production model
  REGRESS   — new model is worse → alert and block promotion
  ABLATE    — identify which component (backbone, loss, augmentation) drove change
  SUNSET    — mark models older than N versions for archival

Outputs
───────
  outputs/product/evolution_report.json
  outputs/product/evolution_report.md
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from boneseg3d.utils.logging_utils import get_logger

log = get_logger("evolve")

OUTPUT_DIR = Path("outputs/product")

# Minimum improvement required to justify a model swap (research taste)
MIN_DICE_IMPROVEMENT    = 0.02   # absolute Dice points
MIN_HD95_IMPROVEMENT_MM = 1.0    # mm reduction in HD95
MAX_VERSIONS_RETAINED   = 3      # sunset anything older


@dataclass
class ModelSnapshot:
    version: str
    val_dice: float
    val_hd95_mm: float
    per_lesion_sensitivity: float
    false_positives_per_scan: float
    training_hours: float
    backbone: str
    notes: str = ""


@dataclass
class EvolutionDecision:
    decision: str              # "PROMOTE" | "HOLD" | "REGRESS" | "ABLATE"
    reason: str
    current_version: str
    candidate_version: str
    delta_dice: float
    delta_hd95: float
    recommended_action: str


# ── Comparison logic ──────────────────────────────────────────────────────────

def _compare_snapshots(
    baseline: ModelSnapshot,
    candidate: ModelSnapshot,
) -> EvolutionDecision:
    delta_dice = candidate.val_dice - baseline.val_dice
    delta_hd95 = baseline.val_hd95_mm - candidate.val_hd95_mm  # positive = improved

    if delta_dice < -0.01 or delta_hd95 < -2.0:
        decision = "REGRESS"
        reason = (
            f"Candidate degrades Dice by {delta_dice:+.3f} and "
            f"HD95 by {-delta_hd95:+.1f} mm vs. baseline."
        )
        action = "Block promotion. Investigate training instability or data leakage."

    elif delta_dice >= MIN_DICE_IMPROVEMENT or delta_hd95 >= MIN_HD95_IMPROVEMENT_MM:
        decision = "PROMOTE"
        reason = (
            f"Candidate improves Dice by {delta_dice:+.3f} "
            f"and HD95 by {delta_hd95:+.1f} mm — exceeds promotion threshold."
        )
        action = (
            "Update model_registry.json: set candidate to 'production', "
            "baseline to 'deprecated'."
        )

    elif 0 <= delta_dice < MIN_DICE_IMPROVEMENT:
        decision = "HOLD"
        reason = (
            f"Improvement ({delta_dice:+.3f} Dice) is below the "
            f"minimum threshold ({MIN_DICE_IMPROVEMENT}). Not worth the swap cost."
        )
        action = (
            "Publish as workshop/blog post. Start ablation to find higher-impact lever. "
            "Do NOT replace production model."
        )

    else:
        decision = "ABLATE"
        reason = "Mixed signals: Dice improved but HD95 regressed (or vice-versa)."
        action = (
            "Run ablation: (1) freeze backbone, vary loss. "
            "(2) freeze loss, vary augmentation. "
            "(3) identify which component drives HD95 regression."
        )

    return EvolutionDecision(
        decision=decision,
        reason=reason,
        current_version=baseline.version,
        candidate_version=candidate.version,
        delta_dice=round(delta_dice, 4),
        delta_hd95=round(delta_hd95, 2),
        recommended_action=action,
    )


# ── Sunset logic ──────────────────────────────────────────────────────────────

def _flag_for_sunset(registry_path: Path) -> list[str]:
    """Return version IDs that should be archived (oldest beyond retention window)."""
    if not registry_path.exists():
        return []
    registry = json.loads(registry_path.read_text())
    entries  = registry.get("entries", [])
    # Sort by registration date, oldest first
    sorted_entries = sorted(entries, key=lambda e: e.get("registered_at", ""))
    deprecated = [
        e["model_id"]
        for e in sorted_entries[: max(0, len(sorted_entries) - MAX_VERSIONS_RETAINED)]
    ]
    return deprecated


# ── Entry point ───────────────────────────────────────────────────────────────

def evaluate_product_iteration(cfg: Any) -> None:
    """
    Load the latest metrics, compare against the registered baseline,
    emit a PROMOTE / HOLD / REGRESS / ABLATE decision, and flag sunset candidates.

    Called by: main.py → evolve()
    Role:      Researcher
    """
    log.info("Evaluating product iteration against registered baseline...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = Path("results/metrics.json")
    if not metrics_path.exists():
        log.warning(
            "results/metrics.json not found — run benchmark_segmentation_performance() first."
        )
        return

    metrics  = json.loads(metrics_path.read_text())
    registry_path = OUTPUT_DIR / "model_registry.json"

    # Build candidate snapshot from latest benchmark results
    candidate = ModelSnapshot(
        version=getattr(cfg, "version", "0.1.0"),
        val_dice=metrics.get("ensemble", {}).get("dice_mean", 0.0),
        val_hd95_mm=metrics.get("ensemble", {}).get("hd95_mean", 999.0),
        per_lesion_sensitivity=metrics.get("ensemble", {}).get("sensitivity_per_lesion", 0.0),
        false_positives_per_scan=metrics.get("ensemble", {}).get("fp_per_scan", 99.0),
        training_hours=metrics.get("training_hours_total", 0.0),
        backbone="ensemble",
        notes=metrics.get("notes", ""),
    )

    # Try to load the current production baseline
    baseline_path = OUTPUT_DIR / "baseline_snapshot.json"
    if baseline_path.exists():
        baseline = ModelSnapshot(**json.loads(baseline_path.read_text()))
        decision = _compare_snapshots(baseline, candidate)
        log.info(
            f"[EVOLVE] Decision: {decision.decision} — {decision.reason}"
        )
    else:
        # First run — no baseline exists yet; candidate becomes the baseline
        decision = EvolutionDecision(
            decision="PROMOTE",
            reason="No prior baseline found. Candidate is the first registered model.",
            current_version="none",
            candidate_version=candidate.version,
            delta_dice=0.0,
            delta_hd95=0.0,
            recommended_action=(
                "Save this run as the baseline snapshot. "
                "Future iterations will be compared against it."
            ),
        )
        log.info("[EVOLVE] First run — saving candidate as baseline.")
        baseline_path.write_text(
            json.dumps(asdict(candidate), indent=2), encoding="utf-8"
        )

    # Sunset check
    sunset_candidates = _flag_for_sunset(registry_path)
    if sunset_candidates:
        log.info(f"[EVOLVE] Models flagged for sunset: {sunset_candidates}")

    # Persist evolution report
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate":    asdict(candidate),
        "decision":     asdict(decision),
        "sunset_candidates": sunset_candidates,
        "thresholds": {
            "min_dice_improvement": MIN_DICE_IMPROVEMENT,
            "min_hd95_improvement_mm": MIN_HD95_IMPROVEMENT_MM,
            "max_versions_retained": MAX_VERSIONS_RETAINED,
        },
    }
    report_path = OUTPUT_DIR / "evolution_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info(f"[EVOLVE] Evolution report → {report_path}")

    _write_evolution_md(report, OUTPUT_DIR / "evolution_report.md")
    log.info(f"[EVOLVE] ✓ Iteration evaluation complete. Decision: {decision.decision}")


def _write_evolution_md(report: dict, path: Path) -> None:
    d = report["decision"]
    lines = [
        "# BoneSeg3D — Evolution Report",
        f"Generated: {report['generated_at']}",
        "",
        f"## Decision: **{d['decision']}**",
        f"> {d['reason']}",
        "",
        f"**Recommended action**: {d['recommended_action']}",
        "",
        "| Metric | Delta |",
        "|---|---|",
        f"| Dice | {d['delta_dice']:+.4f} |",
        f"| HD95 (mm) | {d['delta_hd95']:+.2f} |",
        "",
    ]
    if report["sunset_candidates"]:
        lines += [
            "## Models Flagged for Sunset",
            *[f"- {m}" for m in report["sunset_candidates"]],
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")