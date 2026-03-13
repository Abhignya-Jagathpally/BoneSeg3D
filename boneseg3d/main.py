"""
BoneSeg3D — Root Orchestrator
==============================
Single entry point. All logic lives in boneseg3d/.
This file contains only top-level function calls and lifecycle dispatch.

Data Product Lifecycle
──────────────────────
  IDEATE  →  DESIGN  →  OPERATIONALIZE  →  EVOLVE
  (PM)        (Steward)   (Engineer)         (Researcher)

Usage
──────────────────────
  python main.py                        # full pipeline
  python main.py --phase operationalize # single phase
  python main.py --task train_nnunet    # single named task
  python main.py --resume               # skip completed checkpoints
"""

import argparse
import asyncio
import sys
from pathlib import Path

from boneseg3d.utils.config      import load_config
from boneseg3d.utils.checkpoint  import is_complete, mark_complete
from boneseg3d.utils.logging_utils import get_logger
from boneseg3d.orchestrator      import Orchestrator

# ── Lifecycle phase modules ──────────────────────────────────────────────────
from boneseg3d.product.ideation      import define_clinical_objectives
from boneseg3d.product.design        import specify_data_contracts
from boneseg3d.product.operationalize import register_monitoring_hooks
from boneseg3d.product.evolve        import evaluate_product_iteration

# ── Agent task callables (imported by name, not by step number) ──────────────
from boneseg3d.agents.ingest_tcia_cohort       import ingest_and_structure_tcia_cohort
from boneseg3d.agents.train_nnunet_segmentor   import train_nnunet_lytic_segmentor
from boneseg3d.agents.train_swinunetr_backbone import train_swinunetr_ensemble_backbone
from boneseg3d.agents.benchmark_segmentation  import benchmark_segmentation_performance
from boneseg3d.agents.track_lesion_burden      import register_and_quantify_lesion_burden
from boneseg3d.agents.serve_clinical_dashboard import serve_clinical_reporting_dashboard

log = get_logger("main")


# ════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — IDEATION
#  Role: Data Product Manager
#  Goal: Lock objectives, use-cases, and stakeholder needs before any code runs
# ════════════════════════════════════════════════════════════════════════════

def ideate(cfg) -> None:
    """Validate and surface the clinical problem statement and use-case registry."""
    log.info("── IDEATION ──────────────────────────────────────────────────")
    if is_complete("ideation"):
        log.info("Ideation artefacts already present — skipping.")
        return

    define_clinical_objectives(cfg)
    mark_complete("ideation", metadata={"use_cases": cfg.ideation.use_cases})


# ════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — DESIGN
#  Role: Data Steward
#  Goal: Data contracts, schema governance, IRB/TCIA compliance checkpoints
# ════════════════════════════════════════════════════════════════════════════

def design(cfg) -> None:
    """Emit data specifications, governance contracts, and schema validation rules."""
    log.info("── DESIGN ────────────────────────────────────────────────────")
    if is_complete("design"):
        log.info("Design artefacts already present — skipping.")
        return

    specify_data_contracts(cfg)
    mark_complete("design", metadata={"schema_version": cfg.design.schema_version})


# ════════════════════════════════════════════════════════════════════════════
#  PHASE 3 — OPERATIONALIZE
#  Role: Data Engineer  (+ Researcher for model tasks)
#  Goal: Build, train, evaluate, register, and deploy the segmentation system
# ════════════════════════════════════════════════════════════════════════════

async def operationalize(cfg, orchestrator: Orchestrator) -> None:
    """
    Runs the full ML pipeline.

    Dependency graph
    ─────────────────
    ingest_and_structure_tcia_cohort()          ← must finish first
           ├── train_nnunet_lytic_segmentor()   ┐
           └── train_swinunetr_ensemble_backbone() ┘  ← parallel
                       └── benchmark_segmentation_performance()   ← waits for both
                                   ├── register_and_quantify_lesion_burden()  ┐
                                   └── serve_clinical_reporting_dashboard()   ┘ ← parallel
    """
    log.info("── OPERATIONALIZE ────────────────────────────────────────────")

    # ── Ingest ───────────────────────────────────────────────────────────────
    if not is_complete("ingest_and_structure_tcia_cohort"):
        await orchestrator.run_in_worktree(
            branch="feat/ingest-tcia-cohort",
            task_fn=ingest_and_structure_tcia_cohort,
            task_args=(cfg,),
            checkpoint_key="ingest_and_structure_tcia_cohort",
        )

    # ── Parallel training ─────────────────────────────────────────────────────
    training_tasks = []
    if not is_complete("train_nnunet_lytic_segmentor"):
        training_tasks.append(
            orchestrator.run_in_worktree(
                branch="feat/train-nnunet-segmentor",
                task_fn=train_nnunet_lytic_segmentor,
                task_args=(cfg,),
                checkpoint_key="train_nnunet_lytic_segmentor",
            )
        )
    if not is_complete("train_swinunetr_ensemble_backbone"):
        training_tasks.append(
            orchestrator.run_in_worktree(
                branch="feat/train-swinunetr-backbone",
                task_fn=train_swinunetr_ensemble_backbone,
                task_args=(cfg,),
                checkpoint_key="train_swinunetr_ensemble_backbone",
            )
        )
    if training_tasks:
        await asyncio.gather(*training_tasks)
        log.info("Both training backbones complete.")

    # ── Benchmark (depends on both trained models) ────────────────────────────
    if not is_complete("benchmark_segmentation_performance"):
        await orchestrator.run_in_worktree(
            branch="feat/benchmark-segmentation",
            task_fn=benchmark_segmentation_performance,
            task_args=(cfg,),
            checkpoint_key="benchmark_segmentation_performance",
        )

    # ── Parallel post-processing ──────────────────────────────────────────────
    post_tasks = []
    if not is_complete("register_and_quantify_lesion_burden"):
        post_tasks.append(
            orchestrator.run_in_worktree(
                branch="feat/track-lesion-burden",
                task_fn=register_and_quantify_lesion_burden,
                task_args=(cfg,),
                checkpoint_key="register_and_quantify_lesion_burden",
            )
        )
    if not is_complete("serve_clinical_reporting_dashboard"):
        post_tasks.append(
            orchestrator.run_in_worktree(
                branch="feat/clinical-dashboard",
                task_fn=serve_clinical_reporting_dashboard,
                task_args=(cfg,),
                checkpoint_key="serve_clinical_reporting_dashboard",
            )
        )
    if post_tasks:
        await asyncio.gather(*post_tasks)

    register_monitoring_hooks(cfg)
    mark_complete("operationalize")


# ════════════════════════════════════════════════════════════════════════════
#  PHASE 4 — EVOLVE
#  Role: Researcher
#  Goal: Performance drift detection, model versioning, ablation, deprecation
# ════════════════════════════════════════════════════════════════════════════

def evolve(cfg) -> None:
    """Compare current model version against prior checkpoint; flag regressions."""
    log.info("── EVOLVE ────────────────────────────────────────────────────")
    evaluate_product_iteration(cfg)


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args   = _parse_args()
    cfg    = load_config(args.config)
    orch   = Orchestrator(repo_root=Path(cfg.repo_root), cfg=cfg)

    phase_map = {
        "ideation":        lambda: ideate(cfg),
        "design":          lambda: design(cfg),
        "operationalize":  lambda: asyncio.run(operationalize(cfg, orch)),
        "evolve":          lambda: evolve(cfg),
    }

    # Single named task shortcut
    if args.task:
        task_map = {
            "ingest":     ingest_and_structure_tcia_cohort,
            "train_nnunet":   train_nnunet_lytic_segmentor,
            "train_monai":    train_swinunetr_ensemble_backbone,
            "benchmark":      benchmark_segmentation_performance,
            "track_burden":   register_and_quantify_lesion_burden,
            "dashboard":      serve_clinical_reporting_dashboard,
        }
        if args.task not in task_map:
            log.error(f"Unknown task '{args.task}'. Choose from: {list(task_map)}")
            sys.exit(1)
        task_map[args.task](cfg)
        return

    # Single phase shortcut
    if args.phase:
        phase_map[args.phase]()
        return

    # Full lifecycle run
    for phase_name, phase_fn in phase_map.items():
        log.info(f"Starting phase: {phase_name.upper()}")
        phase_fn()

    log.info("BoneSeg3D pipeline complete.")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BoneSeg3D — 3D Bone Lesion Segmentation for Multiple Myeloma"
    )
    p.add_argument("--config", default="configs/default.yaml",
                   help="Path to YAML config file")
    p.add_argument("--phase",  choices=["ideation","design","operationalize","evolve"],
                   default=None, help="Run a single lifecycle phase")
    p.add_argument("--task",   default=None,
                   help="Run a single named task (skips phase logic)")
    p.add_argument("--resume", action="store_true",
                   help="Skip tasks that already have a checkpoint")
    return p.parse_args()


if __name__ == "__main__":
    main()