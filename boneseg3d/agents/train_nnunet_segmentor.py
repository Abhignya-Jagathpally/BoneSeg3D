"""
boneseg3d/agents/train_nnunet_segmentor.py
============================================
AGENT: train_nnunet_lytic_segmentor
Lifecycle phase: OPERATIONALIZE
Role: Researcher / Data Engineer

Responsibility
──────────────
Run nnU-Net automatic configuration, preprocessing, 5-fold cross-validation
training, and ensemble inference for lytic bone lesion segmentation.

nnU-Net handles its own hyperparameter search — our role is to configure
the environment, run the three nnU-Net CLI phases, and checkpoint each fold.

Worktree branch: feat/train-nnunet-segmentor
Checkpoint key:  train_nnunet_lytic_segmentor
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from boneseg3d.utils.checkpoint    import mark_complete, is_complete
from boneseg3d.utils.logging_utils import get_logger

log = get_logger("train_nnunet_segmentor")

TASK_ID       = "001"
TASK_NAME     = "Task001_BoneLesion"
CONFIGURATION = "3d_fullres"
TRAINER       = "nnUNetTrainerV2"
FOLDS         = list(range(5))     # 0, 1, 2, 3, 4


# ── Environment setup ─────────────────────────────────────────────────────────

def configure_nnunet_environment(cfg: Any) -> dict[str, str]:
    """
    Set nnU-Net environment variables and return them as a dict
    (also injected into subprocess env so nnU-Net CLI picks them up).
    """
    nnunet_raw       = str(Path(getattr(cfg.data, "nnunet_dir", "data/nnunet")).resolve())
    nnunet_preproc   = str(Path("data/nnunet_preprocessed").resolve())
    nnunet_results   = str(Path("outputs/nnunet_results").resolve())

    Path(nnunet_raw).mkdir(parents=True, exist_ok=True)
    Path(nnunet_preproc).mkdir(parents=True, exist_ok=True)
    Path(nnunet_results).mkdir(parents=True, exist_ok=True)

    env = {
        **os.environ,
        "nnUNet_raw_data_base": nnunet_raw,
        "nnUNet_preprocessed":  nnunet_preproc,
        "RESULTS_FOLDER":       nnunet_results,
    }
    log.info(f"nnU-Net env configured: raw={nnunet_raw}")
    return env


# ── Phase 1: Plan and preprocess ──────────────────────────────────────────────

def plan_and_preprocess_nnunet(env: dict) -> None:
    """
    Run nnUNet_plan_and_preprocess to auto-configure patch size, batch size,
    and intensity normalisation from the dataset statistics.
    """
    if is_complete("nnunet_preprocess"):
        log.info("nnU-Net preprocessing already complete — skipping.")
        return

    log.info("Running nnUNet_plan_and_preprocess ...")
    result = subprocess.run(
        [
            "nnUNet_plan_and_preprocess",
            "-t", TASK_ID,
            "--verify_dataset_integrity",
        ],
        env=env,
        capture_output=False,  # stream to stdout so logs are visible
    )
    if result.returncode != 0:
        raise RuntimeError("nnUNet_plan_and_preprocess failed.")

    mark_complete("nnunet_preprocess")
    log.info("✓ nnU-Net plan and preprocess done.")


# ── Phase 2: Training (per fold) ─────────────────────────────────────────────

def train_single_fold(fold: int, env: dict) -> None:
    """Train one cross-validation fold. Skipped if checkpoint exists."""
    ck_key = f"nnunet_train_fold{fold}"
    if is_complete(ck_key):
        log.info(f"Fold {fold} already trained — skipping.")
        return

    log.info(f"Training nnU-Net fold {fold} ...")
    t0 = time.time()
    result = subprocess.run(
        [
            "nnUNet_train",
            CONFIGURATION,
            TRAINER,
            TASK_NAME,
            str(fold),
        ],
        env=env,
        capture_output=False,
    )
    elapsed = (time.time() - t0) / 3600
    if result.returncode != 0:
        raise RuntimeError(f"nnUNet_train fold {fold} failed.")

    mark_complete(ck_key, metadata={"fold": fold, "training_hours": round(elapsed, 2)})
    log.info(f"✓ Fold {fold} complete in {elapsed:.1f}h.")


def train_all_folds(env: dict, cfg: Any) -> None:
    """Train all 5 folds sequentially (GPU memory constraint on single H100)."""
    folds = getattr(cfg.nnunet, "folds", FOLDS)
    for fold in folds:
        train_single_fold(fold, env)


# ── Phase 3: Ensemble inference on test set ───────────────────────────────────

def run_nnunet_ensemble_inference(env: dict, cfg: Any) -> Path:
    """
    Run nnUNet_predict with all trained folds to generate ensemble predictions
    on the held-out test set. Returns the prediction output directory.
    """
    test_dir    = Path(getattr(cfg.data, "nnunet_dir", "data/nnunet")) / TASK_NAME / "imagesTs"
    pred_dir    = Path("outputs/predictions/nnunet")
    pred_dir.mkdir(parents=True, exist_ok=True)

    if is_complete("nnunet_inference"):
        log.info("nnU-Net inference already complete — skipping.")
        return pred_dir

    log.info("Running nnU-Net ensemble inference on test set ...")
    result = subprocess.run(
        [
            "nnUNet_predict",
            "-i",  str(test_dir),
            "-o",  str(pred_dir),
            "-t",  TASK_ID,
            "-m",  CONFIGURATION,
            "--save_npz",   # save softmax probabilities for ensemble fusion
        ],
        env=env,
        capture_output=False,
    )
    if result.returncode != 0:
        raise RuntimeError("nnUNet_predict failed.")

    mark_complete("nnunet_inference", metadata={"pred_dir": str(pred_dir)})
    log.info(f"✓ nnU-Net inference complete → {pred_dir}")
    return pred_dir


# ── Softmax probability extraction ───────────────────────────────────────────

def export_softmax_probabilities(pred_dir: Path) -> Path:
    """
    Collect .npz softmax files from nnU-Net for downstream ensemble fusion
    with the MONAI SwinUNETR model. Returns the directory containing .npz files.
    """
    npz_files = list(pred_dir.glob("*.npz"))
    log.info(f"Found {len(npz_files)} nnU-Net softmax .npz files for ensemble fusion.")
    # Record manifest for the ensemble step
    manifest_path = pred_dir / "softmax_manifest.json"
    manifest_path.write_text(
        json.dumps([str(f) for f in npz_files], indent=2), encoding="utf-8"
    )
    return pred_dir


# ── Entry point ───────────────────────────────────────────────────────────────

def train_nnunet_lytic_segmentor(cfg: Any) -> None:
    """
    End-to-end nnU-Net training for MM lytic bone lesion segmentation:
      1. Configure nnU-Net environment variables
      2. Plan and preprocess (auto-configures patch/batch size)
      3. Train 5-fold cross-validation (checkpoint per fold)
      4. Run ensemble inference on test set
      5. Export softmax probabilities for ensemble fusion

    Called by: main.py → operationalize() → orchestrator.run_in_worktree()
    Worktree:  feat/train-nnunet-segmentor
    """
    log.info("=== train_nnunet_lytic_segmentor ===")

    env = configure_nnunet_environment(cfg)
    plan_and_preprocess_nnunet(env)
    train_all_folds(env, cfg)
    pred_dir = run_nnunet_ensemble_inference(env, cfg)
    export_softmax_probabilities(pred_dir)

    log.info("✓ train_nnunet_lytic_segmentor complete.")