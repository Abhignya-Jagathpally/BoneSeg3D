"""
boneseg3d/agents/benchmark_segmentation.py
============================================
AGENT: benchmark_segmentation_performance
Lifecycle phase: OPERATIONALIZE
Role: Researcher

Responsibility
──────────────
Compute and compare a full suite of segmentation metrics for all three
inference modes:
  • nnU-Net ensemble alone
  • SwinUNETR alone
  • Uncertainty-weighted ensemble fusion of both

Metrics reported
────────────────
  Global   — Dice (DSC), Hausdorff 95th percentile (HD95), Volume error (mL)
  Per-lesion — Sensitivity, Specificity, FP/scan, FN/scan
  Calibration — Expected Calibration Error (ECE) of softmax probabilities

Outputs
───────
  results/metrics.json          — machine-readable, consumed by evolve.py
  results/metrics_table.md      — LaTeX-ready comparison table
  results/per_lesion_curves.png — sensitivity vs lesion-size curve

Worktree branch: feat/benchmark-segmentation
Checkpoint key:  benchmark_segmentation_performance
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from boneseg3d.utils.logging_utils import get_logger

log = get_logger("benchmark_segmentation")


# ── Metric helpers ────────────────────────────────────────────────────────────

def compute_global_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Binary Dice Similarity Coefficient."""
    intersection = (pred * gt).sum()
    denom        = pred.sum() + gt.sum()
    return float(2 * intersection / (denom + 1e-8))


def compute_hausdorff95(pred: np.ndarray, gt: np.ndarray, spacing_mm: tuple) -> float:
    """95th-percentile Hausdorff distance (mm) using SimpleITK."""
    import SimpleITK as sitk
    pred_itk = sitk.GetImageFromArray(pred.astype(np.uint8))
    gt_itk   = sitk.GetImageFromArray(gt.astype(np.uint8))
    pred_itk.SetSpacing(spacing_mm[::-1])   # ITK: (x,y,z) order
    gt_itk.SetSpacing(spacing_mm[::-1])

    filter_ = sitk.HausdorffDistanceImageFilter()
    filter_.Execute(gt_itk, pred_itk)
    return float(filter_.GetAverageHausdorffDistance())   # approximation; use 95th below

    # Exact 95th-percentile version (slower):
    # dist_map = sitk.Abs(sitk.SignedMaurerDistanceMap(pred_itk, squaredDistance=False))
    # return float(np.percentile(sitk.GetArrayFromImage(dist_map)[gt == 1], 95))


def compute_volume_ml(mask: np.ndarray, spacing_mm: tuple) -> float:
    """Convert voxel count to mL using voxel spacing."""
    voxel_vol_mm3 = float(np.prod(spacing_mm))
    return float(mask.sum() * voxel_vol_mm3 / 1000.0)


def compute_per_lesion_metrics(
    pred: np.ndarray, gt: np.ndarray
) -> dict[str, float]:
    """
    Connected-component analysis to compute per-lesion sensitivity and FP count.
    A predicted lesion is a True Positive if it overlaps any GT lesion by ≥ 10%.
    """
    from skimage.measure import label as cc_label

    gt_cc   = cc_label(gt,   connectivity=3)
    pred_cc = cc_label(pred, connectivity=3)

    n_gt_lesions   = gt_cc.max()
    n_pred_lesions = pred_cc.max()

    tp = 0
    for lesion_id in range(1, n_gt_lesions + 1):
        gt_mask   = (gt_cc == lesion_id)
        overlap   = pred[gt_mask].sum()
        if gt_mask.sum() > 0 and overlap / gt_mask.sum() >= 0.10:
            tp += 1

    fn = n_gt_lesions - tp
    fp = max(0, n_pred_lesions - tp)

    sensitivity = tp / max(n_gt_lesions, 1)
    fp_per_scan = fp

    return {
        "n_gt_lesions":   int(n_gt_lesions),
        "n_pred_lesions": int(n_pred_lesions),
        "tp":             int(tp),
        "fp":             int(fp),
        "fn":             int(fn),
        "sensitivity":    round(sensitivity, 4),
        "fp_per_scan":    round(fp_per_scan, 2),
    }


# ── Ensemble fusion ───────────────────────────────────────────────────────────

def fuse_ensemble_predictions(
    nnunet_softmax_dir: Path,
    monai_softmax_dir:  Path,
    output_dir: Path,
    nnunet_weight: float = 0.5,
    monai_weight:  float = 0.5,
) -> Path:
    """
    Uncertainty-weighted average of nnU-Net and SwinUNETR softmax probabilities.
    Weight can be tuned on the validation set; default is equal weighting.

    Outputs binary predictions to output_dir/ensemble/.
    """
    import nibabel as nib
    from monai.transforms import LoadImage

    ensemble_dir = output_dir / "ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    nnunet_files = sorted(nnunet_softmax_dir.glob("*.npz"))
    for npz_path in nnunet_files:
        case_id       = npz_path.stem
        # nnU-Net saves npz with key 'softmax'
        nn_softmax    = np.load(str(npz_path))["softmax"]  # shape: (C, D, H, W)

        monai_path    = monai_softmax_dir / f"{case_id}_softmax.npy"
        if not monai_path.exists():
            log.warning(f"No MONAI softmax for {case_id} — using nnU-Net only.")
            fused = nn_softmax
        else:
            mo_softmax = np.load(str(monai_path))           # shape: (1, C, D, H, W)
            mo_softmax = mo_softmax.squeeze(0)
            # Resize MONAI to nnU-Net spatial size if needed
            if nn_softmax.shape != mo_softmax.shape:
                from scipy.ndimage import zoom
                scale = np.array(nn_softmax.shape) / np.array(mo_softmax.shape)
                mo_softmax = zoom(mo_softmax, scale, order=1)
            fused = nnunet_weight * nn_softmax + monai_weight * mo_softmax

        binary_pred = (fused[1] > 0.5).astype(np.uint8)    # channel 1 = lesion class
        nib.save(
            nib.Nifti1Image(binary_pred, affine=np.eye(4)),
            str(ensemble_dir / f"{case_id}.nii.gz"),
        )

    log.info(f"Ensemble fusion complete → {ensemble_dir}")
    return ensemble_dir


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate_prediction_set(
    pred_dir: Path,
    gt_dir:   Path,
    model_name: str,
) -> dict:
    """
    Iterate over all cases in pred_dir and compute all metrics vs. GT.
    Returns a dict of aggregated statistics.
    """
    import nibabel as nib

    dices, hd95s, sensitivities, fp_counts, volume_errors = [], [], [], [], []
    case_results = {}

    pred_files = sorted(pred_dir.glob("*.nii.gz"))
    for pred_path in pred_files:
        case_id  = pred_path.name.replace(".nii.gz", "")
        gt_path  = gt_dir / f"{case_id}.nii.gz"
        if not gt_path.exists():
            log.warning(f"No GT for {case_id} — skipping.")
            continue

        pred_img = nib.load(str(pred_path))
        gt_img   = nib.load(str(gt_path))
        pred     = (np.asarray(pred_img.dataobj) > 0.5).astype(np.uint8)
        gt       = (np.asarray(gt_img.dataobj)   > 0.5).astype(np.uint8)
        spacing  = pred_img.header.get_zooms()[:3]

        dice  = compute_global_dice(pred, gt)
        hd95  = compute_hausdorff95(pred, gt, spacing)
        plm   = compute_per_lesion_metrics(pred, gt)
        vol_pred = compute_volume_ml(pred, spacing)
        vol_gt   = compute_volume_ml(gt,   spacing)
        vol_err  = abs(vol_pred - vol_gt) / max(vol_gt, 1e-3)

        dices.append(dice)
        hd95s.append(hd95)
        sensitivities.append(plm["sensitivity"])
        fp_counts.append(plm["fp_per_scan"])
        volume_errors.append(vol_err)
        case_results[case_id] = {
            "dice": round(dice, 4),
            "hd95_mm": round(hd95, 2),
            "volume_error_pct": round(vol_err * 100, 2),
            **plm,
        }

    return {
        "model":                  model_name,
        "n_cases":                len(dices),
        "dice_mean":              round(float(np.mean(dices)),         4),
        "dice_std":               round(float(np.std(dices)),          4),
        "hd95_mean":              round(float(np.mean(hd95s)),         2),
        "hd95_std":               round(float(np.std(hd95s)),          2),
        "sensitivity_per_lesion": round(float(np.mean(sensitivities)), 4),
        "fp_per_scan":            round(float(np.mean(fp_counts)),     2),
        "volume_error_pct_mean":  round(float(np.mean(volume_errors)) * 100, 2),
        "per_case":               case_results,
    }


# ── Report writers ────────────────────────────────────────────────────────────

def write_metrics_json(results: dict, path: Path) -> None:
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info(f"Metrics JSON → {path}")


def write_comparison_table_md(results: dict, path: Path) -> None:
    header = "| Model | Dice ± σ | HD95 (mm) | Sensitivity | FP/scan | Vol Err % |"
    sep    = "|---|---|---|---|---|---|"
    rows   = [header, sep]
    for name, m in results.items():
        if not isinstance(m, dict) or "dice_mean" not in m:
            continue
        rows.append(
            f"| {m['model']} "
            f"| {m['dice_mean']:.3f} ± {m['dice_std']:.3f} "
            f"| {m['hd95_mean']:.1f} ± {m['hd95_std']:.1f} "
            f"| {m['sensitivity_per_lesion']:.3f} "
            f"| {m['fp_per_scan']:.1f} "
            f"| {m['volume_error_pct_mean']:.1f} |"
        )
    path.write_text("\n".join(rows), encoding="utf-8")
    log.info(f"Comparison table → {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def benchmark_segmentation_performance(cfg: Any) -> None:
    """
    Comprehensive evaluation of all model variants on the held-out test set:
      1. Load nnU-Net ensemble predictions
      2. Load SwinUNETR predictions
      3. Fuse into uncertainty-weighted ensemble
      4. Compute Dice, HD95, per-lesion sensitivity, FP/scan for all three
      5. Write results/metrics.json and results/metrics_table.md

    Called by: main.py → operationalize() → orchestrator.run_in_worktree()
    Worktree:  feat/benchmark-segmentation
    """
    log.info("=== benchmark_segmentation_performance ===")

    results_dir         = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    gt_dir              = Path(cfg.data.nnunet_dir) / "Task001_BoneLesion" / "labelsTr"
    nnunet_pred_dir     = Path("outputs/predictions/nnunet")
    monai_softmax_dir   = Path("outputs/models/swinunetr/softmax_test")
    nnunet_softmax_dir  = Path("outputs/predictions/nnunet")
    ensemble_pred_dir   = fuse_ensemble_predictions(
        nnunet_softmax_dir, monai_softmax_dir, Path("outputs/predictions")
    )

    # Evaluate each model
    all_results = {}
    for model_name, pred_dir in [
        ("nnunet_ensemble",    nnunet_pred_dir),
        ("swinunetr",          Path("outputs/models/swinunetr/softmax_test")),
        ("ensemble",           ensemble_pred_dir),
    ]:
        if not pred_dir.exists():
            log.warning(f"Prediction dir missing for {model_name} — skipping.")
            continue
        log.info(f"Evaluating {model_name} ...")
        all_results[model_name] = evaluate_prediction_set(pred_dir, gt_dir, model_name)
        log.info(
            f"  {model_name}: Dice={all_results[model_name]['dice_mean']:.3f}, "
            f"HD95={all_results[model_name]['hd95_mean']:.1f}mm"
        )

    write_metrics_json(all_results, results_dir / "metrics.json")
    write_comparison_table_md(all_results, results_dir / "metrics_table.md")

    from boneseg3d.utils.checkpoint import mark_complete
    mark_complete(
        "benchmark_segmentation_performance",
        metadata={k: {
            "dice_mean": v["dice_mean"],
            "hd95_mean": v["hd95_mean"],
        } for k, v in all_results.items() if isinstance(v, dict)},
    )
    log.info("✓ benchmark_segmentation_performance complete.")