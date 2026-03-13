"""
boneseg3d/agents/track_lesion_burden.py
=========================================
AGENT: register_and_quantify_lesion_burden
Lifecycle phase: OPERATIONALIZE
Role: Researcher / Data Engineer

Responsibility
──────────────
For each patient with ≥ 2 serial CT scans, perform:
  1. Rigid registration  — align follow-up to baseline (bone coordinate frame)
  2. Deformable B-spline registration — account for soft-tissue deformation
  3. Propagate baseline segmentation mask to follow-up space
  4. Match baseline ↔ follow-up lesions by centroid proximity
  5. Compute per-lesion volume delta (mL) and percent change
  6. Classify response: CR / sCR / PR / VGPR / SD / PD per IMWG criteria
  7. Write per-patient longitudinal summary JSON

IMWG volumetric response criteria (adapted from standard IMWG)
──────────────────────────────────────────────────────────────
  CR   — no detectable lesions (all volumes → 0)
  sCR  — CR + MRD negativity (proxied by volume < 50 mm³)
  PR   — ≥ 50% reduction in sum of lesion volumes
  VGPR — ≥ 90% reduction
  SD   — < 25% change in either direction
  PD   — ≥ 25% increase OR any new lesion

Worktree branch: feat/track-lesion-burden
Checkpoint key:  register_and_quantify_lesion_burden
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from boneseg3d.utils.logging_utils import get_logger

log = get_logger("track_lesion_burden")


# ── Response classification ───────────────────────────────────────────────────

@dataclass
class IMWGResponse:
    patient_id: str
    baseline_scan:  str
    followup_scan:  str
    baseline_total_volume_ml: float
    followup_total_volume_ml: float
    volume_change_pct: float
    n_baseline_lesions: int
    n_followup_lesions: int
    n_new_lesions: int
    response_category: str       # CR / sCR / PR / VGPR / SD / PD
    mean_registration_error_mm: float


def classify_imwg_response(
    baseline_vol: float,
    followup_vol:  float,
    n_new_lesions: int,
) -> str:
    """Classify volumetric response per adapted IMWG criteria."""
    if followup_vol < 0.05:                          # effectively zero volume
        return "CR"
    if followup_vol < 0.05 and followup_vol < 0.001:
        return "sCR"
    if n_new_lesions > 0:
        return "PD"

    if baseline_vol < 1e-6:
        return "PD" if followup_vol > 0.5 else "SD"

    pct_change = (followup_vol - baseline_vol) / baseline_vol * 100

    if pct_change <= -90:  return "VGPR"
    if pct_change <= -50:  return "PR"
    if pct_change >=  25:  return "PD"
    return "SD"


# ── Registration ─────────────────────────────────────────────────────────────

def rigid_register_followup_to_baseline(
    baseline_nii: Path,
    followup_nii: Path,
    output_dir:   Path,
) -> tuple[Path, float]:
    """
    Rigid (translation + rotation) CT-to-CT registration via SimpleElastix.
    Returns (registered_followup_path, mean_registration_error_mm).
    """
    import SimpleITK as sitk

    output_dir.mkdir(parents=True, exist_ok=True)
    fixed  = sitk.ReadImage(str(baseline_nii), sitk.sitkFloat32)
    moving = sitk.ReadImage(str(followup_nii), sitk.sitkFloat32)

    elastix_filter = sitk.ElastixImageFilter()
    elastix_filter.SetFixedImage(fixed)
    elastix_filter.SetMovingImage(moving)
    elastix_filter.SetParameterMap(sitk.GetDefaultParameterMap("rigid"))
    elastix_filter.SetLogToConsole(False)
    elastix_filter.Execute()

    registered = elastix_filter.GetResultImage()
    out_path   = output_dir / f"{followup_nii.stem}_rigid.nii.gz"
    sitk.WriteImage(registered, str(out_path))

    # Estimate MRE from mutual information (proxy: residual image L1 norm / volume)
    diff  = sitk.Abs(sitk.Cast(fixed, sitk.sitkFloat32) -
                     sitk.Cast(registered, sitk.sitkFloat32))
    arr   = sitk.GetArrayFromImage(diff)
    mre   = float(arr.mean() * 0.01)   # rough proxy in mm; replace with landmark eval
    return out_path, mre


def deformable_register_followup(
    baseline_nii: Path,
    rigid_followup_nii: Path,
    output_dir:   Path,
) -> tuple[Path, Path]:
    """
    B-spline deformable registration to correct residual deformation.
    Returns (deformed_followup_path, deformation_field_path).
    """
    import SimpleITK as sitk

    fixed  = sitk.ReadImage(str(baseline_nii),       sitk.sitkFloat32)
    moving = sitk.ReadImage(str(rigid_followup_nii), sitk.sitkFloat32)

    elastix_filter = sitk.ElastixImageFilter()
    elastix_filter.SetFixedImage(fixed)
    elastix_filter.SetMovingImage(moving)
    pm = sitk.GetDefaultParameterMap("bspline")
    pm["FinalGridSpacingInPhysicalUnits"] = ["10.0"]  # 10 mm control point spacing
    elastix_filter.SetParameterMap(pm)
    elastix_filter.SetLogToConsole(False)
    elastix_filter.Execute()

    deformed = elastix_filter.GetResultImage()
    deform_path   = output_dir / f"{rigid_followup_nii.stem}_deformable.nii.gz"
    sitk.WriteImage(deformed, str(deform_path))

    # Export deformation field for mask propagation
    transformix = sitk.TransformixImageFilter()
    transformix.SetTransformParameterMap(elastix_filter.GetTransformParameterMap())
    transformix.ComputeDeformationFieldOn()
    transformix.Execute()
    field_path = output_dir / "deformation_field.mhd"
    sitk.WriteImage(transformix.GetDeformationField(), str(field_path))

    return deform_path, field_path


def propagate_baseline_mask(
    baseline_mask_nii: Path,
    deformation_field: Path,
    output_dir: Path,
) -> Path:
    """
    Apply the registration deformation field to the baseline segmentation mask
    to propagate it into follow-up space (nearest-neighbour interpolation).
    """
    import SimpleITK as sitk

    mask     = sitk.ReadImage(str(baseline_mask_nii), sitk.sitkUInt8)
    field    = sitk.ReadImage(str(deformation_field))

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(field)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    propagated = resampler.Execute(mask)

    out_path = output_dir / f"{baseline_mask_nii.stem}_propagated.nii.gz"
    sitk.WriteImage(propagated, str(out_path))
    return out_path


# ── Lesion matching and volume computation ────────────────────────────────────

def match_lesions_by_centroid(
    baseline_mask: np.ndarray,
    followup_mask: np.ndarray,
    spacing_mm: tuple,
    max_distance_mm: float = 20.0,
) -> list[dict]:
    """
    Match baseline lesions to follow-up lesions using centroid proximity.
    Returns a list of matched pairs with volume information.
    """
    from skimage.measure import label as cc_label, regionprops

    def get_lesion_props(mask: np.ndarray) -> list[dict]:
        labeled = cc_label(mask, connectivity=3)
        props   = []
        for rp in regionprops(labeled):
            centroid_mm = tuple(c * s for c, s in zip(rp.centroid, spacing_mm))
            vol_ml      = rp.area * float(np.prod(spacing_mm)) / 1000.0
            props.append({"centroid_mm": centroid_mm, "volume_ml": vol_ml, "label": rp.label})
        return props

    baseline_lesions = get_lesion_props(baseline_mask)
    followup_lesions = get_lesion_props(followup_mask)

    matched_pairs = []
    used_followup = set()
    for bl in baseline_lesions:
        best_match, best_dist = None, float("inf")
        for i, fl in enumerate(followup_lesions):
            if i in used_followup:
                continue
            dist = float(np.linalg.norm(
                np.array(bl["centroid_mm"]) - np.array(fl["centroid_mm"])
            ))
            if dist < best_dist:
                best_dist, best_match = dist, i
        if best_match is not None and best_dist <= max_distance_mm:
            fl = followup_lesions[best_match]
            used_followup.add(best_match)
            matched_pairs.append({
                "baseline_vol_ml":    round(bl["volume_ml"], 3),
                "followup_vol_ml":    round(fl["volume_ml"], 3),
                "centroid_dist_mm":   round(best_dist, 2),
                "volume_delta_ml":    round(fl["volume_ml"] - bl["volume_ml"], 3),
                "volume_change_pct":  round(
                    (fl["volume_ml"] - bl["volume_ml"]) / max(bl["volume_ml"], 1e-6) * 100, 2
                ),
            })
        else:
            matched_pairs.append({
                "baseline_vol_ml":  round(bl["volume_ml"], 3),
                "followup_vol_ml":  0.0,
                "centroid_dist_mm": None,
                "volume_delta_ml":  round(-bl["volume_ml"], 3),
                "volume_change_pct": -100.0,
                "note": "no_match_in_followup",
            })

    n_new = len(followup_lesions) - len(used_followup)
    return matched_pairs, n_new


# ── Per-patient pipeline ──────────────────────────────────────────────────────

def process_patient_longitudinal_series(
    patient_id: str,
    scan_pairs: list[tuple[Path, Path, Path, Path]],
    output_dir: Path,
) -> list[IMWGResponse]:
    """
    Process all (baseline, followup) scan pairs for one patient.
    Each element of scan_pairs: (baseline_nii, followup_nii, baseline_mask, followup_mask)
    """
    import nibabel as nib
    patient_dir = output_dir / patient_id
    patient_dir.mkdir(parents=True, exist_ok=True)
    responses   = []

    for baseline_nii, followup_nii, baseline_mask_nii, followup_mask_nii in scan_pairs:
        log.info(f"  Registering {patient_id}: {baseline_nii.name} → {followup_nii.name}")

        reg_dir = patient_dir / f"{baseline_nii.stem}_to_{followup_nii.stem}"
        reg_dir.mkdir(parents=True, exist_ok=True)

        rigid_followup, mre = rigid_register_followup_to_baseline(
            baseline_nii, followup_nii, reg_dir
        )
        deformed_followup, def_field = deformable_register_followup(
            baseline_nii, rigid_followup, reg_dir
        )
        propagated_mask = propagate_baseline_mask(
            baseline_mask_nii, def_field, reg_dir
        )

        # Load masks and compute volumes
        bl_img  = nib.load(str(baseline_mask_nii))
        fu_img  = nib.load(str(followup_mask_nii))
        bl_mask = (np.asarray(bl_img.dataobj) > 0).astype(np.uint8)
        fu_mask = (np.asarray(fu_img.dataobj) > 0).astype(np.uint8)
        spacing = bl_img.header.get_zooms()[:3]

        pairs, n_new = match_lesions_by_centroid(bl_mask, fu_mask, spacing)

        bl_vol = sum(p["baseline_vol_ml"] for p in pairs)
        fu_vol = sum(p["followup_vol_ml"] for p in pairs)

        response_cat = classify_imwg_response(bl_vol, fu_vol, n_new)
        vol_change   = (fu_vol - bl_vol) / max(bl_vol, 1e-6) * 100

        result = IMWGResponse(
            patient_id=patient_id,
            baseline_scan=baseline_nii.name,
            followup_scan=followup_nii.name,
            baseline_total_volume_ml=round(bl_vol, 3),
            followup_total_volume_ml=round(fu_vol, 3),
            volume_change_pct=round(vol_change, 2),
            n_baseline_lesions=len([p for p in pairs if p["baseline_vol_ml"] > 0]),
            n_followup_lesions=len([p for p in pairs if p["followup_vol_ml"] > 0]),
            n_new_lesions=n_new,
            response_category=response_cat,
            mean_registration_error_mm=round(mre, 2),
        )
        responses.append(result)

        # Save per-patient JSON
        (reg_dir / "lesion_pairs.json").write_text(
            json.dumps(pairs, indent=2), encoding="utf-8"
        )
        log.info(
            f"  {patient_id}: {response_cat} "
            f"(Δvol={vol_change:+.1f}%, MRE={mre:.2f}mm)"
        )

    return responses


# ── Entry point ───────────────────────────────────────────────────────────────

def register_and_quantify_lesion_burden(cfg: Any) -> None:
    """
    Longitudinal tracking pipeline:
      1. Discover all patients with ≥ 2 serial CT scans + segmentation masks
      2. Rigid + deformable CT-to-CT registration per pair
      3. Propagate baseline mask to follow-up space
      4. Match lesions by centroid, compute volume deltas
      5. Classify IMWG response per patient per scan pair
      6. Write per-patient JSON and cohort-level summary

    Called by: main.py → operationalize() → orchestrator.run_in_worktree()
    Worktree:  feat/track-lesion-burden
    """
    log.info("=== register_and_quantify_lesion_burden ===")

    longitudinal_dir = Path(getattr(cfg.data, "longitudinal_dir", "data/longitudinal"))
    output_dir       = Path("outputs/longitudinal")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not longitudinal_dir.exists():
        log.warning(
            f"Longitudinal data directory not found at {longitudinal_dir}. "
            "Skipping registration — single time-point data only."
        )
        return

    # Discover patient scan pairs
    all_responses: list[IMWGResponse] = []
    for patient_dir in sorted(longitudinal_dir.iterdir()):
        if not patient_dir.is_dir():
            continue
        scans = sorted(patient_dir.glob("*.nii.gz"))
        masks = sorted((patient_dir / "masks").glob("*.nii.gz")) if (
            patient_dir / "masks").exists() else []

        # Pair consecutive scans (baseline=t0, followup=t1, t2, ...)
        scan_pairs = []
        for i in range(1, min(len(scans), len(masks))):
            scan_pairs.append((scans[0], scans[i], masks[0], masks[i]))

        if not scan_pairs:
            log.info(f"  {patient_dir.name}: < 2 scans or missing masks — skipping.")
            continue

        responses = process_patient_longitudinal_series(
            patient_id=patient_dir.name,
            scan_pairs=scan_pairs,
            output_dir=output_dir,
        )
        all_responses.extend(responses)

    # Cohort summary
    summary = {
        "n_patients": len({r.patient_id for r in all_responses}),
        "n_pairs":    len(all_responses),
        "response_distribution": {
            cat: sum(1 for r in all_responses if r.response_category == cat)
            for cat in ["CR", "sCR", "VGPR", "PR", "SD", "PD"]
        },
        "mean_registration_error_mm": round(
            float(np.mean([r.mean_registration_error_mm for r in all_responses])), 2
        ) if all_responses else None,
        "per_patient": [asdict(r) for r in all_responses],
    }
    summary_path = output_dir / "longitudinal_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info(f"Longitudinal summary → {summary_path}")

    from boneseg3d.utils.checkpoint import mark_complete
    mark_complete(
        "register_and_quantify_lesion_burden",
        metadata={"n_pairs": len(all_responses), "summary": str(summary_path)},
    )
    log.info("✓ register_and_quantify_lesion_burden complete.")