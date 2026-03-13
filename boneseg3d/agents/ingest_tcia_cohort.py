"""
boneseg3d/agents/ingest_tcia_cohort.py
========================================
AGENT: ingest_and_structure_tcia_cohort
Lifecycle phase: OPERATIONALIZE
Role: Data Engineer

Responsibility
──────────────
Download the TCIA Multiple Myeloma CT collection, convert DICOM series
to NIfTI using dcm2niix, apply PHI scrubbing, and organise into an
nnU-Net Task directory with a validated dataset.json.

Worktree branch: feat/ingest-tcia-cohort
Checkpoint key:  ingest_and_structure_tcia_cohort
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from boneseg3d.utils.logging_utils import get_logger

log = get_logger("ingest_tcia_cohort")

# TCIA collection identifier
TCIA_COLLECTION = "Multiple_Myeloma_seg"

# nnU-Net task structure
TASK_ID   = "001"
TASK_NAME = "Task001_BoneLesion"


# ── TCIA download ─────────────────────────────────────────────────────────────

def _authenticate_tcia(api_key: str | None) -> "tcia_utils.tciaapi.TCIA":
    """Return an authenticated TCIA client."""
    from tcia_utils import nbia
    if api_key:
        client = nbia.getToken(user="nbia_guest", pw="", apiKey=api_key)
    else:
        client = nbia.getToken()          # anonymous guest access
    return client


def _fetch_series_manifest(collection: str, output_dir: Path) -> list[dict]:
    """Query TCIA for all CT series in the MM collection."""
    from tcia_utils import nbia
    log.info(f"Fetching series manifest for '{collection}'...")
    series_list = nbia.getSeries(collection=collection, modality="CT")
    log.info(f"Found {len(series_list)} CT series.")
    manifest_path = output_dir / "tcia_manifest.json"
    manifest_path.write_text(json.dumps(series_list, indent=2), encoding="utf-8")
    return series_list


def _download_single_series(series_uid: str, dest: Path) -> Path:
    """Download one DICOM series and return the directory path."""
    from tcia_utils import nbia
    series_dir = dest / series_uid
    series_dir.mkdir(parents=True, exist_ok=True)
    if any(series_dir.iterdir()):
        log.debug(f"Series {series_uid} already downloaded — skipping.")
        return series_dir
    nbia.downloadSeries(series_uid, path=str(series_dir))
    return series_dir


def download_tcia_collection(cfg: Any, raw_dicom_dir: Path) -> list[Path]:
    """
    Parallel download of all TCIA MM CT series.
    Returns list of per-series DICOM directories.
    """
    raw_dicom_dir.mkdir(parents=True, exist_ok=True)
    series_list  = _fetch_series_manifest(TCIA_COLLECTION, raw_dicom_dir)
    series_uids  = [s["SeriesInstanceUID"] for s in series_list]

    n_workers = getattr(cfg.data, "download_workers", 4)
    log.info(f"Downloading {len(series_uids)} series with {n_workers} workers...")

    series_dirs: list[Path] = []
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_download_single_series, uid, raw_dicom_dir): uid
            for uid in series_uids
        }
        for future in as_completed(futures):
            uid = futures[future]
            try:
                series_dirs.append(future.result())
                log.info(f"✓ Downloaded series {uid}")
            except Exception as exc:
                log.error(f"✗ Failed series {uid}: {exc}")

    return series_dirs


# ── DICOM → NIfTI conversion ──────────────────────────────────────────────────

def convert_dicom_series_to_nifti(series_dir: Path, nifti_dir: Path) -> Path | None:
    """
    Run dcm2niix on a single DICOM series directory.
    Returns the output NIfTI path or None on failure.
    """
    subject_id = series_dir.name
    out_dir    = nifti_dir / subject_id
    out_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            "dcm2niix",
            "-z", "y",           # gzip output
            "-f", "%i_%s",       # filename pattern: patientID_seriesNumber
            "-o", str(out_dir),
            str(series_dir),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log.error(f"dcm2niix failed for {series_dir.name}: {result.stderr[:200]}")
        return None

    nifti_files = list(out_dir.glob("*.nii.gz"))
    if not nifti_files:
        log.warning(f"No NIfTI output for series {series_dir.name}")
        return None

    return nifti_files[0]


def convert_all_series_to_nifti(series_dirs: list[Path], nifti_dir: Path) -> list[Path]:
    """Batch convert all DICOM series to NIfTI."""
    nifti_dir.mkdir(parents=True, exist_ok=True)
    nifti_files = []
    for series_dir in series_dirs:
        nii = convert_dicom_series_to_nifti(series_dir, nifti_dir)
        if nii:
            nifti_files.append(nii)
    log.info(f"Converted {len(nifti_files)}/{len(series_dirs)} series to NIfTI.")
    return nifti_files


# ── PHI scrubbing ─────────────────────────────────────────────────────────────

def scrub_phi_from_nifti_headers(nifti_files: list[Path]) -> None:
    """
    Zero out description fields that dcm2niix may populate with PHI.
    Modifies files in-place.
    """
    import nibabel as nib
    import numpy as np

    ZERO_FIELDS = ["descrip", "aux_file", "intent_name"]
    scrubbed = 0
    for nii_path in nifti_files:
        try:
            img = nib.load(str(nii_path))
            hdr = img.header.copy()
            changed = False
            for field in ZERO_FIELDS:
                try:
                    val = hdr[field]
                    if val and val.tobytes().strip(b"\x00"):
                        hdr[field] = b""
                        changed = True
                except (KeyError, TypeError):
                    pass
            if changed:
                clean_img = nib.Nifti1Image(np.asarray(img.dataobj), img.affine, hdr)
                nib.save(clean_img, str(nii_path))
                scrubbed += 1
        except Exception as e:
            log.warning(f"Could not scrub {nii_path.name}: {e}")
    log.info(f"PHI scrubbing: {scrubbed}/{len(nifti_files)} files updated.")


# ── nnU-Net structure ─────────────────────────────────────────────────────────

def organise_into_nnunet_dataset(
    nifti_files: list[Path],
    label_dir: Path,
    nnunet_dir: Path,
    cfg: Any,
) -> Path:
    """
    Build the nnU-Net Task directory:
      Task001_BoneLesion/
        imagesTr/  — training images  (suffix _0000.nii.gz)
        labelsTr/  — training labels
        imagesTs/  — test images
        dataset.json

    Returns path to dataset.json.
    """
    task_dir = nnunet_dir / TASK_NAME
    for sub in ["imagesTr", "labelsTr", "imagesTs"]:
        (task_dir / sub).mkdir(parents=True, exist_ok=True)

    split_seed = getattr(cfg.data, "split_seed", 42)
    import random
    rng = random.Random(split_seed)
    rng.shuffle(nifti_files)

    n         = len(nifti_files)
    n_train   = int(n * 0.70)
    n_val     = int(n * 0.15)
    train_set = nifti_files[:n_train]
    # val is handled inside nnU-Net 5-fold; test set is separate
    test_set  = nifti_files[n_train + n_val:]

    training_cases = []
    for nii in train_set:
        case_id    = nii.stem.replace(".nii", "")
        image_dest = task_dir / "imagesTr" / f"{case_id}_0000.nii.gz"
        label_src  = label_dir / f"{case_id}.nii.gz"
        label_dest = task_dir / "labelsTr" / f"{case_id}.nii.gz"
        shutil.copy2(str(nii), str(image_dest))
        if label_src.exists():
            shutil.copy2(str(label_src), str(label_dest))
        training_cases.append({
            "image": f"./imagesTr/{image_dest.name}",
            "label": f"./labelsTr/{label_dest.name}",
        })

    for nii in test_set:
        case_id    = nii.stem.replace(".nii", "")
        image_dest = task_dir / "imagesTs" / f"{case_id}_0000.nii.gz"
        shutil.copy2(str(nii), str(image_dest))

    dataset_json = _write_dataset_json(task_dir, training_cases, test_set)
    log.info(f"nnU-Net dataset → {task_dir}  ({len(train_set)} train, {len(test_set)} test)")
    return dataset_json


def _write_dataset_json(task_dir: Path, training_cases: list, test_set: list) -> Path:
    content = {
        "name": TASK_NAME,
        "description": "Osteolytic bone lesion segmentation in multiple myeloma whole-body CT",
        "tensorImageSize": "3D",
        "reference": "TCIA Multiple Myeloma CT Collection",
        "licence": "TCIA DUA",
        "release": "1.0",
        "modality": {"0": "CT"},
        "labels": {"0": "background", "1": "lytic_bone_lesion"},
        "numTraining": len(training_cases),
        "numTest": len(test_set),
        "training": training_cases,
        "test": [
            f"./imagesTs/{nii.stem.replace('.nii','')}_0000.nii.gz"
            for nii in test_set
        ],
    }
    path = task_dir / "dataset.json"
    path.write_text(json.dumps(content, indent=2), encoding="utf-8")
    return path


# ── Validation ────────────────────────────────────────────────────────────────

def validate_nnunet_dataset_integrity(dataset_json: Path) -> bool:
    """Verify every listed training image has a corresponding label."""
    data    = json.loads(dataset_json.read_text())
    task_dir = dataset_json.parent
    missing  = []
    for case in data["training"]:
        img_path   = task_dir / case["image"].lstrip("./")
        label_path = task_dir / case["label"].lstrip("./")
        if not img_path.exists():
            missing.append(str(img_path))
        if not label_path.exists():
            missing.append(str(label_path))
    if missing:
        log.error(f"Dataset integrity check FAILED — {len(missing)} missing files.")
        return False
    log.info(f"✓ Dataset integrity validated ({data['numTraining']} training cases).")
    return True


# ── Entry point ───────────────────────────────────────────────────────────────

def ingest_and_structure_tcia_cohort(cfg: Any) -> None:
    """
    Full ingestion pipeline:
      1. Download TCIA MM CT series (parallel)
      2. Convert DICOM → NIfTI via dcm2niix
      3. Scrub PHI from NIfTI headers
      4. Organise into nnU-Net Task001_BoneLesion directory
      5. Validate dataset integrity

    Called by: main.py → operationalize() → orchestrator.run_in_worktree()
    Worktree:  feat/ingest-tcia-cohort
    """
    raw_dicom_dir = Path(getattr(cfg.data, "raw_dicom_dir", "data/raw_dicom"))
    nifti_dir     = Path(getattr(cfg.data, "nifti_dir",     "data/nifti"))
    label_dir     = Path(getattr(cfg.data, "label_dir",     "data/labels"))
    nnunet_dir    = Path(getattr(cfg.data, "nnunet_dir",    "data/nnunet"))

    log.info("=== ingest_and_structure_tcia_cohort ===")

    series_dirs  = download_tcia_collection(cfg, raw_dicom_dir)
    nifti_files  = convert_all_series_to_nifti(series_dirs, nifti_dir)
    scrub_phi_from_nifti_headers(nifti_files)
    dataset_json = organise_into_nnunet_dataset(nifti_files, label_dir, nnunet_dir, cfg)

    ok = validate_nnunet_dataset_integrity(dataset_json)
    if not ok:
        raise RuntimeError("Dataset integrity check failed — cannot proceed to training.")

    log.info("✓ ingest_and_structure_tcia_cohort complete.")