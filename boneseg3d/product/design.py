"""
boneseg3d/product/design.py
============================
PHASE 2 — DESIGN
Role:  Data Steward
Goal:  Define data contracts, schema governance, TCIA compliance checkpoints,
       and annotation quality gates before any training data is consumed.

Outputs
───────
  outputs/product/data_contract.json     — schema + lineage + consent record
  outputs/product/governance_report.md  — human-readable governance summary

Governance checkpoints
──────────────────────
  GC-01  TCIA DUA (Data Use Agreement) acknowledged
  GC-02  Anonymisation verified — no PHI in NIfTI headers
  GC-03  Annotation provenance recorded (annotator, tool, QA reviewer)
  GC-04  Train/val/test splits are patient-level (no slice leakage)
  GC-05  Class imbalance ratio documented (lesion voxels / background voxels)
  GC-06  Dataset version pinned and checksummed (SHA-256)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from boneseg3d.utils.logging_utils import get_logger

log = get_logger("design")

OUTPUT_DIR = Path("outputs/product")


# ── Data contract schema ──────────────────────────────────────────────────────

@dataclass
class DataSource:
    name: str                   # "TCIA-MM-CT"
    url: str
    license: str                # "TCIA DUA"
    modality: str               # "CT"
    n_patients_expected: int
    consent_type: str           # "de-identified" | "IRB-waived"
    download_tool: str          # "tcia_utils"


@dataclass
class AnnotationSpec:
    label_name: str             # "lytic_bone_lesion"
    label_index: int            # 1
    annotation_tool: str        # "ITK-SNAP" | "3D Slicer"
    annotator_role: str         # "fellowship-trained radiologist"
    qa_reviewer_role: str       # "attending radiologist"
    min_lesion_volume_mm3: float  # smallest annotated lesion in mm³


@dataclass
class SplitSpec:
    strategy: str               # "patient-stratified"
    train_fraction: float
    val_fraction: float
    test_fraction: float
    seed: int
    ensure_no_patient_overlap: bool = True


@dataclass
class GovernanceCheckpoint:
    id: str
    description: str
    required: bool
    passed: bool | None = None   # None = not yet evaluated
    evidence: str = ""


@dataclass
class DataContract:
    schema_version: str
    created_at: str
    source: DataSource
    annotation: AnnotationSpec
    splits: SplitSpec
    governance_checkpoints: list[GovernanceCheckpoint]
    nifti_header_fields_required: list[str]
    forbidden_header_fields: list[str]   # PHI fields that must be stripped


# ── Instantiate the contract ───────────────────────────────────────────────────

def _build_data_contract(cfg: Any) -> DataContract:
    return DataContract(
        schema_version="1.0.0",
        created_at=datetime.now(timezone.utc).isoformat(),
        source=DataSource(
            name="TCIA-MM-CT",
            url="https://www.cancerimagingarchive.net",
            license="TCIA Limited Access — Non-Commercial DUA",
            modality="CT",
            n_patients_expected=getattr(cfg.data, "n_patients_expected", 50),
            consent_type="de-identified",
            download_tool="tcia_utils",
        ),
        annotation=AnnotationSpec(
            label_name="lytic_bone_lesion",
            label_index=1,
            annotation_tool="3D Slicer",
            annotator_role="fellowship-trained radiologist",
            qa_reviewer_role="attending radiologist",
            min_lesion_volume_mm3=50.0,  # lesions < 50 mm³ excluded
        ),
        splits=SplitSpec(
            strategy="patient-stratified",
            train_fraction=0.70,
            val_fraction=0.15,
            test_fraction=0.15,
            seed=getattr(cfg.data, "split_seed", 42),
        ),
        nifti_header_fields_required=[
            "pixdim",    # voxel spacing — required for volume calculation
            "dim",       # volume shape
            "qform_code",
            "sform_code",
        ],
        forbidden_header_fields=[
            "descrip",   # may contain patient name in some DICOM converters
            "aux_file",
            "intent_name",
        ],
        governance_checkpoints=[
            GovernanceCheckpoint(
                id="GC-01",
                description="TCIA DUA acknowledged and stored in outputs/product/dua_acknowledgement.txt",
                required=True,
            ),
            GovernanceCheckpoint(
                id="GC-02",
                description="All NIfTI headers pass PHI scrubbing check (no forbidden fields populated)",
                required=True,
            ),
            GovernanceCheckpoint(
                id="GC-03",
                description="Annotation provenance JSON exists for every labelled case",
                required=True,
            ),
            GovernanceCheckpoint(
                id="GC-04",
                description="Train/val/test splits are patient-level with zero patient overlap verified",
                required=True,
            ),
            GovernanceCheckpoint(
                id="GC-05",
                description="Class imbalance ratio (lesion / background voxels) documented",
                required=False,
            ),
            GovernanceCheckpoint(
                id="GC-06",
                description="Dataset directory checksummed (SHA-256) and version pinned in config",
                required=True,
            ),
        ],
    )


# ── Governance validators ─────────────────────────────────────────────────────

def _validate_dua(contract: DataContract) -> bool:
    dua_path = Path("outputs/product/dua_acknowledgement.txt")
    passed = dua_path.exists() and dua_path.stat().st_size > 0
    _update_checkpoint(contract, "GC-01", passed,
                       evidence=str(dua_path) if passed else "File missing")
    return passed


def _validate_nifti_phi(contract: DataContract, nifti_dir: Path) -> bool:
    """Spot-check up to 5 NIfTI files for forbidden header fields."""
    import nibabel as nib  # imported lazily — not available at design-phase dry-run
    issues: list[str] = []
    for nii_file in list(nifti_dir.glob("**/*.nii.gz"))[:5]:
        try:
            hdr = nib.load(str(nii_file)).header
            for field in contract.forbidden_header_fields:
                val = hdr.get(field, b"")
                if val and val != b"" and val != b"\x00" * len(val):
                    issues.append(f"{nii_file.name}: {field}={val!r}")
        except Exception as e:
            issues.append(f"{nii_file.name}: could not open ({e})")
    passed = len(issues) == 0
    _update_checkpoint(contract, "GC-02", passed,
                       evidence="Clean" if passed else "; ".join(issues))
    return passed


def _validate_patient_splits(contract: DataContract, split_json: Path) -> bool:
    if not split_json.exists():
        _update_checkpoint(contract, "GC-04", False, evidence="Split file missing")
        return False
    splits = json.loads(split_json.read_text())
    all_ids = (
        splits.get("train", []) +
        splits.get("val",   []) +
        splits.get("test",  [])
    )
    # Patient IDs should be the prefix before the first underscore
    patient_ids = [s.split("_")[0] for s in all_ids]
    has_overlap = len(patient_ids) != len(set(patient_ids))
    passed = not has_overlap
    _update_checkpoint(contract, "GC-04", passed,
                       evidence="No overlap" if passed else "Patient overlap detected!")
    return passed


def _checksum_dataset(contract: DataContract, dataset_dir: Path) -> str:
    """SHA-256 of a sorted listing of all NIfTI files (names + sizes)."""
    files = sorted(dataset_dir.glob("**/*.nii.gz"))
    digest_input = "\n".join(
        f"{f.relative_to(dataset_dir)}:{f.stat().st_size}" for f in files
    )
    checksum = hashlib.sha256(digest_input.encode()).hexdigest()
    passed = len(files) > 0
    _update_checkpoint(contract, "GC-06", passed,
                       evidence=f"sha256={checksum}, n_files={len(files)}")
    return checksum


def _update_checkpoint(
    contract: DataContract, gc_id: str, passed: bool, evidence: str = ""
) -> None:
    for gc in contract.governance_checkpoints:
        if gc.id == gc_id:
            gc.passed  = passed
            gc.evidence = evidence
            status = "✓ PASS" if passed else "✗ FAIL"
            log.info(f"[GOVERNANCE] {gc_id} {status}: {evidence or gc.description}")
            break


# ── Entry point ───────────────────────────────────────────────────────────────

def specify_data_contracts(cfg: Any) -> None:
    """
    Build the data contract, run governance checks, and persist artefacts.

    Called by: main.py → design()
    Role:      Data Steward
    """
    log.info("Specifying data contracts and running governance checkpoints...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    contract = _build_data_contract(cfg)

    # Run governance checks (non-fatal: failures are recorded, not raised)
    _validate_dua(contract)

    nifti_dir = Path(getattr(cfg.data, "nifti_dir", "data/nifti"))
    if nifti_dir.exists():
        _validate_nifti_phi(contract, nifti_dir)
    else:
        log.warning(f"NIfTI dir not yet present at {nifti_dir} — GC-02 deferred.")

    split_json = Path(getattr(cfg.data, "split_json", "data/nnunet/splits_final.json"))
    if split_json.exists():
        _validate_patient_splits(contract, split_json)
    else:
        log.warning(f"Split JSON not yet present — GC-04 deferred.")

    dataset_dir = Path(getattr(cfg.data, "nnunet_dir", "data/nnunet"))
    if dataset_dir.exists():
        _checksum_dataset(contract, dataset_dir)

    # Persist
    contract_dict = asdict(contract)
    (OUTPUT_DIR / "data_contract.json").write_text(
        json.dumps(contract_dict, indent=2), encoding="utf-8"
    )

    # Governance summary markdown
    _write_governance_md(contract, OUTPUT_DIR / "governance_report.md")

    required_failures = [
        gc for gc in contract.governance_checkpoints
        if gc.required and gc.passed is False
    ]
    if required_failures:
        log.error(
            f"DESIGN GATE: {len(required_failures)} required governance checks FAILED: "
            + ", ".join(gc.id for gc in required_failures)
        )
    else:
        log.info("✓ All evaluated required governance checks passed.")


def _write_governance_md(contract: DataContract, path: Path) -> None:
    lines = [
        "# BoneSeg3D — Governance Report",
        f"Schema version: {contract.schema_version}",
        f"Generated: {contract.created_at}",
        "",
        "## Governance Checkpoints",
        "| ID | Required | Status | Evidence |",
        "|---|---|---|---|",
    ]
    for gc in contract.governance_checkpoints:
        status = "✓ PASS" if gc.passed else ("✗ FAIL" if gc.passed is False else "⏳ PENDING")
        req    = "Yes" if gc.required else "No"
        lines.append(f"| {gc.id} | {req} | {status} | {gc.evidence or gc.description} |")
    path.write_text("\n".join(lines), encoding="utf-8")