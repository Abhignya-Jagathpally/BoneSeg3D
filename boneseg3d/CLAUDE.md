# BoneSeg3D — 3D Bone Lesion Detection in Multiple Myeloma CT Scans
## CLAUDE.md — Agent Orchestration & Development Guide

> **Model**: Claude Sonnet / Opus via Claude Code
> **Compute**: UNT H100 GPU
> **Stack**: MONAI · nnU-Net · PyTorch · SimpleElastix · Streamlit
> **Timeline**: 2–3 weeks
> **Repo root**: all `main()` calls live in `boneseg3d/main.py`

---

## 0. Research Context & Motivation

### Problem Statement
Multiple myeloma (MM) is a plasma cell malignancy that causes osteolytic (lytic) bone lesions detectable on whole-body low-dose CT (WBLDCT). Radiologists must manually review hundreds of axial slices per patient — a time-intensive, error-prone process. Automated segmentation can accelerate diagnosis, response monitoring, and clinical trial endpoints.

### Related Work (What Already Exists)
| Paper | Method | Limitation |
|---|---|---|
| Faghani et al. 2023 (*Skeletal Radiology*) | 2D UNet + YOLO two-step | 2D slice-based; no volumetric tracking |
| van Leeuwen et al. 2025 (*Springer/BNAIC*) | CNN false-positive classifier | Post-hoc FP reduction, not end-to-end |
| Xu et al. 2018 (*Contrast Media & Mol. Imaging*) | W-Net on PET/CT | Requires PET; not CT-only |
| nnU-Net BMS (*Academic Radiology*, 2025) | nnU-Net on WB-MRI | MRI-only, no CT lesion detection |
| VISTA3D (*CVPR 2025*) | 127-class 3D foundation model | Not fine-tuned for MM lytic lesions |
| Zhao et al. MMNet 2024 (*QIMS*) | Multiscale encoder-decoder on MRI | MRI only, no CT pipeline |

### Identified Gaps (Your Niche)
1. **No open-source, end-to-end CT pipeline** that combines DICOM ingestion → nnU-Net/MONAI 3D segmentation → per-lesion volumetric metrics → longitudinal registration-based tracking → clinical dashboard. Each prior work addresses at most 2 of these.
2. **False-positive suppression is unsolved in 3D**. Current FP reducers operate on 2D patches post-hoc. A 3D attention-gated or confidence-calibrated approach is missing.
3. **Longitudinal response tracking (serial CT)** with deep-learning-driven deformable registration has not been combined with automated lesion segmentation for MM. RECIST 1.1 remains manual.
4. **VISTA3D zero-shot benchmark on MM CT** has not been published. Fine-tuning VISTA3D on TCIA MM data constitutes a clear contribution.
5. **No reproducible benchmark** on TCIA MM datasets with standardized Dice, Hausdorff, per-lesion sensitivity/specificity metrics — making it impossible to compare methods.

### Unique Approach / Out-of-the-Box Angle
- **Comparative advantage lever**: Your pipeline is a *data + systems* contribution, not just a new architecture. This is hard to replicate and clinically impactful.
- **Dual-backbone ensemble**: nnU-Net (automated tuning, strong baseline) + MONAI SwinUNETR (long-range context via Transformer attention) fused via uncertainty-weighted voting — outperforms either alone and is novel for MM-CT.
- **VISTA3D zero-shot → fine-tune ablation**: First published study benchmarking the CVPR 2025 VISTA3D model on MM lesions — timely, fast-follows a hot paper.
- **Lesion burden trajectory**: Frame response assessment as a *temporal regression problem* — predict lesion volume at t+1 given CT at t, enabling prognosis beyond binary RECIST categories.
- **Reproducible TCIA benchmark**: Open leaderboard contribution (metrics, splits, JSON) that the community can build on — high citation value.

---

## 1. Workflow Architecture

```
main.py
  ├── step1_data_pipeline()       # DICOM → NIfTI → nnU-Net structure
  ├── step2_nnunet_baseline()     # nnU-Net plan, preprocess, train
  ├── step3_monai_model()         # SwinUNETR + residual 3D U-Net, sliding window
  ├── step4_evaluation()          # Dice, Hausdorff, per-lesion sensitivity
  ├── step5_longitudinal()        # SimpleElastix registration + volume delta
  └── step6_dashboard()           # Streamlit DICOM → inference → 3D render
```

**Subagent strategy**: each step runs in its own Claude Code worktree branch. Agents work in parallel where dependencies allow (steps 2 & 3 can run concurrently after step 1).

---

## 2. Subagent & Worktree Execution

### Parallel Worktree Branches
```bash
# Create isolated worktrees per task
git worktree add ../boneseg3d-data    -b feat/data-pipeline
git worktree add ../boneseg3d-nnunet  -b feat/nnunet-baseline
git worktree add ../boneseg3d-monai   -b feat/monai-model
git worktree add ../boneseg3d-eval    -b feat/evaluation
git worktree add ../boneseg3d-long    -b feat/longitudinal
git worktree add ../boneseg3d-dash    -b feat/dashboard
```

### Subagent Dispatch (in CLAUDE.md tasks)
Each subagent receives a scoped prompt referencing its worktree. The orchestrating agent in `main.py` calls:
```python
# Pseudocode: orchestrator dispatches via Task tool
dispatch_subagent("feat/data-pipeline",   task="step1_data_pipeline")
dispatch_subagent("feat/nnunet-baseline", task="step2_nnunet_baseline")
dispatch_subagent("feat/monai-model",     task="step3_monai_model")
# step4 depends on step2 & step3 artifacts → wait for both
dispatch_subagent("feat/evaluation",      task="step4_evaluation",    depends=["step2","step3"])
dispatch_subagent("feat/longitudinal",    task="step5_longitudinal",  depends=["step3"])
dispatch_subagent("feat/dashboard",       task="step6_dashboard",     depends=["step3"])
```

### Rules for Subagents
- Each subagent MUST write a `CHECKPOINT.json` upon successful completion (see §4).
- Subagents MUST NOT modify files outside their worktree directory.
- Subagents MUST log GPU utilization and peak memory to `logs/gpu_<step>.log`.
- If a subagent fails, the orchestrator retries once, then surfaces the error.

---

## 3. Root `main.py` — Clean Orchestration Entry Point

```python
# boneseg3d/main.py
"""
BoneSeg3D — Main Orchestration Entry Point
All logic lives in step modules. This file contains only top-level calls.
"""

from boneseg3d.steps.step1_data import step1_data_pipeline
from boneseg3d.steps.step2_nnunet import step2_nnunet_baseline
from boneseg3d.steps.step3_monai import step3_monai_model
from boneseg3d.steps.step4_eval import step4_evaluation
from boneseg3d.steps.step5_longitudinal import step5_longitudinal_tracking
from boneseg3d.steps.step6_dashboard import step6_clinical_dashboard
from boneseg3d.utils.checkpoint import load_checkpoint, save_checkpoint
from boneseg3d.utils.config import load_config
import argparse


def main():
    args = parse_args()
    cfg  = load_config(args.config)  # configs/default.yaml

    # ── Step 1: Data pipeline (DICOM → NIfTI → nnU-Net layout) ──────────
    if not load_checkpoint("step1"):
        step1_data_pipeline(cfg)
        save_checkpoint("step1")

    # ── Step 2: nnU-Net baseline (runs in parallel with step 3) ──────────
    if not load_checkpoint("step2"):
        step2_nnunet_baseline(cfg)
        save_checkpoint("step2")

    # ── Step 3: MONAI SwinUNETR + residual 3D U-Net ───────────────────────
    if not load_checkpoint("step3"):
        step3_monai_model(cfg)
        save_checkpoint("step3")

    # ── Step 4: Evaluation metrics (Dice, Hausdorff, per-lesion) ─────────
    if not load_checkpoint("step4"):
        step4_evaluation(cfg)
        save_checkpoint("step4")

    # ── Step 5: Longitudinal registration & volume tracking ───────────────
    if not load_checkpoint("step5"):
        step5_longitudinal_tracking(cfg)
        save_checkpoint("step5")

    # ── Step 6: Streamlit clinical dashboard ──────────────────────────────
    if not load_checkpoint("step6"):
        step6_clinical_dashboard(cfg)
        save_checkpoint("step6")


def parse_args():
    parser = argparse.ArgumentParser(description="BoneSeg3D Pipeline")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints")
    parser.add_argument("--step",   type=int, default=None, help="Run single step only")
    return parser.parse_args()


if __name__ == "__main__":
    main()
```

---

## 4. Checkpoint System

Every major step writes a JSON checkpoint to `checkpoints/` so the pipeline can resume without re-running completed steps (critical for long H100 training runs).

```python
# boneseg3d/utils/checkpoint.py
import json, os
from pathlib import Path
from datetime import datetime

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

def save_checkpoint(step_name: str, metadata: dict = None):
    record = {
        "step": step_name,
        "completed_at": datetime.utcnow().isoformat(),
        "metadata": metadata or {}
    }
    path = CHECKPOINT_DIR / f"{step_name}.json"
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"[CHECKPOINT] ✓ {step_name} saved → {path}")

def load_checkpoint(step_name: str) -> bool:
    path = CHECKPOINT_DIR / f"{step_name}.json"
    return path.exists()
```

### Checkpoint Gates (when to write them)
| Checkpoint | Written after |
|---|---|
| `step1` | NIfTI conversion + nnU-Net JSON config verified |
| `step2_preprocess` | `nnUNet_plan_and_preprocess` completes |
| `step2_train_fold{0..4}` | Each training fold completes |
| `step2` | Best model selected and saved |
| `step3_epoch{N}` | Every 50 training epochs (MONAI) |
| `step3` | Final MONAI model exported to TorchScript |
| `step4` | All metrics computed and saved to `results/metrics.json` |
| `step5` | Registration + delta volumes computed for all serial pairs |
| `step6` | Dashboard verified running on localhost |

---

## 5. Step-by-Step Implementation Guide

### Step 1 — DICOM Pipeline
**Prompt for subagent:**
```
Write a Python pipeline to download TCIA DICOM CT data for multiple myeloma,
convert to NIfTI format using dcm2niix, and organize into nnU-Net dataset
structure (imagesTr/, labelsTr/, imagesTs/) with dataset.json config.
Include TCIA REST API authentication, parallel download with tqdm, and
integrity checks (file size, dcm2niix exit code).
Target dataset: TCIA "Multiple Myeloma" collection.
```
**Key libraries**: `tcia_utils`, `dcm2niix`, `nibabel`, `concurrent.futures`
**Checkpoint**: written after `dataset.json` validates against nnU-Net schema

### Step 2 — nnU-Net Baseline
**Prompt for subagent:**
```
Set up nnU-Net training pipeline for 3D bone lesion segmentation on MM CT data.
Run nnUNet_plan_and_preprocess for Task001_BoneLesion with 3d_fullres config.
Train 5-fold cross-validation. Log GPU utilization. Save best model checkpoint.
Export inference predictions to predictions/ folder.
```
**Commands**:
```bash
nnUNet_plan_and_preprocess -t 1 --verify_dataset_integrity
nnUNet_train 3d_fullres nnUNetTrainerV2 Task001_BoneLesion 0  # per fold
nnUNet_predict -i imagesTs/ -o predictions/ -t 1 -m 3d_fullres
```
**Checkpoint**: per-fold + final ensemble

### Step 3 — MONAI Custom Model
**Prompt for subagent:**
```
Build a 3D SwinUNETR (from MONAI) for bone lesion segmentation. Also implement
a 3D U-Net with residual blocks as a secondary backbone. Use sliding window
inference (roi_size=128^3, sw_batch_size=4, overlap=0.5) for full CT volumes.
Include:
  - DiceCELoss with class weighting for lesion imbalance
  - CosineAnnealingLR scheduler
  - Mixed precision (torch.cuda.amp)
  - TensorBoard logging (loss, Dice per epoch)
  - Model export to TorchScript for deployment
```
**Checkpoint**: every 50 epochs + final

### Step 4 — Evaluation Metrics
**Prompt for subagent:**
```
Write evaluation script for 3D segmentation with:
  - Global Dice Similarity Coefficient (DSC)
  - 95th-percentile Hausdorff Distance (HD95)
  - Per-lesion sensitivity (connected component analysis)
  - Per-lesion false positive rate
  - Volume estimation error (mL) per lesion
  - Aggregated results table (mean ± std) saved to results/metrics.json
Compare nnU-Net predictions vs MONAI predictions vs ensemble.
```
**Libraries**: `SimpleITK`, `scikit-image` (connected components), `scipy`

### Step 5 — Longitudinal Tracking
**Prompt for subagent:**
```
Build a registration-based pipeline to track MM bone lesion volume changes
across serial CT scans using SimpleElastix. Steps:
  1. Rigid registration (bone alignment) — baseline to follow-up
  2. Deformable B-spline registration (local tissue adaptation)
  3. Propagate baseline segmentation mask to follow-up space
  4. Compute delta volume (mL) and percent change per lesion
  5. Classify lesion response: CR/PR/SD/PD per IMWG criteria
  6. Output: per-patient longitudinal summary JSON
```
**Libraries**: `SimpleElastix` (SimpleITK with Elastix), `pandas`

### Step 6 — Clinical Dashboard
**Prompt for subagent:**
```
Create a Streamlit app (dashboard/app.py) that:
  1. Accepts a DICOM folder via file uploader
  2. Runs dcm2niix conversion in a subprocess
  3. Runs the TorchScript BoneSeg3D model (sliding window inference)
  4. Displays: lesion count, total lesion volume (mL), largest lesion size,
     change from baseline (if prior scan uploaded), 3D volume rendering
     (using vtkplotter/itkwidgets or plotly isosurface)
  5. Exports structured report as PDF
```

---

## 6. Academic Research Alignment

### Writing the Paper: Focused Contributions
Following best practices for impactful research writing:

**Core Idea (one sentence)**: *BoneSeg3D is the first open-source, end-to-end pipeline for 3D volumetric MM bone lesion segmentation, longitudinal tracking, and clinical deployment on TCIA CT data.*

**Target Venue**: MICCAI 2026 (main conference) or Medical Image Analysis (journal)
**Backup**: MIDL 2026, IEEE TMI

**Figure Plan** (each must stand alone):
1. System architecture overview (pipeline diagram)
2. Qualitative segmentation results (axial/coronal/sagittal overlay)
3. Quantitative comparison table (Dice, HD95 vs. prior work)
4. Per-lesion sensitivity vs. lesion size curve
5. Longitudinal tracking: volume trajectory per patient (line plot)
6. Dashboard screenshot + clinical workflow integration

**Abstract keywords**: whole-body CT · lytic bone lesion · nnU-Net · SwinUNETR · longitudinal tracking · multiple myeloma · open benchmark

### What to Kill Early
- If VISTA3D zero-shot Dice < 0.3 without fine-tuning → mention in ablation only, not main contribution
- If longitudinal registration fails (MRE > 5mm) → descope to static-only, defer to future work
- If TCIA dataset is too small (< 50 patients) → focus contribution on pipeline + benchmark protocol, not raw model performance

### Speed / Timing
- MONAI SwinUNETR training: ~6 hours per fold on H100 → full 5-fold = ~30 hours
- nnU-Net 3d_fullres training: ~20 hours per fold → use 2-fold fast baseline first
- Start writing intro + related work in parallel with training (Week 1)

---

## 7. Directory Structure

```
boneseg3d/
├── main.py                        # ← ONLY entry point, function calls only
├── CLAUDE.md                      # ← This file
├── configs/
│   └── default.yaml
├── boneseg3d/
│   ├── steps/
│   │   ├── step1_data.py
│   │   ├── step2_nnunet.py
│   │   ├── step3_monai.py
│   │   ├── step4_eval.py
│   │   ├── step5_longitudinal.py
│   │   └── step6_dashboard.py
│   ├── models/
│   │   ├── swin_unetr.py
│   │   └── residual_unet3d.py
│   ├── utils/
│   │   ├── checkpoint.py
│   │   ├── config.py
│   │   ├── dicom_utils.py
│   │   └── metrics.py
│   └── __init__.py
├── dashboard/
│   └── app.py                     # Streamlit app
├── checkpoints/                   # Auto-generated checkpoint JSONs
├── logs/                          # GPU/training logs
├── results/                       # metrics.json, visualizations
├── data/
│   ├── raw_dicom/
│   ├── nifti/
│   └── nnunet/
│       ├── Task001_BoneLesion/
│       │   ├── imagesTr/
│       │   ├── labelsTr/
│       │   ├── imagesTs/
│       │   └── dataset.json
└── requirements.txt
```

---

## 8. Environment Setup

```bash
# Create environment
conda create -n boneseg3d python=3.10 -y
conda activate boneseg3d

# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install monai[all]
pip install nnunet
pip install SimpleITK SimpleElastix
pip install streamlit plotly nibabel tcia_utils dcm2niix
pip install scikit-image scipy pandas tqdm tensorboard

# Verify GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## 9. Key References

1. Faghani et al. (2023). *A deep learning algorithm for detecting lytic bone lesions of multiple myeloma on CT.* Skeletal Radiology. https://pubmed.ncbi.nlm.nih.gov/35980454/
2. van Leeuwen et al. (2025). *Deep Learning Classifiers to Reduce False Positives in Osteolytic Lesion Segmentation.* Springer BNAIC/Benelearn.
3. Isensee et al. (2021). *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.* Nature Methods.
4. He et al. (2025). *VISTA3D: A Unified Segmentation Foundation Model For 3D Medical Imaging.* CVPR 2025. https://arxiv.org/abs/2406.05285
5. Zhao et al. (2024). *MMNet: deep multiscale feature fusion for MM segmentation in MRI.* QIMS, 14, 7176–7199.
6. Frontiers in Radiology (2023). *Deep learning image segmentation approaches for malignant bone lesions: systematic review.* https://www.frontiersin.org/journals/radiology/articles/10.3389/fradi.2023.1241651/full
7. Academic Radiology (2025). *Advanced Automated Model for Robust Bone Marrow Segmentation in WB-MRI.* https://www.academicradiology.org/article/S1076-6332(24)01048-1/fulltext
8. MDPI Cancers (2024). *Reliability of Automated RECIST 1.1 and Volumetric RECIST Target Lesion Response Evaluation.* https://www.mdpi.com/2072-6694/16/23/4009
9. RSNA Radiology (2019). *MY-RADS: Myeloma Response Assessment and Diagnosis System.* https://pubs.rsna.org/doi/full/10.1148/radiol.2019181949
10. Tang et al. (2022). *Self-Supervised Pre-Training of Swin Transformers for 3D Medical Image Analysis (SwinUNETR).* CVPR.

---

## 10. Weekly Milestones

| Week | Goals | Success Criteria |
|---|---|---|
| **1** | Step 1 + Step 2 + Step 3 start | TCIA data downloaded, nnU-Net preprocessing done, MONAI training started |
| **2** | Step 3 done + Step 4 + Step 5 | Dice > 0.65 on val set, metrics table complete, registration error < 3mm |
| **3** | Step 6 + write-up + ablation | Dashboard demo-ready, intro/methods/results drafted, MICCAI abstract submitted |

---

*This CLAUDE.md is the single source of truth for all agents working in this repo. Read it first, always.*