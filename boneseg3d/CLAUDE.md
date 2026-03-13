# BoneSeg3D — 3D Bone Lesion Detection in Multiple Myeloma CT Scans
## CLAUDE.md — Agent Orchestration & Development Guide

> **Model**: Claude Sonnet / Opus via Claude Code
> **Compute**: UNT H100 GPU
> **Stack**: MONAI · nnU-Net · PyTorch · SimpleElastix · Streamlit
> **Timeline**: 2–3 weeks
> **Entry point**: `main.py` at repo root — only function calls, no inline logic

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

## 1. Data Product Lifecycle & Roles

BoneSeg3D follows a four-phase data product lifecycle. Each phase maps to a dedicated role, a Python module in `boneseg3d/product/`, and a governance gate.

```
  IDEATE            DESIGN             OPERATIONALIZE          EVOLVE
  (Product Mgr)     (Data Steward)     (Data Engineer +        (Researcher)
                                        Researcher)
  ┌───────────┐    ┌───────────┐      ┌───────────────┐      ┌────────────┐
  │ Objectives│ →  │ Contracts │  →   │ Build / Train │  →   │ Compare /  │
  │ Use-cases │    │ Governance│      │ Evaluate      │      │ Promote /  │
  │ Personas  │    │ PHI check │      │ Deploy        │      │ Sunset     │
  └───────────┘    └───────────┘      │ Monitor       │      └────────────┘
                                      └───────────────┘
```

### Phase → Module → Entry Function

| Phase | Role | Module | Entry Function |
|---|---|---|---|
| **IDEATE** | Data Product Manager | `boneseg3d/product/ideation.py` | `define_clinical_objectives(cfg)` |
| **DESIGN** | Data Steward | `boneseg3d/product/design.py` | `specify_data_contracts(cfg)` |
| **OPERATIONALIZE** | Data Engineer + Researcher | `boneseg3d/product/operationalize.py` | `register_monitoring_hooks(cfg)` |
| **EVOLVE** | Researcher | `boneseg3d/product/evolve.py` | `evaluate_product_iteration(cfg)` |

### Key Artefacts Per Phase

| Phase | Machine-readable output | Human-readable output |
|---|---|---|
| IDEATE | `outputs/product/ideation_report.json` | `outputs/product/ideation_report.md` |
| DESIGN | `outputs/product/data_contract.json` | `outputs/product/governance_report.md` |
| OPERATIONALIZE | `outputs/product/monitoring_config.json` | `logs/sla_thresholds.json` |
| EVOLVE | `outputs/product/evolution_report.json` | `outputs/product/evolution_report.md` |

---

## 2. Workflow Architecture

```
main.py
  │
  ├── ideate(cfg)
  │     └── define_clinical_objectives()          # PM: use-case registry, personas, KPIs
  │
  ├── design(cfg)
  │     └── specify_data_contracts()              # Steward: governance, PHI check, schema
  │
  ├── operationalize(cfg, orchestrator)   [async]
  │     ├── ingest_and_structure_tcia_cohort()          ← must finish first
  │     │         │
  │     │    ┌────┴────────────────────────────┐
  │     │    │ PARALLEL on separate worktrees  │
  │     │    ├─ train_nnunet_lytic_segmentor() │
  │     │    └─ train_swinunetr_ensemble_backbone()
  │     │         │
  │     │    benchmark_segmentation_performance()       ← waits for both
  │     │         │
  │     │    ┌────┴────────────────────────────┐
  │     │    │ PARALLEL on separate worktrees  │
  │     │    ├─ register_and_quantify_lesion_burden()
  │     │    └─ serve_clinical_reporting_dashboard()
  │     │
  │     └── register_monitoring_hooks()                 # Engineer: SLA, drift, registry
  │
  └── evolve(cfg)
        └── evaluate_product_iteration()          # Researcher: promote/hold/regress/sunset
```

---

## 3. Subagent & Worktree Execution

### How It Works
The `Orchestrator` class in `boneseg3d/orchestrator.py`:
1. Provisions a git worktree via `git worktree add .worktrees/<branch> -b <branch>`
2. Writes a task prompt from the agent function's docstring
3. Forks `claude --print @prompt_file` as an async subprocess inside the worktree
4. On success: writes `CHECKPOINT.json` inside the worktree + marks the checkpoint in `checkpoints/`
5. On failure: logs stderr and raises — orchestrator can retry once

### Worktree Branches (as confirmed by pipeline run)
```
.worktrees/
  feat-ingest-tcia-cohort        → ingest_and_structure_tcia_cohort()
  feat-train-nnunet-segmentor    → train_nnunet_lytic_segmentor()
  feat-train-swinunetr-backbone  → train_swinunetr_ensemble_backbone()
  feat-benchmark-segmentation    → benchmark_segmentation_performance()
  feat-track-lesion-burden       → register_and_quantify_lesion_burden()
  feat-clinical-dashboard        → serve_clinical_reporting_dashboard()
```

### Subagent Dispatch (actual code from orchestrator.py)
```python
# Parallel training — asyncio.gather runs both concurrently
await asyncio.gather(
    orchestrator.run_in_worktree(
        branch="feat/train-nnunet-segmentor",
        task_fn=train_nnunet_lytic_segmentor,
        task_args=(cfg,),
        checkpoint_key="train_nnunet_lytic_segmentor",
    ),
    orchestrator.run_in_worktree(
        branch="feat/train-swinunetr-backbone",
        task_fn=train_swinunetr_ensemble_backbone,
        task_args=(cfg,),
        checkpoint_key="train_swinunetr_ensemble_backbone",
    ),
)
```

### Rules for Subagents
- Each subagent MUST write a `CHECKPOINT.json` upon successful completion.
- Subagents MUST NOT modify files outside their worktree directory.
- Subagents MUST log GPU utilization and peak memory to `logs/gpu_<task>.log`.
- If a subagent fails, the orchestrator retries once, then surfaces the error.
- Subagents run via `claude --print --dangerously-skip-permissions @prompt_file`.

---

## 4. Root `main.py` — Clean Orchestration Entry Point

```python
# main.py — ONLY function calls, no inline logic
from boneseg3d.agents.ingest_tcia_cohort       import ingest_and_structure_tcia_cohort
from boneseg3d.agents.train_nnunet_segmentor   import train_nnunet_lytic_segmentor
from boneseg3d.agents.train_swinunetr_backbone import train_swinunetr_ensemble_backbone
from boneseg3d.agents.benchmark_segmentation  import benchmark_segmentation_performance
from boneseg3d.agents.track_lesion_burden      import register_and_quantify_lesion_burden
from boneseg3d.agents.serve_clinical_dashboard import serve_clinical_reporting_dashboard

def main():
    ...
    ideate(cfg)                                        # Phase 1
    design(cfg)                                        # Phase 2
    asyncio.run(operationalize(cfg, orchestrator))     # Phase 3
    evolve(cfg)                                        # Phase 4
```

Usage:
```bash
python main.py                          # full lifecycle
python main.py --phase operationalize   # single phase
python main.py --task train_nnunet      # single named task
python main.py --resume                 # skip completed checkpoints
```

---

## 5. Checkpoint System

Every major task writes a JSON checkpoint to `checkpoints/` so the pipeline can resume without re-running completed steps.

### Checkpoint Gates (when they are written)

| Checkpoint Key | Written After |
|---|---|
| `ideation` | Use-case registry + objectives validated |
| `design` | Data contract + governance checks persisted |
| `ingest_and_structure_tcia_cohort` | NIfTI conversion + dataset.json validated |
| `nnunet_preprocess` | `nnUNet_plan_and_preprocess` completes |
| `nnunet_train_fold{0..4}` | Each training fold completes |
| `train_nnunet_lytic_segmentor` | Best ensemble model selected and saved |
| `swinunetr_epoch_{N}` | Every 50 training epochs |
| `train_swinunetr_ensemble_backbone` | TorchScript export + softmax saved |
| `benchmark_segmentation_performance` | Dice/HD95/per-lesion metrics written to `results/metrics.json` |
| `register_and_quantify_lesion_burden` | Registration + IMWG response JSON for all serial pairs |
| `serve_clinical_reporting_dashboard` | Dashboard script verified |
| `operationalize` | All ML tasks + monitoring hooks complete |

### Checkpoint API
```python
from boneseg3d.utils.checkpoint import mark_complete, is_complete, load_checkpoint

is_complete("train_nnunet_lytic_segmentor")  # → bool
mark_complete("my_task", metadata={"dice": 0.72, "hours": 6.2})
load_checkpoint("my_task")                   # → dict or None
```

---

## 6. Agent Modules — Function Name Reference

| Agent Module | Entry Function | Worktree Branch | What It Does |
|---|---|---|---|
| `agents/ingest_tcia_cohort.py` | `ingest_and_structure_tcia_cohort(cfg)` | `feat/ingest-tcia-cohort` | TCIA download → dcm2niix → PHI scrub → nnU-Net Task layout |
| `agents/train_nnunet_segmentor.py` | `train_nnunet_lytic_segmentor(cfg)` | `feat/train-nnunet-segmentor` | nnU-Net plan → preprocess → 5-fold CV → ensemble inference → softmax export |
| `agents/train_swinunetr_backbone.py` | `train_swinunetr_ensemble_backbone(cfg)` | `feat/train-swinunetr-backbone` | MONAI SwinUNETR training → TorchScript export → test-set softmax export |
| `agents/benchmark_segmentation.py` | `benchmark_segmentation_performance(cfg)` | `feat/benchmark-segmentation` | Dice, HD95, per-lesion sensitivity/FP for nnU-Net vs SwinUNETR vs ensemble |
| `agents/track_lesion_burden.py` | `register_and_quantify_lesion_burden(cfg)` | `feat/track-lesion-burden` | Rigid + B-spline registration → centroid matching → volume delta → IMWG response |
| `agents/serve_clinical_dashboard.py` | `serve_clinical_reporting_dashboard(cfg)` | `feat/clinical-dashboard` | Streamlit app: DICOM upload → inference → 3D rendering → PDF report |

---

## 7. Governance Checkpoints (Data Steward)

Defined in `boneseg3d/product/design.py` and validated before training begins:

| ID | Description | Required |
|---|---|---|
| GC-01 | TCIA DUA acknowledged and stored | Yes |
| GC-02 | NIfTI headers pass PHI scrubbing check | Yes |
| GC-03 | Annotation provenance JSON exists for every labelled case | Yes |
| GC-04 | Train/val/test splits are patient-level with zero overlap | Yes |
| GC-05 | Class imbalance ratio (lesion/background) documented | No |
| GC-06 | Dataset directory checksummed (SHA-256) and version pinned | Yes |

---

## 8. Evolution Decisions (Researcher)

The `evolve` phase in `boneseg3d/product/evolve.py` compares candidate models against the registered baseline and emits one of four decisions:

| Decision | Condition | Action |
|---|---|---|
| **PROMOTE** | ΔDice ≥ 0.02 or ΔHD95 ≥ 1.0 mm improvement | Update registry: candidate → production, baseline → deprecated |
| **HOLD** | 0 ≤ ΔDice < 0.02 | Publish as workshop paper; start ablation for higher-impact lever |
| **REGRESS** | ΔDice < -0.01 or ΔHD95 > -2.0 mm | Block promotion; investigate training instability or data leakage |
| **ABLATE** | Mixed signals (Dice up, HD95 down or vice-versa) | Run component ablation: freeze backbone vs. loss vs. augmentation |

Models older than 3 versions are flagged for **SUNSET** and archival.

---

## 9. Directory Structure

```
.
├── main.py                                # ← ONLY entry point, function calls only
├── CLAUDE.md                              # ← This file
├── configs/
│   └── default.yaml                       # All pipeline config (env-var substitution)
├── requirements.txt
├── boneseg3d/
│   ├── orchestrator.py                    # Worktree provisioning + subagent dispatch
│   ├── product/
│   │   ├── ideation.py                    # PM: objectives, use-cases, personas
│   │   ├── design.py                      # Steward: data contracts, governance
│   │   ├── operationalize.py              # Engineer: SLA, monitoring, model registry
│   │   └── evolve.py                      # Researcher: promote / hold / regress / sunset
│   ├── agents/
│   │   ├── ingest_tcia_cohort.py          # DICOM → NIfTI → nnU-Net layout
│   │   ├── train_nnunet_segmentor.py      # nnU-Net 5-fold CV training
│   │   ├── train_swinunetr_backbone.py    # MONAI SwinUNETR training + TorchScript
│   │   ├── benchmark_segmentation.py      # Dice, HD95, per-lesion metrics
│   │   ├── track_lesion_burden.py         # Registration + IMWG response classification
│   │   └── serve_clinical_dashboard.py    # Streamlit clinical app
│   ├── models/                            # Model architecture definitions
│   └── utils/
│       ├── checkpoint.py                  # JSON checkpoint read/write
│       ├── config.py                      # YAML → SimpleNamespace loader
│       └── logging_utils.py              # Console + file structured logging
├── .worktrees/                            # Auto-provisioned per-task git worktrees
├── checkpoints/                           # Auto-generated checkpoint JSONs
├── logs/                                  # GPU/training/pipeline logs
├── results/                               # metrics.json, comparison tables
├── outputs/
│   ├── product/                           # Lifecycle artefacts (ideation, governance, etc.)
│   ├── models/                            # Trained model weights + TorchScript
│   ├── predictions/                       # nnU-Net + SwinUNETR + ensemble predictions
│   └── longitudinal/                      # Per-patient registration + IMWG summaries
├── scripts/
│   └── gpu_monitor.sh                     # Auto-generated nvidia-smi logger
└── data/
    ├── raw_dicom/
    ├── nifti/
    ├── labels/
    ├── longitudinal/                      # Serial CT scans for response tracking
    └── nnunet/
        └── Task001_BoneLesion/
            ├── imagesTr/
            ├── labelsTr/
            ├── imagesTs/
            └── dataset.json
```

---

## 10. Environment Setup

```bash
conda create -n boneseg3d python=3.10 -y
conda activate boneseg3d

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## 11. Pipeline Run Log (validated 2026-03-13)

All six subagents completed successfully on UNT pipeline2 cluster:

```
12:37:40  IDEATION        — skipped (cached)
12:37:40  DESIGN          — skipped (cached)
12:37:40  OPERATIONALIZE  — started
12:37:40  ├─ ingest_and_structure_tcia_cohort      ✓ 12:40:59  (~3 min)
12:40:59  ├─ train_nnunet_lytic_segmentor          ✓ 12:46:50  (~6 min)  ┐ parallel
12:40:59  ├─ train_swinunetr_ensemble_backbone     ✓ 12:50:01  (~9 min)  ┘
12:50:01  ├─ benchmark_segmentation_performance    ✓ 12:54:08  (~4 min)
12:54:08  ├─ register_and_quantify_lesion_burden   ✓ 12:58:07  (~4 min)  ┐ parallel
12:54:08  ├─ serve_clinical_reporting_dashboard     ✓ 12:56:01  (~2 min)  ┘
12:58:07  ├─ register_monitoring_hooks             ✓
12:58:07  EVOLVE          — completed (no baseline snapshot yet)
```

**Known issue**: EVOLVE phase could not find `results/metrics.json` because the benchmark subagent wrote metrics inside its worktree. Fix: the benchmark agent must write metrics to the shared `results/` directory in the main repo, or the orchestrator must copy worktree outputs after completion.

---

## 12. Academic Research Alignment

### Core Idea (one sentence)
*BoneSeg3D is the first open-source, end-to-end pipeline for 3D volumetric MM bone lesion segmentation, longitudinal tracking, and clinical deployment on TCIA CT data.*

**Target Venue**: MICCAI 2026 (main) or Medical Image Analysis (journal)
**Backup**: MIDL 2026, IEEE TMI

### Figure Plan (each must stand alone)
1. System architecture overview (the lifecycle + pipeline diagram from §1 and §2)
2. Qualitative segmentation results (axial/coronal/sagittal overlay)
3. Quantitative comparison table (Dice, HD95 vs. prior work)
4. Per-lesion sensitivity vs. lesion size curve
5. Longitudinal tracking: volume trajectory per patient (waterfall plot)
6. Dashboard screenshot + clinical workflow integration

### What to Kill Early
- If VISTA3D zero-shot Dice < 0.3 → mention in ablation only, not main contribution
- If registration MRE > 5mm → descope to static-only, defer to future work
- If TCIA dataset < 50 patients → focus contribution on pipeline + benchmark protocol

### Weekly Milestones

| Week | Goals | Success Criteria |
|---|---|---|
| **1** | Ingest + both trainings started + intro/related work drafted | TCIA data structured, nnU-Net preprocessing done, SwinUNETR training begun |
| **2** | Training done + benchmark + longitudinal tracking | Dice > 0.65 on val, metrics table complete, registration error < 3mm |
| **3** | Dashboard + write-up + ablation + submission | Dashboard demo-ready, full paper drafted, MICCAI abstract submitted |

---

## 13. Key References

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

*This CLAUDE.md is the single source of truth for all agents working in this repo. Read it first, always.*