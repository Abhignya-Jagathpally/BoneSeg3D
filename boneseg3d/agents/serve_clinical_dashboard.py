"""
boneseg3d/agents/serve_clinical_dashboard.py
=============================================
AGENT: serve_clinical_reporting_dashboard
Lifecycle phase: OPERATIONALIZE
Role: Data Engineer

Responsibility
──────────────
Streamlit clinical dashboard that wraps the deployed TorchScript model.
Clinicians can upload a DICOM folder, view 3D lesion renderings,
review quantitative metrics, and export a structured PDF report.

Features
────────
  • DICOM folder upload → dcm2niix conversion → sliding window inference
  • Lesion count, total volume (mL), largest lesion size
  • Change-from-baseline comparison (if prior scan provided)
  • Interactive 3D isosurface rendering (Plotly)
  • IMWG response category (requires prior scan)
  • One-click structured PDF report export

Run standalone
──────────────
  streamlit run boneseg3d/agents/serve_clinical_dashboard.py

Worktree branch: feat/clinical-dashboard
Checkpoint key:  serve_clinical_reporting_dashboard
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
import torch


# ── Model loader (cached across sessions) ─────────────────────────────────────

@st.cache_resource(show_spinner="Loading BoneSeg3D model...")
def load_deployed_model(model_path: str) -> torch.jit.ScriptModule:
    model = torch.jit.load(model_path, map_location="cpu")
    model.eval()
    return model


# ── DICOM → NIfTI conversion ──────────────────────────────────────────────────

def convert_uploaded_dicom_to_nifti(dicom_dir: Path, output_dir: Path) -> Path | None:
    """Run dcm2niix on an uploaded DICOM folder. Returns NIfTI path or None."""
    result = subprocess.run(
        ["dcm2niix", "-z", "y", "-f", "scan", "-o", str(output_dir), str(dicom_dir)],
        capture_output=True, text=True,
    )
    nifti_files = list(output_dir.glob("*.nii.gz"))
    if result.returncode != 0 or not nifti_files:
        st.error(f"dcm2niix failed: {result.stderr[:300]}")
        return None
    return nifti_files[0]


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_nifti_for_inference(nifti_path: Path) -> tuple[torch.Tensor, object]:
    """
    Load NIfTI, resample to 1×1×1.5mm, clip HU [-500,1500] → [0,1].
    Returns (input_tensor, nibabel_image) for affine recovery.
    """
    import nibabel as nib
    from monai.transforms import (
        Compose, LoadImage, EnsureChannelFirst, Spacing,
        ScaleIntensityRange, EnsureType,
    )

    transforms = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Spacing(pixdim=(1.0, 1.0, 1.5), mode="bilinear"),
        ScaleIntensityRange(a_min=-500, a_max=1500, b_min=0.0, b_max=1.0, clip=True),
        EnsureType(),
    ])
    tensor = transforms(str(nifti_path))
    original_img = nib.load(str(nifti_path))
    return tensor.unsqueeze(0), original_img   # add batch dim


# ── Inference ─────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Running segmentation model...", ttl=3600)
def run_sliding_window_inference(
    _model,
    nifti_path_str: str,
    roi_size: tuple = (128, 128, 128),
    sw_batch_size: int = 4,
    overlap: float = 0.5,
) -> np.ndarray:
    """
    Cache-wrapped sliding window inference. Returns binary lesion mask (D×H×W).
    Uses str key for cache hashing (Tensors are not hashable by Streamlit).
    """
    from monai.inferers import sliding_window_inference

    tensor, _ = preprocess_nifti_for_inference(Path(nifti_path_str))
    with torch.no_grad():
        pred = sliding_window_inference(
            tensor, roi_size, sw_batch_size, _model, overlap=overlap,
        )
    softmax = torch.softmax(pred, dim=1)
    mask    = (softmax[0, 1] > 0.5).numpy().astype(np.uint8)
    return mask


# ── Metric computation ────────────────────────────────────────────────────────

def compute_lesion_metrics(
    mask: np.ndarray, spacing_mm: tuple
) -> dict:
    """Return lesion count, total volume, and largest lesion size."""
    from skimage.measure import label as cc_label, regionprops

    labeled = cc_label(mask, connectivity=3)
    regions = regionprops(labeled)
    voxel_vol_ml = float(np.prod(spacing_mm)) / 1000.0

    lesion_volumes = sorted(
        [rp.area * voxel_vol_ml for rp in regions], reverse=True
    )
    return {
        "lesion_count":       len(lesion_volumes),
        "total_volume_ml":    round(sum(lesion_volumes), 3),
        "largest_lesion_ml":  round(lesion_volumes[0], 3) if lesion_volumes else 0.0,
        "lesion_volumes_ml":  [round(v, 3) for v in lesion_volumes],
    }


def compute_change_from_baseline(
    prior_metrics: dict, current_metrics: dict
) -> dict:
    """Compute volume deltas and IMWG response category."""
    from boneseg3d.agents.track_lesion_burden import classify_imwg_response

    bl_vol = prior_metrics["total_volume_ml"]
    fu_vol = current_metrics["total_volume_ml"]
    n_new  = max(0, current_metrics["lesion_count"] - prior_metrics["lesion_count"])
    delta_pct = (fu_vol - bl_vol) / max(bl_vol, 1e-6) * 100

    return {
        "volume_change_ml":  round(fu_vol - bl_vol, 3),
        "volume_change_pct": round(delta_pct, 2),
        "n_new_lesions":     n_new,
        "imwg_response":     classify_imwg_response(bl_vol, fu_vol, n_new),
    }


# ── 3D rendering ──────────────────────────────────────────────────────────────

def render_3d_isosurface(mask: np.ndarray, spacing_mm: tuple) -> "go.Figure":
    """Create a Plotly 3D isosurface figure of the lesion mask."""
    import plotly.graph_objects as go
    from skimage import measure

    if mask.sum() == 0:
        fig = go.Figure()
        fig.add_annotation(text="No lesions detected", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    verts, faces, _, _ = measure.marching_cubes(mask, level=0.5, spacing=spacing_mm)

    fig = go.Figure(data=[go.Mesh3d(
        x=verts[:, 2], y=verts[:, 1], z=verts[:, 0],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color="firebrick",
        opacity=0.75,
        name="Lytic Lesions",
    )])
    fig.update_layout(
        scene=dict(
            xaxis_title="R-L (mm)", yaxis_title="A-P (mm)", zaxis_title="S-I (mm)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        title="3D Lesion Rendering",
    )
    return fig


# ── PDF report export ─────────────────────────────────────────────────────────

def export_pdf_report(
    patient_id: str,
    current_metrics: dict,
    change_metrics: dict | None,
    output_path: Path,
) -> None:
    """
    Generate a structured PDF report using reportlab.
    Falls back to a plain-text .txt if reportlab is unavailable.
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet

        doc    = SimpleDocTemplate(str(output_path), pagesize=letter)
        styles = getSampleStyleSheet()
        story  = [
            Paragraph("BoneSeg3D Clinical Report", styles["Title"]),
            Paragraph(f"Patient: {patient_id}", styles["Heading2"]),
            Spacer(1, 12),
            Paragraph(f"Lesion Count: {current_metrics['lesion_count']}", styles["Normal"]),
            Paragraph(f"Total Volume: {current_metrics['total_volume_ml']:.2f} mL", styles["Normal"]),
            Paragraph(f"Largest Lesion: {current_metrics['largest_lesion_ml']:.2f} mL", styles["Normal"]),
        ]
        if change_metrics:
            story += [
                Spacer(1, 12),
                Paragraph("Response Assessment (vs. Prior Scan)", styles["Heading2"]),
                Paragraph(f"Volume Change: {change_metrics['volume_change_pct']:+.1f}%", styles["Normal"]),
                Paragraph(f"IMWG Response: {change_metrics['imwg_response']}", styles["Normal"]),
                Paragraph(f"New Lesions: {change_metrics['n_new_lesions']}", styles["Normal"]),
            ]
        doc.build(story)
    except ImportError:
        output_path.with_suffix(".txt").write_text(
            json.dumps({"patient": patient_id, **current_metrics,
                        **(change_metrics or {})}, indent=2)
        )


# ── Streamlit UI ──────────────────────────────────────────────────────────────

def serve_clinical_reporting_dashboard(cfg=None) -> None:
    """
    Streamlit application entry point.
    When called from main.py, writes a launch-ready indicator.
    When run directly via `streamlit run`, displays the full UI.
    """
    # If called programmatically from main.py, just verify the script exists
    if cfg is not None and not st.runtime.exists():
        from boneseg3d.utils.checkpoint import mark_complete
        script_path = Path(__file__).resolve()
        st.write = lambda *a, **k: None   # suppress calls outside streamlit
        mark_complete("serve_clinical_reporting_dashboard",
                      metadata={"dashboard_script": str(script_path)})
        return

    # ─── Streamlit UI ──────────────────────────────────────────────────────
    st.set_page_config(
        page_title="BoneSeg3D | MM Lesion Analysis",
        page_icon="🦴",
        layout="wide",
    )
    st.title("🦴 BoneSeg3D — Multiple Myeloma Lesion Analysis")
    st.caption("Whole-body CT · Lytic lesion segmentation · IMWG response assessment")

    model_path = str(Path("outputs/models/swinunetr/swinunetr_deployment.pt"))
    model      = load_deployed_model(model_path)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Upload Scans")
        dicom_files   = st.file_uploader("Current CT (DICOM folder — ZIP)",
                                         type=["zip"], key="current")
        prior_files   = st.file_uploader("Prior CT (optional, for Δ comparison)",
                                         type=["zip"], key="prior")
        patient_id    = st.text_input("Patient ID", value="PT-001")
        run_button    = st.button("▶  Run Analysis", type="primary")

    with col2:
        if run_button and dicom_files:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)

                # Unzip and convert current scan
                import zipfile
                current_zip = tmp / "current.zip"
                current_zip.write_bytes(dicom_files.read())
                with zipfile.ZipFile(current_zip) as zf:
                    zf.extractall(tmp / "current_dicom")
                current_nii = convert_uploaded_dicom_to_nifti(
                    tmp / "current_dicom", tmp / "current_nifti"
                )
                if not current_nii:
                    st.stop()

                # Inference
                with st.spinner("Segmenting lesions..."):
                    mask    = run_sliding_window_inference(model, str(current_nii))
                    import nibabel as nib
                    spacing = nib.load(str(current_nii)).header.get_zooms()[:3]
                    metrics = compute_lesion_metrics(mask, spacing)

                # Optional prior scan
                change_metrics = None
                if prior_files:
                    prior_zip = tmp / "prior.zip"
                    prior_zip.write_bytes(prior_files.read())
                    with zipfile.ZipFile(prior_zip) as zf:
                        zf.extractall(tmp / "prior_dicom")
                    prior_nii = convert_uploaded_dicom_to_nifti(
                        tmp / "prior_dicom", tmp / "prior_nifti"
                    )
                    if prior_nii:
                        prior_mask    = run_sliding_window_inference(model, str(prior_nii))
                        prior_spacing = nib.load(str(prior_nii)).header.get_zooms()[:3]
                        prior_metrics = compute_lesion_metrics(prior_mask, prior_spacing)
                        change_metrics = compute_change_from_baseline(prior_metrics, metrics)

                # ── Results display ────────────────────────────────────────
                st.subheader("Lesion Metrics")
                m1, m2, m3 = st.columns(3)
                m1.metric("Lesion Count",   metrics["lesion_count"])
                m2.metric("Total Volume",   f"{metrics['total_volume_ml']:.2f} mL")
                m3.metric("Largest Lesion", f"{metrics['largest_lesion_ml']:.2f} mL")

                if change_metrics:
                    st.subheader("Response Assessment")
                    r1, r2, r3 = st.columns(3)
                    r1.metric("Volume Δ",
                              f"{change_metrics['volume_change_ml']:+.2f} mL",
                              delta=f"{change_metrics['volume_change_pct']:+.1f}%")
                    r2.metric("IMWG Response", change_metrics["imwg_response"])
                    r3.metric("New Lesions",   change_metrics["n_new_lesions"])

                st.subheader("3D Lesion Rendering")
                fig = render_3d_isosurface(mask, spacing)
                st.plotly_chart(fig, use_container_width=True)

                # ── PDF export ─────────────────────────────────────────────
                report_path = tmp / f"{patient_id}_boneseg3d_report.pdf"
                export_pdf_report(patient_id, metrics, change_metrics, report_path)
                if report_path.exists():
                    with open(report_path, "rb") as f:
                        st.download_button(
                            "📄 Download Report (PDF)",
                            data=f.read(),
                            file_name=report_path.name,
                            mime="application/pdf",
                        )


# ── When run directly via `streamlit run` ─────────────────────────────────────
if __name__ == "__main__":
    serve_clinical_reporting_dashboard()