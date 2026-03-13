"""
boneseg3d/agents/train_swinunetr_backbone.py
=============================================
AGENT: train_swinunetr_ensemble_backbone
Lifecycle phase: OPERATIONALIZE
Role: Researcher

Responsibility
──────────────
Train a MONAI SwinUNETR (Swin Transformer U-Net) as the second backbone
in the dual-model ensemble. Uses sliding window inference for full CT
volumes, mixed precision, and TensorBoard logging.

The trained model is exported to TorchScript for deployment and its
softmax outputs are saved as .npy files for uncertainty-weighted
ensemble fusion with the nnU-Net backbone.

Worktree branch: feat/train-swinunetr-backbone
Checkpoint key:  train_swinunetr_ensemble_backbone
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from boneseg3d.utils.checkpoint    import mark_complete, is_complete
from boneseg3d.utils.logging_utils import get_logger

log = get_logger("train_swinunetr_backbone")

# Training hyper-parameters (overridable via cfg.monai)
DEFAULT_HP = dict(
    roi_size       = (128, 128, 128),
    in_channels    = 1,
    out_channels   = 2,             # background + lytic lesion
    feature_size   = 48,
    max_epochs     = 300,
    val_interval   = 5,
    learning_rate  = 1e-4,
    weight_decay   = 1e-5,
    batch_size     = 2,
    sw_batch_size  = 4,
    overlap        = 0.5,
    amp            = True,
    dice_ce_lambda = 1.0,
    checkpoint_every_n_epochs = 50,
)


# ── Model construction ────────────────────────────────────────────────────────

def build_swinunetr(hp: dict, device: torch.device) -> nn.Module:
    """Instantiate a MONAI SwinUNETR with the given hyper-parameters."""
    from monai.networks.nets import SwinUNETR

    model = SwinUNETR(
        img_size      = hp["roi_size"],
        in_channels   = hp["in_channels"],
        out_channels  = hp["out_channels"],
        feature_size  = hp["feature_size"],
        use_checkpoint= True,   # gradient checkpointing saves GPU memory
    ).to(device)

    log.info(
        f"SwinUNETR built — params: "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M"
    )
    return model


# ── Loss and metrics ──────────────────────────────────────────────────────────

def build_loss_and_optimizer(model: nn.Module, hp: dict):
    from monai.losses import DiceCELoss

    # Weight lesion class higher to compensate for severe class imbalance
    loss_fn   = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        ce_weight=torch.tensor([0.1, 0.9]),  # background, lesion
    )
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=hp["learning_rate"],
        weight_decay=hp["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=hp["max_epochs"]
    )
    return loss_fn, optimiser, scheduler


def compute_dice_metric(preds: torch.Tensor, labels: torch.Tensor) -> float:
    from monai.metrics import DiceMetric
    metric = DiceMetric(include_background=False, reduction="mean")
    metric(y_pred=preds, y=labels)
    return metric.aggregate().item()


# ── Data loading ──────────────────────────────────────────────────────────────

def build_dataloaders(cfg: Any, hp: dict):
    """Build MONAI CacheDataset train/val loaders from the nnU-Net directory."""
    from monai.data import CacheDataset, DataLoader
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
        ScaleIntensityRanged, CropForegroundd, RandCropByPosNegLabeld,
        RandFlipd, RandRotate90d, RandShiftIntensityd, EnsureTyped,
    )

    task_dir  = Path(cfg.data.nnunet_dir) / "Task001_BoneLesion"
    dataset   = json.loads((task_dir / "dataset.json").read_text())
    cases     = dataset["training"]

    # Build file dicts
    data_dicts = [
        {
            "image": str(task_dir / c["image"].lstrip("./")),
            "label": str(task_dir / c["label"].lstrip("./")),
        }
        for c in cases
        if Path(task_dir / c["image"].lstrip("./")).exists()
    ]

    n_val  = max(1, int(len(data_dicts) * 0.15))
    train_dicts = data_dicts[n_val:]
    val_dicts   = data_dicts[:n_val]

    # Hounsfield clipping: bone window [-500, 1500] HU
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.5),
                 mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-500, a_max=1500,
                             b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"], label_key="label",
            spatial_size=hp["roi_size"], pos=1, neg=1, num_samples=4,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.5),
        EnsureTyped(keys=["image", "label"]),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.5),
                 mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-500, a_max=1500,
                             b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        EnsureTyped(keys=["image", "label"]),
    ])

    train_ds = CacheDataset(train_dicts, train_transforms, cache_rate=1.0, num_workers=4)
    val_ds   = CacheDataset(val_dicts,   val_transforms,   cache_rate=1.0, num_workers=4)

    train_loader = DataLoader(train_ds, batch_size=hp["batch_size"],
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1,
                              shuffle=False, num_workers=4, pin_memory=True)
    log.info(f"Data: {len(train_dicts)} train | {len(val_dicts)} val")
    return train_loader, val_loader


# ── Training loop ─────────────────────────────────────────────────────────────

def run_training_loop(
    model:       nn.Module,
    train_loader,
    val_loader,
    loss_fn:     nn.Module,
    optimiser:   torch.optim.Optimizer,
    scheduler:   torch.optim.lr_scheduler._LRScheduler,
    hp:          dict,
    device:      torch.device,
    output_dir:  Path,
) -> float:
    """
    Standard MONAI training loop with:
      - Mixed precision (AMP)
      - Per-epoch TensorBoard logging
      - Epoch checkpoint every N epochs
      - Best-model saving by validation Dice
    Returns best validation Dice.
    """
    from monai.inferers import sliding_window_inference
    from monai.data    import decollate_batch
    from monai.transforms import AsDiscrete, Compose

    writer    = SummaryWriter(log_dir=str(Path("logs/tensorboard/swinunetr")))
    scaler    = GradScaler() if hp["amp"] else None
    best_dice = 0.0

    post_pred  = Compose([AsDiscrete(argmax=True, to_onehot=hp["out_channels"])])
    post_label = Compose([AsDiscrete(to_onehot=hp["out_channels"])])

    for epoch in range(1, hp["max_epochs"] + 1):
        model.train()
        epoch_loss, steps = 0.0, 0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimiser.zero_grad()

            if hp["amp"]:
                with autocast():
                    logits = model(images)
                    loss   = loss_fn(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimiser)
                scaler.update()
            else:
                logits = model(images)
                loss   = loss_fn(logits, labels)
                loss.backward()
                optimiser.step()

            epoch_loss += loss.item()
            steps      += 1

        scheduler.step()
        mean_loss = epoch_loss / max(steps, 1)
        writer.add_scalar("Loss/train", mean_loss, epoch)
        log.info(f"[Epoch {epoch:03d}/{hp['max_epochs']}] loss={mean_loss:.4f}")

        # ── Validation ──────────────────────────────────────────────────────
        if epoch % hp["val_interval"] == 0:
            model.eval()
            val_dices = []
            with torch.no_grad():
                for val_batch in val_loader:
                    val_img   = val_batch["image"].to(device)
                    val_label = val_batch["label"].to(device)
                    val_pred  = sliding_window_inference(
                        val_img, hp["roi_size"],
                        hp["sw_batch_size"], model,
                        overlap=hp["overlap"],
                    )
                    val_pred_post  = [post_pred(p)  for p in decollate_batch(val_pred)]
                    val_label_post = [post_label(l) for l in decollate_batch(val_label)]
                    dice = compute_dice_metric(
                        torch.stack(val_pred_post), torch.stack(val_label_post)
                    )
                    val_dices.append(dice)

            mean_val_dice = float(np.mean(val_dices))
            writer.add_scalar("Dice/val", mean_val_dice, epoch)
            log.info(f"  Val Dice = {mean_val_dice:.4f}")

            if mean_val_dice > best_dice:
                best_dice = mean_val_dice
                torch.save(model.state_dict(), str(output_dir / "swinunetr_best.pth"))
                log.info(f"  ✓ New best model saved (Dice={best_dice:.4f})")

        # ── Epoch checkpoint ─────────────────────────────────────────────────
        if epoch % hp["checkpoint_every_n_epochs"] == 0:
            ck_path = output_dir / f"swinunetr_epoch{epoch:04d}.pth"
            torch.save(model.state_dict(), str(ck_path))
            mark_complete(
                f"swinunetr_epoch_{epoch}",
                metadata={"epoch": epoch, "best_dice_so_far": best_dice},
            )

    writer.close()
    return best_dice


# ── TorchScript export ────────────────────────────────────────────────────────

def export_model_to_torchscript(model: nn.Module, hp: dict, output_dir: Path) -> Path:
    """Export the best SwinUNETR weights to TorchScript for deployment."""
    best_weights = output_dir / "swinunetr_best.pth"
    if not best_weights.exists():
        raise FileNotFoundError(f"Best weights not found at {best_weights}")

    model.load_state_dict(torch.load(str(best_weights), map_location="cpu"))
    model.eval()

    dummy_input = torch.zeros(1, hp["in_channels"], *hp["roi_size"])
    with torch.no_grad():
        traced = torch.jit.trace(model.cpu(), dummy_input)

    ts_path = output_dir / "swinunetr_deployment.pt"
    traced.save(str(ts_path))
    log.info(f"TorchScript model → {ts_path}")
    return ts_path


# ── Softmax export for ensemble ───────────────────────────────────────────────

def export_test_softmax_probabilities(
    model: nn.Module, hp: dict, cfg: Any, output_dir: Path
) -> Path:
    """
    Run sliding window inference over the test set and save softmax
    probabilities as .npy files for ensemble fusion with nnU-Net.
    """
    from monai.inferers import sliding_window_inference
    from monai.data import DataLoader, Dataset
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
        ScaleIntensityRanged, EnsureTyped,
    )

    test_dir  = Path(cfg.data.nnunet_dir) / "Task001_BoneLesion" / "imagesTs"
    softmax_dir = output_dir / "softmax_test"
    softmax_dir.mkdir(parents=True, exist_ok=True)

    test_files = [{"image": str(f)} for f in sorted(test_dir.glob("*.nii.gz"))]
    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.5), mode="bilinear"),
        ScaleIntensityRanged(keys=["image"], a_min=-500, a_max=1500,
                             b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image"]),
    ])
    device = next(model.parameters()).device
    loader = DataLoader(Dataset(test_files, transforms), batch_size=1)

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            img  = batch["image"].to(device)
            pred = sliding_window_inference(
                img, hp["roi_size"], hp["sw_batch_size"], model,
                overlap=hp["overlap"],
            )
            softmax = torch.softmax(pred, dim=1).cpu().numpy()
            case_id = Path(test_files[i]["image"]).stem.replace(".nii", "")
            np.save(str(softmax_dir / f"{case_id}_softmax.npy"), softmax)

    log.info(f"SwinUNETR softmax probs → {softmax_dir} ({len(test_files)} cases)")
    return softmax_dir


# ── Entry point ───────────────────────────────────────────────────────────────

def train_swinunetr_ensemble_backbone(cfg: Any) -> None:
    """
    End-to-end SwinUNETR training pipeline:
      1. Build model and dataloaders
      2. Train with AMP, CosineAnnealingLR, per-epoch TensorBoard logging
      3. Checkpoint every 50 epochs + best model
      4. Export best model to TorchScript
      5. Export test-set softmax probabilities for ensemble fusion

    Called by: main.py → operationalize() → orchestrator.run_in_worktree()
    Worktree:  feat/train-swinunetr-backbone
    """
    log.info("=== train_swinunetr_ensemble_backbone ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    hp = {**DEFAULT_HP}
    if hasattr(cfg, "monai"):
        hp.update({k: v for k, v in vars(cfg.monai).items() if k in hp})

    output_dir = Path("outputs/models/swinunetr")
    output_dir.mkdir(parents=True, exist_ok=True)

    model                  = build_swinunetr(hp, device)
    loss_fn, opt, sched    = build_loss_and_optimizer(model, hp)
    train_loader, val_loader = build_dataloaders(cfg, hp)

    t0         = time.time()
    best_dice  = run_training_loop(
        model, train_loader, val_loader, loss_fn, opt, sched,
        hp, device, output_dir,
    )
    hours = (time.time() - t0) / 3600

    ts_path     = export_model_to_torchscript(model, hp, output_dir)
    softmax_dir = export_test_softmax_probabilities(model, hp, cfg, output_dir)

    mark_complete(
        "train_swinunetr_ensemble_backbone",
        metadata={
            "best_val_dice":     round(best_dice, 4),
            "training_hours":    round(hours, 2),
            "torchscript_path":  str(ts_path),
            "softmax_dir":       str(softmax_dir),
        },
    )
    log.info(f"✓ train_swinunetr_ensemble_backbone complete. Best val Dice = {best_dice:.4f}")