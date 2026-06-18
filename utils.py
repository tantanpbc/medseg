"""
Training / validation loops, checkpointing, metrics (HD95),
efficiency benchmarking, and dataloader construction.
"""

import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import distance_transform_edt, binary_erosion, generate_binary_structure
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

import albumentations as A

from config import SEED, NUM_CLASSES
from datasets.registry import build_dataset, get_dataset_kwargs
from datasets.splitter import split_dataset_indices


# ────────────────────────────── Patient ID extractors ──────────────────────────────

def _acdc_patient_id(filepath):
    """'/.../patient042_slice03.h5' → 'patient042'"""
    return os.path.basename(filepath).split("_")[0]

def _camus_patient_id(filepath):
    """'/.../patient0001_2CH_ED_gt.mhd' → 'patient0001'"""
    return os.path.basename(filepath).split("_")[0]

def _kvasir_patient_id(filepath):
    """Kvasir has no patient structure — treat each file as its own patient."""
    return os.path.splitext(os.path.basename(filepath))[0]

def _chestxray_patient_id(filepath):
    """
    '/.../train_images/<case_id>/images/<case_id>.png' → '<case_id>'

    The case ID is the parent directory of 'images/', which is unique per case.
    Using the grandparent dirname (the case folder) as the patient ID ensures
    all images from one case stay in the same split.
    """
    # filepath = .../train_images/<case_id>/images/<filename>.png
    # dirname  = .../train_images/<case_id>/images
    # parent   = .../train_images/<case_id>   ← this is the case/patient ID
    return os.path.basename(os.path.dirname(os.path.dirname(filepath)))

PATIENT_ID_FN = {
    "ACDC":      _acdc_patient_id,
    "CAMUS":     _camus_patient_id,
    "KVASIR":    _kvasir_patient_id,
    "CHESTXRAY": _chestxray_patient_id,
}


# ────────────────────────────── DataLoaders ──────────────────────────────

def get_loaders(
    dataset_name="ACDC",
    batch_size=16,
    train_transform=None,
    val_transform=None,
    num_workers=2,
    pin_memory=True,
    val_split=0.1,          # now used as val ratio in the 3-way split
    test_split=0.2,         # held-out test ratio
    **dataset_kwargs,
):
    """
    Build train / val / test DataLoaders with a deterministic patient-level split.

    The split is patient-aware: all slices from a given patient land in exactly
    one of {train, val, test}. This prevents data leakage between splits.

    Returns:
        train_dataset, train_loader,
        val_dataset,   val_loader,
        test_dataset,  test_loader
    """
    # Build three separate dataset objects so each can have its own transform
    train_ds = build_dataset(dataset_name, transform=train_transform, **dataset_kwargs)
    val_ds   = build_dataset(dataset_name, transform=val_transform,   **dataset_kwargs)
    test_ds  = build_dataset(dataset_name, transform=val_transform,   **dataset_kwargs)

    # Patient-level split on the underlying sample list
    patient_id_fn = PATIENT_ID_FN.get(dataset_name, lambda p: os.path.basename(p))
    train_ratio   = 1.0 - val_split - test_split

    train_idx, val_idx, test_idx = split_dataset_indices(
        samples       = train_ds.samples,   # raw file paths before any transform
        patient_id_fn = patient_id_fn,
        train_ratio   = train_ratio,
        val_ratio     = val_split,
        seed          = SEED,
    )

    train_dataset = Subset(train_ds, train_idx)
    val_dataset   = Subset(val_ds,   val_idx)
    test_dataset  = Subset(test_ds,  test_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
    )

    return (
        train_dataset, train_loader,
        val_dataset,   val_loader,
        test_dataset,  test_loader,
    )


def get_default_transforms():
    """Return the standard training augmentation pipeline."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            scale=(0.85, 1.15),
            rotate=(-15, 15),
            p=0.5,
        ),
        A.ElasticTransform(alpha=20, sigma=5, p=0.2,
                           border_mode=cv2.BORDER_CONSTANT),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    ])


# ────────────────────────────── Checkpointing ──────────────────────────────

def save_checkpoint(state, filepath):
    print("=> Saving checkpoint")
    torch.save(state, filepath)


def load_checkpoint(filepath, model, optimizer):
    print("=> Loading checkpoint")
    checkpoint = torch.load(filepath, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


# ────────────────────────────── Training loop ──────────────────────────────

def train_fn(train_loader, model, optimizer, loss_fn, scaler, device, scheduler=None):
    model.train()
    loop = tqdm(train_loader, desc="Training", leave=True)

    for imgs, targets in loop:
        imgs    = imgs.to(device)
        targets = targets.long().to(device)

        with torch.autocast("cuda"):
            outputs = model(imgs)
            if isinstance(outputs, tuple):
                main_pred, aux_pred = outputs
                loss = loss_fn(main_pred, targets) + 0.4 * loss_fn(aux_pred, targets)
            else:
                loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        loop.set_postfix(loss=f"{loss.item():.4f}")


# ────────────────────────────── Validation loop ──────────────────────────────

def val_fn(val_loader, model, device, num_cls=NUM_CLASSES):
    model.eval()
    num_correct = 0
    num_pixels  = 0
    dice_sum    = torch.zeros(num_cls, device=device)
    iou_sum     = torch.zeros(num_cls, device=device)
    cls_count   = torch.zeros(num_cls, device=device)

    loop = tqdm(val_loader, desc="Validating", leave=True)

    with torch.no_grad():
        for imgs, targets in loop:
            imgs    = imgs.to(device)
            targets = targets.long().to(device)

            outputs = model(imgs)
            if isinstance(outputs, tuple):
                preds = outputs[0].argmax(dim=1)
            else:
                preds = outputs.argmax(dim=1)

            num_correct += (preds == targets).sum()
            num_pixels  += torch.numel(preds)

            for cls in range(num_cls):
                pred_cls   = (preds == cls)
                target_cls = (targets == cls)
                TP = (pred_cls & target_cls).sum()
                FP = (pred_cls & ~target_cls).sum()
                FN = (~pred_cls & target_cls).sum()
                denom_dice = (2 * TP + FP + FN).float()
                denom_iou  = (TP + FP + FN).float()
                if denom_iou > 0:
                    dice_sum[cls] += (2 * TP.float()) / (denom_dice + 1e-8)
                    iou_sum[cls]  += TP.float() / (denom_iou + 1e-8)
                    cls_count[cls] += 1

            mean_dice = (dice_sum / (cls_count + 1e-8)).mean().item()
            mean_iou  = (iou_sum  / (cls_count + 1e-8)).mean().item()
            loop.set_postfix(dice=f"{mean_dice:.4f}", iou=f"{mean_iou:.4f}")

    pixel_acc  = (num_correct / num_pixels).item() * 100
    dice_score = (dice_sum[1:] / (cls_count[1:] + 1e-8)).mean().item()
    iou_score  = (iou_sum[1:]  / (cls_count[1:] + 1e-8)).mean().item()

    print(f"Pixel Acc: {pixel_acc:.2f}% | Dice (fg): {dice_score:.4f} | IoU (fg): {iou_score:.4f}")
    return dice_score, iou_score, pixel_acc


# ────────────────────────────── Efficiency benchmarking ──────────────────────────────

def compute_model_size(model):
    if isinstance(model, nn.DataParallel):
        model = model.module
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_bytes / (1024 ** 2)


def benchmark_latency(model, device, input_size=(1, 1, 256, 256), warmup=20, runs=100):
    model.eval()
    dummy = torch.randn(input_size, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.time() - t0) * 1000)
    return np.mean(times) / input_size[0]


# ────────────────────────────── HD95 metric ──────────────────────────────

def compute_hd95(pred, target, num_classes=NUM_CLASSES):
    struct = generate_binary_structure(2, 1)
    hd95_per_class = []
    for cls in range(1, num_classes):
        pred_mask   = (pred == cls).astype(bool)
        target_mask = (target == cls).astype(bool)
        if not pred_mask.any() and not target_mask.any():
            continue
        if not pred_mask.any() or not target_mask.any():
            h, w = pred.shape
            hd95_per_class.append(float(np.sqrt(h ** 2 + w ** 2)))
            continue
        pred_border   = pred_mask ^ binary_erosion(pred_mask, struct)
        target_border = target_mask ^ binary_erosion(target_mask, struct)
        if not pred_border.any() or not target_border.any():
            continue
        dt_target = distance_transform_edt(~pred_border)
        dt_pred   = distance_transform_edt(~target_border)
        surface_distances = np.concatenate([dt_target[target_border], dt_pred[pred_border]])
        hd95_per_class.append(np.percentile(surface_distances, 95))
    return np.mean(hd95_per_class) if hd95_per_class else None


# ────────────────────────────── Freeze / unfreeze backbone ──────────────────────────────

def set_backbone_grad(model, requires_grad: bool, backbone_attr: str = "encoder"):
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    backbone   = getattr(base_model, backbone_attr)
    for param in backbone.parameters():
        param.requires_grad = requires_grad
    if requires_grad:
        backbone.train()
    else:
        backbone.eval()
    print(f"Backbone {'unfrozen' if requires_grad else 'frozen'}")