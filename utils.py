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


# ────────────────────────────── DataLoaders ──────────────────────────────

def get_loaders(
    dataset_name="ACDC",
    batch_size=16,
    train_transform=None,
    val_transform=None,
    num_workers=2,
    pin_memory=True,
    val_split=0.2,
    **dataset_kwargs,
):
    """Build train / validation DataLoaders with a deterministic split."""
    train_ds = build_dataset(dataset_name, transform=train_transform, **dataset_kwargs)
    val_ds   = build_dataset(dataset_name, transform=val_transform,   **dataset_kwargs)

    total = len(train_ds)
    indices = list(range(total))
    rng = np.random.RandomState(SEED)
    rng.shuffle(indices)

    val_size = int(total * val_split)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    print(f"Train: {len(train_indices)} | Val: {len(val_indices)}")

    train_dataset = Subset(train_ds, train_indices)
    val_dataset   = Subset(val_ds, val_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
    )
    return train_dataset, train_loader, val_dataset, val_loader


def get_default_transforms():
    """Return the standard training augmentation pipeline used in the notebook."""
    train_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            scale=(0.85, 1.15),
            rotate=(-15, 15),
            cval_mask=0,
            p=0.5,
        ),
        A.ElasticTransform(alpha=20, sigma=5, p=0.2,
                           border_mode=cv2.BORDER_CONSTANT),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    ])
    return train_transforms


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
        imgs = imgs.to(device)
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

    pixel_acc = (num_correct / num_pixels).item() * 100
    # Foreground-only mean (excludes class 0 background)
    dice_score = (dice_sum[1:] / (cls_count[1:] + 1e-8)).mean().item()
    iou_score  = (iou_sum[1:]  / (cls_count[1:] + 1e-8)).mean().item()

    print(f"Pixel Acc: {pixel_acc:.2f}% | Dice (fg): {dice_score:.4f} | IoU (fg): {iou_score:.4f}")
    return dice_score, iou_score, pixel_acc


# ────────────────────────────── Efficiency benchmarking ──────────────────────────────

def compute_model_size(model):
    """Model size in MB (float32 parameters)."""
    if isinstance(model, nn.DataParallel):
        model = model.module
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_bytes / (1024 ** 2)


def benchmark_latency(model, device, input_size=(1, 1, 256, 256), warmup=20, runs=100):
    """Average forward pass latency in ms per sample."""
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

    latency_ms = np.mean(times) / input_size[0]
    return latency_ms


# ────────────────────────────── HD95 metric ──────────────────────────────

def compute_hd95(pred, target, num_classes=NUM_CLASSES):
    """95th percentile Hausdorff Distance averaged across foreground classes."""
    struct = generate_binary_structure(2, 1)
    hd95_per_class = []

    for cls in range(1, num_classes):
        pred_mask   = (pred == cls).astype(bool)
        target_mask = (target == cls).astype(bool)

        if not pred_mask.any() and not target_mask.any():
            continue
        if not pred_mask.any() or not target_mask.any():
            h, w = pred.shape
            max_dist = float(np.sqrt(h ** 2 + w ** 2))
            hd95_per_class.append(max_dist)
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
    """Toggle backbone gradient computation, robust to DataParallel wrapping."""
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    backbone = getattr(base_model, backbone_attr)
    for param in backbone.parameters():
        param.requires_grad = requires_grad
    if requires_grad:
        backbone.train()
    else:
        backbone.eval()
    print(f"Backbone {'unfrozen' if requires_grad else 'frozen'}")
