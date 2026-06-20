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
    """'frame_0000.png' / 'mask_0000.png' -> '0000'"""
    stem = os.path.splitext(os.path.basename(filepath))[0]
    return stem.split("_")[-1]

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


def get_default_transforms(dataset_name=None):
    """
    Return the training augmentation pipeline, tailored per dataset.

    Each dataset gets a small, deliberately simple pipeline matched to its
    imaging modality and the kind of variation that's actually clinically
    plausible — rather than one generic pipeline applied everywhere.

    ACDC (cardiac MRI, small irregular structures: RV/MYO/LV):
        The heart genuinely changes shape across the cardiac cycle, so mild
        elastic deformation is a realistic augmentation here, not just noise.
        Affine covers patient positioning/breathing variation. Blur covers
        scanner/motion blur.

    CAMUS (echocardiography, MYO/LV, fan-shaped ultrasound view):
        Ultrasound's primary real-world variation is speckle noise and probe
        angle, not tissue shape warping — elastic deformation would distort
        the heart shape unrealistically. GaussNoise simulates speckle;
        rotation is kept conservative since the probe's acoustic window
        constrains achievable angles in practice.

    KVASIR (colonoscopy RGB, polyps):
        Endoscope lighting and color rendering vary significantly between
        captures (white-balance, mucosal redness, specular highlights) —
        color/brightness/contrast jitter matters more here than anywhere
        else. Polyps have no canonical orientation, so full rotation range
        is anatomically fine, unlike the other three modalities.

    CHESTXRAY (PA chest X-ray, lung fields):
        Lung fields are large and smooth; patient positioning (slight
        rotation/translation, inspiration depth) is the main real variation,
        not tissue-level shape change — so no elastic deformation. Kept
        deliberately gentle since this is normally a very easy segmentation
        task and the only reason to augment at all is positioning variance.
    """
    if dataset_name == "ACDC":
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

    if dataset_name == "CAMUS":
        return A.Compose([
            A.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                scale=(0.90, 1.10),
                rotate=(-8, 8),
                p=0.5,
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        ])

    if dataset_name == "KVASIR":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                scale=(0.85, 1.15),
                rotate=(-180, 180),
                p=0.5,
            ),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.4),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        ])

    if dataset_name == "CHESTXRAY":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Affine(
                translate_percent={"x": (-0.08, 0.08), "y": (-0.08, 0.08)},
                scale=(0.90, 1.10),
                rotate=(-10, 10),
                p=0.5,
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        ])

    # Fallback for any unregistered dataset — same as the original generic pipeline
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

def val_fn(val_loader, model, device, dataset_name=None, num_cls=None):
    """
    Run validation and compute sample-averaged Dice / IoU / pixel accuracy.

    IMPORTANT — metric definition matches evaluate.py exactly:
        Dice/IoU are computed PER SAMPLE (one score per image per class),
        then averaged across samples. This is NOT the same as pooling
        TP/FP/FN across an entire batch/epoch and computing one Dice from
        the totals — that "batch-averaged" approach systematically
        produces different (often higher) numbers than sample-averaging,
        especially for small structures or larger batch sizes, since
        large-foreground samples dominate a pooled denominator while
        sample-averaging weights every image equally.

        Using the same averaging convention here as in evaluate.py means
        the Dice you see during training (used for checkpoint selection)
        is directly comparable to the final test-set Dice you see after
        training — they are the same metric, not two different ones that
        happen to share a name.

    Dataset-aware foreground classes:
        Some datasets exclude certain label indices from the reported
        "foreground" mean (e.g. CAMUS may report classes [2,3] only, while
        ACDC reports [1,2,3]). Passing dataset_name looks this up from
        DATASET_CONFIGS so the correct checkpoint-selection metric is used
        for every dataset, not just ACDC's [1,2,3] assumed by older code.

    Args:
        val_loader:   DataLoader yielding (image, mask) batches.
        model:        Model (or DataParallel-wrapped model) to evaluate.
        device:       torch device.
        dataset_name: Dataset key (e.g. "ACDC", "CAMUS") used to look up
                      num_classes and foreground_classes from DATASET_CONFIGS.
                      If None, falls back to NUM_CLASSES and "all classes
                      except background" — matching the old ACDC-only behaviour.
        num_cls:      Explicit override for number of classes. Only used if
                      dataset_name is None.

    Returns:
        (dice_score, iou_score, pixel_acc) — sample-averaged, restricted to
        the dataset's foreground_classes, matching evaluate.py's definition.
    """
    if dataset_name is not None:
        from config import DATASET_CONFIGS
        dataset_cfg     = DATASET_CONFIGS[dataset_name.upper()]
        num_cls         = dataset_cfg["num_classes"]
        active_classes  = dataset_cfg["foreground_classes"]
    else:
        num_cls        = num_cls or NUM_CLASSES
        active_classes = list(range(1, num_cls))   # legacy fallback: all non-background

    model.eval()
    class_dice_lists = {cls: [] for cls in range(1, num_cls)}
    class_iou_lists  = {cls: [] for cls in range(1, num_cls)}
    all_pixel_acc    = []

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

            # Per-SAMPLE metrics (loop over the batch dimension) — this is the
            # part that differs from the old pooled-batch implementation.
            preds_np   = preds.cpu().numpy()
            targets_np = targets.cpu().numpy()
            batch_size = preds_np.shape[0]

            for b in range(batch_size):
                pred_b   = preds_np[b]
                target_b = targets_np[b]

                all_pixel_acc.append((pred_b == target_b).mean() * 100)

                for cls in range(1, num_cls):
                    pred_cls = (pred_b == cls)
                    gt_cls   = (target_b == cls)
                    intersection = np.logical_and(pred_cls, gt_cls).sum()
                    union        = np.logical_or(pred_cls, gt_cls).sum()

                    if gt_cls.sum() > 0 or pred_cls.sum() > 0:
                        dice = (2 * intersection) / (pred_cls.sum() + gt_cls.sum() + 1e-8)
                        iou  = intersection / (union + 1e-8)
                        class_dice_lists[cls].append(dice)
                        class_iou_lists[cls].append(iou)

            running_dice = np.mean([
                np.mean(class_dice_lists[c]) for c in active_classes if class_dice_lists[c]
            ]) if any(class_dice_lists[c] for c in active_classes) else 0.0
            running_iou = np.mean([
                np.mean(class_iou_lists[c]) for c in active_classes if class_iou_lists[c]
            ]) if any(class_iou_lists[c] for c in active_classes) else 0.0
            loop.set_postfix(dice=f"{running_dice:.4f}", iou=f"{running_iou:.4f}")

    pixel_acc = float(np.mean(all_pixel_acc)) if all_pixel_acc else 0.0

    # Only average over the dataset's defined foreground classes —
    # this is the dataset-aware fix (previously hardcoded to classes[1:]).
    dice_per_class = [np.mean(class_dice_lists[c]) for c in active_classes if class_dice_lists[c]]
    iou_per_class  = [np.mean(class_iou_lists[c])  for c in active_classes if class_iou_lists[c]]
    dice_score = float(np.mean(dice_per_class)) if dice_per_class else 0.0
    iou_score  = float(np.mean(iou_per_class))  if iou_per_class  else 0.0

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
    """
    95th-percentile Hausdorff Distance, averaged over foreground classes.

    Missing-structure convention: if a class is present in target but
    absent from pred (or vice versa), there's no valid surface-distance
    pairing. This function penalizes that case with the image diagonal
    (sqrt(H^2 + W^2) pixels) rather than skipping it — consistent with
    evaluate.py's convention. See the docstring in evaluate.evaluate()
    for the full rationale; this is a deliberate choice, not universal
    across the literature, so check conventions before comparing HD95
    numbers against other papers.
    """
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