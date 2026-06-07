"""
SegFormer training strategy — multi-stage training with encoder freeze/unfreeze.

Stages:
    A  – Frozen encoder, decoder-only warm-up      (epochs_a epochs)
    B  – Unfrozen encoder, differential LR          (epochs_b epochs)
    C  – Extended fine-tuning                       (epochs_b epochs)
    D  – Final fine-tuning                          (epochs_b epochs)
"""

import gc

import torch
import torch.nn as nn
import torch.optim as optim

from config import PIN_MEMORY
from models.registry import get_backbone_attr
from utils import (
    get_loaders, get_default_transforms,
    save_checkpoint, load_checkpoint,
    train_fn, val_fn, set_backbone_grad,
)


def train(model_name, model, device, args, model_config,
          train_dataset=None, train_loader=None, val_dataset=None, val_loader=None):
    """
    Run the 4-stage SegFormer training loop.

    Returns:
        (model, device, args, train_dataset, train_loader, val_dataset, val_loader)
    """
    backbone_attr = get_backbone_attr(model_name)
    hp = model_config["hyperparams"]

    # Allow CLI overrides
    lr_a = args.lr if args.lr else hp["lr_a"]
    lr_b = hp["lr_b"]
    batch_size_a = args.batch_size if args.batch_size else hp["batch_size_a"]
    batch_size_b = args.batch_size if args.batch_size else hp["batch_size_b"]
    epochs_a = hp["epochs_a"]
    epochs_b = hp["epochs_b"]

    loss_fn = nn.CrossEntropyLoss()
    scaler  = torch.amp.GradScaler("cuda")
    best_dice = 0.0

    checkpoint_path = args.checkpoint or f"{args.output_dir}/{model_name}.pth"

    if args.load_model and args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        best_dice = ckpt.get("best_dice", 0.0)
        print(f"Resumed from checkpoint (best_dice={best_dice:.4f})")

    if args.skip_train:
        print("Skipping training (--skip_train).")
        return model, device, args, train_dataset, train_loader, val_dataset, val_loader

    # ── Stage A: Frozen encoder warm-up ──
    set_backbone_grad(model, requires_grad=False, backbone_attr=backbone_attr)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_a,
        weight_decay=args.weight_decay,
    )

    for epoch in range(epochs_a):
        print(f"\n[Stage A - Epoch {epoch + 1}/{epochs_a}]")
        train_fn(train_loader, model, optimizer, loss_fn, scaler, device)
        dice_score, iou_score, pixel_acc = val_fn(val_loader, model, device)
        if dice_score >= best_dice:
            best_dice = dice_score
            save_checkpoint(
                {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "best_dice": best_dice},
                checkpoint_path,
            )

    # ── Rebuild loaders with smaller batch for fine-tuning ──
    from datasets.registry import get_dataset_kwargs
    ds_kwargs = get_dataset_kwargs(args.dataset, args)
    train_dataset, train_loader, val_dataset, val_loader = get_loaders(
        dataset_name=args.dataset,
        batch_size=batch_size_b,
        train_transform=get_default_transforms(),
        val_transform=None,
        num_workers=args.num_workers,
        pin_memory=PIN_MEMORY,
        val_split=args.val_split,
        **ds_kwargs,
    )

    # ── Stage B: Unfrozen encoder fine-tuning ──
    set_backbone_grad(model, requires_grad=True, backbone_attr=backbone_attr)
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    optimizer = torch.optim.AdamW([
        {"params": getattr(base_model, backbone_attr).parameters(), "lr": lr_b},
        {"params": base_model.decoder.parameters(), "lr": lr_a * 0.5},
    ])

    for epoch in range(epochs_b):
        print(f"\n[Stage B - Epoch {epoch + 1}/{epochs_b}]")
        train_fn(train_loader, model, optimizer, loss_fn, scaler, device)
        dice_score, iou_score, pixel_acc = val_fn(val_loader, model, device)
        if dice_score >= best_dice:
            best_dice = dice_score
            save_checkpoint(
                {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "best_dice": best_dice},
                checkpoint_path,
            )

    # ── Stage C: Extended fine-tuning ──
    for epoch in range(epochs_b):
        print(f"\n[Stage C - Epoch {epoch + 1}/{epochs_b}]")
        train_fn(train_loader, model, optimizer, loss_fn, scaler, device)
        dice_score, iou_score, pixel_acc = val_fn(val_loader, model, device)
        if dice_score >= best_dice:
            best_dice = dice_score
            save_checkpoint(
                {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "best_dice": best_dice},
                checkpoint_path,
            )

    # ── Stage D: Final fine-tuning ──
    for epoch in range(epochs_b):
        print(f"\n[Stage D - Epoch {epoch + 1}/{epochs_b}]")
        train_fn(train_loader, model, optimizer, loss_fn, scaler, device)
        dice_score, iou_score, pixel_acc = val_fn(val_loader, model, device)
        if dice_score >= best_dice:
            best_dice = dice_score
            save_checkpoint(
                {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "best_dice": best_dice},
                checkpoint_path,
            )

    gc.collect()
    torch.cuda.empty_cache()

    print(f"\nTraining complete. Best Dice: {best_dice:.4f}")
    print(f"Checkpoint saved to: {checkpoint_path}")

    return model, device, args, train_dataset, train_loader, val_dataset, val_loader
