"""
UNet-ResNet34 training strategy — three-stage training with
encoder freeze/unfreeze and progressive LR reduction.

Stages:
    1 – Frozen encoder (ResNet34), decoder-only warm-up   (epochs_a, lr_a)
    2 – Unfrozen encoder, full-model fine-tuning           (epochs_b, lr_b)
    3 – Reduced LR fine-tuning on same optimizer           (epochs_c, lr_c)
"""

import gc

import torch
import torch.nn as nn
import torch.optim as optim

from models.registry import get_backbone_attr
from utils import (
    save_checkpoint, load_checkpoint,
    train_fn, val_fn, set_backbone_grad,
)


def train(model_name, model, device, args, model_config,
          train_dataset=None, train_loader=None,
          val_dataset=None,   val_loader=None,
          test_dataset=None,  test_loader=None):
    """
    Run the 3-stage UNet-ResNet34 training loop.

    Returns:
        (model, device, args, train_dataset, train_loader, val_dataset, val_loader)
    """
    backbone_attr = get_backbone_attr(model_name)
    hp = model_config["hyperparams"]

    # Allow CLI overrides
    lr_a = args.lr if args.lr else hp["lr_a"]
    lr_b = hp["lr_b"]
    lr_c = hp["lr_c"]
    epochs_a = hp["epochs_a"]
    epochs_b = hp["epochs_b"]
    epochs_c = hp["epochs_c"]

    from losses import make_loss
    loss_fn = make_loss(args.dataset, device)
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
        return (model, device, args,
            train_dataset, train_loader,
            val_dataset,   val_loader,
            test_dataset,  test_loader)

    # ── Stage 1: Frozen backbone warm-up ──
    set_backbone_grad(model, requires_grad=False, backbone_attr=backbone_attr)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_a,
        weight_decay=args.weight_decay,
    )

    for epoch in range(epochs_a):
        print(f"\n[Stage 1 - Epoch {epoch + 1}/{epochs_a}]")
        train_fn(train_loader, model, optimizer, loss_fn, scaler, device)
        dice_score, iou_score, pixel_acc = val_fn(val_loader, model, device)
        if dice_score >= best_dice:
            best_dice = dice_score
            save_checkpoint(
                {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "best_dice": best_dice},
                checkpoint_path,
            )

    # ── Stage 2: Unfrozen backbone fine-tuning ──
    set_backbone_grad(model, requires_grad=True, backbone_attr=backbone_attr)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_b,
        weight_decay=args.weight_decay,
    )

    for epoch in range(epochs_b):
        print(f"\n[Stage 2 - Epoch {epoch + 1}/{epochs_b}]")
        train_fn(train_loader, model, optimizer, loss_fn, scaler, device)
        dice_score, iou_score, pixel_acc = val_fn(val_loader, model, device)
        if dice_score >= best_dice:
            best_dice = dice_score
            save_checkpoint(
                {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "best_dice": best_dice},
                checkpoint_path,
            )

    # ── Stage 3: Reduced LR fine-tuning ──
    for pg in optimizer.param_groups:
        pg["lr"] = lr_c

    for epoch in range(epochs_c):
        print(f"\n[Stage 3 - Epoch {epoch + 1}/{epochs_c}]")
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

    return (model, device, args,
            train_dataset, train_loader,
            val_dataset,   val_loader,
            test_dataset,  test_loader)