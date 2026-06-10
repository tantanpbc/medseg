"""
UNet-Vanilla training strategy — two-stage training with progressive
LR reduction. No encoder freeze/unfreeze is needed since the model
contains no pretrained backbone.

Stages:
    1 – Full-model training at higher LR     (epochs_a, lr_a)
    2 – Reduced LR fine-tuning               (epochs_b, lr_b)
"""

import gc

import torch
import torch.nn as nn
import torch.optim as optim

from utils import (
    save_checkpoint, load_checkpoint,
    train_fn, val_fn,
)


def train(model_name, model, device, args, model_config,
          train_dataset=None, train_loader=None, val_dataset=None, val_loader=None):
    """
    Run the 2-stage UNet-Vanilla training loop.

    Returns:
        (model, device, args, train_dataset, train_loader, val_dataset, val_loader)
    """
    hp = model_config["hyperparams"]

    # Allow CLI overrides for the primary LR
    lr_a = args.lr if args.lr else hp["lr_a"]
    lr_b = hp["lr_b"]
    epochs_a = args.epochs if args.epochs else hp["epochs_a"]
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

    # ── Stage 1: Full-model training ──
    optimizer = optim.AdamW(
        model.parameters(),
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

    # ── Stage 2: Reduced LR fine-tuning ──
    for pg in optimizer.param_groups:
        pg["lr"] = lr_b

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

    gc.collect()
    torch.cuda.empty_cache()

    print(f"\nTraining complete. Best Dice: {best_dice:.4f}")
    print(f"Checkpoint saved to: {checkpoint_path}")

    return model, device, args, train_dataset, train_loader, val_dataset, val_loader