"""
DeepLabv3+ training strategy — single-stage training with per-component
differential learning rates and LambdaLR power-decay scheduler.

Key characteristics:
    - Backbone trained at lower LR (lr_b), ASPP+decoder at full LR (lr_a)
    - LambdaLR scheduler stepping per batch with polynomial decay (power=0.9)
    - Auxiliary classifier loss (0.4 weighting) active during training
"""

import gc

import torch
import torch.nn as nn

from config import PIN_MEMORY
from models.registry import get_backbone_attr
from utils import (
    save_checkpoint, load_checkpoint,
    train_fn, val_fn,
)


def train(model_name, model, device, args, model_config,
          train_dataset=None, train_loader=None,
          val_dataset=None,   val_loader=None,
          test_dataset=None,  test_loader=None):
    """
    Run the single-stage DeepLabv3+ training loop.

    Returns:
        (model, device, args, train_dataset, train_loader, val_dataset, val_loader)
    """
    hp = model_config["hyperparams"]

    # Allow CLI overrides
    lr_a = args.lr if args.lr else hp["lr_a"]
    lr_b = hp["lr_b"]
    num_epochs = args.epochs if args.epochs else hp["epochs"]

    backbone_attr = get_backbone_attr(model_name)
    checkpoint_path = args.checkpoint or f"{args.output_dir}/{model_name}.pth"

    from losses import make_loss
    loss_fn = make_loss(args.dataset, device)
    scaler  = torch.amp.GradScaler("cuda")
    best_dice = 0.0

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

    # ── Optimizer: per-component differential LR ──
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    optimizer = torch.optim.AdamW([
        {"params": getattr(base_model, backbone_attr).parameters(), "lr": lr_b},
        {"params": base_model.aspp.parameters(), "lr": lr_a},
        {"params": base_model.decoder.parameters(), "lr": lr_a},
    ], weight_decay=args.weight_decay)

    # ── Scheduler: polynomial (power) decay, stepping per batch ──
    max_iters = num_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: (1 - step / max_iters) ** 0.9
    )

    # ── Training loop ──
    for epoch in range(num_epochs):
        print(f"\n[Epoch {epoch + 1}/{num_epochs}]")
        train_fn(train_loader, model, optimizer, loss_fn, scaler, device, scheduler=scheduler)

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