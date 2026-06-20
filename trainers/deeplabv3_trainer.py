"""
DeepLabv3+ training strategy — single-stage training with per-component
differential learning rates and a polynomial (power-law) LR decay schedule.

This matches the original reference script's training loop exactly:
    - No freeze/unfreeze phases — backbone, ASPP, and decoder are all
      trainable from epoch 0, just at different learning rates.
    - backbone        @ lr_b  (lower LR — preserve pretrained ImageNet features)
    - ASPP + decoder  @ lr_a  (full LR  — randomly initialised, need to learn fast)
    - LambdaLR scheduler with lr_lambda = (1 - step/max_iters) ** 0.9,
      stepped once per training BATCH (not per epoch) across all NUM_EPOCHS.

Loss function:
    Uses the shared losses.make_loss() (weighted CrossEntropy + 0.5 * soft Dice),
    same as every other model in this repo, so Dice/IoU results are directly
    comparable across models. This differs from the original reference script
    (plain unweighted CrossEntropyLoss) — only the training schedule/optimizer
    structure is kept notebook-faithful, not the loss.
"""

import gc

import torch
import torch.nn as nn

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
        (model, device, args,
         train_dataset, train_loader,
         val_dataset,   val_loader,
         test_dataset,  test_loader)
    """
    hp = model_config["hyperparams"]

    # Allow CLI overrides
    lr_a         = args.lr if args.lr else hp["lr_a"]
    lr_b         = hp["lr_b"]
    weight_decay = hp.get("weight_decay", args.weight_decay)
    num_epochs   = args.epochs if args.epochs else hp["epochs"]

    backbone_attr   = get_backbone_attr(model_name)
    checkpoint_path = args.checkpoint or f"{args.output_dir}/{model_name}_{args.dataset.lower()}.pth"

    # Shared loss — weighted CE + 0.5 * soft Dice, same as all other trainers.
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

    # ── Optimizer: per-component differential LR, no freezing ──
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    optimizer = torch.optim.AdamW([
        {"params": getattr(base_model, backbone_attr).parameters(), "lr": lr_b},
        {"params": base_model.aspp.parameters(),                    "lr": lr_a},
        {"params": base_model.decoder.parameters(),                 "lr": lr_a},
    ], weight_decay=weight_decay)

    # ── Scheduler: polynomial (power) decay, stepping per batch across the FULL run ──
    max_iters = num_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: (1 - step / max_iters) ** 0.9
    )

    # ── Single-stage training loop ──
    for epoch in range(num_epochs):
        print(f"\n[Epoch {epoch + 1}/{num_epochs}]")
        train_fn(train_loader, model, optimizer, loss_fn, scaler, device, scheduler=scheduler)

        dice_score, iou_score, pixel_acc = val_fn(val_loader, model, device, dataset_name=args.dataset)
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