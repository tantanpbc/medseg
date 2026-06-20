"""
UNet-ResNet34 training strategy — 2-phase training matching the original
notebook plan exactly (medseg-unet-acdc-camus-ii.ipynb, TRAIN_PHASES):

    Phase 1 (warmup):    encoder frozen,   10 epochs @ lr_a=1e-3
    Phase 2 (finetune):  encoder unfrozen, 20 epochs @ lr_b=1e-4

Each phase builds a FRESH Adam optimizer (only over currently-trainable
params) plus its own CosineAnnealingLR(T_max=phase_epochs), exactly as the
notebook does — this is not the same as decaying a single optimizer across
phase boundaries, since starting a fresh cosine schedule at full lr_a/lr_b
gives a different LR trajectory than continuing a single decay curve.

weight_decay defaults to 1e-5 here (notebook-specific), not the repo-wide
--weight_decay CLI default, unless explicitly overridden via hyperparams.
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
    Run the 2-phase UNet-ResNet34 training loop (warmup → finetune).

    Returns:
        (model, device, args,
         train_dataset, train_loader,
         val_dataset,   val_loader,
         test_dataset,  test_loader)
    """
    backbone_attr = get_backbone_attr(model_name)
    hp = model_config["hyperparams"]

    # Allow CLI overrides
    lr_a         = args.lr if args.lr else hp["lr_a"]
    lr_b         = hp["lr_b"]
    epochs_a     = hp["epochs_a"]
    epochs_b     = hp["epochs_b"]
    weight_decay = hp.get("weight_decay", args.weight_decay)   # notebook uses 1e-5

    from losses import make_loss
    loss_fn = make_loss(args.dataset, device)
    scaler  = torch.amp.GradScaler("cuda")
    best_dice = 0.0

    checkpoint_path = args.checkpoint or f"{args.output_dir}/{model_name}_{args.dataset.lower()}.pth"

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

    # ── Phase 1: Encoder frozen, warm-up ──
    set_backbone_grad(model, requires_grad=False, backbone_attr=backbone_attr)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_a,
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_a)

    print(f"\n{'='*60}")
    print(f"PHASE 1: WARM-UP (Encoder Frozen) | Epochs: {epochs_a} | LR: {lr_a}")
    print(f"{'='*60}")

    for epoch in range(epochs_a):
        print(f"\n[Warm-up Epoch {epoch + 1}/{epochs_a}]")
        train_fn(train_loader, model, optimizer, loss_fn, scaler, device)
        scheduler.step()   # stepped once per epoch, matching the notebook
        dice_score, iou_score, pixel_acc = val_fn(val_loader, model, device, dataset_name=args.dataset)
        if dice_score >= best_dice:
            best_dice = dice_score
            save_checkpoint(
                {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "best_dice": best_dice},
                checkpoint_path,
            )

    # ── Phase 2: Encoder unfrozen, fine-tuning ──
    # A fresh optimizer + fresh cosine schedule, exactly as the notebook does
    # at the top of its `for phase in TRAIN_PHASES` loop.
    set_backbone_grad(model, requires_grad=True, backbone_attr=backbone_attr)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_b,
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_b)

    print(f"\n{'='*60}")
    print(f"PHASE 2: FINE-TUNE (Encoder Unfrozen) | Epochs: {epochs_b} | LR: {lr_b}")
    print(f"{'='*60}")

    for epoch in range(epochs_b):
        print(f"\n[Fine-tune Epoch {epoch + 1}/{epochs_b}]")
        train_fn(train_loader, model, optimizer, loss_fn, scaler, device)
        scheduler.step()
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