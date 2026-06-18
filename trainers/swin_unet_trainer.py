"""
Swin-UNet training strategy — 2-stage training with encoder freeze/unfreeze.

Stages:
    A  – Frozen encoder, decoder-only warm-up      (epochs_a epochs)
    B  – Unfrozen encoder, differential LR          (epochs_b epochs)

Key design decisions:
    - Combined CE + Dice loss: directly optimises the evaluation metric
    - Weighted CE: handles ACDC class imbalance (BG ~90% of pixels)
    - Cosine annealing: prevents plateau, no warmup in Stage B (decoder already trained)
    - Gradient clipping: stabilises Swin attention blocks during fine-tuning
    - Stage A exits early if val Dice stops improving (patience=5)
    - Stage A shortened vs original: decoder converges fast, unlock encoder sooner
"""

import gc

import torch
import torch.nn as nn
import torch.optim as optim

from config import PIN_MEMORY
from models.registry import get_backbone_attr
from utils import (
    get_loaders, get_default_transforms,
    save_checkpoint,
    train_fn, val_fn, set_backbone_grad,
)


from losses import make_loss


# ────────────────────────────── Scheduler ──────────────────────────────

def _make_scheduler(optimizer, steps_per_epoch, epochs, warmup_epochs=0):
    """Cosine annealing with optional linear warm-up."""
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps  = epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.05, 0.5 * (1.0 + torch.cos(torch.tensor(3.14159265 * progress)).item()))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ────────────────────────────── Training helpers ──────────────────────────────

def _train_epoch(train_loader, model, optimizer, loss_fn, scaler, device, scheduler,
                 clip_grad=1.0):
    """One training epoch with gradient clipping."""
    model.train()
    from tqdm.auto import tqdm
    loop = tqdm(train_loader, desc="Training", leave=True)

    for imgs, targets in loop:
        imgs    = imgs.to(device)
        targets = targets.long().to(device)

        with torch.autocast("cuda"):
            outputs = model(imgs)
            loss    = loss_fn(outputs, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()

        # Gradient clipping — critical for Swin attention blocks in Stage B
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        loop.set_postfix(loss=f"{loss.item():.4f}")


# ────────────────────────────── Main trainer ──────────────────────────────

def train(model_name, model, device, args, model_config,
          train_dataset=None, train_loader=None,
          val_dataset=None,   val_loader=None,
          test_dataset=None,  test_loader=None):
    """
    Run the 2-stage Swin-UNet training loop.

    Stage A: Frozen encoder warm-up  — decoder only, high LR, early-stop on plateau
    Stage B: Full fine-tuning        — encoder slow LR, decoder keeps warm-up LR

    test_dataset / test_loader are passed through untouched — evaluation on the
    held-out test set happens in evaluate.py after training, never here.

    Returns:
        (model, device, args,
         train_dataset, train_loader,
         val_dataset,   val_loader,
         test_dataset,  test_loader)
    """
    backbone_attr = get_backbone_attr(model_name)
    hp = model_config["hyperparams"]

    lr_a         = args.lr if args.lr else hp["lr_a"]
    lr_b         = hp["lr_b"]
    batch_size_a = args.batch_size if args.batch_size else hp["batch_size_a"]
    batch_size_b = args.batch_size if args.batch_size else hp["batch_size_b"]
    epochs_a     = hp["epochs_a"]
    epochs_b     = hp["epochs_b"]

    from config import DATASET_CONFIGS
    num_classes = DATASET_CONFIGS[args.dataset]["num_classes"]

    loss_fn = make_loss(args.dataset, device, num_classes)
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

    # ── Stage A: Frozen encoder, decoder warm-up ─────────────────────────────
    set_backbone_grad(model, requires_grad=False, backbone_attr=backbone_attr)

    optimizer_a = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_a,
        weight_decay=args.weight_decay,
    )
    scheduler_a = _make_scheduler(optimizer_a, len(train_loader), epochs_a, warmup_epochs=2)

    # Early stopping: if Dice doesn't improve for `patience` epochs, move to Stage B early.
    # The decoder typically converges well before epochs_a is exhausted.
    patience      = hp.get("stage_a_patience", 5)
    no_improve    = 0

    print(f"\n── Stage A: decoder warm-up ({epochs_a} epochs, lr={lr_a}) ──")
    for epoch in range(epochs_a):
        # Keep frozen encoder in eval mode — model.train() inside train_fn would
        # otherwise re-enable its BatchNorm running-stat updates.
        base = model.module if isinstance(model, nn.DataParallel) else model
        getattr(base, backbone_attr).eval()

        print(f"\n[Stage A - Epoch {epoch + 1}/{epochs_a}]")
        _train_epoch(train_loader, model, optimizer_a, loss_fn, scaler, device,
                     scheduler_a, clip_grad=1.0)
        dice_score, iou_score, pixel_acc = val_fn(val_loader, model, device)

        if dice_score > best_dice:
            best_dice  = dice_score
            no_improve = 0
            save_checkpoint(
                {"state_dict": model.state_dict(),
                 "optimizer":  optimizer_a.state_dict(),
                 "best_dice":  best_dice},
                checkpoint_path,
            )
        else:
            no_improve += 1
            print(f"  No improvement for {no_improve}/{patience} epochs.")
            if no_improve >= patience:
                print(f"  Early stopping Stage A — moving to fine-tuning.")
                break

    # ── Rebuild loaders with (possibly) smaller batch for Stage B ────────────
    from datasets.registry import get_dataset_kwargs
    ds_kwargs = get_dataset_kwargs(args.dataset, args)
    (train_dataset, train_loader,
     val_dataset,   val_loader,
     test_dataset,  test_loader) = get_loaders(
        dataset_name    = args.dataset,
        batch_size      = batch_size_b,
        train_transform = get_default_transforms(),
        val_transform   = None,
        num_workers     = args.num_workers,
        pin_memory      = PIN_MEMORY,
        val_split       = args.val_split,
        test_split      = args.test_split,
        **ds_kwargs,
    )

    # ── Stage B: Full fine-tuning with differential LRs ──────────────────────
    # Encoder:  lr_b (very small, e.g. 1e-6) — preserve pretrained features
    # Decoder:  lr_a (same as warm-up)        — keep adapting
    # No warmup here — decoder is already well-trained, jump straight to cosine decay.
    set_backbone_grad(model, requires_grad=True, backbone_attr=backbone_attr)
    base_model = model.module if isinstance(model, nn.DataParallel) else model

    optimizer_b = optim.AdamW([
        {"params": getattr(base_model, backbone_attr).parameters(), "lr": lr_b},
        {"params": base_model.decoder.parameters(),                 "lr": lr_a},
        {"params": base_model.final_conv.parameters(),              "lr": lr_a},
    ], weight_decay=args.weight_decay)
    # No warmup for Stage B — cosine decays smoothly from lr_a/lr_b
    scheduler_b = _make_scheduler(optimizer_b, len(train_loader), epochs_b, warmup_epochs=0)

    print(f"\n── Stage B: full fine-tuning ({epochs_b} epochs, "
          f"encoder_lr={lr_b}, decoder_lr={lr_a}) ──")
    for epoch in range(epochs_b):
        print(f"\n[Stage B - Epoch {epoch + 1}/{epochs_b}]")
        _train_epoch(train_loader, model, optimizer_b, loss_fn, scaler, device,
                     scheduler_b, clip_grad=0.5)   # tighter clip for fine-tuning
        dice_score, iou_score, pixel_acc = val_fn(val_loader, model, device)

        if dice_score > best_dice:
            best_dice = dice_score
            save_checkpoint(
                {"state_dict": model.state_dict(),
                 "optimizer":  optimizer_b.state_dict(),
                 "best_dice":  best_dice},
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