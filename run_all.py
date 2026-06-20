"""
Convenience script — runs training, evaluation on the held-out test set,
and visualisation in sequence.

Usage:
    python run_all.py --model swin_unet_base --dataset ACDC
    python run_all.py --model segformer_b5   --dataset ACDC
"""

import os

import torch

from config import parse_args
from train import main as train_main
from evaluate import evaluate
from visualize import visualize_predictions


def main():
    args = parse_args()

    print("=" * 60)
    print(f"  Model:   {args.model}")
    print(f"  Dataset: {args.dataset}")
    print("=" * 60)

    # ── Train ──
    (model, device, args,
     _,             _,
     _,             _,
     test_dataset,  _) = train_main()

    # ── Reload the BEST checkpoint before final evaluation ──
    # The `model` returned by train_main() holds whatever weights were live
    # at the end of the LAST training epoch, which is not necessarily the
    # best-Dice checkpoint saved during training (cosine/poly LR schedules
    # can leave the final epoch slightly worse than an earlier peak).
    # Always reload from disk so test-set numbers reflect the checkpoint
    # that was actually selected by validation, not an arbitrary final state.
    checkpoint_path = args.checkpoint or os.path.join(args.output_dir, f"{args.model}_{args.dataset.lower()}.pth")
    if os.path.isfile(checkpoint_path):
        print(f"\nLoading best checkpoint for evaluation: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        base_model = model.module if hasattr(model, "module") else model
        base_model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
        best_dice = ckpt.get("best_dice")
        if best_dice is not None:
            print(f"  (checkpoint best_dice = {best_dice:.4f})")
    else:
        print(f"\nWARNING: no checkpoint found at {checkpoint_path} — "
              f"evaluating with the final in-memory model state instead.")

    # ── Evaluate on held-out TEST set ──
    print("\n" + "=" * 60)
    print("FINAL TEST SET EVALUATION")
    print("=" * 60)
    metrics = evaluate(model, device, test_dataset, args.output_dir, args.dataset,
                       model_name=args.model, split="test")

    # ── Visualise ──
    if not args.skip_viz:
        print("\n" + "=" * 60)
        print("VISUALISATION")
        print("=" * 60)
        visualize_predictions(model, device, test_dataset, args.output_dir, args.dataset,
                              model_name=args.model)


if __name__ == "__main__":
    main()