"""
Convenience script — runs training, evaluation on the held-out test set,
and visualisation in sequence.

Usage:
    python run_all.py --model swin_unet_base --dataset ACDC
    python run_all.py --model segformer_b5   --dataset ACDC
"""

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