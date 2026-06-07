"""
Convenience script — runs training, evaluation, and visualisation
in sequence with a single command.

Usage:
    python run_all.py --model segformer_b0 --dataset ACDC --acdc_dir /data/ACDC
    python run_all.py --model deeplabv3_resnet50 --dataset ACDC --acdc_dir /data/ACDC
    python run_all.py --model segformer_b0 --dataset CAMUS --camus_dir /data/CAMUS
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
    model, device, args, _, _, val_dataset, _ = train_main()

    # ── Evaluate ──
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    metrics = evaluate(model, device, val_dataset, args.output_dir, args.dataset,
                       model_name=args.model)

    # ── Visualise ──
    if not args.skip_viz:
        print("\n" + "=" * 60)
        print("VISUALISATION")
        print("=" * 60)
        visualize_predictions(model, device, val_dataset, args.output_dir, args.dataset,
                              model_name=args.model)


if __name__ == "__main__":
    main()
