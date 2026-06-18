"""
Evaluation script — computes per-class Dice, IoU, HD95, pixel accuracy,
model size, and inference latency.  Exports a CSV summary and
an NPZ file with predictions.

Usage:
    python evaluate.py --model segformer_b0 --checkpoint ./outputs/segformer_b0.pth
                       --acdc_dir /data/ACDC --output_dir ./outputs
    python evaluate.py --model deeplabv3_resnet50 --dataset ACDC
    python evaluate.py --model unet_vanilla --dataset KVASIR
                       --kvasir_dir /data/kvasir-seg/Kvasir-SEG
"""

import gc
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.ndimage import distance_transform_edt, binary_erosion, generate_binary_structure
from tqdm.auto import tqdm

from config import (
    seed_everything, parse_args,
    MODEL_CONFIGS, DATASET_CONFIGS, FIXED_VIZ_IDX,
)
from models.registry import build_model
from datasets.registry import get_dataset_kwargs, get_num_classes
from utils import (
    get_loaders,
    compute_model_size, benchmark_latency,
)


def evaluate(model, device, val_dataset, output_dir, dataset_name, model_name, split="test"):
    """Run full evaluation with per-class metrics and export results."""
    # ── Dataset-aware config (replaces all hardcoded class lists) ──
    dataset_cfg    = DATASET_CONFIGS[dataset_name.upper()]
    num_classes    = dataset_cfg["num_classes"]
    ACTIVE_CLASSES = dataset_cfg["foreground_classes"]   # [1,2,3] ACDC | [2,3] CAMUS | [1] KVASIR
    CLASS_NAMES    = {
        i: name
        for i, name in enumerate(dataset_cfg["class_names"])
        if i > 0   # skip background
    }

    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    # ---- Efficiency metrics ----
    model_size_mb = compute_model_size(model)
    latency_ms    = benchmark_latency(
        model, device,
        input_size=(1, dataset_cfg["in_channels"], 256, 256),
        warmup=5, runs=20,
    )
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Model Size: {model_size_mb:.2f} MB | Latency: {latency_ms:.2f} ms/sample")

    # Initialize storage for per-class metrics (excluding background class 0)
    class_dice_lists = {cls: [] for cls in range(1, num_classes)}
    class_iou_lists  = {cls: [] for cls in range(1, num_classes)}
    class_hd95_lists = {cls: [] for cls in range(1, num_classes)}
    all_pixel_acc = []
    all_preds     = []

    struct = generate_binary_structure(2, 1)

    model.eval()
    with torch.no_grad():
        for idx in tqdm(range(len(val_dataset)), desc="Computing metrics"):
            img, gt_mask = val_dataset[idx]
            img_input = img.unsqueeze(0).to(device)

            logits = model(img_input)
            if isinstance(logits, tuple):
                logits = logits[0]
            pred_mask = logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
            gt_np     = gt_mask.cpu().numpy().astype(np.uint8)

            # Calculate sample-level global pixel accuracy
            all_pixel_acc.append((pred_mask == gt_np).mean() * 100)

            # Calculate per-class Dice and IoU for the current sample
            for cls in range(1, num_classes):
                pred_cls     = (pred_mask == cls)
                gt_cls       = (gt_np == cls)
                intersection = (pred_cls & gt_cls).sum()
                union        = (pred_cls | gt_cls).sum()

                if gt_cls.sum() > 0 or pred_cls.sum() > 0:
                    dice = (2 * intersection) / (pred_cls.sum() + gt_cls.sum() + 1e-8)
                    iou  = intersection / (union + 1e-8)
                    class_dice_lists[cls].append(dice)
                    class_iou_lists[cls].append(iou)

            # Calculate per-class 95th percentile Hausdorff Distance (HD95)
            for cls in range(1, num_classes):
                pred_mask_cls   = (pred_mask == cls).astype(bool)
                target_mask_cls = (gt_np == cls).astype(bool)

                if not pred_mask_cls.any() and not target_mask_cls.any():
                    continue
                if not pred_mask_cls.any() or not target_mask_cls.any():
                    # Handle missing structure: penalize with maximum possible image distance
                    h, w = pred_mask.shape
                    max_dist = float(np.sqrt(h ** 2 + w ** 2))
                    class_hd95_lists[cls].append(max_dist)
                    continue

                pred_border   = pred_mask_cls ^ binary_erosion(pred_mask_cls, struct)
                target_border = target_mask_cls ^ binary_erosion(target_mask_cls, struct)

                if not pred_border.any() or not target_border.any():
                    continue

                dt_target = distance_transform_edt(~pred_border)
                dt_pred   = distance_transform_edt(~target_border)

                surface_distances = np.concatenate([dt_target[target_border], dt_pred[pred_border]])
                class_hd95_lists[cls].append(np.percentile(surface_distances, 95))

            # Accumulate predictions only for the fixed viz indices to save RAM
            if idx in FIXED_VIZ_IDX:
                all_preds.append(pred_mask)

    # Compute final mean metrics across all items for each class
    final_metrics          = {}
    mean_dice_accumulator  = []
    mean_iou_accumulator   = []
    mean_hd95_accumulator  = []

    for cls in range(1, num_classes):
        c_dice = np.mean(class_dice_lists[cls]) if class_dice_lists[cls] else 0.0
        c_iou  = np.mean(class_iou_lists[cls])  if class_iou_lists[cls]  else 0.0
        c_hd95 = np.mean(class_hd95_lists[cls]) if class_hd95_lists[cls] else 0.0

        final_metrics[f"dice_cls_{cls}"] = c_dice
        final_metrics[f"iou_cls_{cls}"]  = c_iou
        final_metrics[f"hd95_cls_{cls}"] = c_hd95

        # Only accumulate into the foreground mean if class belongs to active dataset classes
        if cls in ACTIVE_CLASSES:
            mean_dice_accumulator.append(c_dice)
            mean_iou_accumulator.append(c_iou)
            mean_hd95_accumulator.append(c_hd95)

    # Calculate overall foreground averages using active valid subsets
    dice_score = np.mean(mean_dice_accumulator) if mean_dice_accumulator else 0.0
    iou_score  = np.mean(mean_iou_accumulator)  if mean_iou_accumulator  else 0.0
    hd95_score = np.mean(mean_hd95_accumulator) if mean_hd95_accumulator else 0.0
    pixel_acc  = np.mean(all_pixel_acc)

    # ---- Print evaluation summary to console ----
    print(f"\n==================================================")
    print(f"FINAL EVALUATION RESULTS ({dataset_name})")
    print(f"==================================================")
    print(f"Overall Metrics (Valid Foreground Classes Only):")
    print(f"  Global Pixel Acc : {pixel_acc:.2f}%")
    print(f"  Mean Dice (fg)   : {dice_score:.4f}")
    print(f"  Mean IoU (fg)    : {iou_score:.4f}")
    print(f"  Mean HD95 (fg)   : {hd95_score:.2f} px")
    print(f"  Model Weight Size: {model_size_mb:.2f} MB")
    print(f"  Inference Latency: {latency_ms:.2f} ms/sample")
    print(f"--------------------------------------------------")
    print(f"Per-Class Breakdown:")
    for cls in range(1, num_classes):
        status_label = "" if cls in ACTIVE_CLASSES else " [OMITTED FROM MEAN]"
        print(f"  Class {cls} - {CLASS_NAMES.get(cls, f'Class {cls}')}{status_label}:")
        print(f"    Dice: {final_metrics[f'dice_cls_{cls}']:.4f}")
        print(f"    IoU : {final_metrics[f'iou_cls_{cls}']:.4f}")
        print(f"    HD95: {final_metrics[f'hd95_cls_{cls}']:.2f} px")
    print(f"==================================================")

    # ---- Export extended metrics CSV ----
    tag         = dataset_name.lower()
    export_data = {
        "model":          model_name,
        "dataset":        tag,
        "split":          split,          # "val" during training, "test" for final evaluation
        "pixel_accuracy": pixel_acc,
        "mean_dice":      dice_score,
        "mean_iou":       iou_score,
        "mean_hd95":      hd95_score,
        "model_size_mb":  model_size_mb,
        "latency_ms":     latency_ms,
    }
    # Dynamically flatten class-specific metrics into the dataframe row
    for cls in range(1, num_classes):
        export_data[f"dice_cls_{cls}"] = final_metrics[f"dice_cls_{cls}"]
        export_data[f"iou_cls_{cls}"]  = final_metrics[f"iou_cls_{cls}"]
        export_data[f"hd95_cls_{cls}"] = final_metrics[f"hd95_cls_{cls}"]

    csv_path   = os.path.join(output_dir, f"{model_name}_{tag}_{split}_metrics.csv")
    metrics_df = pd.DataFrame([export_data])
    metrics_df.to_csv(csv_path, index=False)

    # Export compressed predictions matrix
    npz_path = os.path.join(output_dir, f"{model_name}_{tag}_{split}_predictions.npz")
    np.savez(
        npz_path,
        predictions=np.array(all_preds),
        fixed_viz_idx=np.array(FIXED_VIZ_IDX),
    )

    print(f"Artifacts successfully exported to {output_dir}")

    return {
        "dice":          dice_score,
        "iou":           iou_score,
        "hd95":          hd95_score,
        "pixel_acc":     pixel_acc,
        "model_size_mb": model_size_mb,
        "latency_ms":    latency_ms,
        "per_class":     final_metrics,
    }


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config = MODEL_CONFIGS[args.model]
    ds_kwargs    = get_dataset_kwargs(args.dataset, args)
    hp           = model_config["hyperparams"]
    batch_size   = args.batch_size or hp.get("batch_size", hp.get("batch_size_b", 16))

    # Use the same deterministic split as training — test set is always the held-out partition
    (_, _,
     _, _,
     test_dataset, test_loader) = get_loaders(
        dataset_name = args.dataset,
        batch_size   = batch_size,
        val_split    = args.val_split,
        test_split   = args.test_split,
        num_workers  = args.num_workers,
        **ds_kwargs,
    )

    checkpoint_path = args.checkpoint or os.path.join(args.output_dir, f"{args.model}.pth")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Pass --checkpoint /path/to/model.pth or set --output_dir"
        )

    model_kwargs   = dict(model_config["model_kwargs"])
    encoder_config = model_config.get("encoder_config", {})
    model_kwargs.update(encoder_config)
    dataset_cfg = DATASET_CONFIGS[args.dataset.upper()]
    model_kwargs["in_channels"]  = dataset_cfg["in_channels"]
    model_kwargs["out_channels"] = dataset_cfg["num_classes"]
    model = build_model(args.model, **model_kwargs)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model = nn.DataParallel(model)
    model.to(device)

    # Evaluate on the held-out TEST set only
    evaluate(model, device, test_dataset, args.output_dir, args.dataset,
             model_name=args.model, split="test")


if __name__ == "__main__":
    main()