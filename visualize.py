"""
Visualisation utilities — ground-truth / prediction / overlay grids.

Usage:
    python visualize.py --model segformer_b0 --checkpoint ./outputs/segformer_b0.pth
                        --acdc_dir /data/ACDC --output_dir ./outputs
    python visualize.py --model deeplabv3_resnet50 --dataset ACDC
"""

import os

import cv2
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless servers
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from config import (
    seed_everything, parse_args,
    MODEL_CONFIGS, FIXED_VIZ_IDX, ACDC_COLORMAP,
)
from models.registry import build_model
from datasets.registry import get_dataset_kwargs
from utils import get_loaders


def colorize_mask(mask, colormap=None):
    """Map class indices to RGB colours."""
    if colormap is None:
        colormap = ACDC_COLORMAP
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in colormap.items():
        color_mask[mask == cls_id] = color
    return color_mask


def visualize_predictions(model, device, val_dataset, output_dir,
                          dataset_name, model_name, fixed_viz_idx=None):
    """Save a side-by-side GT / Prediction / Overlay grid as PNG."""
    if fixed_viz_idx is None:
        fixed_viz_idx = FIXED_VIZ_IDX

    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    n_rows = len(fixed_viz_idx)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))

    for row, idx in enumerate(fixed_viz_idx):
        real_idx = idx % len(val_dataset)
        img, gt_mask = val_dataset[real_idx]
        img_np = img.squeeze().cpu().numpy()
        gt_np  = gt_mask.cpu().numpy().astype(np.uint8)

        img_input = img.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(img_input)
            if isinstance(logits, tuple):
                logits = logits[0]
            pred_mask = logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)

        gt_color   = colorize_mask(gt_np)
        pred_color = colorize_mask(pred_mask)

        # Normalise image for display
        img_display = np.clip(img_np, -3, 3)
        img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min() + 1e-8)
        img_rgb = cv2.cvtColor((img_display * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        overlay = cv2.addWeighted(img_rgb, 0.5, pred_color, 0.5, 0)

        axes[row, 0].imshow(gt_color)
        axes[row, 0].set_title("Ground Truth")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(pred_color)
        axes[row, 1].set_title("Prediction")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title("Overlay")
        axes[row, 2].axis("off")

    plt.tight_layout()
    tag = dataset_name.lower()
    out_path = os.path.join(output_dir, f"{model_name}_{tag}_visualisations.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Visualisation saved to: {out_path}")


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config = MODEL_CONFIGS[args.model]
    hp = model_config["hyperparams"]
    batch_size = args.batch_size or hp.get("batch_size", hp.get("batch_size_b", 16))

    ds_kwargs = get_dataset_kwargs(args.dataset, args)
    _, _, val_dataset, _ = get_loaders(
        dataset_name=args.dataset,
        batch_size=batch_size,
        train_transform=None,
        val_transform=None,
        num_workers=args.num_workers,
        val_split=args.val_split,
        **ds_kwargs,
    )

    # Resolve checkpoint path
    checkpoint_path = args.checkpoint or os.path.join(args.output_dir, f"{args.model}.pth")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Pass --checkpoint /path/to/model.pth or --output_dir with the saved .pth"
        )

    model_kwargs = dict(model_config["model_kwargs"])
    encoder_config = model_config.get("encoder_config", {})
    model_kwargs.update(encoder_config)
    model = build_model(args.model, **model_kwargs)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model = nn.DataParallel(model)
    model.to(device)

    os.makedirs(args.output_dir, exist_ok=True)
    visualize_predictions(model, device, val_dataset, args.output_dir, args.dataset, model_name=args.model)


if __name__ == "__main__":
    main()
