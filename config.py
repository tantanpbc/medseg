"""
Global configuration: seeds, constants, model configs, dataset configs,
and CLI argument parser.

Override paths via CLI arguments when running on a different server.
"""

import os
import random
import argparse

import numpy as np
import torch


# ────────────────────────────── Determinism ──────────────────────────────

SEED = 42


def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ────────────────────────────── Pipeline constants ──────────────────────────────

NUM_CLASSES = 4          # 0:BG, 1:RV, 2:MYO, 3:LV
IMAGE_SIZE = (256, 256)

# Cardiac structure colormap for visualization
ACDC_COLORMAP = {
    0: (0, 0, 0),
    1: (255, 0, 0),       # RV  - Right Ventricle
    2: (0, 200, 0),       # MYO - Myocardium
    3: (0, 0, 255),       # LV  - Left Ventricle
}

# Fixed indices for reproducible visualisation grids
FIXED_VIZ_IDX = [5, 16, 25, 35, 45, 55, 65, 75, 86, 95,
                 105, 115, 125, 135, 145, 155, 165, 175, 185, 195]

# Shared training defaults
WEIGHT_DECAY = 1e-4
VAL_SPLIT    = 0.2
NUM_WORKERS  = 2
PIN_MEMORY   = True


# ────────────────────────────── Model registry ──────────────────────────────

MODEL_CONFIGS = {
    "segformer_b0": {
        "backbone_attr": "encoder",
        "pretrained_repo": "nvidia/segformer-b0-finetuned-ade-512-512",
        "encoder_config": {
            "embed_dims": [32, 64, 160, 256],
            "num_heads":  [1, 2, 5, 8],
            "depths":     [2, 2, 2, 2],
            "sr_ratios":  [8, 4, 2, 1],
            "mlp_ratio":  4,
            "drop_path_rate": 0.1,
        },
        "hyperparams": {
            "lr_a": 8e-4,
            "lr_b": 1e-5,
            "batch_size_a": 64,
            "batch_size_b": 32,
            "epochs_a": 15,
            "epochs_b": 10,
        },
        "model_kwargs": {
            "num_classes": NUM_CLASSES,
            "decoder_dim": 256,
            "in_channels": 1,
        },
    },
    "deeplabv3_resnet50": {
        "backbone_attr": "backbone",
        "hyperparams": {
            "lr_a": 2e-4,
            "lr_b": 2e-5,
            "batch_size": 16,
            "epochs": 50,
        },
        "model_kwargs": {
            "output_stride": 16,
            "in_channels": 1,
            "out_channels": NUM_CLASSES,
        },
    },
    "unet_resnet34": {
        "backbone_attr": "encoder",
        "hyperparams": {
            "lr_a": 1e-4,
            "lr_b": 1e-5,
            "lr_c": 2e-6,
            "batch_size": 16,
            "epochs_a": 15,
            "epochs_b": 25,
            "epochs_c": 10,
        },
        "model_kwargs": {
            "in_channels": 1,
            "out_channels": NUM_CLASSES,
        },
    },
}


# ────────────────────────────── Dataset registry ──────────────────────────────

DATASET_CONFIGS = {
    "ACDC": {
        "num_classes": 4,
        "class_names": {1: "RV (Right Ventricle)", 2: "MYO (Myocardium)", 3: "LV (Left Ventricle)"},
        "default_acdc_dir": "/kaggle/input/acdc-dataset/ACDC_preprocessed",
    },
    "CAMUS": {
        "num_classes": 4,  # stored as 0,1,2,3 but only 2,3 are foreground
        "class_names": {2: "MYO (Myocardium)", 3: "LV (Left Ventricle)"},
        "default_camus_dir": "/kaggle/input/parsakh/camus-echocardiography-image-dataset/camus-echocardiography-image-dataset",
    },
}


# ────────────────────────────── CLI argument parser ──────────────────────────────

def parse_args():
    """Command-line interface for overriding defaults on a lab server."""
    p = argparse.ArgumentParser(description="Medical Image Segmentation — multi-model pipeline")

    # Model and dataset selection
    p.add_argument("--model",    default="segformer_b0",
                   choices=list(MODEL_CONFIGS.keys()),
                   help="Model architecture to train / evaluate")
    p.add_argument("--dataset",  default="ACDC",
                   choices=list(DATASET_CONFIGS.keys()),
                   help="Dataset to use")

    # Data paths — ACDC
    p.add_argument("--acdc_dir",
                   default=DATASET_CONFIGS["ACDC"]["default_acdc_dir"],
                   help="Root of ACDC preprocessed dataset")

    # Data paths — CAMUS
    p.add_argument("--camus_dir",
                   default=DATASET_CONFIGS["CAMUS"]["default_camus_dir"],
                   help="Root of CAMUS dataset")
    p.add_argument("--camus_frames_dir", default=None,
                   help="CAMUS frames/ directory (derived from camus_dir if omitted)")
    p.add_argument("--camus_masks_dir",  default=None,
                   help="CAMUS masks/  directory (derived from camus_dir if omitted)")

    # Output
    p.add_argument("--output_dir", default="/kaggle/working",
                   help="Directory for checkpoints & exports")

    # Generic training overrides (applied on top of model-specific defaults)
    p.add_argument("--batch_size", type=int, default=None,
                   help="Override batch size (both stages for SegFormer)")
    p.add_argument("--epochs",     type=int, default=None,
                   help="Override total epochs")
    p.add_argument("--lr",         type=float, default=None,
                   help="Override learning rate")
    p.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--val_split",    type=float, default=VAL_SPLIT)
    p.add_argument("--num_workers",  type=int,   default=NUM_WORKERS)
    p.add_argument("--seed",         type=int,   default=SEED)

    # Checkpoint
    p.add_argument("--load_model", action="store_true", help="Resume from checkpoint")
    p.add_argument("--checkpoint", default=None,        help="Path to .pth checkpoint")

    # Misc
    p.add_argument("--skip_train", action="store_true", help="Skip training; run evaluation only")
    p.add_argument("--skip_viz",   action="store_true", help="Skip matplotlib visualisation")

    args = p.parse_args()

    # Derive CAMUS sub-paths from camus_dir if not explicitly provided
    if args.camus_frames_dir is None:
        args.camus_frames_dir = os.path.join(args.camus_dir, "frames")
    if args.camus_masks_dir is None:
        args.camus_masks_dir = os.path.join(args.camus_dir, "masks")

    return args
