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

SEED = 123


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
IMAGE_SIZE = (256, 256)  # <-- Global resolution anchor constant

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
VAL_SPLIT    = 0.05   # 10% of patients for validation during training
TEST_SPLIT   = 0.15   # 10% of patients held out for final test evaluation
# → train_ratio = 1 - VAL_SPLIT - TEST_SPLIT = 0.80, giving an 8-1-1 split
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
            "out_channels": NUM_CLASSES,
            "decoder_dim": 256,
            "in_channels": 1,
        },
    },
    "segformer_b5": {
        "backbone_attr": "encoder",
        "pretrained_repo": "nvidia/segformer-b5-finetuned-ade-640-640",
        "encoder_config": {
            "embed_dims": [64, 128, 320, 512],
            "num_heads":  [1, 2, 5, 8],
            "depths":     [3, 6, 40, 3],
            "sr_ratios":  [8, 4, 2, 1],
            "mlp_ratio":  4,
            "drop_path_rate": 0.1,
        },
        "hyperparams": {
            "lr_a": 8e-4,
            "lr_b": 2e-5,
            "batch_size_a": 12,
            "batch_size_b": 6,
            "epochs_a": 20,
            "epochs_b": 15,
        },
        "model_kwargs": {
            "out_channels": NUM_CLASSES,
            "decoder_dim": 768,
            "in_channels": 1,
        },
    },
    "deeplabv3_resnet50": {
        "backbone_attr": "backbone",
        "hyperparams": {
            # Single-stage training, no freeze/unfreeze phases.
            # Per-component differential LR for the whole run:
            #   backbone (lower LR)  @ lr_b
            #   ASPP + decoder       @ lr_a
            # Polynomial LR decay (power=0.9) applied across all NUM_EPOCHS, per batch.
            "lr_a": 5e-4,       # LEARNING_RATE_A — ASPP + decoder
            "lr_b": 5e-5,       # LEARNING_RATE_B — backbone
            "weight_decay": 1e-4,
            "batch_size": 12,
            "epochs": 45,
        },
        "model_kwargs": {
            "output_stride": 32,
            "in_channels": 1,
            "out_channels": NUM_CLASSES,
        },
    },
    "unet_vanilla": {
        "backbone_attr": None,          
        "hyperparams": {
            "lr_a":      1e-4,          # Stage 1: full-model warm-up
            "lr_b":      1e-5,          # Stage 2: reduced-LR fine-tuning
            "batch_size": 16,
            "epochs_a":  25,
            "epochs_b":  25,
        },
        "model_kwargs": {
            "in_channels":  1,
            "out_channels": NUM_CLASSES,
            "features":     [64, 128, 256, 512],
        },
    },
    "unet_resnet34": {
        "backbone_attr": "encoder",
        "hyperparams": {
            # Matches original notebook exactly (medseg-unet-acdc-camus-ii.ipynb,
            # TRAIN_PHASES list):
            #   Phase 1 (warmup):   encoder frozen,   10 epochs @ 1e-3
            #   Phase 2 (finetune): encoder unfrozen, 20 epochs @ 1e-4
            # Each phase gets its OWN fresh Adam optimizer + CosineAnnealingLR
            # (T_max = that phase's epoch count) — not a single optimizer
            # carried across phases with a manually-lowered LR.
            "lr_a": 8e-4,        # Phase 1 (warmup) LR
            "lr_b": 8e-6,        # Phase 2 (finetune) LR
            "weight_decay": 1e-4,   # Notebook uses 1e-5 here, NOT the global default
            "batch_size": 16,        # Notebook's BATCH_SIZE
            "epochs_a": 10,         # Phase 1 epoch count
            "epochs_b": 20,         # Phase 2 epoch count
        },
        "model_kwargs": {
            "in_channels": 1,
            "out_channels": NUM_CLASSES,
        },
    },
    "swin_unet_tiny": {
        "backbone_attr": "encoder",
        "hyperparams": {
            "lr_a": 1e-3,               # Stage A: frozen encoder — decoder trains at full LR
            "lr_b": 1e-6,               # Stage B: unfrozen encoder — encoder learns slowly
            "batch_size_a": 32,
            "batch_size_b": 16,
            "epochs_a": 15,             # Decoder converges fast; early-stop handles the rest
            "epochs_b": 35,             # More budget for full fine-tuning (where gains come from)
            "stage_a_patience": 5,      # Exit Stage A early if Dice plateaus for 5 epochs
        },
        "model_kwargs": {
            "in_channels": 1,
            "out_channels": NUM_CLASSES,
            "pretrained": True,
            "swin_model": "swin_tiny_patch4_window7_224",
            # decoder_channels intentionally omitted — auto-computed from encoder channels
        },
    },
    "swin_unet_small": {
        "backbone_attr": "encoder",
        "hyperparams": {
            "lr_a": 1e-3,
            "lr_b": 5e-7,
            "batch_size_a": 16,
            "batch_size_b": 8,
            "epochs_a": 15,
            "epochs_b": 35,
            "stage_a_patience": 5,
        },
        "model_kwargs": {
            "in_channels": 1,
            "out_channels": NUM_CLASSES,
            "pretrained": True,
            "swin_model": "swin_small_patch4_window7_224",
        },
    },
    "swin_unet_base": {
        # Swin-Base: 88M params — comparable scale to SegFormer-B5 (82M params).
        # Channels: [128, 256, 512, 1024]. Auto-detected at build time.
        "backbone_attr": "encoder",
        "hyperparams": {
            "lr_a": 1e-3,               # decoder warm-up (encoder frozen)
            "lr_b": 4e-5,               # encoder fine-tune (very conservative for large model)
            "batch_size_a": 16,         # Base is ~3× larger than Tiny — reduce batch size
            "batch_size_b": 8,
            "epochs_a": 20,
            "epochs_b": 35,
            "stage_a_patience": 5,
        },
        "model_kwargs": {
            "in_channels": 1,
            "out_channels": NUM_CLASSES,
            "pretrained": True,
            "swin_model": "swin_base_patch4_window7_224",
        },
    },
}


# ────────────────────────────── Dataset registry ──────────────────────────────

DATASET_CONFIGS = {
    "ACDC": {
        "num_classes":        4,
        "in_channels":        1,
        "foreground_classes": [1, 2, 3],
        "class_names":        ["Background", "RV (Right Ventricle)", "MYO (Myocardium)", "LV (Left Ventricle)"],
        "default_acdc_dir":   "/home/tanht/medseg/data/ACDC/ACDC_preprocessed",
    },
    "CAMUS": {
        "num_classes":        4,  
        "in_channels":        1,
        "foreground_classes": [2, 3],
        "class_names":        ["Background", "RV (Right Ventricle)", "MYO (Myocardium)", "LV (Left Ventricle)"],
        "default_camus_dir":  "/home/tanht/medseg/data/CAMUS",
    },
    "KVASIR": {
        "num_classes":        2,
        "in_channels":        3,
        "foreground_classes": [1],
        "class_names":        ["Background", "Polyp"],
        "default_kvasir_dir": "/home/tanht/medseg/data/KVASIR/kvasir-seg",
    },
    "CHESTXRAY": {
        "num_classes":           2,
        "in_channels":           1,
        "foreground_classes":    [1],
        "class_names":           ["Background", "Lung"],
        "default_chestxray_dir": "/home/tanht/medseg/data/CHESTXRAY",
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

    # Global Image Size Override Config
    p.add_argument("--img_size", type=int, default=IMAGE_SIZE[0],
                   help="Global input image resolution (width/height dimensions)")

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

    # Data paths — KVASIR
    p.add_argument("--kvasir_dir",
                   default=DATASET_CONFIGS["KVASIR"]["default_kvasir_dir"],
                   help="Root of Kvasir-SEG dataset (contains images/ and masks/)")

    # Data paths — CHESTXRAY
    p.add_argument("--chestxray_dir",
                   default=DATASET_CONFIGS["CHESTXRAY"]["default_chestxray_dir"],
                   help="Root of Chest X-Ray dataset (contains train_images/ and test_images/)")

    # Output
    p.add_argument("--output_dir", default="/home/tanht/medseg/output",
                   help="Directory for checkpoints & exports")

    # Generic training overrides (applied on top of model-specific defaults)
    p.add_argument("--batch_size", type=int, default=None,
                   help="Override batch size (both stages for SegFormer)")
    p.add_argument("--epochs",     type=int, default=None,
                   help="Override total epochs")
    p.add_argument("--lr",         type=float, default=None,
                   help="Override learning rate")
    p.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--val_split",    type=float, default=VAL_SPLIT,
                   help="Fraction of patients for validation (default: 0.10, giving an 8-1-1 split)")
    p.add_argument("--test_split",   type=float, default=TEST_SPLIT,
                   help="Fraction of patients held out for final test evaluation (default: 0.10, giving an 8-1-1 split)")
    p.add_argument("--num_workers",  type=int,   default=NUM_WORKERS)
    p.add_argument("--seed",         type=int,   default=SEED)

    # Checkpoint
    p.add_argument("--load_model", action="store_true", help="Resume from checkpoint")
    p.add_argument("--checkpoint", default=None,        help="Path to .pth checkpoint")

    # Pretrained weights
    p.add_argument("--pretrained_path", default=None,
                   help="Local path to pretrained encoder weights (.pth). "
                        "Use download_swin_weights.py to generate. "
                        "If not set, weights are downloaded from timm on first use.")

    # Misc
    p.add_argument("--skip_train", action="store_true", help="Skip training; run evaluation only")
    p.add_argument("--skip_viz",   action="store_true", help="Skip matplotlib visualisation")

    args = p.parse_args()

    if args.camus_frames_dir is None:
        args.camus_frames_dir = os.path.join(args.camus_dir, "frames")
    if args.camus_masks_dir is None:
        args.camus_masks_dir = os.path.join(args.camus_dir, "masks")

    return args