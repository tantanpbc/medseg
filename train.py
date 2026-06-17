"""
Unified training entry point — builds model and data, dispatches to the
model-specific trainer.

Usage:
    python train.py --model segformer_b0 --dataset ACDC --acdc_dir /data/ACDC
    python train.py --model deeplabv3_resnet50 --dataset ACDC --acdc_dir /data/ACDC
    python train.py --model segformer_b0 --dataset CAMUS --camus_dir /data/CAMUS
    python train.py --skip_train --model deeplabv3_resnet50 --checkpoint ./outputs/deeplabv3_resnet50.pth
"""

import os

import torch
import torch.nn as nn

from config import (
    seed_everything, parse_args,
    MODEL_CONFIGS, PIN_MEMORY,
)
from models.registry import build_model, get_backbone_attr
from datasets.registry import get_dataset_kwargs
from utils import get_loaders, get_default_transforms


# ────────────────────────────── Trainer dispatch ──────────────────────────────

_TRAINERS = {
    "segformer_b0":       "trainers.segformer_trainer",
    "deeplabv3_resnet50": "trainers.deeplabv3_trainer",
    "unet_vanilla":        "trainers.unet_vanilla_trainer",
    "unet_resnet34":      "trainers.unet_resnet34_trainer",
    "swin_unet_tiny":     "trainers.swin_unet_trainer",
    "swin_unet_small":    "trainers.swin_unet_trainer",
}


def _get_trainer(model_name):
    """Lazy-import and return the trainer module for the given model."""
    module_path = _TRAINERS.get(model_name)
    if module_path is None:
        raise ValueError(f"No trainer registered for model: {model_name!r}")
    import importlib
    return importlib.import_module(module_path)


# ────────────────────────────── Main ──────────────────────────────

def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name  = args.model
    model_config = MODEL_CONFIGS[model_name]

    print(f"Model:   {model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Device:  {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Model ──
    model_kwargs = dict(model_config["model_kwargs"])

    from config import DATASET_CONFIGS
    dataset_cfg = DATASET_CONFIGS[args.dataset]
    model_kwargs["in_channels"]  = dataset_cfg["in_channels"]
    model_kwargs["out_channels"] = dataset_cfg["num_classes"]
    # Merge encoder config into model kwargs (SegFormer-specific)
    encoder_config = model_config.get("encoder_config", {})
    model_kwargs.update(encoder_config)
    model = build_model(model_name, **model_kwargs)

    # Load pretrained weights (SegFormer: HuggingFace encoder; DeepLabv3: torchvision backbone)
    if model_name.startswith("segformer"):
        from models.segformer import load_pretrained_encoder
        encoder_config = model_config.get("encoder_config", {})
        # Pass encoder config to reinitialize the encoder if needed
        repo_id = model_config.get("pretrained_repo")
        print("Loading pretrained encoder weights...")
        load_pretrained_encoder(model.encoder, repo_id=repo_id, in_channels=dataset_cfg["in_channels"])

    model = nn.DataParallel(model)
    model.to(device)

    # ── Data ──
    hp = model_config["hyperparams"]
    batch_size = args.batch_size or hp.get("batch_size", hp.get("batch_size_a", 16))

    train_transforms = get_default_transforms()
    ds_kwargs = get_dataset_kwargs(args.dataset, args)

    train_dataset, train_loader, val_dataset, val_loader = get_loaders(
        dataset_name=args.dataset,
        batch_size=batch_size,
        train_transform=train_transforms,
        val_transform=None,
        num_workers=args.num_workers,
        pin_memory=PIN_MEMORY,
        val_split=args.val_split,
        **ds_kwargs,
    )

    # ── Dispatch to model-specific trainer ──
    trainer = _get_trainer(model_name)
    return trainer.train(
        model_name, model, device, args, model_config,
        train_dataset=train_dataset,
        train_loader=train_loader,
        val_dataset=val_dataset,
        val_loader=val_loader,
    )


if __name__ == "__main__":
    main()