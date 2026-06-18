"""
Unified training entry point — builds model and data, dispatches to the
model-specific trainer.

Usage:
    python train.py --model swin_unet_base --dataset ACDC --acdc_dir /data/ACDC
    python train.py --model segformer_b5   --dataset ACDC --acdc_dir /data/ACDC
"""

import os
import importlib

import torch
import torch.nn as nn

from config import seed_everything, parse_args, MODEL_CONFIGS, PIN_MEMORY
from models.registry import build_model
from datasets.registry import get_dataset_kwargs
from utils import get_loaders, get_default_transforms


_TRAINERS = {
    "segformer_b0":       "trainers.segformer_trainer",
    "segformer_b5":       "trainers.segformer_trainer",
    "deeplabv3_resnet50": "trainers.deeplabv3_trainer",
    "unet_vanilla":       "trainers.unet_vanilla_trainer",
    "unet_resnet34":      "trainers.unet_resnet34_trainer",
    "swin_unet_tiny":     "trainers.swin_unet_trainer",
    "swin_unet_small":    "trainers.swin_unet_trainer",
    "swin_unet_base":     "trainers.swin_unet_trainer",
}


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name   = args.model
    model_config = MODEL_CONFIGS[model_name]

    print(f"Model:   {model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Device:  {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Model ──
    model_kwargs   = dict(model_config["model_kwargs"])
    encoder_config = model_config.get("encoder_config", {})
    model_kwargs.update(encoder_config)

    from config import DATASET_CONFIGS
    dataset_cfg = DATASET_CONFIGS[args.dataset]
    model_kwargs["in_channels"]  = dataset_cfg["in_channels"]
    model_kwargs["out_channels"] = dataset_cfg["num_classes"]

    if getattr(args, "pretrained_path", None):
        model_kwargs["pretrained_path"] = args.pretrained_path

    model = build_model(model_name, **model_kwargs)

    if model_name.startswith("segformer"):
        from models.segformer import load_pretrained_encoder
        repo_id = model_config.get("pretrained_repo")
        print("Loading pretrained encoder weights...")
        load_pretrained_encoder(model.encoder, repo_id=repo_id,
                                in_channels=dataset_cfg["in_channels"])

    model = nn.DataParallel(model)
    model.to(device)

    # ── Data (3-way patient-level split) ──
    hp         = model_config["hyperparams"]
    batch_size = args.batch_size or hp.get("batch_size", hp.get("batch_size_a", 16))
    ds_kwargs  = get_dataset_kwargs(args.dataset, args)

    (train_dataset, train_loader,
     val_dataset,   val_loader,
     test_dataset,  test_loader) = get_loaders(
        dataset_name    = args.dataset,
        batch_size      = batch_size,
        train_transform = get_default_transforms(),
        val_transform   = None,
        num_workers     = args.num_workers,
        pin_memory      = PIN_MEMORY,
        val_split       = args.val_split,
        test_split      = args.test_split,
        **ds_kwargs,
    )

    # ── Dispatch to model-specific trainer ──
    trainer = importlib.import_module(_TRAINERS[model_name])
    return trainer.train(
        model_name, model, device, args, model_config,
        train_dataset = train_dataset,
        train_loader  = train_loader,
        val_dataset   = val_dataset,
        val_loader    = val_loader,
        test_dataset  = test_dataset,
        test_loader   = test_loader,
    )


if __name__ == "__main__":
    main()