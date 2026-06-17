"""
One-time download of Swin Transformer pretrained weights from timm.

Run this once on a machine with a good internet connection.
The weights are saved locally so future runs never need to re-download.

Usage:
    python scripts/download_swin_weights.py

    # Specify a custom save directory
    python scripts/download_swin_weights.py --save_dir /data/pretrained_weights

After running, pass the saved path to training:
    python run_all.py --model swin_unet_tiny --dataset ACDC \
                      --pretrained_path ./pretrained_weights/swin_tiny_patch4_window7_224.pth
"""

import argparse
import os
import torch
import timm


MODELS = {
    "swin_tiny_patch4_window7_224":  "swin_tiny_patch4_window7_224.pth",
    "swin_small_patch4_window7_224": "swin_small_patch4_window7_224.pth",
}


def download(model_name, save_path):
    print(f"\nDownloading: {model_name}")
    print(f"Save path  : {save_path}")

    if os.path.exists(save_path):
        print(f"  Already exists, skipping.")
        return

    model = timm.create_model(model_name, pretrained=True)
    torch.save(model.state_dict(), save_path)
    size_mb = os.path.getsize(save_path) / (1024 ** 2)
    print(f"  Saved ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Download Swin Transformer pretrained weights")
    parser.add_argument("--save_dir", default="D:\Project Repo\ML MedSeg Project\medseg\data\pretrained_weights",
                        help="Directory to save weight files")
    parser.add_argument("--model", default="all",
                        choices=["all"] + list(MODELS.keys()),
                        help="Which model to download (default: all)")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    targets = MODELS if args.model == "all" else {args.model: MODELS[args.model]}

    for model_name, filename in targets.items():
        save_path = os.path.join(args.save_dir, filename)
        download(model_name, save_path)

    print(f"\nAll done. Weights saved to: {args.save_dir}/")
    print("\nTo use in training, add --pretrained_path to your command:")
    for model_name, filename in targets.items():
        variant = "swin_unet_tiny" if "tiny" in model_name else "swin_unet_small"
        print(f"  python run_all.py --model {variant} --dataset ACDC "
              f"--pretrained_path {args.save_dir}/{filename}")


if __name__ == "__main__":
    main()
