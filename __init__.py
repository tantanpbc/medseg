"""
medseg — unified multi-model, multi-dataset medical image segmentation.

Supported models:
    - UNet-ResNet34
    - DeepLabv3+-ResNet50
    - SegFormer-B0

Supported datasets:
    - ACDC (HDF5 slices)
    - CAMUS (PNG frames)

Usage:
    python run_all.py --model unet_resnet34 --dataset ACDC
    python run_all.py --model deeplabv3_resnet50 --dataset CAMUS
    python run_all.py --model segformer_b0 --dataset ACDC
"""
