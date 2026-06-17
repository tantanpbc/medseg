"""
Model registry — factory function for building model instances by name.

To add a new model:
    1. Create ``models/your_model.py`` with the architecture class.
    2. Add an entry to ``build_model()`` and ``get_backbone_attr()`` below.
    3. Add a corresponding entry in ``config.MODEL_CONFIGS``.
"""


def build_model(model_name, **kwargs):
    """
    Instantiate the correct model class by name.

    Args:
        model_name: One of ``"segformer_b0"``, ``"deeplabv3_resnet50"``.
        **kwargs:   Forwarded to the model constructor.

    Returns:
        A ``nn.Module`` instance.
    """
    if model_name == "deeplabv3_resnet50":
        from models.deeplabv3 import DeepLabv3plus
        return DeepLabv3plus(**kwargs)
    elif model_name == "segformer_b0":
        from models.segformer import SegFormer
        return SegFormer(**kwargs)
    elif model_name == "unet_resnet34":
        from models.unet_resnet34 import UNetResnet34
        return UNetResnet34(**kwargs)
    elif model_name == "unet_vanilla":
        from models.unet_vanilla import UNetVanilla
        return UNetVanilla(**kwargs)
    elif model_name == "swin_unet_tiny":
        from models.swin_unet import SwinUNet
        return SwinUNet(
            swin_model="swin_tiny_patch4_window7_224",
            decoder_channels=[768, 384, 192, 96],
            **kwargs
        )
    else:
        raise ValueError(
            f"Unknown model: {model_name!r}. "
            f"Available: 'segformer_b0', 'deeplabv3_resnet50', 'unet_resnet34', 'unet_vanilla'."
        )


def get_backbone_attr(model_name):
    """
    Return the name of the backbone/encoder attribute for freeze/unfreeze.

    DeepLabv3+ exposes its pretrained encoder as ``model.backbone``,
    while SegFormer uses ``model.encoder``.
    """
    if model_name == "deeplabv3_resnet50":
        return "backbone"
    elif model_name == "segformer_b0":
        return "encoder"
    elif model_name == "unet_resnet34":
        return "encoder"
    elif model_name == "unet_vanilla":
        return "encoder"
    else:
        raise ValueError(f"Unknown model: {model_name!r}")
