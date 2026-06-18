"""
Model registry — factory function for building model instances by name.

To add a new model:
    1. Create ``models/your_model.py`` with the architecture class.
    2. Add an entry to ``build_model()`` and ``get_backbone_attr()`` below.
    3. Add a corresponding entry in ``config.MODEL_CONFIGS``.
"""


def build_model(model_name, **kwargs):
    if model_name == "deeplabv3_resnet50":
        from models.deeplabv3 import DeepLabv3plus
        return DeepLabv3plus(**kwargs)
    elif model_name in ("segformer_b0", "segformer_b5"):
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
        kwargs.pop("swin_model", None)
        kwargs.pop("decoder_channels", None)
        return SwinUNet(swin_model="swin_tiny_patch4_window7_224", **kwargs)
    elif model_name == "swin_unet_small":
        from models.swin_unet import SwinUNet
        kwargs.pop("swin_model", None)
        kwargs.pop("decoder_channels", None)
        return SwinUNet(swin_model="swin_small_patch4_window7_224", **kwargs)
    elif model_name == "swin_unet_base":
        from models.swin_unet import SwinUNet
        kwargs.pop("swin_model", None)
        kwargs.pop("decoder_channels", None)
        return SwinUNet(swin_model="swin_base_patch4_window7_224", **kwargs)
    else:
        available = [
            "segformer_b0", "segformer_b5", "deeplabv3_resnet50",
            "unet_resnet34", "unet_vanilla",
            "swin_unet_tiny", "swin_unet_small", "swin_unet_base",
        ]
        raise ValueError(f"Unknown model: {model_name!r}. Available: {available}")


def get_backbone_attr(model_name):
    """Return the attribute name of the encoder/backbone for freeze/unfreeze."""
    _BACKBONE_ATTRS = {
        "deeplabv3_resnet50": "backbone",
        "segformer_b0":       "encoder",
        "segformer_b5":       "encoder",
        "unet_resnet34":      "encoder",
        "unet_vanilla":       "encoder",
        "swin_unet_tiny":     "encoder",
        "swin_unet_small":    "encoder",
        "swin_unet_base":     "encoder",
    }
    if model_name not in _BACKBONE_ATTRS:
        raise ValueError(f"Unknown model: {model_name!r}")
    return _BACKBONE_ATTRS[model_name]