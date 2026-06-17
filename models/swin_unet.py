"""
Swin-UNet architecture: Hybrid vision transformer + CNN decoder.
 
A Swin Transformer encoder (with pretrained weights from timm) combined with
a CNN-style decoder featuring skip connections and progressive upsampling.
 
Architecture:
    - Encoder: Swin-Tiny pretrained backbone (4 stages with shifted windows)
    - Decoder: 4 progressive ConvTranspose2d blocks with skip concatenation
    - Final head: 1×1 convolution to class logits
 
The encoder is loaded from timm pretrained weights (ImageNet-1K), with automatic
grayscale input adaptation by averaging RGB channel weights.
 
Key components:
    - SwinTransformer encoder: hierarchical feature extraction via local self-attention
    - Skip connections: concatenated at each decoder level
    - Decoder blocks: transposed conv + double conv
    - Hybrid approach: transformer for global context, CNN for precise boundaries
 
References:
    Swin-Unet: Unet Transformers for Semantic Segmentation
    (Cao et al., arXiv 2021)
    
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    (Liu et al., ICCV 2021)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NUM_CLASSES, IMAGE_SIZE

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SwinTransformerEncoder(nn.Module):
    def __init__(self, in_channels=1, pretrained=True, model_name="swin_tiny_patch4_window7_224",
                 pretrained_path=None, img_size=IMAGE_SIZE[0]):
        super(SwinTransformerEncoder, self).__init__()

        if not TIMM_AVAILABLE:
            raise RuntimeError(
                "timm library required for Swin-UNet. Install with: pip install timm"
            )

        if pretrained and pretrained_path is None:
            local_check_path = f"/home/tanht/medseg/data/pretrained_weights/{model_name}.pth"
            if os.path.exists(local_check_path):
                pretrained_path = local_check_path

        if pretrained and pretrained_path is not None:
            print(f"Loading Swin weights from local file via timm remapper: {pretrained_path}")
            self.swin = timm.create_model(
                model_name,
                pretrained=True,
                features_only=True,
                img_size=img_size,
                pretrained_cfg_overlay=dict(file=pretrained_path)
            )
        else:
            self.swin = timm.create_model(model_name, pretrained=pretrained, features_only=True, img_size=img_size)

        if in_channels == 1:
            self._adapt_to_grayscale()

    def _adapt_to_grayscale(self):
        for name, module in self.swin.named_modules():
            if isinstance(module, nn.Conv2d) and module.in_channels == 3:
                old_weight = module.weight.data
                new_weight = old_weight.mean(dim=1, keepdim=True)
                new_conv = nn.Conv2d(
                    1,
                    module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    bias=(module.bias is not None),
                )
                new_conv.weight.data = new_weight
                if module.bias is not None:
                    new_conv.bias.data = module.bias.data
                parts = name.rsplit(".", 1)
                if len(parts) == 2:
                    parent_name, child_name = parts
                    parent = dict(self.swin.named_modules())[parent_name]
                    setattr(parent, child_name, new_conv)
                break

    def forward(self, x):
        features = self.swin(x)
        skip_connections = features[:-1][::-1]  
        bottleneck = features[-1]  
        return bottleneck, skip_connections


class DecoderModule(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderModule, self).__init__()
        self.up_sample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x, skip_connection):
        x = self.up_sample(x)
        if skip_connection.shape[2:] != x.shape[2:]:
            x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip_connection, x], dim=1))


class Decoder(nn.Module):
    def __init__(self, channels):
        super(Decoder, self).__init__()
        self.stages = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.stages.append(DecoderModule(channels[i], channels[i+1], channels[i+1]))
        
        self.final_up = nn.ConvTranspose2d(channels[-1], channels[-1], kernel_size=4, stride=4)

    def forward(self, x, skip_connections):
        for i, stage in enumerate(self.stages):
            x = stage(x, skip_connections[i])
        x = self.final_up(x)
        return x


class SwinUNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=NUM_CLASSES,
        pretrained=True,
        swin_model="swin_tiny_patch4_window7_224",
        decoder_channels=[768, 384, 192, 96],
        pretrained_path=None,
        img_size=IMAGE_SIZE[0], 
    ):
        super(SwinUNet, self).__init__()

        self.encoder = SwinTransformerEncoder(
            in_channels=in_channels,
            pretrained=pretrained,
            model_name=swin_model,
            pretrained_path=pretrained_path,
            img_size=img_size,
        )

        self.decoder = Decoder(decoder_channels)
        self.final_conv = nn.Conv2d(decoder_channels[-1], out_channels, kernel_size=1)

    def forward(self, x):
        bottleneck, skip_connections = self.encoder(x)

        if bottleneck.dim() == 4 and bottleneck.shape[-1] in [96, 192, 384, 768]:
            bottleneck = bottleneck.permute(0, 3, 1, 2).contiguous()
            
        permuted_skips = []
        for skip in skip_connections:
            if skip.dim() == 4 and skip.shape[-1] in [96, 192, 384, 768]:
                permuted_skips.append(skip.permute(0, 3, 1, 2).contiguous())
            else:
                permuted_skips.append(skip)

        x = self.decoder(bottleneck, permuted_skips)
        return self.final_conv(x)


def swin_unet_tiny(in_channels=1, out_channels=NUM_CLASSES, pretrained=True, img_size=IMAGE_SIZE[0], pretrained_path=None):
    return SwinUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        pretrained=pretrained,
        swin_model="swin_tiny_patch4_window7_224",
        decoder_channels=[768, 384, 192, 96],
        pretrained_path=pretrained_path,
        img_size=img_size,
    )


def swin_unet_small(in_channels=1, out_channels=NUM_CLASSES, pretrained=True, img_size=IMAGE_SIZE[0], pretrained_path=None):
    return SwinUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        pretrained=pretrained,
        swin_model="swin_small_patch4_window7_224",
        decoder_channels=[768, 384, 192, 96],
        pretrained_path=pretrained_path,
        img_size=img_size,
    )