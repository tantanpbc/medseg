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

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NUM_CLASSES

# Optional: try to import timm for pretrained Swin weights
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


# ────────────────────────────── Building blocks ──────────────────────────────

class DoubleConv(nn.Module):
    """Two consecutive Conv→BN→ReLU layers."""

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


# ────────────────────────────── Encoder (Swin Transformer) ──────────────────────────────

class SwinTransformerEncoder(nn.Module):
    """
    Swin Transformer encoder backbone using pretrained weights from timm.

    Extracts hierarchical features at 4 resolution levels (1/4, 1/8, 1/16, 1/32 of input)
    with skip connections at each stage.

    Supports grayscale input by averaging pretrained RGB weights.
    """

    def __init__(self, in_channels=1, pretrained=True, model_name="swin_tiny_patch4_window7_224"):
        super(SwinTransformerEncoder, self).__init__()

        if not TIMM_AVAILABLE:
            raise RuntimeError(
                "timm library required for Swin-UNet. "
                "Install with: pip install timm"
            )

        # Load pretrained Swin model from timm
        self.swin = timm.create_model(model_name, pretrained=pretrained, features_only=True)

        # Adapt for grayscale input if needed
        if in_channels == 1:
            self._adapt_to_grayscale()

    def _adapt_to_grayscale(self):
        """Average pretrained RGB patch embedding weights to single channel."""
        # Access the patch embed layer (first layer of Swin)
        # timm models structure: swin.patch_embed or similar
        # Safe approach: iterate and find Conv2d with in_channels=3
        for name, module in self.swin.named_modules():
            if isinstance(module, nn.Conv2d) and module.in_channels == 3:
                # Found RGB input layer, adapt to grayscale
                old_weight = module.weight.data  # (out_channels, 3, ks, ks)
                new_weight = old_weight.mean(dim=1, keepdim=True)  # (out_channels, 1, ks, ks)
                # Replace with a new Conv2d that accepts single channel
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
                # Replace in the model
                # Find parent and replace
                parts = name.rsplit(".", 1)
                if len(parts) == 2:
                    parent_name, child_name = parts
                    parent = dict(self.swin.named_modules())[parent_name]
                    setattr(parent, child_name, new_conv)
                break

    def forward(self, x):
        """
        Forward pass returns features at 4 scales + list of skip connections.

        Returns:
            bottleneck: deepest feature map (1/32 resolution)
            skip_connections: list of features at progressively deeper levels
        """
        # Swin with features_only=True returns a list of 4 feature maps
        # out[0]: 1/4 resolution
        # out[1]: 1/8 resolution
        # out[2]: 1/16 resolution
        # out[3]: 1/32 resolution (bottleneck)
        features = self.swin(x)

        # Reverse so we have [1/4, 1/8, 1/16] and bottleneck separate
        skip_connections = features[:-1][::-1]  # [1/16, 1/8, 1/4]
        bottleneck = features[-1]  # 1/32

        return bottleneck, skip_connections


# ────────────────────────────── Decoder ──────────────────────────────

class DecoderModule(nn.Module):
    """Single decoder block: transposed conv upsample + skip concat + double conv."""

    def __init__(self, in_channels, out_channels):
        super(DecoderModule, self).__init__()
        self.up_sample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x, skip_connection):
        x = self.up_sample(x)
        # Guard against spatial mismatches (especially with transformer features)
        if skip_connection.shape[2:] != x.shape[2:]:
            x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip_connection, x], dim=1))


class Decoder(nn.Module):
    """Stack of DecoderModules that progressively upsample."""

    def __init__(self, features):
        """
        Args:
            features: List of channel dimensions for decoder levels.
                      E.g., [768, 384, 192, 96] for Swin-Tiny
        """
        super(Decoder, self).__init__()
        self.stages = nn.ModuleList([
            DecoderModule(features[i], features[i + 1])
            for i in range(len(features) - 1)
        ])

    def forward(self, x, skip_connections):
        """
        Args:
            x: bottleneck feature map
            skip_connections: list of skip features from encoder (deepest → shallowest)
        """
        for i, stage in enumerate(self.stages):
            x = stage(x, skip_connections[i])
        return x


# ────────────────────────────── Full model ──────────────────────────────

class SwinUNet(nn.Module):
    """
    Swin-UNet: Hybrid Vision Transformer + CNN decoder for semantic segmentation.

    A Swin Transformer backbone serves as the encoder, extracting hierarchical
    features with shifted-window self-attention. A CNN-style decoder with skip
    connections progressively upsamples and refines predictions.

    Args:
        in_channels:   Number of input image channels (1 for grayscale, 3 for RGB).
        out_channels:  Number of output segmentation classes.
        pretrained:    If True, load ImageNet-pretrained Swin weights.
        swin_model:    Name of timm Swin variant ('swin_tiny_patch4_window7_224',
                       'swin_small_patch4_window7_224', etc.)
        decoder_channels: Channel dimensions for decoder stages.
                          Default matches Swin-Tiny (768 → 384 → 192 → 96).

    Returns:
        logits: (batch_size, out_channels, height, width)
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=NUM_CLASSES,
        pretrained=True,
        swin_model="swin_tiny_patch4_window7_224",
        decoder_channels=[768, 384, 192, 96],
    ):
        super(SwinUNet, self).__init__()

        self.encoder = SwinTransformerEncoder(
            in_channels=in_channels,
            pretrained=pretrained,
            model_name=swin_model,
        )

        # Decoder: progressively upsample from bottleneck to output resolution
        # Bottleneck (768 for Swin-Tiny) → decoder_channels[0]
        decoder_in_channels = [decoder_channels[0]] + decoder_channels[:-1]
        self.decoder = Decoder(decoder_in_channels + [decoder_channels[-1]])

        # Final classification head: map final decoder features to class logits
        self.final_conv = nn.Conv2d(decoder_channels[-1], out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass: encode → decode → classify.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            logits: Predicted class logits (batch_size, out_channels, height, width)
        """
        # Encode
        bottleneck, skip_connections = self.encoder(x)

        # Decode
        x = self.decoder(bottleneck, skip_connections)

        # Classify
        return self.final_conv(x)


# ────────────────────────────── Convenience factory ──────────────────────────────

def swin_unet_tiny(in_channels=1, out_channels=NUM_CLASSES, pretrained=True):
    """Swin-UNet with Swin-Tiny encoder."""
    return SwinUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        pretrained=pretrained,
        swin_model="swin_tiny_patch4_window7_224",
        decoder_channels=[768, 384, 192, 96],
    )


def swin_unet_small(in_channels=1, out_channels=NUM_CLASSES, pretrained=True):
    """Swin-UNet with Swin-Small encoder."""
    return SwinUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        pretrained=pretrained,
        swin_model="swin_small_patch4_window7_224",
        decoder_channels=[768, 384, 192, 96],
    )
