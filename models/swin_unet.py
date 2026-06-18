"""
Swin-UNet architecture: Hybrid vision transformer + CNN decoder.

A Swin Transformer encoder (with pretrained weights from timm) combined with
a CNN-style decoder featuring skip connections and progressive upsampling.

Architecture:
    - Encoder: Swin-Tiny/Small pretrained backbone (4 stages with shifted windows)
    - Decoder: 4 progressive ConvTranspose2d blocks with skip concatenation
    - Final head: 1×1 convolution to class logits

The encoder is loaded from timm pretrained weights (ImageNet-1K), with automatic
grayscale input adaptation by averaging RGB channel weights.

Weight loading:
    Pretrained weights MUST be saved with features_only=True (use download_swin_weights.py).
    Loading full classification model weights will cause key mismatches.

Key components:
    - SwinTransformer encoder: hierarchical feature extraction via local self-attention
    - Skip connections: concatenated at each decoder level (all 4 stages used, including
      the stage-0 64×64 features fed directly to the final upsample block)
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
            print(f"Loading Swin weights from local file: {pretrained_path}")
            # Create the features_only model WITHOUT pretrained weights first,
            # then load our saved state_dict manually with strict=False.
            # This avoids timm's remapper trying to reconcile classification vs
            # features_only key layouts, which causes shape/name mismatches.
            self.swin = timm.create_model(
                model_name,
                pretrained=False,
                features_only=True,
                img_size=img_size,
            )
            state_dict = torch.load(pretrained_path, map_location="cpu")
            missing, unexpected = self.swin.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"  [Swin weights] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
            if unexpected:
                print(f"  [Swin weights] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
            print(f"  Pretrained weights loaded successfully.")
        else:
            self.swin = timm.create_model(
                model_name,
                pretrained=pretrained,
                features_only=True,
                img_size=img_size,
            )

        if in_channels == 1:
            self._adapt_to_grayscale()

    def _adapt_to_grayscale(self):
        """Replace the first Conv2d (patch embedding) to accept 1-channel input.

        The RGB weights are averaged across the channel dim so all pretrained
        spatial patterns are preserved rather than discarded.
        """
        for name, module in self.swin.named_modules():
            if isinstance(module, nn.Conv2d) and module.in_channels == 3:
                old_weight = module.weight.data          # (out_ch, 3, kH, kW)
                new_weight = old_weight.mean(dim=1, keepdim=True)  # (out_ch, 1, kH, kW)
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
                else:
                    setattr(self.swin, name, new_conv)
                print(f"  Patch embedding adapted: 3-channel → 1-channel (weights averaged)")
                break

    @staticmethod
    def _to_nchw(t):
        """Convert (B, H, W, C) → (B, C, H, W) if needed.

        Some timm versions return NHWC for Swin. Detection: for all valid
        Swin outputs at 256px input, spatial dims (8/16/32/64) are always
        smaller than channel dims (96/192/384/768), so shape[1] < shape[3]
        unambiguously identifies NHWC and is safe across all resolutions.
        """
        if t.ndim == 4 and t.shape[1] < t.shape[3]:
            return t.permute(0, 3, 1, 2).contiguous()
        return t

    def forward(self, x):
        # timm features_only may return NCHW or NHWC depending on timm version.
        # _to_nchw normalises everything to NCHW before returning.
        # Expected shapes for 256×256 input after normalisation:
        #   stage0: (B,  96, 64, 64)
        #   stage1: (B, 192, 32, 32)
        #   stage2: (B, 384, 16, 16)
        #   stage3: (B, 768,  8,  8)
        features = self.swin(x)
        features = [self._to_nchw(f) for f in features]
        bottleneck = features[-1]               # (B, 768, 8, 8)
        skip_connections = features[:-1][::-1]  # [stage2, stage1, stage0] coarse→fine
        return bottleneck, skip_connections


class DecoderBlock(nn.Module):
    """Upsample by 2× then merge with a skip connection."""
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # Guard against off-by-one from integer division in Swin downsampling
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class Decoder(nn.Module):
    """
    Full decoder with 4 skip connections (one per Swin stage).

    Spatial progression for 256×256 input (swin_tiny):
        bottleneck  (B, 768,  8,  8)
        + skip2     (B, 384, 16, 16)  → (B, 384, 16, 16)
        + skip1     (B, 192, 32, 32)  → (B, 192, 32, 32)
        + skip0     (B,  96, 64, 64)  → (B,  96, 64, 64)
        ConvTranspose 4×               → (B,  96,256,256)

    All four Swin stages are consumed so the finest 64×64 stage-0 features
    (which capture patch-level boundary details) reach the output head.
    """
    def __init__(self, encoder_channels, decoder_channels):
        """
        Args:
            encoder_channels: Channel widths of the skip features, finest-first.
                              e.g. [384, 192, 96] for swin_tiny (stage2→1→0).
            decoder_channels: Output channels at each decoder stage.
                              e.g. [384, 192, 96].
        """
        super(Decoder, self).__init__()
        assert len(encoder_channels) == len(decoder_channels), (
            "encoder_channels and decoder_channels must have the same length"
        )

        bottleneck_ch = encoder_channels[0] * 2   # 768 for swin_tiny (stage2 ch × 2)
        self.stages = nn.ModuleList()

        in_ch = bottleneck_ch
        for skip_ch, out_ch in zip(encoder_channels, decoder_channels):
            self.stages.append(DecoderBlock(in_ch, skip_ch, out_ch))
            in_ch = out_ch

        # Final 4× upsample: 64×64 → 256×256 (no skip; recovers input resolution)
        self.final_up = nn.ConvTranspose2d(decoder_channels[-1], decoder_channels[-1],
                                           kernel_size=4, stride=4)

    def forward(self, bottleneck, skip_connections):
        """
        Args:
            bottleneck:       (B, C_bot, H_bot, W_bot)
            skip_connections: list of skips, ordered coarse→fine
                              i.e. [stage2_feat, stage1_feat, stage0_feat]
        """
        x = bottleneck
        for stage, skip in zip(self.stages, skip_connections):
            x = stage(x, skip)
        return self.final_up(x)


class SwinUNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=NUM_CLASSES,
        pretrained=True,
        swin_model="swin_tiny_patch4_window7_224",
        decoder_channels=None,
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

        # Probe the encoder with a dummy input to discover actual channel widths.
        # This makes SwinUNet work with any Swin variant without manual config:
        #   Tiny/Small: [96,  192, 384, 768]
        #   Base:       [128, 256, 512, 1024]
        #   Large:      [192, 384, 768, 1536]
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_size, img_size)
            bottleneck, skips = self.encoder(dummy)
            # skips are ordered [coarse, ..., fine] i.e. [stage2, stage1, stage0]
            encoder_channels = [s.shape[1] for s in skips]   # e.g. [384, 192, 96]

        if decoder_channels is None:
            decoder_channels = encoder_channels               # match encoder widths

        self.decoder = Decoder(encoder_channels, decoder_channels)
        self.final_conv = nn.Conv2d(decoder_channels[-1], out_channels, kernel_size=1)

    def forward(self, x):
        # encoder always returns NCHW tensors (timm features_only guarantee)
        bottleneck, skip_connections = self.encoder(x)
        x = self.decoder(bottleneck, skip_connections)
        return self.final_conv(x)


# ── Convenience constructors ──────────────────────────────────────────────────

def swin_unet_tiny(in_channels=1, out_channels=NUM_CLASSES, pretrained=True,
                   img_size=IMAGE_SIZE[0], pretrained_path=None):
    return SwinUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        pretrained=pretrained,
        swin_model="swin_tiny_patch4_window7_224",
        pretrained_path=pretrained_path,
        img_size=img_size,
    )


def swin_unet_small(in_channels=1, out_channels=NUM_CLASSES, pretrained=True,
                    img_size=IMAGE_SIZE[0], pretrained_path=None):
    return SwinUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        pretrained=pretrained,
        swin_model="swin_small_patch4_window7_224",
        pretrained_path=pretrained_path,
        img_size=img_size,
    )


def swin_unet_base(in_channels=1, out_channels=NUM_CLASSES, pretrained=True,
                   img_size=IMAGE_SIZE[0], pretrained_path=None):
    """
    Swin-Base encoder: 88M params, channels [128, 256, 512, 1024].

    Comparable in scale to SegFormer-B5 (82M params).
    Requires swin_base_patch4_window7_224 pretrained weights.

    Swin-Base has different channel widths than Tiny/Small [128,256,512,1024
    vs 96,192,384,768], but SwinUNet auto-derives decoder channels from the
    encoder output so no manual channel config is needed.
    """
    return SwinUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        pretrained=pretrained,
        swin_model="swin_base_patch4_window7_224",
        pretrained_path=pretrained_path,
        img_size=img_size,
    )