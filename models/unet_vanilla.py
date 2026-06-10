"""
Vanilla UNet architecture: classic encoder-decoder with skip connections,
built entirely from scratch with no pretrained weights.

Each encoder stage doubles the channel width and halves spatial resolution
via MaxPool. Each decoder stage halves the channel width and restores
spatial resolution via transposed convolution, concatenating the matching
encoder skip connection before applying a double conv block.

References:
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    (Ronneberger et al., MICCAI 2015)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NUM_CLASSES


# ────────────────────────────── Building blocks ──────────────────────────────

class DoubleConv(nn.Module):
    """Two consecutive Conv→BN→ReLU layers — the core UNet building block."""

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


# ────────────────────────────── Encoder ──────────────────────────────

class Encoder(nn.Module):
    """
    Vanilla UNet encoder: a stack of DoubleConv blocks separated by MaxPool2d.

    Produces one skip connection per stage (before pooling) and a bottleneck
    output (after the final DoubleConv, no pooling).
    """

    def __init__(self, in_channels=1, features=[64, 128, 256, 512]):
        super(Encoder, self).__init__()
        self.stages = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        ch = in_channels
        for f in features:
            self.stages.append(DoubleConv(ch, f))
            ch = f

        # Bottleneck: double the last feature width, no pooling follows
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

    def forward(self, x):
        skip_connections = []
        for stage in self.stages:
            x = stage(x)
            skip_connections.append(x)   # save before pooling
            x = self.pool(x)
        x = self.bottleneck(x)
        return x, skip_connections


# ────────────────────────────── Decoder ──────────────────────────────

class DecoderModule(nn.Module):
    """Single decoder block: transposed conv upsample + skip concat + double conv."""

    def __init__(self, in_channels, out_channels):
        super(DecoderModule, self).__init__()
        self.up_sample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x, skip_connection):
        x = self.up_sample(x)
        # Guard against off-by-one spatial mismatches
        if skip_connection.shape[2:] != x.shape[2:]:
            x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip_connection, x], dim=1))


class Decoder(nn.Module):
    """Stack of DecoderModules that progressively upsample back to input resolution."""

    def __init__(self, features=[512, 256, 128, 64]):
        super(Decoder, self).__init__()
        # features[i] is the in_channels for stage i; out is features[i+1]
        self.stages = nn.ModuleList([
            DecoderModule(features[i], features[i + 1])
            for i in range(len(features) - 1)
        ])

    def forward(self, x, skip_connections):
        # Skip connections come in encoder order (shallow → deep);
        # reverse to match decoder order (deep → shallow).
        for i, stage in enumerate(self.stages):
            x = stage(x, skip_connections[::-1][i])
        return x


# ────────────────────────────── Full model ──────────────────────────────

class UNetVanilla(nn.Module):
    """
    Classic UNet built from scratch — no pretrained encoder.

    Args:
        in_channels:  Number of input image channels (1 for grayscale MRI).
        out_channels: Number of output segmentation classes.
        features:     Channel widths for each encoder stage.
                      Bottleneck will be ``features[-1] * 2``.
                      Decoder mirrors this list in reverse.
    """

    def __init__(self, in_channels=1, out_channels=NUM_CLASSES, features=[64, 128, 256, 512]):
        super(UNetVanilla, self).__init__()
        self.encoder = Encoder(in_channels=in_channels, features=features)

        # Decoder channel schedule: bottleneck → reversed encoder features
        decoder_features = [features[-1] * 2] + features[::-1]
        self.decoder = Decoder(decoder_features)

        # 1×1 head maps final feature maps to class logits
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        x, skip_connections = self.encoder(x)
        x = self.decoder(x, skip_connections)
        return self.final_conv(x)