"""
UNet-ResNet34 architecture: ResNet34 encoder with skip connections
and a convolutional decoder.

The encoder is initialised from ImageNet-pretrained ResNet34 weights
(via torchvision), with automatic grayscale input adaptation.

References:
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    (Ronneberger et al., MICCAI 2015)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from config import NUM_CLASSES


# ────────────────────────────── Encoder ──────────────────────────────

class EncoderResnet34(nn.Module):
    """ResNet34-based encoder producing skip connections at multiple scales."""

    def __init__(self, in_channels=1, use_pretrained=True):
        super(EncoderResnet34, self).__init__()
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if use_pretrained else None
        self.resnet = models.resnet34(weights=weights)

        # Grayscale adaptation: average pretrained RGB conv1 weights
        if in_channels == 1:
            old_conv = self.resnet.conv1
            new_conv = nn.Conv2d(
                1, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=(old_conv.bias is not None),
            )
            new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
            self.resnet.conv1 = new_conv

        self.encoder = nn.ModuleList([
            nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu),
            self.resnet.maxpool,
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
        ])
        self.bottleneck = self.resnet.layer4

    def forward(self, x):
        skip_connections = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i != 1:
                skip_connections.append(x)
        x = self.bottleneck(x)
        return x, skip_connections


# ────────────────────────────── Decoder ──────────────────────────────

class DecoderModule(nn.Module):
    """Single decoder block: transposed conv upsample + skip concat + double conv."""

    def __init__(self, in_channels, out_channels):
        super(DecoderModule, self).__init__()
        self.up_sample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip_connection):
        x = self.up_sample(x)
        if skip_connection.shape[2:] != x.shape[2:]:
            x = F.resize(x, size=skip_connection.shape[2:])
        return self.conv(torch.cat([skip_connection, x], dim=1))


class Decoder(nn.Module):
    """Stack of DecoderModules that progressively upsample."""

    def __init__(self, features=[512, 256, 128, 64, 64]):
        super(Decoder, self).__init__()
        self.decoder = nn.ModuleList([
            DecoderModule(features[i], features[i + 1])
            for i in range(len(features) - 1)
        ])

    def forward(self, x, skip_connections):
        for i, layer in enumerate(self.decoder):
            x = layer(x, skip_connections[::-1][i])
        return x


# ────────────────────────────── Full model ──────────────────────────────

class UNetResnet34(nn.Module):
    """
    UNet with a ResNet34 encoder backbone and a learned decoder.

    Args:
        in_channels:  Number of input channels (1 for grayscale).
        out_channels: Number of output segmentation classes.
        features:     Channel widths for each decoder level.
    """

    def __init__(self, in_channels=1, out_channels=NUM_CLASSES, features=[512, 256, 128, 64, 64]):
        super(UNetResnet34, self).__init__()
        self.encoder = EncoderResnet34(in_channels=in_channels)
        self.decoder = Decoder(features)
        last_ch = features[-1]
        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(last_ch, last_ch, kernel_size=2, stride=2),
            nn.Conv2d(last_ch, last_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(last_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(last_ch, last_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(last_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(last_ch, out_channels, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x, skip_connections = self.encoder(x)
        x = self.decoder(x, skip_connections)
        return self.final_conv(x)
