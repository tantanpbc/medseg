"""
DeepLabv3+ architecture with ResNet-50 backbone.

Components:
    - ResNet-50 backbone (torchvision pretrained, adapted for grayscale)
    - ASPP (Atrous Spatial Pyramid Pooling) module
    - Decoder with low-level feature fusion
    - Auxiliary classifier (active during training only)

References:
    Encoder-Decoder with Atrous Separable Convolution for Semantic Image
    Segmentation  (Chen et al., ECCV 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from config import NUM_CLASSES


# ────────────────────────────── Building blocks ──────────────────────────────

class CBR(nn.Module):
    """Conv2d-BatchNorm-ReLU building block."""
    def __init__(self, in_channels, out_channels, k, p=0, d=1):
        super(CBR, self).__init__()
        self.cbr = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=p, dilation=d, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.cbr(x)


# ────────────────────────────── Backbone ──────────────────────────────

class Resnet50Backbone(nn.Module):
    def __init__(self, output_stride=16, in_channels=1, use_pretrained=True):
        super(Resnet50Backbone, self).__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if use_pretrained else None
        self.resnet = models.resnet50(weights=weights)

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

        self.output_stride = output_stride
        if output_stride == 16:
            self._replace_with_dilated(self.resnet.layer4, d_rate=2)
        elif output_stride == 8:
            self._replace_with_dilated(self.resnet.layer3, d_rate=2)
            self._replace_with_dilated(self.resnet.layer4, d_rate=4)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        low_feature = self.resnet.layer1(x)
        x = self.resnet.layer2(low_feature)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        return x, low_feature

    def _replace_with_dilated(self, layer, d_rate):
        """Replace strided convolutions with dilated convolutions to control output stride."""
        layer[0].conv2.stride = (1, 1)
        layer[0].downsample[0].stride = (1, 1)
        for block in layer:
            block.conv2.dilation = (d_rate, d_rate)
            block.conv2.padding = (d_rate, d_rate)
            block.conv2.stride = (1, 1)


# ────────────────────────────── ASPP ──────────────────────────────

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module."""
    def __init__(self, in_channels=2048, out_channels=256, d_rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        self.branches = nn.ModuleList([
            CBR(in_channels, out_channels, k=1),
            CBR(in_channels, out_channels, k=3, p=d_rates[0], d=d_rates[0]),
            CBR(in_channels, out_channels, k=3, p=d_rates[1], d=d_rates[1]),
            CBR(in_channels, out_channels, k=3, p=d_rates[2], d=d_rates[2]),
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                CBR(in_channels, out_channels, k=1),
            ),
        ])
        self.project = CBR(len(self.branches) * out_channels, out_channels, k=1)

    def forward(self, x):
        H, W = x.shape[2:]
        outputs = []
        for branch in self.branches:
            y = branch(x)
            if y.shape[2:] != (H, W):
                y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
            outputs.append(y)
        return self.project(torch.cat(outputs, dim=1))


# ────────────────────────────── Decoder ──────────────────────────────

class DeepLabDecoder(nn.Module):
    def __init__(self, in_channels=256, reduced_channels=48, out_channels=NUM_CLASSES):
        super(DeepLabDecoder, self).__init__()
        self.low_process = CBR(256, reduced_channels, k=1)
        self.decoder = nn.Sequential(
            CBR(in_channels + reduced_channels, in_channels, k=3, p=1),
            CBR(in_channels, in_channels, k=3, p=1),
        )
        self.classifier = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, high_x, low_x, size):
        low_x = self.low_process(low_x)
        high_x = F.interpolate(high_x, size=low_x.shape[2:], mode="bilinear", align_corners=False)
        x = self.decoder(torch.cat([low_x, high_x], dim=1))
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return self.classifier(x)


# ────────────────────────────── Full model ──────────────────────────────

class DeepLabv3plus(nn.Module):
    def __init__(self, output_stride=16, in_channels=1, out_channels=NUM_CLASSES):
        super(DeepLabv3plus, self).__init__()
        self.backbone = Resnet50Backbone(output_stride=output_stride, in_channels=in_channels)
        self.aspp = ASPP()
        self.decoder = DeepLabDecoder(out_channels=out_channels)
        self.aux_classifier = nn.Conv2d(256, out_channels, kernel_size=1)

    def forward(self, x):
        size = x.shape[2:]
        high_x, low_x = self.backbone(x)
        aspp_feat = self.aspp(high_x)
        main_out = self.decoder(aspp_feat, low_x, size)

        if self.training:
            aux_out = self.aux_classifier(aspp_feat)
            aux_out = F.interpolate(aux_out, size=size, mode="bilinear", align_corners=False)
            return main_out, aux_out

        return main_out
