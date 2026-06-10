"""
SegFormer architecture: MiT encoder + lightweight MLP decoder.

The encoder (MiTEncoder) is fully configurable via constructor parameters
so any SegFormer variant (B0–B5) can be swapped in by passing the
appropriate config values.

References:
    SegFormer: Simple and Efficient Design for Semantic Segmentation
    with Transformers  (Xie et al., NeurIPS 2021)
"""

import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from config import NUM_CLASSES


# ────────────────────────────── Building blocks ──────────────────────────────

class DropPath(nn.Module):
    """Stochastic depth per sample."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class ConvEmbedding(nn.Module):
    """Patch embedding via convolution."""

    def __init__(self, in_channels, embed_dim, kernel_size, stride, padding):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size, stride, padding)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.norm(tokens)
        return tokens, H, W


class Attention(nn.Module):
    """Efficient self-attention with optional spatial reduction."""

    def __init__(self, embed_dim, num_heads, sr_ratio):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.sr_ratio = sr_ratio
        self.scale = self.head_dim ** (-0.5)

        self.q = nn.Linear(embed_dim, embed_dim)
        self.kv = nn.Linear(embed_dim, 2 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(embed_dim, embed_dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(embed_dim)
        else:
            self.sr = None

    def forward(self, tokens, H, W):
        B, L, C = tokens.shape
        q = self.q(tokens).reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr is not None:
            x = tokens.reshape(B, H, W, C).permute(0, 3, 1, 2)
            x_sr = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x_sr = self.norm(x_sr)
            kv = self.kv(x_sr)
        else:
            kv = self.kv(tokens)

        kv = kv.reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        y = (attn @ v).permute(0, 2, 1, 3).reshape(B, L, C)
        return self.proj(y)


class MixFFN(nn.Module):
    """Feed-forward network with depthwise convolution."""

    def __init__(self, embed_dim, mlp_ratio, drop=0.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, tokens, H, W):
        B, L, _ = tokens.shape
        tokens = self.fc1(tokens)
        x = tokens.permute(0, 2, 1).reshape(B, -1, H, W)
        x = self.act(self.dwconv(x))
        tokens = x.flatten(2).permute(0, 2, 1)
        tokens = self.drop(self.fc2(tokens))
        return tokens


class TransformerBlock(nn.Module):

    def __init__(self, embed_dim, num_heads, sr_ratio, mlp_ratio, drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads, sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MixFFN(embed_dim, mlp_ratio)

    def forward(self, tokens, H, W):
        tokens = tokens + self.drop_path(self.attn(self.norm1(tokens), H, W))
        tokens = tokens + self.mlp(self.norm2(tokens), H, W)
        return tokens


# ────────────────────────────── Encoder ──────────────────────────────

# Default encoder config: SegFormer-B0
_DEFAULT_EMBED_DIMS = [32, 64, 160, 256]
_DEFAULT_NUM_HEADS  = [1, 2, 5, 8]
_DEFAULT_DEPTHS     = [2, 2, 2, 2]
_DEFAULT_SR_RATIOS  = [8, 4, 2, 1]


class MiTEncoder(nn.Module):
    """
    Mixed Transformer encoder — supports any SegFormer variant (B0–B5)
    via configuration parameters.

    Internal naming convention (used by the pretrained weight loader):
        patch_embeds.{0-3}    — ConvEmbedding modules
        stages.{0-3}.{0-N}   — TransformerBlock modules per stage
        norms.{0-3}           — LayerNorm after each stage
    """

    def __init__(
        self,
        in_channels=1,
        embed_dims=None,
        depths=None,
        num_heads=None,
        sr_ratios=None,
        mlp_ratio=4,
        drop_path_rate=0.1,
    ):
        super().__init__()
        if embed_dims is None:
            embed_dims = _DEFAULT_EMBED_DIMS
        if depths is None:
            depths = _DEFAULT_DEPTHS
        if num_heads is None:
            num_heads = _DEFAULT_NUM_HEADS
        if sr_ratios is None:
            sr_ratios = _DEFAULT_SR_RATIOS

        # ── Patch embeddings: [in_ch → ed[0]], [ed[0] → ed[1]], ... ──
        self.patch_embeds = nn.ModuleList()
        for i in range(4):
            in_ch  = in_channels if i == 0 else embed_dims[i - 1]
            ks     = 7 if i == 0 else 3
            stride = 4 if i == 0 else 2
            pad    = 3 if i == 0 else 1
            self.patch_embeds.append(ConvEmbedding(in_ch, embed_dims[i], ks, stride, pad))

        # ── Transformer blocks per stage ──
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        self.stages = nn.ModuleList()
        cur = 0
        for i in range(4):
            stage_blocks = nn.ModuleList([
                TransformerBlock(
                    embed_dims[i], num_heads[i], sr_ratios[i], mlp_ratio,
                    drop_path=dpr[cur + j],
                )
                for j in range(depths[i])
            ])
            self.stages.append(stage_blocks)
            cur += depths[i]

        # ── Stage norms ──
        self.norms = nn.ModuleList([nn.LayerNorm(ed) for ed in embed_dims])

    def _stage_forward(self, tokens, stage_blocks, norm, H, W):
        for block in stage_blocks:
            tokens = block(tokens, H, W)
        tokens = norm(tokens)
        B, _, C = tokens.shape
        return tokens.permute(0, 2, 1).reshape(B, C, H, W)

    def forward(self, x):
        feature_maps = []
        for i in range(4):
            inp = x if i == 0 else feature_maps[-1]
            tokens, H, W = self.patch_embeds[i](inp)
            x = self._stage_forward(tokens, self.stages[i], self.norms[i], H, W)
            feature_maps.append(x)
        return feature_maps


# ────────────────────────────── Pretrained weight loader ──────────────────────────────

def load_pretrained_encoder(model, repo_id, in_channels=1):
    """
    Load pretrained SegFormer encoder weights from HuggingFace
    and adapt for grayscale or RGB input.

    Works for any SegFormer variant (B0–B5) — just pass the right
    ``repo_id``.
    """
    print(f"Downloading pretrained encoder weights from {repo_id} ...")
    ckpt_path = hf_hub_download(
        repo_id=repo_id,
        filename="pytorch_model.bin",
    )
    state_dict = torch.load(ckpt_path, map_location="cpu")

    converted = {}
    kv_parts = {}

    for hf_key, value in state_dict.items():
        if not hf_key.startswith("segformer.encoder."):
            continue
        key = hf_key.replace("segformer.encoder.", "")

        # ── Patch embeddings ──
        #   HF:   patch_embeddings.{i}.proj / .layer_norm
        #   Ours: patch_embeds.{i}.proj / .norm
        key = re.sub(r"patch_embeddings\.(\d)\.proj",       r"patch_embeds.\1.proj", key)
        key = re.sub(r"patch_embeddings\.(\d)\.layer_norm",  r"patch_embeds.\1.norm", key)

        # ── Stage output norms ──
        #   HF:   layer_norm.{i}
        #   Ours: norms.{i}
        key = re.sub(r"^layer_norm\.(\d)", r"norms.\1", key)

        # ── Transformer blocks ──
        #   HF:   block.{stage}.{block}.xxx
        #   Ours: stages.{stage}.{block}.xxx
        key = re.sub(r"^block\.(\d)\.", r"stages.\1.", key)

        # ── Block internals ──
        key = key.replace("layer_norm_1",               "norm1")
        key = key.replace("layer_norm_2",               "norm2")
        key = key.replace("attention.self.layer_norm",   "attn.norm")
        key = key.replace("attention.self.query",        "attn.q")
        key = key.replace("attention.self.sr",           "attn.sr")
        key = key.replace("attention.output.dense",      "attn.proj")

        # Fuse key/value into single linear (attention.self.key/value → attn.kv)
        if "attention.self.key" in key:
            base = key.replace("attention.self.key", "attn.kv")
            kv_parts.setdefault(base, {})["k"] = value
            continue
        elif "attention.self.value" in key:
            base = key.replace("attention.self.value", "attn.kv")
            kv_parts.setdefault(base, {})["v"] = value
            continue

        # ── MLP ──
        key = key.replace("mlp.dense1",       "mlp.fc1")
        key = key.replace("mlp.dense2",       "mlp.fc2")
        key = key.replace("mlp.dwconv.dwconv", "mlp.dwconv")

        converted[key] = value

    # Concatenate K and V into fused kv weight
    for kv_key, parts in kv_parts.items():
        if "k" in parts and "v" in parts:
            converted[kv_key] = torch.cat([parts["k"], parts["v"]], dim=0)

    # Average the RGB input projection weights to a single channel
    proj_key = "patch_embeds.0.proj.weight"
    if proj_key in converted and converted[proj_key].shape[1] == 3:
        if in_channels == 1:
            # Grayscale: average the 3 RGB channels into 1
            converted[proj_key] = converted[proj_key].mean(dim=1, keepdim=True)
        # in_channels == 3: ImageNet weights already match RGB input

    result = model.load_state_dict(converted, strict=False)
    print(
        f"Loaded pretrained weights | "
        f"Missing: {len(result.missing_keys)} | "
        f"Unexpected: {len(result.unexpected_keys)}"
    )
    return model


# ────────────────────────────── Decoder ──────────────────────────────

class SegFormerDecoder(nn.Module):

    def __init__(self, in_channels=None, decoder_dim=256, out_channels=NUM_CLASSES, dropout=0.1):
        super().__init__()
        if in_channels is None:
            in_channels = _DEFAULT_EMBED_DIMS
        self.decoder_dim = decoder_dim
        self.proj = nn.ModuleList([nn.Linear(ed, decoder_dim) for ed in in_channels])
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * 4, decoder_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(decoder_dim, out_channels, kernel_size=1)

    def forward(self, feature_maps, input_size=None):
        _, _, H, W = feature_maps[0].shape
        projected = []
        for i, x in enumerate(feature_maps):
            B, C, h, w = x.shape
            tokens = x.reshape(B, C, -1).permute(0, 2, 1)
            tokens = self.proj[i](tokens)
            x = tokens.permute(0, 2, 1).reshape(B, self.decoder_dim, h, w)
            projected.append(F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False))

        x = self.fuse(torch.cat(projected, dim=1))
        x = self.dropout(x)
        x = self.classifier(x)
        if input_size is not None:
            x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x


# ────────────────────────────── Full model ──────────────────────────────

class SegFormer(nn.Module):

    def __init__(self, out_channels=NUM_CLASSES, decoder_dim=256, in_channels=1,
                 embed_dims=None, depths=None, num_heads=None, sr_ratios=None,
                 mlp_ratio=4, drop_path_rate=0.1):
        super().__init__()
        self.encoder = MiTEncoder(
            in_channels=in_channels,
            embed_dims=embed_dims,
            depths=depths,
            num_heads=num_heads,
            sr_ratios=sr_ratios,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
        )
        # Derive in_channels for decoder from embed_dims
        encoder_out_channels = embed_dims if embed_dims is not None else _DEFAULT_EMBED_DIMS
        self.decoder = SegFormerDecoder(
            in_channels=encoder_out_channels,
            decoder_dim=decoder_dim,
            out_channels=out_channels,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        feature_maps = self.encoder(x)
        return self.decoder(feature_maps, input_size=(H, W))
