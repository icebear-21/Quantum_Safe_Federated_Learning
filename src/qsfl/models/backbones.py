"""Backbone definitions: TinyCNN, ResNet (CIFAR-style), ViT (small).

All accept ``in_channels`` and ``num_classes`` and adapt to small (e.g. 28x28)
MedMNIST images. Layers are named so the white-box watermarker can target a
specific weight tensor (see qsfl.watermark).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TinyCNN(nn.Module):
    """Smallest backbone — used by the smoke test."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Linear(32 * 4 * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(torch.flatten(x, 1))


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return torch.relu(out)


class ResNet(nn.Module):
    """CIFAR-style ResNet (3x3 stem, no max-pool) suitable for small images."""

    def __init__(
        self, blocks: tuple[int, ...] = (2, 2, 2, 2), in_channels: int = 3, num_classes: int = 10
    ) -> None:
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, blocks[0], 1)
        self.layer2 = self._make_layer(128, blocks[1], 2)
        self.layer3 = self._make_layer(256, blocks[2], 2)
        self.layer4 = self._make_layer(512, blocks[3], 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(_BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out)
        return self.fc(torch.flatten(out, 1))


class ViT(nn.Module):
    """Compact Vision Transformer for small images."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        image_size: int = 28,
        patch_size: int = 4,
        dim: int = 192,
        depth: int = 6,
        heads: int = 3,
        mlp_dim: int = 384,
    ) -> None:
        super().__init__()
        # Pad so the image is divisible by patch_size, then patch-embed via conv.
        self.pad = (patch_size - image_size % patch_size) % patch_size
        padded = image_size + self.pad
        n_patches = (padded // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=mlp_dim, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad:
            x = nn.functional.pad(x, (0, self.pad, 0, self.pad))
        x = self.patch_embed(x)  # B, dim, H', W'
        x = x.flatten(2).transpose(1, 2)  # B, N, dim
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.encoder(x)
        return self.head(self.norm(x[:, 0]))
