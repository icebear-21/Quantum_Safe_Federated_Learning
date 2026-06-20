"""Config-driven model construction."""

from __future__ import annotations

import torch.nn as nn

from qsfl.models.backbones import ResNet, TinyCNN, ViT

_RESNET_VARIANTS = {
    "resnet18": (2, 2, 2, 2),
    "resnet34": (3, 4, 6, 3),
}


def list_backbones() -> list[str]:
    return ["tiny_cnn", "resnet", "vit"]


def build_model(cfg, num_classes: int, in_channels: int = 3, image_size: int = 28) -> nn.Module:
    """Build a backbone from ``cfg.model``.

    ``cfg`` is the full config; we read ``cfg.model.backbone`` and the matching
    sub-tree. The smoke config sets ``model.resnet.variant=tiny_cnn`` to select
    the tiny backbone through the resnet slot.
    """
    backbone = str(cfg.model.backbone).lower()

    if backbone == "tiny_cnn":
        return TinyCNN(in_channels=in_channels, num_classes=num_classes)

    if backbone == "resnet":
        variant = str(cfg.model.resnet.variant).lower()
        if variant == "tiny_cnn":
            return TinyCNN(in_channels=in_channels, num_classes=num_classes)
        if variant not in _RESNET_VARIANTS:
            raise ValueError(f"unknown resnet variant: {variant}")
        return ResNet(
            blocks=_RESNET_VARIANTS[variant], in_channels=in_channels, num_classes=num_classes
        )

    if backbone == "vit":
        v = cfg.model.vit
        return ViT(
            in_channels=in_channels,
            num_classes=num_classes,
            image_size=image_size,
            patch_size=int(v.patch_size),
            dim=int(v.dim),
            depth=int(v.depth),
            heads=int(v.heads),
            mlp_dim=int(v.get("mlp_dim", v.dim * 2)),
        )

    raise ValueError(f"unknown backbone: {backbone!r} (one of {list_backbones()})")
