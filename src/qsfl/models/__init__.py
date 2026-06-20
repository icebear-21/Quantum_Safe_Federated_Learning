"""Model backbones (CNN / ResNet / ViT) and a config-driven registry."""

from qsfl.models.registry import build_model, list_backbones

__all__ = ["build_model", "list_backbones"]
