"""Dual watermark embedding/extraction over a torch model's weights.

WM_1 (primary, visible image P) and WM_2 (secondary, invisible bits S) are
embedded into two *different* large weight tensors so they don't interfere; the
embedding keys (target layer, key_seed, payload) form the secret ``WatermarkBundle``
needed for extraction/verification.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from qsfl.utils.logging import get_logger
from qsfl.watermark.payload import make_primary_image, make_secondary_bits
from qsfl.watermark.uchida import embed_bits, extract_bits

logger = get_logger("watermark")


@dataclass
class WatermarkBundle:
    """Secret material + originals needed to extract and verify ownership."""

    primary_bits: np.ndarray  # 2D (size,size) original P
    primary_layer: str
    primary_key_seed: int
    secondary_bits: np.ndarray  # 1D original S
    secondary_layer: str
    secondary_key_seed: int
    meta: dict = field(default_factory=dict)


def _weight_tensors(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    params = [(n, p) for n, p in model.named_parameters() if p.dim() >= 2]
    return sorted(params, key=lambda kv: kv[1].numel(), reverse=True)


def _rms(t: torch.Tensor) -> float:
    return float(t.detach().float().pow(2).mean().sqrt().item())


class DualWatermarker:
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def _pick_layers(self, model: nn.Module, t_primary: int, t_secondary: int) -> tuple[str, str]:
        ranked = _weight_tensors(model)
        prim = next((n for n, p in ranked if p.numel() >= t_primary), None)
        if prim is None:
            raise ValueError(f"no weight tensor large enough for {t_primary}-bit primary watermark")
        sec = next((n for n, p in ranked if n != prim and p.numel() >= t_secondary), None)
        if sec is None:
            logger.warning("only one large layer; embedding both watermarks into %s", prim)
            sec = prim
        return prim, sec

    def embed(self, model: nn.Module) -> WatermarkBundle:
        """Embed both watermarks in place; return the secret bundle."""
        wcfg = self.cfg.watermark
        primary_bits = make_primary_image(wcfg.primary)
        secondary_bits = make_secondary_bits(int(wcfg.secondary.length), int(wcfg.secondary.payload_seed))

        prim_layer, sec_layer = self._pick_layers(model, primary_bits.size, secondary_bits.size)
        params = dict(model.named_parameters())

        # WM_1 primary
        self._embed_into(
            params[prim_layer], primary_bits.reshape(-1),
            int(wcfg.primary.key_seed), float(wcfg.primary.strength),
        )
        # WM_2 secondary
        self._embed_into(
            params[sec_layer], secondary_bits,
            int(wcfg.secondary.key_seed), float(wcfg.secondary.strength),
        )
        logger.info(
            "embedded primary(%d bits)->%s, secondary(%d bits)->%s",
            primary_bits.size, prim_layer, secondary_bits.size, sec_layer,
        )
        return WatermarkBundle(
            primary_bits=primary_bits,
            primary_layer=prim_layer,
            primary_key_seed=int(wcfg.primary.key_seed),
            secondary_bits=secondary_bits,
            secondary_layer=sec_layer,
            secondary_key_seed=int(wcfg.secondary.key_seed),
            meta={"primary_shape": list(primary_bits.shape)},
        )

    @staticmethod
    def _embed_into(param: nn.Parameter, bits: np.ndarray, key_seed: int, strength: float) -> None:
        w = param.detach().cpu().numpy()
        # margin lives in projection space (~ rms scale); strength is the knob
        # trading watermark robustness against weight perturbation.
        margin = strength * (_rms(param) + 1e-8)
        new_w = embed_bits(w, bits, key_seed, margin)
        with torch.no_grad():
            param.copy_(torch.as_tensor(new_w, dtype=param.dtype, device=param.device))

    def extract(self, model: nn.Module, bundle: WatermarkBundle) -> dict:
        """Extract P' and S' from a (possibly attacked) model."""
        params = dict(model.named_parameters())
        prim_w = params[bundle.primary_layer].detach().cpu().numpy()
        sec_w = params[bundle.secondary_layer].detach().cpu().numpy()
        p_prime = extract_bits(prim_w, bundle.primary_bits.size, bundle.primary_key_seed)
        s_prime = extract_bits(sec_w, bundle.secondary_bits.size, bundle.secondary_key_seed)
        return {
            "primary_extracted": p_prime.reshape(bundle.primary_bits.shape),
            "secondary_extracted": s_prime,
        }
