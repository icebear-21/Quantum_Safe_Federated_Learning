"""Watermark-removal attack.

The adversary perturbs the weight tensors (overwrite/noise) to scrub the
watermark while trying to keep accuracy. We report the resulting NC/BER and
accuracy; a robust watermark keeps NC high (BER low) until accuracy collapses.
"""

from __future__ import annotations

import torch

from qsfl.attacks.base import AttackContext, watermark_metrics
from qsfl.federated.trainer import evaluate


@torch.no_grad()
def run(ctx: AttackContext) -> dict:
    a = ctx.cfg.attacks.watermark_removal
    strength = float(a.strength)
    model = ctx.fresh_model()
    for p in model.parameters():
        if p.dim() >= 2:
            p.add_(strength * (p.std() + 1e-8) * torch.randn_like(p))
    acc = evaluate(model, ctx.data.test_set, ctx.device)
    return {
        "method": str(a.get("method", "overwrite")),
        "strength": strength,
        "accuracy_after": acc,
        "accuracy_drop": ctx.baseline_accuracy - acc,
        **watermark_metrics(model, ctx.bundle, ctx.cfg),
    }
