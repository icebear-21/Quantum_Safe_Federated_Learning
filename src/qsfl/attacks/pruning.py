"""Pruning attack (watermark robustness).

Global magnitude pruning zeros the smallest-|w| fraction of each weight tensor;
we sweep prune amounts and report accuracy + watermark NC/BER at each.
"""

from __future__ import annotations

import torch

from qsfl.attacks.base import AttackContext, watermark_metrics
from qsfl.federated.trainer import evaluate


@torch.no_grad()
def _prune(model, amount: float) -> None:
    for p in model.parameters():
        if p.dim() >= 2 and amount > 0:
            k = int(amount * p.numel())
            if k <= 0:
                continue
            thresh = p.abs().flatten().kthvalue(k).values
            p[p.abs() <= thresh] = 0.0


def run(ctx: AttackContext) -> dict:
    amounts = list(ctx.cfg.attacks.pruning.amounts)
    by_amount = {}
    for amount in amounts:
        model = ctx.fresh_model()
        _prune(model, float(amount))
        acc = evaluate(model, ctx.data.test_set, ctx.device)
        by_amount[str(amount)] = {
            "amount": float(amount),
            "accuracy_after": acc,
            "accuracy_drop": ctx.baseline_accuracy - acc,
            **watermark_metrics(model, ctx.bundle, ctx.cfg),
        }
    return {"by_amount": by_amount}
