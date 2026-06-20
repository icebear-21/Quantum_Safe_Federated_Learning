"""Fine-tuning attack (watermark robustness).

The adversary fine-tunes the stolen model on their own data; we measure how much
the watermark (NC/BER) survives and the accuracy after fine-tuning.
"""

from __future__ import annotations

from qsfl.attacks.base import AttackContext, quick_train, watermark_metrics
from qsfl.federated.trainer import evaluate


def run(ctx: AttackContext) -> dict:
    a = ctx.cfg.attacks.fine_tuning
    model = ctx.fresh_model()
    quick_train(
        model, ctx.data.test_set, int(a.epochs), float(a.lr), ctx.device,
        num_workers=int(ctx.cfg.num_workers),
    )
    acc = evaluate(model, ctx.data.test_set, ctx.device)
    return {
        "accuracy_after": acc,
        "accuracy_drop": ctx.baseline_accuracy - acc,
        **watermark_metrics(model, ctx.bundle, ctx.cfg),
    }
