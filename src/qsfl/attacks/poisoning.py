"""Data-poisoning attack (label flipping).

A fraction of a client's labels are flipped to random classes, then the model is
trained on the poisoned data; we report the accuracy drop (and watermark NC/BER,
to check the attack doesn't incidentally destroy the watermark).
"""

from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset

from qsfl.attacks.base import AttackContext, quick_train, watermark_metrics
from qsfl.federated.trainer import evaluate


class _LabelFlip(Dataset):
    def __init__(self, base: Dataset, fraction: float, num_classes: int, seed: int) -> None:
        self.base = base
        self.num_classes = num_classes
        rng = np.random.default_rng(seed)
        n = len(base)
        flip_idx = rng.choice(n, size=int(fraction * n), replace=False)
        self.new_label = {int(i): int(rng.integers(0, num_classes)) for i in flip_idx}

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        x, y = self.base[i]
        return x, self.new_label.get(i, int(np.asarray(y).reshape(-1)[0]))


def run(ctx: AttackContext) -> dict:
    a = ctx.cfg.attacks.poisoning
    model = ctx.fresh_model()
    poisoned = _LabelFlip(
        ctx.data.client_train[0], float(a.fraction), ctx.data.num_classes, int(ctx.cfg.seed)
    )
    quick_train(model, poisoned, epochs=2, lr=1e-3, device=ctx.device, num_workers=int(ctx.cfg.num_workers))
    acc = evaluate(model, ctx.data.test_set, ctx.device)
    return {
        "fraction": float(a.fraction),
        "type": str(a.get("type", "label_flip")),
        "accuracy_after": acc,
        "accuracy_drop": ctx.baseline_accuracy - acc,
        **watermark_metrics(model, ctx.bundle, ctx.cfg),
    }
