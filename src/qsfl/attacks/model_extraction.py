"""Model-extraction attack.

The adversary queries the victim and trains a fresh student on the predicted
labels. We report the student's accuracy + fidelity (agreement with the victim)
and the watermark NC in the student — which should be ~chance, since a white-box
weight watermark does not transfer to an independently-trained student.
"""

from __future__ import annotations

import torch
from torch.utils.data import TensorDataset

from qsfl.attacks.base import AttackContext, collect_queries, quick_train, watermark_metrics
from qsfl.federated.trainer import evaluate


@torch.no_grad()
def _agreement(student, victim, dataset, device) -> float:
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    student.to(device).eval()
    victim.to(device).eval()
    agree = total = 0
    for x, _ in loader:
        x = x.to(device)
        agree += int((student(x).argmax(1) == victim(x).argmax(1)).sum().item())
        total += x.shape[0]
    return agree / max(total, 1)


def run(ctx: AttackContext) -> dict:
    a = ctx.cfg.attacks.model_extraction
    victim = ctx.fresh_model().eval()
    x, y = collect_queries(victim, ctx.data.test_set, int(a.queries), ctx.device)
    student = ctx.random_model()
    quick_train(student, TensorDataset(x, y), int(a.epochs), 1e-3, ctx.device,
                num_workers=int(ctx.cfg.num_workers))
    student_acc = evaluate(student, ctx.data.test_set, ctx.device)
    fidelity = _agreement(student, victim, ctx.data.test_set, ctx.device)
    wm = watermark_metrics(student, ctx.bundle, ctx.cfg)
    return {
        "student_accuracy": student_acc,
        "victim_accuracy": ctx.baseline_accuracy,
        "fidelity": fidelity,
        "watermark_transferred": {f"student_{k}": v for k, v in wm.items()},
    }
