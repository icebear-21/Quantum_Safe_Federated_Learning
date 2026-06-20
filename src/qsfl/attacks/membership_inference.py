"""Membership-inference attack (privacy leakage).

A loss-threshold attack: training members tend to have lower loss than unseen
samples. We report attack AUC and accuracy; ``privacy_leakage = AUC - 0.5``
(0 = no leakage, 0.5 = perfect membership distinguishability).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from qsfl.attacks.base import AttackContext


@torch.no_grad()
def _per_sample_loss(model, dataset, device, limit: int) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    crit = nn.CrossEntropyLoss(reduction="none")
    model.to(device).eval()
    out: list[np.ndarray] = []
    seen = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device).long().view(-1)
        out.append(crit(model(x), y).cpu().numpy())
        seen += len(out[-1])
        if seen >= limit:
            break
    return np.concatenate(out)[:limit]


def run(ctx: AttackContext) -> dict:
    from sklearn.metrics import roc_auc_score

    model = ctx.fresh_model().eval()
    limit = 512
    members = _per_sample_loss(model, ctx.data.client_train[0], ctx.device, limit)
    nonmembers = _per_sample_loss(model, ctx.data.test_set, ctx.device, limit)

    losses = np.concatenate([members, nonmembers])
    labels = np.concatenate([np.ones_like(members), np.zeros_like(nonmembers)])
    scores = -losses  # higher score (lower loss) -> predicted member
    auc = float(roc_auc_score(labels, scores))

    thr = float(np.median(losses))
    pred_member = losses < thr
    acc = float((pred_member == labels.astype(bool)).mean())
    return {
        "auc": auc,
        "privacy_leakage": auc - 0.5,
        "attack_accuracy": acc,
        "n_members": int(len(members)),
        "n_nonmembers": int(len(nonmembers)),
    }
