"""Gradient-leakage attack (DLG-style, privacy leakage).

Given the gradient of the loss on a single private sample, the adversary
optimizes a dummy input to match that gradient and thereby reconstruct the
sample. We report the reconstruction MSE (lower = more leakage). This is the
threat that secure aggregation / masking is meant to mitigate.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from qsfl.attacks.base import AttackContext


def run(ctx: AttackContext) -> dict:
    a = ctx.cfg.attacks.gradient_leakage
    device = ctx.device
    model = ctx.fresh_model().to(device).eval()
    for p in model.parameters():
        p.requires_grad_(True)
    params = [p for p in model.parameters() if p.requires_grad]
    crit = nn.CrossEntropyLoss()

    x0, y0 = ctx.data.client_train[0][0]
    x = x0.unsqueeze(0).to(device)
    y = torch.tensor([int(y0)], device=device)

    true_grads = [g.detach() for g in torch.autograd.grad(crit(model(x), y), params)]

    dummy = torch.randn_like(x, requires_grad=True)
    opt = torch.optim.Adam([dummy], lr=float(a.lr))
    for _ in range(int(a.iters)):
        opt.zero_grad()
        grads = torch.autograd.grad(crit(model(dummy), y), params, create_graph=True)
        g_loss = sum(((g - tg) ** 2).sum() for g, tg in zip(grads, true_grads))
        g_loss.backward()
        opt.step()

    mse = float(((dummy.detach() - x) ** 2).mean().item())
    return {
        "reconstruction_mse": mse,
        "iters": int(a.iters),
        "privacy_leakage_inv_mse": float(1.0 / (mse + 1e-8)),
    }
