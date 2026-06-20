"""Shared attack context + helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from qsfl.federated.trainer import numpy_to_state
from qsfl.models import build_model
from qsfl.utils.metrics import bit_error_rate, normalized_correlation
from qsfl.watermark import DualWatermarker


@dataclass
class AttackContext:
    """Everything an attack needs; ``fresh_model`` hands out independent copies."""

    cfg: Any
    data: Any
    bundle: Any
    protected_state: dict
    baseline_accuracy: float
    device: str

    def fresh_model(self) -> nn.Module:
        m = build_model(self.cfg, self.data.num_classes, self.data.in_channels, self.data.image_size)
        numpy_to_state(m, self.protected_state)
        return m

    def random_model(self) -> nn.Module:
        return build_model(self.cfg, self.data.num_classes, self.data.in_channels, self.data.image_size)


def quick_train(
    model: nn.Module, dataset: Dataset, epochs: int, lr: float, device: str,
    batch_size: int = 64, num_workers: int = 0,
) -> nn.Module:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for _ in range(int(epochs)):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).long().view(-1)
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()
    return model


def watermark_metrics(model: nn.Module, bundle, cfg) -> dict:
    """NC(P,P'), NC(S,S'), BER(S,S') for a (possibly attacked) model."""
    if bundle is None:
        return {}
    ext = DualWatermarker(cfg).extract(model, bundle)
    return {
        "nc_primary": normalized_correlation(bundle.primary_bits, ext["primary_extracted"], binary=True),
        "nc_secondary": normalized_correlation(bundle.secondary_bits, ext["secondary_extracted"], binary=True),
        "ber_secondary": bit_error_rate(bundle.secondary_bits, ext["secondary_extracted"]),
    }


@torch.no_grad()
def collect_queries(model: nn.Module, dataset: Dataset, n: int, device: str):
    """Query ``model`` on up to ``n`` inputs; return (X, predicted_labels)."""
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    model.to(device).eval()
    xs, ys = [], []
    seen = 0
    for x, _ in loader:
        x = x.to(device)
        preds = model(x).argmax(dim=1)
        xs.append(x.cpu())
        ys.append(preds.cpu())
        seen += x.shape[0]
        if seen >= n:
            break
    return torch.cat(xs)[:n], torch.cat(ys)[:n]
