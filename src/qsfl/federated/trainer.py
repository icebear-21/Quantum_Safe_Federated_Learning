"""Local client training + evaluation + state<->numpy helpers."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def state_to_numpy(model: nn.Module) -> dict[str, np.ndarray]:
    """Snapshot a model's state_dict as numpy arrays (CPU copies)."""
    return {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}


def numpy_to_state(model: nn.Module, state: dict[str, np.ndarray]) -> None:
    """Load a numpy state into ``model`` (casting back to each tensor's dtype)."""
    sd = model.state_dict()
    new = {k: torch.as_tensor(state[k]).to(sd[k].dtype) for k in sd}
    model.load_state_dict(new)


def _build_optimizer(model: nn.Module, cfg) -> torch.optim.Optimizer:
    f = cfg.federated
    if str(f.optimizer).lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=float(f.lr), weight_decay=float(f.weight_decay))
    return torch.optim.SGD(
        model.parameters(),
        lr=float(f.lr),
        momentum=float(f.momentum),
        weight_decay=float(f.weight_decay),
    )


class LocalTrainer:
    """Runs ``local_epochs`` of SGD/Adam on one client's data."""

    def __init__(self, cfg, device: str) -> None:
        self.cfg = cfg
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def train(self, model: nn.Module, dataset: Dataset) -> dict[str, np.ndarray]:
        f = self.cfg.federated
        loader = DataLoader(
            dataset,
            batch_size=int(f.batch_size),
            shuffle=True,
            num_workers=int(self.cfg.num_workers),
            drop_last=False,
        )
        model.to(self.device).train()
        opt = _build_optimizer(model, self.cfg)
        for _ in range(int(f.local_epochs)):
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device).long().view(-1)
                opt.zero_grad()
                loss = self.criterion(model(x), y)
                loss.backward()
                opt.step()
        return state_to_numpy(model)


@torch.no_grad()
def evaluate(model: nn.Module, dataset: Dataset, device: str, batch_size: int = 256) -> float:
    """Top-1 accuracy over ``dataset``."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.to(device).eval()
    correct = total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device).long().view(-1)
        pred = model(x).argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return correct / max(total, 1)
