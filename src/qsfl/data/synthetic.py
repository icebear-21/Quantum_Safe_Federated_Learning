"""Synthetic in-memory dataset (no network) for hermetic tests / CI.

Enabled with ``data.name=synthetic``. Produces random images + labels so the
full pipeline can run end-to-end without downloading MedMNIST.
"""

from __future__ import annotations

import torch
from torch.utils.data import TensorDataset

from qsfl.data.medmnist_data import FederatedData


def build_synthetic(cfg) -> FederatedData:
    d = cfg.data
    num_clients = int(cfg.federated.num_clients)
    num_classes = int(d.get("synthetic_classes", 4))
    in_channels = 3 if bool(d.as_rgb) else 1
    size = int(d.image_size)
    per_client = int(d.get("synthetic_per_client", 64))
    n_test = int(d.get("synthetic_test", 64))
    g = torch.Generator().manual_seed(int(cfg.seed))

    def _make(n: int) -> TensorDataset:
        x = torch.randn(n, in_channels, size, size, generator=g)
        y = torch.randint(0, num_classes, (n,), generator=g)
        return TensorDataset(x, y)

    clients = [_make(per_client) for _ in range(num_clients)]
    return FederatedData(
        client_train=clients,
        test_set=_make(n_test),
        num_classes=num_classes,
        in_channels=in_channels,
        image_size=size,
        client_names=[f"synthetic_{i+1}" for i in range(num_clients)],
    )
