"""MedMNIST v2 loading + federated client construction.

Two heterogeneity modes (both produce ONE global model with a shared head):

  * single subset  -> partition that dataset across clients (IID or Dirichlet
    non-IID). This is standard federated label/quantity skew.
  * multiple subsets -> "modality-heterogeneous": each hospital gets a different
    MedMNIST subset (MRI/CT/X-ray/pathology-like), and the label spaces are
    concatenated with per-subset offsets into a unified output space. Several
    clients may share a subset (round-robin), in which case that subset is
    IID-split among them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import Dataset

from qsfl.data.partition import dirichlet_partition, iid_partition
from qsfl.utils.logging import get_logger

logger = get_logger("data")


@dataclass
class FederatedData:
    client_train: list[Dataset]
    test_set: Dataset
    num_classes: int
    in_channels: int
    image_size: int
    client_names: list[str] = field(default_factory=list)

    @property
    def num_clients(self) -> int:
        return len(self.client_train)


class _Remapped(Dataset):
    """A view over a MedMNIST dataset: selected indices, scalar+offset labels."""

    def __init__(self, base: Dataset, indices, label_offset: int = 0) -> None:
        self.base = base
        self.indices = list(indices)
        self.label_offset = label_offset

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        img, target = self.base[self.indices[i]]
        target = int(np.asarray(target).reshape(-1)[0]) + self.label_offset
        return img, target


def _load_medmnist(name: str, split: str, root: str, image_size: int, as_rgb: bool, download: bool):
    try:
        import medmnist
        from medmnist import INFO
        from torchvision import transforms
    except Exception as exc:  # pragma: no cover - import guard
        raise ImportError("medmnist and torchvision are required for the data layer.") from exc

    info = INFO[name]
    DataClass = getattr(medmnist, info["python_class"])
    n_classes = len(info["label"])
    channels = 3 if as_rgb else info["n_channels"]
    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5] * channels, [0.5] * channels)]
    )
    kwargs = dict(split=split, transform=tfm, download=download, as_rgb=as_rgb, root=root)
    try:
        ds = DataClass(size=image_size, **kwargs)
    except TypeError:
        # Older medmnist without the `size` kwarg (fixed 28x28).
        ds = DataClass(**kwargs)
    return ds, n_classes, channels


def build_federated_data(cfg) -> FederatedData:
    """Construct per-client training sets + a combined test set from config."""
    if str(cfg.data.get("name", "medmnist")).lower() == "synthetic":
        from qsfl.data.synthetic import build_synthetic

        return build_synthetic(cfg)

    d = cfg.data
    subsets = list(d.subsets)
    num_clients = int(cfg.federated.num_clients)
    seed = int(cfg.seed)
    as_rgb = bool(d.as_rgb)
    image_size = int(d.image_size)
    root = str(d.root)
    download = bool(d.download)
    max_train = d.get("max_train_per_client", None)
    max_test = d.get("max_test", None)

    in_channels = 3 if as_rgb else None

    # Assign subsets to clients round-robin -> groups of clients per subset.
    assignment = [subsets[i % len(subsets)] for i in range(num_clients)]
    groups: dict[str, list[int]] = {}
    for client_id, sub in enumerate(assignment):
        groups.setdefault(sub, []).append(client_id)

    # Unified label space: stable offset per subset (sorted for determinism).
    offsets: dict[str, int] = {}
    running = 0
    per_subset_classes: dict[str, int] = {}

    client_train: list[Dataset | None] = [None] * num_clients
    test_views: list[Dataset] = []
    rng = np.random.default_rng(seed)

    for sub in sorted(set(subsets)):
        train_ds, n_classes, channels = _load_medmnist(
            sub, "train", root, image_size, as_rgb, download
        )
        test_ds, _, _ = _load_medmnist(sub, "test", root, image_size, as_rgb, download)
        if in_channels is None:
            in_channels = channels
        offsets[sub] = running
        per_subset_classes[sub] = n_classes
        running += n_classes

        member_clients = groups.get(sub, [])
        if not member_clients:
            continue

        labels = np.asarray(train_ds.labels).reshape(-1)
        if len(member_clients) == 1 and len(subsets) > 1:
            shards = [np.arange(len(labels))]
        elif d.partition == "noniid":
            shards = dirichlet_partition(labels, len(member_clients), float(d.dirichlet_alpha), seed)
        else:
            shards = iid_partition(len(labels), len(member_clients), seed)

        for local_i, client_id in enumerate(member_clients):
            idx = shards[local_i]
            if max_train is not None and len(idx) > int(max_train):
                idx = rng.choice(idx, size=int(max_train), replace=False)
            client_train[client_id] = _Remapped(train_ds, idx, offsets[sub])

        test_idx = np.arange(len(test_ds))
        if max_test is not None and len(test_idx) > int(max_test):
            test_idx = rng.choice(test_idx, size=int(max_test), replace=False)
        test_views.append(_Remapped(test_ds, test_idx, offsets[sub]))

    num_classes = running if len(subsets) > 1 else per_subset_classes[subsets[0]]
    test_set: Dataset = (
        test_views[0] if len(test_views) == 1 else torch.utils.data.ConcatDataset(test_views)
    )
    client_names = [f"hospital_{i+1}:{assignment[i]}" for i in range(num_clients)]

    logger.info(
        "Built %d clients over subsets=%s | num_classes=%d | in_channels=%d",
        num_clients,
        subsets,
        num_classes,
        in_channels,
    )
    return FederatedData(
        client_train=[c for c in client_train if c is not None],
        test_set=test_set,
        num_classes=int(num_classes),
        in_channels=int(in_channels),
        image_size=image_size,
        client_names=client_names,
    )
