"""Client partitioning: IID and Dirichlet non-IID."""

from __future__ import annotations

import numpy as np


def iid_partition(num_items: int, num_clients: int, seed: int = 0) -> list[np.ndarray]:
    """Shuffle indices and split into ``num_clients`` roughly equal shards."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(num_items)
    return [np.sort(s) for s in np.array_split(idx, num_clients)]


def dirichlet_partition(
    labels: np.ndarray, num_clients: int, alpha: float = 0.5, seed: int = 0
) -> list[np.ndarray]:
    """Non-IID label-skew partition via a per-class Dirichlet draw.

    Lower ``alpha`` -> more skew. Each client's index list is returned sorted.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels).reshape(-1)
    classes = np.unique(labels)
    client_idx: list[list[int]] = [[] for _ in range(num_clients)]
    for c in classes:
        c_idx = np.where(labels == c)[0]
        rng.shuffle(c_idx)
        proportions = rng.dirichlet(alpha=np.full(num_clients, alpha))
        # cut points along the class indices
        cuts = (np.cumsum(proportions) * len(c_idx)).astype(int)[:-1]
        for client, shard in enumerate(np.split(c_idx, cuts)):
            client_idx[client].extend(shard.tolist())
    return [np.sort(np.array(ci, dtype=int)) for ci in client_idx]
