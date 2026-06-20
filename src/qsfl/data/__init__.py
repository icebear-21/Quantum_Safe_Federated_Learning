"""Layer 1 data: MedMNIST loaders + federated partitioning."""

from qsfl.data.medmnist_data import FederatedData, build_federated_data

__all__ = ["FederatedData", "build_federated_data"]
