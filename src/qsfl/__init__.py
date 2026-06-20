"""QSFL — Quantum-Safe Federated Learning & Deep-Learning Model Protection.

Top-level package. Sub-packages map to the framework layers:

    data/          Layer 1  medical data & heterogeneous clients
    quantum/       Layer 2  VQC quantum key generation
    crypto/        Layer 2  hybrid KDF + Kyber KEM + AES-256-GCM
    federated/     Layer 3  local training + secure aggregation
    watermark/     Layer 4  dual (visible + invisible) white-box watermarking
    attacks/       Layer 5  threat-model evaluation suite
    ledger/        Layer 6  hash-chained / web3 ownership ledger
    verification/  Layer 6  NC / BER ownership-acceptance logic
    pipeline/      orchestration (end-to-end runner)
    utils/         config, seeding, logging, metrics, plotting
"""

__version__ = "0.1.0"
