"""Ledger backend factory."""

from __future__ import annotations

from qsfl.ledger.base import Ledger


def get_ledger(cfg) -> Ledger:
    backend = str(cfg.ledger.backend).lower()
    if backend == "hashchain":
        from qsfl.ledger.hashchain import HashChainLedger

        return HashChainLedger(cfg.ledger.path)
    if backend == "web3":
        from qsfl.ledger.web3_backend import Web3Ledger

        return Web3Ledger(cfg)
    raise ValueError(f"unknown ledger backend: {backend!r} (use 'hashchain' or 'web3')")
