"""Layer 6 ledger: ownership registration / verification.

One interface (:class:`Ledger`), two backends:
  * ``hashchain`` (default) — append-only SHA256-linked ledger; tamper-evident,
    fully offline, produces the reported ownership-proof results.
  * ``web3`` (optional) — a minimal Solidity registry on a local dev chain.

A single-node chain is NOT a decentralized trust anchor (see SECURITY.md).
"""

from qsfl.ledger.base import Ledger, OwnershipRecord
from qsfl.ledger.factory import get_ledger

__all__ = ["Ledger", "OwnershipRecord", "get_ledger"]
