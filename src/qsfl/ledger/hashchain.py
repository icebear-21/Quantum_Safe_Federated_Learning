"""Append-only, SHA256-linked ownership ledger (default backend).

Each block hashes ``(index, timestamp, record, prev_hash)``; editing any past
block breaks every subsequent link, which :meth:`verify_integrity` detects. This
is tamper-*evident* (a single party controls the file) — not decentralized
immutability. See SECURITY.md.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from qsfl.ledger.base import Ledger, OwnershipRecord

GENESIS_PREV = "0" * 64


def _block_hash(index: int, timestamp: float, record: dict, prev_hash: str) -> str:
    payload = json.dumps(
        {"index": index, "timestamp": timestamp, "record": record, "prev_hash": prev_hash},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class HashChainLedger(Ledger):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._blocks: list[dict] = []
        if self.path.exists():
            with open(self.path) as fh:
                self._blocks = [json.loads(line) for line in fh if line.strip()]

    # --- internal ---
    def _append(self, block: dict) -> None:
        self._blocks.append(block)
        with open(self.path, "a") as fh:
            fh.write(json.dumps(block, sort_keys=True, separators=(",", ":")) + "\n")

    @property
    def _last_hash(self) -> str:
        return self._blocks[-1]["hash"] if self._blocks else GENESIS_PREV

    # --- Ledger API ---
    def register(self, record: OwnershipRecord) -> dict:
        index = len(self._blocks)
        prev = self._last_hash
        rec = json.loads(record.canonical_json())
        h = _block_hash(index, record.timestamp, rec, prev)
        block = {"index": index, "timestamp": record.timestamp, "record": rec, "prev_hash": prev, "hash": h}
        self._append(block)
        return {"index": index, "block_hash": h, "prev_hash": prev, "backend": "hashchain"}

    def verify(self, model_hash: str) -> dict:
        integrity_ok = self.verify_integrity()
        for block in self._blocks:
            if block["record"].get("model_hash") == model_hash:
                return {
                    "found": True,
                    "integrity_ok": integrity_ok,
                    "index": block["index"],
                    "block_hash": block["hash"],
                    "record": block["record"],
                }
        return {"found": False, "integrity_ok": integrity_ok, "record": None}

    def verify_integrity(self) -> bool:
        prev = GENESIS_PREV
        for block in self._blocks:
            expected = _block_hash(block["index"], block["timestamp"], block["record"], prev)
            if expected != block["hash"] or block["prev_hash"] != prev:
                return False
            prev = block["hash"]
        return True
