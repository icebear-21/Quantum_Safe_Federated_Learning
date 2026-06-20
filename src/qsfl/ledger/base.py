"""Ledger interface + ownership record."""

from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field


@dataclass
class OwnershipRecord:
    """What gets registered: H = SHA256(C) plus ownership/watermark provenance."""

    model_hash: str  # H = SHA256(C)
    owner: str
    watermark_commitment: str = ""  # hash binding the watermark payloads
    timestamp: float = field(default_factory=time.time)
    meta: dict = field(default_factory=dict)

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()


class Ledger(ABC):
    """Append-only ownership ledger."""

    @abstractmethod
    def register(self, record: OwnershipRecord) -> dict:
        """Register ``record``; return a proof (backend-specific dict)."""

    @abstractmethod
    def verify(self, model_hash: str) -> dict:
        """Look up ``model_hash``; return ``{found, record, integrity_ok, ...}``."""

    @abstractmethod
    def verify_integrity(self) -> bool:
        """Return True iff the whole ledger is internally consistent (untampered)."""
