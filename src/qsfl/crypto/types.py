"""Serializable ciphertext container."""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass, field


def _b64e(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))


@dataclass
class EncryptedPayload:
    """Everything needed to decrypt one payload (+ provenance metadata).

    Persisted per encrypted update ``C_i`` and per encrypted model ``C``. ``H =
    SHA256(C)`` (see :meth:`ciphertext_hash`) is what the ledger registers.
    """

    ciphertext: bytes
    kyber_ct: bytes
    nonce: bytes
    salt: bytes
    aad: bytes = b""
    kem_backend: str = "kyber-py"
    kem_algorithm: str = "ML-KEM-768"
    label: str = ""
    meta: dict = field(default_factory=dict)

    def ciphertext_hash(self) -> str:
        """``H = SHA256(C)`` over the ciphertext, hex-encoded."""
        return hashlib.sha256(self.ciphertext).hexdigest()

    def to_dict(self) -> dict:
        return {
            "ciphertext": _b64e(self.ciphertext),
            "kyber_ct": _b64e(self.kyber_ct),
            "nonce": _b64e(self.nonce),
            "salt": _b64e(self.salt),
            "aad": _b64e(self.aad),
            "kem_backend": self.kem_backend,
            "kem_algorithm": self.kem_algorithm,
            "label": self.label,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EncryptedPayload":
        return cls(
            ciphertext=_b64d(d["ciphertext"]),
            kyber_ct=_b64d(d["kyber_ct"]),
            nonce=_b64d(d["nonce"]),
            salt=_b64d(d["salt"]),
            aad=_b64d(d.get("aad", "")),
            kem_backend=d.get("kem_backend", "kyber-py"),
            kem_algorithm=d.get("kem_algorithm", "ML-KEM-768"),
            label=d.get("label", ""),
            meta=d.get("meta", {}),
        )
