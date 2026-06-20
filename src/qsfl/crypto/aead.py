"""AES-256-GCM AEAD with a (key, nonce) reuse guard.

Nonce reuse under the same key is the classic AES-GCM catastrophe (it leaks the
authentication subkey and XOR of plaintexts). Our key-derivation gives a fresh
``data_key`` per update via a random salt, so reuse is already unlikely — but we
also enforce a hard invariant: the same ``(key, nonce)`` pair is never used to
encrypt twice in a process. This is cheap insurance and is unit-tested.
"""

from __future__ import annotations

import hashlib

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

NONCE_LEN = 12  # 96-bit nonce (GCM standard / most efficient)

_SEEN_KEY_NONCE: set[bytes] = set()


class NonceReuseError(RuntimeError):
    """Raised when the same (key, nonce) pair is reused for encryption."""


def reset_nonce_guard() -> None:
    """Clear the in-process (key, nonce) ledger (used by tests)."""
    _SEEN_KEY_NONCE.clear()


def _guard(key: bytes, nonce: bytes) -> None:
    tag = hashlib.sha256(b"kng" + key + nonce).digest()
    if tag in _SEEN_KEY_NONCE:
        raise NonceReuseError("AES-GCM (key, nonce) pair reuse detected — refusing to encrypt")
    _SEEN_KEY_NONCE.add(tag)


def aes_gcm_encrypt(
    key: bytes, nonce: bytes, plaintext: bytes, aad: bytes = b"", *, guard: bool = True
) -> bytes:
    """Encrypt ``plaintext`` with AES-256-GCM. Returns ciphertext (incl. tag).

    Args:
        key: 32-byte AES-256 key.
        nonce: 12-byte nonce, unique per key.
        plaintext: data to encrypt.
        aad: additional authenticated (but not encrypted) data.
        guard: enforce the (key, nonce)-uniqueness invariant (default True).
    """
    if len(key) != 32:
        raise ValueError(f"AES-256-GCM requires a 32-byte key, got {len(key)}")
    if len(nonce) != NONCE_LEN:
        raise ValueError(f"nonce must be {NONCE_LEN} bytes, got {len(nonce)}")
    if guard:
        _guard(key, nonce)
    return AESGCM(key).encrypt(nonce, plaintext, aad)


def aes_gcm_decrypt(key: bytes, nonce: bytes, ciphertext: bytes, aad: bytes = b"") -> bytes:
    """Decrypt + verify AES-256-GCM ciphertext. Raises on tamper (bad tag)."""
    if len(key) != 32:
        raise ValueError(f"AES-256-GCM requires a 32-byte key, got {len(key)}")
    return AESGCM(key).decrypt(nonce, ciphertext, aad)
