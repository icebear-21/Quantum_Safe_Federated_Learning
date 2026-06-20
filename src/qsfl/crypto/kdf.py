"""HKDF-SHA256 hybrid key derivation.

The data-encryption key combines two independent secrets::

    data_key = HKDF-SHA256(IKM = kyber_shared_secret ‖ K_q,
                           salt = per-update random,
                           info = context,
                           length = 32)            # AES-256

This is a hybrid PQC combiner: the resulting key is secure if *either* the Kyber
shared secret or the VQC key ``K_q`` remains secret. The per-update random salt
also guarantees a fresh ``data_key`` per update (see the nonce-reuse argument in
SECURITY.md).
"""

from __future__ import annotations

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

DATA_KEY_LEN = 32  # AES-256


def derive_data_key(
    kyber_shared_secret: bytes,
    quantum_key: bytes,
    salt: bytes,
    info: bytes = b"qsfl/data-key/v1",
    length: int = DATA_KEY_LEN,
) -> bytes:
    """Derive an AES-256 data key from the Kyber secret and the VQC key.

    Args:
        kyber_shared_secret: the KEM shared secret ``ss``.
        quantum_key: the VQC-derived key ``K_q`` (32 bytes for a 256-bit key).
        salt: fresh random salt per update (>= 16 bytes recommended).
        info: domain-separation/context string.
        length: output key length in bytes (32 for AES-256).
    """
    if not kyber_shared_secret or not quantum_key:
        raise ValueError("both kyber_shared_secret and quantum_key must be non-empty")
    ikm = bytes(kyber_shared_secret) + bytes(quantum_key)
    hkdf = HKDF(algorithm=hashes.SHA256(), length=length, salt=salt, info=info)
    return hkdf.derive(ikm)
