"""High-level hybrid encryptor tying KEM + HKDF + AES-256-GCM together.

Roles in the prototype (a simulation of the real-world key distribution):
  * The authorized aggregator/owner holds the KEM keypair ``(pk, sk)`` and the
    VQC key ``K_q``.
  * Clients encrypt updates with ``pk`` + ``K_q`` (``encrypt``).
  * Only the aggregator/owner can ``decrypt`` (needs ``sk``).

In a real deployment ``K_q`` would be distributed over a QKD link / QRNG rather
than shared in-process; here we share it to keep the prototype self-contained.
See SECURITY.md.
"""

from __future__ import annotations

import os

from qsfl.crypto.aead import NONCE_LEN, aes_gcm_decrypt, aes_gcm_encrypt
from qsfl.crypto.kdf import derive_data_key
from qsfl.crypto.kem import KEM, get_kem
from qsfl.crypto.types import EncryptedPayload

SALT_LEN = 16


class HybridEncryptor:
    """Encrypt/decrypt byte payloads with the hybrid (Kyber ‖ K_q) data key."""

    def __init__(
        self,
        quantum_key: bytes,
        kem: KEM | None = None,
        *,
        backend: str = "kyber-py",
        algorithm: str = "ML-KEM-768",
        public_key: bytes | None = None,
        secret_key: bytes | None = None,
        info: bytes = b"qsfl/data-key/v1",
    ) -> None:
        if len(quantum_key) < 16:
            raise ValueError("quantum_key looks too short; expected >= 16 bytes")
        self.kem = kem or get_kem(backend, algorithm)
        self.quantum_key = bytes(quantum_key)
        self.info = info
        if public_key is None or secret_key is None:
            public_key, secret_key = self.kem.keypair()
        self.public_key = public_key
        self.secret_key = secret_key

    def encrypt(self, payload: bytes, aad: bytes = b"", label: str = "") -> EncryptedPayload:
        """Encrypt ``payload``; only the holder of ``secret_key`` can decrypt."""
        kyber_ct, ss = self.kem.encapsulate(self.public_key)
        salt = os.urandom(SALT_LEN)
        data_key = derive_data_key(ss, self.quantum_key, salt, self.info)
        nonce = os.urandom(NONCE_LEN)
        ciphertext = aes_gcm_encrypt(data_key, nonce, payload, aad)
        return EncryptedPayload(
            ciphertext=ciphertext,
            kyber_ct=kyber_ct,
            nonce=nonce,
            salt=salt,
            aad=aad,
            kem_backend=self.kem.name,
            kem_algorithm=self.kem.algorithm,
            label=label,
        )

    def decrypt(self, payload: EncryptedPayload) -> bytes:
        """Decrypt an :class:`EncryptedPayload` (raises on tamper)."""
        ss = self.kem.decapsulate(self.secret_key, payload.kyber_ct)
        data_key = derive_data_key(ss, self.quantum_key, payload.salt, self.info)
        return aes_gcm_decrypt(data_key, payload.nonce, payload.ciphertext, payload.aad)
