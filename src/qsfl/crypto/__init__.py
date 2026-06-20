"""Layer 2 crypto: hybrid KDF + post-quantum KEM + AES-256-GCM AEAD.

Public surface:
    get_kem(cfg)                  -> a KEM backend (kyber-py | liboqs)
    derive_data_key(...)          -> HKDF-SHA256(kyber_ss ‖ K_q)
    HybridEncryptor(...)          -> high-level encrypt/decrypt of byte payloads
    EncryptedPayload              -> serializable ciphertext container
"""

from qsfl.crypto.aead import NonceReuseError, aes_gcm_decrypt, aes_gcm_encrypt, reset_nonce_guard
from qsfl.crypto.encryptor import HybridEncryptor
from qsfl.crypto.kdf import derive_data_key
from qsfl.crypto.kem import KEM, get_kem
from qsfl.crypto.types import EncryptedPayload

__all__ = [
    "KEM",
    "get_kem",
    "derive_data_key",
    "HybridEncryptor",
    "EncryptedPayload",
    "aes_gcm_encrypt",
    "aes_gcm_decrypt",
    "reset_nonce_guard",
    "NonceReuseError",
]
