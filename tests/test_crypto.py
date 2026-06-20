import os

import pytest

from qsfl.crypto import (
    HybridEncryptor,
    NonceReuseError,
    aes_gcm_encrypt,
    derive_data_key,
    get_kem,
    reset_nonce_guard,
)


@pytest.fixture(autouse=True)
def _clear_guard():
    reset_nonce_guard()
    yield
    reset_nonce_guard()


def test_kyber_py_kem_roundtrip():
    kem = get_kem("kyber-py", "ML-KEM-768")
    pk, sk = kem.keypair()
    ct, ss_enc = kem.encapsulate(pk)
    ss_dec = kem.decapsulate(sk, ct)
    assert ss_enc == ss_dec
    assert len(ss_enc) >= 16


def test_hybrid_encrypt_decrypt_roundtrip():
    kq = os.urandom(32)
    enc = HybridEncryptor(quantum_key=kq, backend="kyber-py")
    payload = b"the global model weights" * 100
    blob = enc.encrypt(payload, aad=b"client-1", label="dW_1")
    assert blob.ciphertext != payload
    assert enc.decrypt(blob) == payload


def test_tamper_is_detected():
    enc = HybridEncryptor(quantum_key=os.urandom(32), backend="kyber-py")
    blob = enc.encrypt(b"secret weights")
    blob.ciphertext = blob.ciphertext[:-1] + bytes([blob.ciphertext[-1] ^ 0x01])
    with pytest.raises(Exception):
        enc.decrypt(blob)


def test_wrong_quantum_key_fails_to_decrypt():
    payload = b"hybrid key needs BOTH secrets"
    enc = HybridEncryptor(quantum_key=os.urandom(32), backend="kyber-py")
    blob = enc.encrypt(payload)
    # Same KEM keypair, different K_q -> different data key -> auth failure.
    attacker = HybridEncryptor(
        quantum_key=os.urandom(32),
        backend="kyber-py",
        public_key=enc.public_key,
        secret_key=enc.secret_key,
    )
    with pytest.raises(Exception):
        attacker.decrypt(blob)


def test_nonce_reuse_guard():
    key = os.urandom(32)
    nonce = os.urandom(12)
    aes_gcm_encrypt(key, nonce, b"first")
    with pytest.raises(NonceReuseError):
        aes_gcm_encrypt(key, nonce, b"second")  # same (key, nonce) -> refused


def test_hkdf_combines_both_secrets():
    salt = os.urandom(16)
    k1 = derive_data_key(b"ss-aaaa", b"kq-bbbb", salt)
    k2 = derive_data_key(b"ss-aaaa", b"kq-cccc", salt)  # only K_q differs
    assert k1 != k2 and len(k1) == 32
