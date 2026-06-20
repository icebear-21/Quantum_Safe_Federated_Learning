"""Gated liboqs round-trip — keeps the PRIMARY (reported-results) KEM backend
from silently rotting while everyone develops on the kyber-py fallback.

Skipped automatically when liboqs-python is not installed; runs in the full path
(``pytest -m liboqs``) on the conda / native environment.
"""

import pytest

oqs = pytest.importorskip("oqs", reason="liboqs-python not installed")

from qsfl.crypto import HybridEncryptor, get_kem  # noqa: E402

pytestmark = pytest.mark.liboqs


def test_liboqs_kem_roundtrip():
    kem = get_kem("liboqs", "ML-KEM-768")
    pk, sk = kem.keypair()
    ct, ss_enc = kem.encapsulate(pk)
    assert kem.decapsulate(sk, ct) == ss_enc


def test_liboqs_hybrid_roundtrip():
    import os

    enc = HybridEncryptor(quantum_key=os.urandom(32), backend="liboqs")
    payload = b"reported-results path" * 50
    assert enc.decrypt(enc.encrypt(payload)) == payload
