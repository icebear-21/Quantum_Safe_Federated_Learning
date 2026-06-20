"""VQC key determinism: same (theta_seed) -> same K_q; different seed -> different."""

import pytest

pytest.importorskip("pennylane", reason="pennylane not installed")

from qsfl.quantum import VQCKeyGenerator  # noqa: E402


def test_key_is_deterministic_given_seed():
    g1 = VQCKeyGenerator(n_qubits=4, n_layers=2, key_bits=256, theta_seed=7)
    g2 = VQCKeyGenerator(n_qubits=4, n_layers=2, key_bits=256, theta_seed=7)
    assert g1.generate_key() == g2.generate_key()


def test_key_length_is_256_bits():
    key = VQCKeyGenerator(n_qubits=4, n_layers=2, key_bits=256, theta_seed=7).generate_key()
    assert len(key) == 32


def test_different_seed_changes_key():
    a = VQCKeyGenerator(n_qubits=4, n_layers=2, theta_seed=7).generate_key()
    b = VQCKeyGenerator(n_qubits=4, n_layers=2, theta_seed=8).generate_key()
    assert a != b
