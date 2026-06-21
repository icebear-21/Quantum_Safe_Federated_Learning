"""Variational Quantum Circuit (VQC) key generator.

A parameterized circuit of ``R_Y(θ)`` rotation layers interleaved with CNOT
entanglers over ``n`` qubits and ``L`` layers. We measure the circuit and derive
a 256-bit key ``K_q`` from the outcome.

Determinism (reproducibility requirement): the key is fully determined by
``theta_seed`` (which fixes the circuit parameters) and the analytic statevector,
then expanded to ``key_bits`` via SHA-256. This **emulates** QKD-style entropy —
we do not claim true quantum randomness from a classical simulator (see
SECURITY.md). A separate VQC-given-(theta,seed) determinism test guards this.

Backends: ``default.qubit`` (CPU, default), ``lightning.qubit`` (fast CPU), or
``lightning.gpu`` (GPU statevector on the H100) if the plugin is installed.
"""

from __future__ import annotations

import hashlib

import numpy as np


def _expand(seed: bytes, length: int) -> bytes:
    """Deterministically expand ``seed`` to ``length`` bytes via counter-mode SHA-256."""
    out = bytearray()
    counter = 0
    while len(out) < length:
        out += hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
        counter += 1
    return bytes(out[:length])


class VQCKeyGenerator:
    """Builds the circuit once, then produces the key on demand."""

    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 3,
        key_bits: int = 256,
        theta_seed: int = 7,
        backend: str = "default.qubit",
        shots: int | None = None,
    ) -> None:
        if key_bits % 8 != 0:
            raise ValueError("key_bits must be a multiple of 8")
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.key_bits = key_bits
        self.theta_seed = theta_seed
        self.backend = backend
        self.shots = shots

        rng = np.random.default_rng(theta_seed)
        self.theta = rng.uniform(0.0, 2.0 * np.pi, size=(n_layers, n_qubits))

    def _measure(self) -> np.ndarray:
        """Return the measurement vector used as raw key material."""
        try:
            import pennylane as qml  # noqa: PLC0415
        except Exception as exc:  # pragma: no cover - import guard
            raise ImportError(
                "PennyLane is required for VQC key generation (pip install pennylane)."
            ) from exc

        try:
            dev = qml.device(self.backend, wires=self.n_qubits, shots=self.shots)
        except qml.DeviceError:
            # e.g. lightning.gpu plugin not installed; fall back to the CPU
            # statevector (negligible cost for the small qubit counts used here).
            from qsfl.utils.logging import get_logger

            get_logger("quantum").warning(
                "quantum backend %r unavailable; falling back to default.qubit", self.backend
            )
            self.backend = "default.qubit"
            dev = qml.device(self.backend, wires=self.n_qubits, shots=self.shots)

        @qml.qnode(dev)
        def circuit(theta):
            for layer in range(self.n_layers):
                for q in range(self.n_qubits):
                    qml.RY(theta[layer, q], wires=q)
                for q in range(self.n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
                if self.n_qubits > 1:
                    qml.CNOT(wires=[self.n_qubits - 1, 0])  # ring entanglement
            return qml.probs(wires=list(range(self.n_qubits)))

        return np.asarray(circuit(self.theta), dtype=np.float64)

    def generate_key(self) -> bytes:
        """Produce the deterministic ``key_bits``-bit ``K_q`` as bytes."""
        probs = self._measure()
        # Quantize to stabilize across float jitter, then hash + expand.
        quantized = np.round(probs * (2**20)).astype(np.int64)
        digest = hashlib.sha256(
            quantized.tobytes() + self.theta.tobytes() + str(self.theta_seed).encode()
        ).digest()
        return _expand(digest, self.key_bits // 8)


def generate_quantum_key(cfg) -> bytes:
    """Convenience: build a generator from a config node and return ``K_q``.

    Accepts the ``quantum`` config sub-tree (``cfg.quantum``).
    """
    q = cfg
    return VQCKeyGenerator(
        n_qubits=int(q.n_qubits),
        n_layers=int(q.n_layers),
        key_bits=int(q.key_bits),
        theta_seed=int(q.theta_seed),
        backend=str(q.backend),
        shots=q.get("shots", None),
    ).generate_key()
