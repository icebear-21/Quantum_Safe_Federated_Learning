"""Post-quantum Key Encapsulation Mechanism (ML-KEM / Kyber).

One stateless interface, two backends:

  * ``kyber-py`` — pure-Python ML-KEM. No native build; used by the smoke test
    and as the default. NOT constant-time (fine for a prototype).
  * ``liboqs``   — Open Quantum Safe native implementation. The PRIMARY backend
    for reported results (vetted). Requires the native liboqs library.

Both implement::

    pk, sk = kem.keypair()
    ct, ss = kem.encapsulate(pk)
    ss     = kem.decapsulate(sk, ct)     # equals the ss from encapsulate

``ss`` (the KEM shared secret) is fed, together with the VQC key ``K_q``, into
the HKDF combiner in ``qsfl.crypto.kdf`` — so the data key is safe if *either*
secret holds (a standard hybrid-PQC construction).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

# Canonical algorithm name -> backend-specific aliases (newer liboqs/kyber-py use
# the FIPS-203 "ML-KEM-*" names; older builds exposed "Kyber*").
_ALIASES = {
    "ML-KEM-512": {"liboqs": ["ML-KEM-512", "Kyber512"], "kyber-py": "ML_KEM_512"},
    "ML-KEM-768": {"liboqs": ["ML-KEM-768", "Kyber768"], "kyber-py": "ML_KEM_768"},
    "ML-KEM-1024": {"liboqs": ["ML-KEM-1024", "Kyber1024"], "kyber-py": "ML_KEM_1024"},
}


class KEM(ABC):
    """Stateless KEM interface."""

    name: str
    algorithm: str

    @abstractmethod
    def keypair(self) -> tuple[bytes, bytes]:
        """Return ``(public_key, secret_key)``."""

    @abstractmethod
    def encapsulate(self, public_key: bytes) -> tuple[bytes, bytes]:
        """Return ``(ciphertext, shared_secret)`` for ``public_key``."""

    @abstractmethod
    def decapsulate(self, secret_key: bytes, ciphertext: bytes) -> bytes:
        """Recover and return ``shared_secret`` from ``ciphertext``."""


class KyberPyKEM(KEM):
    """Pure-Python ML-KEM via the ``kyber-py`` package (default backend).

    The tuple ordering of ``keygen``/``encaps`` has shifted across kyber-py
    versions, so we disambiguate by length rather than position: the ML-KEM
    shared secret is always 32 bytes, while ciphertexts and keys are far larger.
    """

    name = "kyber-py"

    def __init__(self, algorithm: str = "ML-KEM-768") -> None:
        try:
            from kyber_py.ml_kem import ML_KEM_512, ML_KEM_768, ML_KEM_1024  # type: ignore
        except Exception as exc:  # pragma: no cover - import guard
            raise ImportError(
                "kyber-py is required for the default KEM backend. "
                "Install it with `pip install kyber-py` (it is in requirements.txt)."
            ) from exc
        impls = {"ML-KEM-512": ML_KEM_512, "ML-KEM-768": ML_KEM_768, "ML-KEM-1024": ML_KEM_1024}
        if algorithm not in impls:
            raise ValueError(f"unsupported KEM algorithm: {algorithm}")
        self.algorithm = algorithm
        self._impl = impls[algorithm]

    def keypair(self) -> tuple[bytes, bytes]:
        a, b = (bytes(x) for x in self._impl.keygen())
        # encapsulation (public) key is shorter than the decapsulation (secret) key
        return (a, b) if len(a) < len(b) else (b, a)

    def encapsulate(self, public_key: bytes) -> tuple[bytes, bytes]:
        a, b = (bytes(x) for x in self._impl.encaps(public_key))
        ss, ct = (a, b) if len(a) == 32 else (b, a)  # shared secret is 32 bytes
        return ct, ss

    def decapsulate(self, secret_key: bytes, ciphertext: bytes) -> bytes:
        try:
            return bytes(self._impl.decaps(secret_key, ciphertext))
        except (TypeError, ValueError):
            # tolerate (ciphertext, dk) arg order in some kyber-py versions
            return bytes(self._impl.decaps(ciphertext, secret_key))


class LibOQSKEM(KEM):
    """Native ML-KEM via Open Quantum Safe (``liboqs-python``). Reported results."""

    name = "liboqs"

    def __init__(self, algorithm: str = "ML-KEM-768") -> None:
        try:
            import oqs  # type: ignore
        except Exception as exc:  # pragma: no cover - import guard
            raise ImportError(
                "liboqs-python (import name `oqs`) is not installed. Use the conda "
                "environment (environment.yml) or `make setup-liboqs`, or fall back "
                "to aggregation.kem.backend=kyber-py."
            ) from exc
        self._oqs = oqs
        self.algorithm = self._resolve_algorithm(algorithm)

    def _resolve_algorithm(self, algorithm: str) -> str:
        enabled = set(self._oqs.get_enabled_kem_mechanisms())
        for candidate in _ALIASES.get(algorithm, {}).get("liboqs", [algorithm]):
            if candidate in enabled:
                return candidate
        raise ValueError(
            f"liboqs has no enabled mechanism for {algorithm}. Enabled: {sorted(enabled)}"
        )

    def keypair(self) -> tuple[bytes, bytes]:
        with self._oqs.KeyEncapsulation(self.algorithm) as kem:
            pk = kem.generate_keypair()
            sk = kem.export_secret_key()
        return bytes(pk), bytes(sk)

    def encapsulate(self, public_key: bytes) -> tuple[bytes, bytes]:
        with self._oqs.KeyEncapsulation(self.algorithm) as kem:
            ct, ss = kem.encap_secret(public_key)
        return bytes(ct), bytes(ss)

    def decapsulate(self, secret_key: bytes, ciphertext: bytes) -> bytes:
        with self._oqs.KeyEncapsulation(self.algorithm, secret_key=secret_key) as kem:
            ss = kem.decap_secret(ciphertext)
        return bytes(ss)


def get_kem(backend: str = "kyber-py", algorithm: str = "ML-KEM-768") -> KEM:
    """Factory: return the requested KEM backend.

    Args:
        backend: ``"kyber-py"`` (default, pure Python) or ``"liboqs"`` (native).
        algorithm: one of ``ML-KEM-512 / ML-KEM-768 / ML-KEM-1024``.
    """
    backend = backend.lower()
    if backend in ("kyber-py", "kyber_py", "kyberpy"):
        return KyberPyKEM(algorithm)
    if backend in ("liboqs", "oqs"):
        return LibOQSKEM(algorithm)
    raise ValueError(f"unknown KEM backend: {backend!r} (use 'kyber-py' or 'liboqs')")
