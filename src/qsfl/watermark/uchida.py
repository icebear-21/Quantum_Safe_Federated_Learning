"""Uchida-style white-box weight watermarking (closed-form embedding).

A secret projection ``X`` (``T`` orthonormal rows in R^M, derived from a private
``key_seed``) maps a flattened weight vector ``w`` to ``T`` projection values. We
embed a ``T``-bit message ``b`` (mapped to ±1 as ``s``) so each projection has the
right sign with at least ``margin`` of slack — but only *nudge* the bits that are
wrong-signed or under-margin, leaving correctly-signed projections untouched
(minimal-perturbation embedding, far gentler on small layers than overwriting all
T directions):

    p = X w
    correction_t = max(0, margin - s_t · p_t) · s_t
    w' = w + Xᵀ correction        # rows orthonormal -> X w' = p + correction

Extraction reads the sign of the projection (no original weights needed):

    b' = 1[ X w' > 0 ]

Clean extraction is perfect (NC = 1, BER = 0) by construction; robustness to
fine-tuning/pruning grows with ``margin`` (the strength knob), which Layer 5
measures.
"""

from __future__ import annotations

import numpy as np


def carriers(num_bits: int, dim: int, key_seed: int) -> np.ndarray:
    """Deterministic ``(num_bits, dim)`` matrix with orthonormal rows."""
    if dim < num_bits:
        raise ValueError(f"weight too small ({dim}) for {num_bits}-bit watermark")
    rng = np.random.default_rng(key_seed)
    g = rng.standard_normal((dim, num_bits))
    q, _ = np.linalg.qr(g)  # q: (dim, num_bits), orthonormal columns
    return q.T  # orthonormal rows


def embed_bits(
    weight_flat: np.ndarray, bits: np.ndarray, key_seed: int, margin: float
) -> np.ndarray:
    """Return a copy of ``weight_flat`` with ``bits`` embedded (minimal nudge)."""
    w = weight_flat.astype(np.float64).reshape(-1)
    bits = np.asarray(bits).reshape(-1)
    x = carriers(len(bits), w.size, key_seed)
    s = 2.0 * bits.astype(np.float64) - 1.0
    p = x @ w
    deficit = float(margin) - s * p
    correction = np.where(deficit > 0.0, deficit * s, 0.0)
    w_new = w + x.T @ correction
    return w_new.astype(weight_flat.dtype).reshape(weight_flat.shape)


def extract_bits(weight_flat: np.ndarray, num_bits: int, key_seed: int) -> np.ndarray:
    """Recover the embedded bits from ``weight_flat``."""
    w = weight_flat.astype(np.float64).reshape(-1)
    x = carriers(num_bits, w.size, key_seed)
    return (x @ w > 0).astype(np.int64)
