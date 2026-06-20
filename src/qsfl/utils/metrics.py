"""Core metrics: Normalized Correlation (NC), Bit Error Rate (BER), accuracy.

These define the ownership-verification decision (see qsfl.verification) and the
attack-evaluation deltas, so they are unit-tested for correctness.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _as_vector(x: ArrayLike) -> np.ndarray:
    return np.asarray(x).reshape(-1).astype(np.float64)


def normalized_correlation(a: ArrayLike, b: ArrayLike, *, binary: bool = False) -> float:
    """Normalized correlation (cosine similarity) between two equal-length vectors.

    NC(a, b) = <a, b> / (||a|| * ||b||), in [-1, 1].

    Args:
        a, b: equal-length arrays (watermark payloads, weight projections, ...).
        binary: if True, map {0,1} -> {-1,+1} before correlating, which is the
            usual convention for bit-valued watermarks (makes NC=1 for an exact
            match and NC=-1 for the complement).
    """
    va, vb = _as_vector(a), _as_vector(b)
    if va.shape != vb.shape:
        raise ValueError(f"shape mismatch for NC: {va.shape} vs {vb.shape}")
    if binary:
        va = 2.0 * va - 1.0
        vb = 2.0 * vb - 1.0
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def bit_error_rate(a: ArrayLike, b: ArrayLike) -> float:
    """Fraction of differing bits between two equal-length bit sequences, in [0,1]."""
    va = _as_vector(a).round().astype(np.int64)
    vb = _as_vector(b).round().astype(np.int64)
    if va.shape != vb.shape:
        raise ValueError(f"shape mismatch for BER: {va.shape} vs {vb.shape}")
    if va.size == 0:
        return 0.0
    return float(np.mean(va != vb))


def accuracy(preds: ArrayLike, targets: ArrayLike) -> float:
    """Top-1 classification accuracy in [0,1]."""
    p = _as_vector(preds).round().astype(np.int64)
    t = _as_vector(targets).round().astype(np.int64)
    if p.shape != t.shape:
        raise ValueError(f"shape mismatch for accuracy: {p.shape} vs {t.shape}")
    if p.size == 0:
        return 0.0
    return float(np.mean(p == t))
