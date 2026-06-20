"""Secure-aggregation math + masking utilities.

Two modes, both producing the SAME weighted-FedAvg result; they differ only in
what the server gets to see:

  * ``authorized_decrypt`` (default, simulation): server decrypts each client's
    raw update and runs weighted average. Simple; server sees individual updates.

  * ``masked`` (Bonawitz-style): each client sends ``w_i·ΔW_i + mask_i`` where
    pairwise masks cancel in the sum, so the server only ever sees masked
    updates. ``test_secure_agg.py`` asserts the masks truly cancel.

Prototype honesty (see SECURITY.md): the pairwise masks here are PRG-derived from
a shared ``mask_seed`` for reproducibility, and masks are computed over the set of
participating clients for the round (so they always cancel among participants). A
real deployment derives masks from authenticated Diffie-Hellman so the server
cannot reconstruct them, and uses secret-sharing to recover from mid-round
dropouts. We expose ``dropout_tolerance`` and recompute masks over survivors.
"""

from __future__ import annotations

import hashlib

import numpy as np

State = dict[str, np.ndarray]


# --- state arithmetic -------------------------------------------------------
def add_states(a: State, b: State) -> State:
    return {k: a[k] + b[k] for k in a}


def subtract_states(a: State, b: State) -> State:
    return {k: a[k] - b[k] for k in a}


def scale_state(s: State, factor: float) -> State:
    return {k: (v.astype(np.float64) * factor).astype(v.dtype) for k, v in s.items()}


def sum_states(states: list[State]) -> State:
    out = {k: np.zeros_like(v) for k, v in states[0].items()}
    for s in states:
        for k in out:
            out[k] = out[k] + s[k]
    return out


def weighted_average(states: list[State], weights: list[float]) -> State:
    """Weighted FedAvg: ``Σ w_i·s_i / Σ w_i``."""
    total = float(sum(weights))
    out = {k: np.zeros_like(v, dtype=np.float64) for k, v in states[0].items()}
    for s, w in zip(states, weights):
        for k in out:
            out[k] += s[k].astype(np.float64) * float(w)
    return {k: (v / total).astype(states[0][k].dtype) for k, v in out.items()}


# --- pairwise masking -------------------------------------------------------
def _pair_seed(seed: int, lo: int, hi: int) -> int:
    h = hashlib.sha256(f"{seed}:{lo}:{hi}".encode()).digest()
    return int.from_bytes(h[:8], "big")


def generate_pairwise_mask(
    client_id: int, participants: list[int], template: State, seed: int
) -> State:
    """Mask for ``client_id`` s.t. ``Σ_i mask_i = 0`` over ``participants``.

    For each pair (i, j) a shared PRG (seeded by the unordered pair) draws a mask;
    the lower id adds it, the higher id subtracts it. Casting to the template
    dtype is symmetric (``trunc(-x) == -trunc(x)``), so cancellation is exact even
    for integer buffers.
    """
    mask: State = {k: np.zeros_like(v) for k, v in template.items()}
    for other in participants:
        if other == client_id:
            continue
        lo, hi = min(client_id, other), max(client_id, other)
        rng = np.random.default_rng(_pair_seed(seed, lo, hi))
        sign = 1.0 if client_id < other else -1.0
        for k, v in template.items():
            draw = rng.standard_normal(v.shape)
            mask[k] = mask[k] + (sign * draw).astype(v.dtype)
    return mask
