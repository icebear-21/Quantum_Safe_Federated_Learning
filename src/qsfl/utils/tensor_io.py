"""Deterministic, pickle-free (de)serialization of weight dicts to bytes.

We avoid ``torch.save``/pickle for the encryption payloads because (a) pickle is
unsafe to load from untrusted ciphertext and (b) we want a canonical, stable
byte layout so ``H = SHA256(C)`` is reproducible. Layout: a small JSON header
(sorted keys, dtype, shape) followed by raw little-endian array bytes.
"""

from __future__ import annotations

import io
import json

import numpy as np

_MAGIC = b"QSFLW1\n"


def pack_state(state: dict[str, np.ndarray]) -> bytes:
    """Serialize ``{name: ndarray}`` to canonical bytes (sorted by name)."""
    names = sorted(state.keys())
    header = []
    buf = io.BytesIO()
    buf.write(_MAGIC)
    for name in names:
        arr = np.ascontiguousarray(state[name])
        raw = arr.tobytes(order="C")
        header.append({"name": name, "dtype": str(arr.dtype), "shape": list(arr.shape), "nbytes": len(raw)})
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    buf.write(len(header_bytes).to_bytes(8, "big"))
    buf.write(header_bytes)
    for name in names:
        buf.write(np.ascontiguousarray(state[name]).tobytes(order="C"))
    return buf.getvalue()


def unpack_state(blob: bytes) -> dict[str, np.ndarray]:
    """Inverse of :func:`pack_state`."""
    if not blob.startswith(_MAGIC):
        raise ValueError("bad magic; not a QSFL weight blob")
    pos = len(_MAGIC)
    header_len = int.from_bytes(blob[pos : pos + 8], "big")
    pos += 8
    header = json.loads(blob[pos : pos + header_len].decode("utf-8"))
    pos += header_len
    out: dict[str, np.ndarray] = {}
    for entry in header:
        nbytes = entry["nbytes"]
        arr = np.frombuffer(blob[pos : pos + nbytes], dtype=np.dtype(entry["dtype"]))
        out[entry["name"]] = arr.reshape(entry["shape"]).copy()
        pos += nbytes
    return out
