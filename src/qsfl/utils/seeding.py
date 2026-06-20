"""Global seeding + determinism.

Reproducibility hazard handled here: CUDA ops are NOT deterministic by default.
For the *reported* headline numbers (accuracy / NC / BER) to be reproducible —
not just the crypto — we set torch's deterministic flags and the cuBLAS
workspace env var. This carries a performance cost (some fast nondeterministic
kernels are disabled), which we document rather than hide.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy and (if present) torch; optionally force determinism.

    Args:
        seed: the global seed.
        deterministic: if True, enable torch deterministic algorithms and disable
            cuDNN autotuning. Slower, but makes GPU results reproducible.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except Exception:
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # cuBLAS determinism (must be set before the first CUDA call to take full
        # effect; we set it here and document running it early).
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            # Older torch may lack warn_only; fall back to strict best-effort.
            torch.use_deterministic_algorithms(True)
