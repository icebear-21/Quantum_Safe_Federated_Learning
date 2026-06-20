"""Layer 5: threat-model evaluation suite.

Each attack is an independent module exposing ``run(ctx) -> dict`` and is
registered in :data:`REGISTRY`. The harness (:func:`run_attacks`) builds the
protected model once and runs every enabled attack against a fresh copy.
"""

from qsfl.attacks.base import AttackContext, watermark_metrics
from qsfl.attacks.registry import REGISTRY, get_attack
from qsfl.attacks.runner import run_attacks

__all__ = ["AttackContext", "watermark_metrics", "REGISTRY", "get_attack", "run_attacks"]
