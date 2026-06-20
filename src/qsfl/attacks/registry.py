"""Attack registry."""

from __future__ import annotations

from collections.abc import Callable

from qsfl.attacks.fine_tuning import run as _fine_tuning
from qsfl.attacks.gradient_leakage import run as _gradient_leakage
from qsfl.attacks.membership_inference import run as _membership_inference
from qsfl.attacks.model_extraction import run as _model_extraction
from qsfl.attacks.poisoning import run as _poisoning
from qsfl.attacks.pruning import run as _pruning
from qsfl.attacks.watermark_removal import run as _watermark_removal

REGISTRY: dict[str, Callable] = {
    "fine_tuning": _fine_tuning,
    "pruning": _pruning,
    "watermark_removal": _watermark_removal,
    "model_extraction": _model_extraction,
    "membership_inference": _membership_inference,
    "gradient_leakage": _gradient_leakage,
    "poisoning": _poisoning,
}


def get_attack(name: str) -> Callable:
    if name not in REGISTRY:
        raise ValueError(f"unknown attack {name!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[name]
