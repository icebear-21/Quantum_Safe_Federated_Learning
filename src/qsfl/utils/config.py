"""Config loading.

Layering (lowest precedence first):
    configs/default.yaml  ->  named config (e.g. federated.yaml)  ->  CLI dotlist

Everything is config-driven; nothing in the codebase hard-codes a path or a
magic number. ``device: auto`` is resolved to ``cpu``/``cuda`` here so the rest
of the code can treat it as concrete.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from omegaconf import DictConfig, OmegaConf

_DEFAULT_NAME = "default.yaml"


def find_repo_root(start: Path | None = None) -> Path:
    """Walk upward until we find a ``configs/default.yaml`` (repo root marker)."""
    here = (start or Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        if (parent / "configs" / _DEFAULT_NAME).is_file():
            return parent
    # Fallback: src/qsfl/utils/config.py -> repo root is parents[3]
    return Path(__file__).resolve().parents[3]


def _resolve_device(cfg: DictConfig) -> None:
    if cfg.get("device", "auto") != "auto":
        return
    try:
        import torch

        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        cfg.device = "cpu"


def load_config(
    config_path: str | Path,
    overrides: Sequence[str] | None = None,
    *,
    resolve_device: bool = True,
) -> DictConfig:
    """Load ``default.yaml`` then merge ``config_path`` then CLI ``overrides``.

    Args:
        config_path: path to a named config (may itself be default.yaml).
        overrides: OmegaConf dotlist overrides, e.g. ``["federated.rounds=5"]``.
        resolve_device: if True, turn ``device: auto`` into ``cpu``/``cuda``.
    """
    root = find_repo_root()
    default_path = root / "configs" / _DEFAULT_NAME
    base = OmegaConf.load(default_path)

    config_path = Path(config_path)
    if config_path.resolve() != default_path.resolve():
        named = OmegaConf.load(config_path)
        cfg = OmegaConf.merge(base, named)
    else:
        cfg = base

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))

    if resolve_device:
        _resolve_device(cfg)

    return cfg  # type: ignore[return-value]


def save_config(cfg: DictConfig, path: str | Path) -> None:
    """Persist the fully-resolved config alongside results for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write(OmegaConf.to_yaml(cfg, resolve=True))
