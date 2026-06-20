"""Lightweight, consistent logging."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "qsfl", level: str = "INFO") -> logging.Logger:
    """Return a process-wide configured logger."""
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S")
        )
        root = logging.getLogger("qsfl")
        root.addHandler(handler)
        root.setLevel(getattr(logging, level.upper(), logging.INFO))
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(name if name.startswith("qsfl") else f"qsfl.{name}")
