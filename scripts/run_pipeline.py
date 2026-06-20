#!/usr/bin/env python3
"""Run the end-to-end pipeline.

    python scripts/run_pipeline.py --config configs/federated.yaml [dotlist overrides]

Example overrides:  federated.rounds=30  aggregation.mode=masked  model.backbone=vit
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qsfl.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
