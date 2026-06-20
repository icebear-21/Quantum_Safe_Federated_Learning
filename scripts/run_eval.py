#!/usr/bin/env python3
"""Run the Layer-5 attack-evaluation suite.

    python scripts/run_eval.py --config configs/federated.yaml [dotlist overrides]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qsfl.utils.config import load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(prog="qsfl-eval", description="QSFL attack evaluation")
    parser.add_argument("--config", required=True, help="path to a YAML config")
    args, overrides = parser.parse_known_args()
    cfg = load_config(args.config, overrides)

    from qsfl.attacks import run_attacks

    run_attacks(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
