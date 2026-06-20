"""``qsfl`` console entrypoint (thin wrapper over the pipeline runner)."""

from __future__ import annotations

import argparse
import sys

from qsfl.utils.config import load_config


def _parse(argv: list[str] | None = None) -> tuple[str, list[str]]:
    parser = argparse.ArgumentParser(
        prog="qsfl", description="Quantum-Safe Federated Learning pipeline"
    )
    parser.add_argument("--config", required=True, help="path to a YAML config")
    args, overrides = parser.parse_known_args(argv)
    return args.config, overrides


def main(argv: list[str] | None = None) -> int:
    config, overrides = _parse(argv or sys.argv[1:])
    cfg = load_config(config, overrides)
    from qsfl.pipeline import run_pipeline

    run_pipeline(cfg)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
