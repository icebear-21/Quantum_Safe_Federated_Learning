"""Attack-evaluation harness: build the protected model once, run every enabled
attack against a fresh copy, emit JSON + plots.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from qsfl.attacks.base import AttackContext
from qsfl.attacks.registry import REGISTRY
from qsfl.federated.trainer import state_to_numpy
from qsfl.utils.logging import get_logger
from qsfl.utils.plotting import plot_attack_summary, plot_pruning

logger = get_logger("attacks")


def run_attacks(cfg) -> dict:
    from qsfl.pipeline.runner import train_and_protect

    art = train_and_protect(cfg)
    ctx = AttackContext(
        cfg=cfg,
        data=art.data,
        bundle=art.bundle,
        protected_state=state_to_numpy(art.model),
        baseline_accuracy=art.accuracy_watermarked,
        device=str(cfg.device),
    )

    results: dict = {
        "run_name": str(cfg.run_name),
        "baseline_accuracy": art.accuracy_watermarked,
        "baseline_accuracy_clean": art.accuracy_clean,
        "attacks": {},
    }
    for name in list(cfg.attacks.enabled):
        if name not in REGISTRY:
            logger.warning("unknown attack '%s' — skipping", name)
            continue
        logger.info("running attack: %s", name)
        t = time.time()
        try:
            res = REGISTRY[name](ctx)
            res["_elapsed_sec"] = round(time.time() - t, 2)
        except Exception as exc:  # one failing attack must not kill the suite
            logger.exception("attack '%s' failed", name)
            res = {"error": str(exc)}
        results["attacks"][name] = res

    out_dir = Path(cfg.output_dir) / str(cfg.run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "attacks.json").write_text(json.dumps(results, indent=2, default=str))

    try:
        pr = results["attacks"].get("pruning", {})
        if "by_amount" in pr:
            plot_pruning(pr["by_amount"], out_dir / "pruning_robustness.png")
        plot_attack_summary(results, out_dir / "attack_summary.png")
    except Exception:
        logger.exception("plotting failed (non-fatal)")

    logger.info("attack evaluation complete -> %s", out_dir / "attacks.json")
    return results
