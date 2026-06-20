"""Result plots (headless Agg backend)."""

from __future__ import annotations

from pathlib import Path


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_pruning(by_amount: dict, path: str | Path) -> None:
    plt = _plt()
    items = sorted(by_amount.values(), key=lambda d: d["amount"])
    xs = [d["amount"] for d in items]
    fig, ax = plt.subplots(figsize=(6, 4))
    if items and "nc_secondary" in items[0]:
        ax.plot(xs, [d["nc_secondary"] for d in items], "-o", label="NC(S,S')")
        ax.plot(xs, [d["ber_secondary"] for d in items], "-s", label="BER(S,S')")
    ax.plot(xs, [d["accuracy_after"] for d in items], "-^", label="accuracy")
    ax.set_xlabel("prune amount")
    ax.set_ylabel("value")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Watermark robustness under pruning")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_attack_summary(results: dict, path: str | Path) -> None:
    plt = _plt()
    names, accs = [], []
    for name, r in results.get("attacks", {}).items():
        a = r.get("accuracy_after", r.get("student_accuracy"))
        if a is not None:
            names.append(name)
            accs.append(a)
    if not names:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, accs, color="#4C72B0")
    ax.axhline(results.get("baseline_accuracy", 0.0), ls="--", color="k", label="baseline")
    ax.set_ylabel("accuracy after attack")
    ax.set_title("Model accuracy under attacks")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
