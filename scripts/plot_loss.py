"""Live training-loss curve (seaborn). Run alongside training in a second terminal.

Rendering uses CPU + the integrated GPU (the desktop) — it does NOT touch CUDA,
so it never competes with the RTX 5050 doing the training.

    uv run python scripts/plot_loss.py
    uv run python scripts/plot_loss.py --run-dir experiments/<run> --refresh 3
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not os.path.exists(path):
        return rows
    for line in open(path, encoding="utf-8", errors="ignore"):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


PER_DATASET = {
    "fineweb_edu": "#2ca02c",
    "code": "#9467bd",
    "wikitext": "#ff7f0e",
    "simplestories": "#8c564b",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="experiments/Calliope-100M-v3-general-run001")
    ap.add_argument("--refresh", type=float, default=3.0)
    args = ap.parse_args()
    rd = Path(args.run_dir)

    sns.set_theme(style="darkgrid", context="talk")
    plt.ion()
    fig, ax = plt.subplots(figsize=(11, 6))
    try:
        fig.canvas.manager.set_window_title("Calliope — live loss")
    except Exception:
        pass

    while plt.fignum_exists(fig.number):
        loss_rows = read_jsonl(rd / "loss_log.jsonl")
        evals = [r for r in read_jsonl(rd / "metrics.jsonl") if "val_loss" in r]
        ax.clear()

        if loss_rows:
            it = [r["iter"] for r in loss_rows]
            ax.plot(it, [r["ema"] for r in loss_rows], color="#1f77b4", lw=1.6, label="train (ema)")
        if evals:
            it = [r["iter"] for r in evals]
            ax.plot(it, [r["val_loss"] for r in evals], "o-", color="#d62728", ms=5, lw=1.6, label="val (weighted)")
            for key, color in PER_DATASET.items():
                ys = [r.get(f"{key}_val_loss", math.nan) for r in evals]
                if not all(isinstance(y, float) and math.isnan(y) for y in ys):
                    ax.plot(it, ys, "--", color=color, lw=1.0, alpha=0.7, label=key)
            last = evals[-1]
            ax.set_title(f"Calliope live loss — iter {last['iter']}  ·  val {last['val_loss']:.3f}")
        else:
            ax.set_title("Calliope live loss — waiting for data…")

        ax.set_xlabel("iter")
        ax.set_ylabel("loss (nats)")
        if loss_rows or evals:
            ax.legend(loc="upper right", fontsize=10, ncol=2)
        try:
            plt.pause(max(0.5, args.refresh))
        except Exception:
            break

    print("plot window closed", file=sys.stderr)


if __name__ == "__main__":
    main()
