import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def collect(run_dir: Path) -> dict:
    stats: dict[str, float] = {}
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        evals = [r for r in read_jsonl(metrics_path) if "val_loss" in r]
        if evals:
            best = min(evals, key=lambda r: r["val_loss"])
            stats["iter (best)"] = best.get("iter")
            stats["val_loss"] = round(best["val_loss"], 4)
            for k, v in best.items():
                if k.endswith("_val_loss"):
                    stats[k] = round(v, 4)

    report_path = run_dir / "behavior_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        # behavior_report val_loss is measured on the base's splits for BOTH base and
        # adapter runs, so it's the comparable source — let it override metrics.jsonl
        # (whose val_loss for a LoRA run is the specialist's own training split).
        vl = report.get("val_loss", {})
        for k, v in vl.items():
            stats[k if k == "weighted_val_loss" else f"{k}_val_loss"] = round(v, 4)
        if "weighted_val_loss" in vl:
            stats["val_loss"] = round(vl["weighted_val_loss"], 4)
        u = report.get("unconditional", {})
        for key in ("once_upon_rate", "unique_opening_ratio", "opening_entropy", "mean_repetition_score"):
            if key in u:
                stats[key] = u[key]
    return stats


def render(labels: list[str], per_run: list[dict], title: str) -> str:
    # Union of metric rows, preserving first-seen order.
    rows: list[str] = []
    for stats in per_run:
        for k in stats:
            if k not in rows:
                rows.append(k)

    header = "| metric | " + " | ".join(labels) + " |"
    sep = "|" + "---|" * (len(labels) + 1)
    lines = [f"# {title}", "", header, sep]
    for row in rows:
        cells = [str(stats.get(row, "-")) for stats in per_run]
        lines.append(f"| {row} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", default=[], help="run directory (repeatable)")
    parser.add_argument("--labels", default="", help="comma-separated labels (default: dir names)")
    parser.add_argument("--title", default="Calliope run comparison")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    if len(args.run) < 2:
        raise SystemExit("pass at least two --run directories")

    run_dirs = [Path(r) for r in args.run]
    labels = args.labels.split(",") if args.labels else [d.name for d in run_dirs]
    if len(labels) != len(run_dirs):
        raise SystemExit("number of --labels must match number of --run")

    per_run = [collect(d) for d in run_dirs]
    md = render(labels, per_run, args.title)
    print(md)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
