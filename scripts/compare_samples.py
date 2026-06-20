import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def by_key(rows: list[dict]) -> dict[tuple[str, int], str]:
    return {(row["prompt"], row["sample_index"]): row["text"] for row in rows}


def final_metric(path: Path) -> dict:
    return read_jsonl(path)[-1]


def clip(text: str, limit: int = 900) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tenm-dir", default="experiments/calliope_10m_20260620_161343")
    parser.add_argument("--thirtym-dir", default="experiments/Calliope-30M-run001")
    parser.add_argument("--left-label", default="10M")
    parser.add_argument("--right-label", default="30M")
    parser.add_argument("--title", default="Calliope 10M vs 30M Baseline Report")
    parser.add_argument("--out", default="experiments/baseline_10m_vs_30m.md")
    args = parser.parse_args()

    tenm_dir = Path(args.tenm_dir)
    thirtym_dir = Path(args.thirtym_dir)
    tenm_metrics = final_metric(tenm_dir / "metrics.jsonl")
    thirtym_metrics = final_metric(thirtym_dir / "metrics.jsonl")
    tenm = by_key(read_jsonl(tenm_dir / "fixed_prompt_samples.jsonl"))
    thirtym = by_key(read_jsonl(thirtym_dir / "fixed_prompt_samples.jsonl"))
    keys = sorted(set(tenm) & set(thirtym))

    winner = args.left_label if tenm_metrics["val_loss"] <= thirtym_metrics["val_loss"] else args.right_label
    lines = [
        f"# {args.title}",
        "",
        "## Summary",
        "",
        f"- {args.left_label} source: `{tenm_dir}`",
        f"- {args.right_label} source: `{thirtym_dir}`",
        f"- prompts compared: {len({prompt for prompt, _ in keys})}",
        f"- samples compared: {len(keys)}",
        "",
        "## Final Metrics",
        "",
        "| Model | Iter | Train Loss | Val Loss | Loss EMA | VRAM GB |",
        "|---|---:|---:|---:|---:|---:|",
        f"| {args.left_label} | {tenm_metrics.get('iter')} | {tenm_metrics.get('train_loss'):.4f} | {tenm_metrics.get('val_loss'):.4f} | {tenm_metrics.get('loss_ema', 0):.4f} | {tenm_metrics.get('vram_gb', 0):.3f} |",
        f"| {args.right_label} | {thirtym_metrics.get('iter')} | {thirtym_metrics.get('train_loss'):.4f} | {thirtym_metrics.get('val_loss'):.4f} | {thirtym_metrics.get('loss_ema', 0):.4f} | {thirtym_metrics.get('vram_gb', 0):.3f} |",
        "",
        "## Readout",
        "",
        f"- {winner} has the lower validation loss on this run.",
        "- Use the side-by-side samples below as a quick qualitative check.",
        "- Keep changes only if they improve validation loss, samples, speed, memory, or capability per parameter.",
        "",
        "## Side-by-Side Samples",
        "",
    ]

    for prompt, sample_index in keys:
        lines += [
            f"### `{prompt}` sample {sample_index}",
            "",
            f"**{args.left_label}**",
            "",
            clip(tenm[(prompt, sample_index)]),
            "",
            f"**{args.right_label}**",
            "",
            clip(thirtym[(prompt, sample_index)]),
            "",
        ]

    out = Path(args.out)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
