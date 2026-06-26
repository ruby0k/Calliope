import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def by_key(rows: list[dict]) -> dict[tuple[str, int], dict]:
    return {(row["prompt"], row["sample_index"]): row for row in rows}


def final_metric(path: Path) -> dict:
    return read_jsonl(path)[-1]


def best_metric(path: Path) -> dict:
    return min(read_jsonl(path), key=lambda row: row["val_loss"])


def avg(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row]
    return sum(values) / len(values) if values else 0.0


def count_true(rows: list[dict], key: str) -> int:
    return sum(bool(row.get(key)) for row in rows)


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
    tenm_best = best_metric(tenm_dir / "metrics.jsonl")
    thirtym_best = best_metric(thirtym_dir / "metrics.jsonl")
    tenm_rows = read_jsonl(tenm_dir / "fixed_prompt_samples.jsonl")
    thirtym_rows = read_jsonl(thirtym_dir / "fixed_prompt_samples.jsonl")
    tenm = by_key(tenm_rows)
    thirtym = by_key(thirtym_rows)
    keys = sorted(set(tenm) & set(thirtym))

    winner = args.left_label if tenm_best["val_loss"] <= thirtym_best["val_loss"] else args.right_label
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
        "## Training Metrics",
        "",
        "| Model | Best Iter | Best Val Loss | Final Iter | Final Val Loss | Final Loss EMA | VRAM GB |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| {args.left_label} | {tenm_best.get('iter')} | {tenm_best.get('val_loss'):.4f} | {tenm_metrics.get('iter')} | {tenm_metrics.get('val_loss'):.4f} | {tenm_metrics.get('loss_ema', 0):.4f} | {tenm_metrics.get('vram_gb', 0):.3f} |",
        f"| {args.right_label} | {thirtym_best.get('iter')} | {thirtym_best.get('val_loss'):.4f} | {thirtym_metrics.get('iter')} | {thirtym_metrics.get('val_loss'):.4f} | {thirtym_metrics.get('loss_ema', 0):.4f} | {thirtym_metrics.get('vram_gb', 0):.3f} |",
        "",
        "## Sample Quality Metrics",
        "",
        "| Model | Avg Repetition | EOS Count | Unfinished Count | Avg Sentence Len | Unique Token Ratio | Name Consistency |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| {args.left_label} | {avg(tenm_rows, 'repetition_score'):.4f} | {count_true(tenm_rows, 'eos_inside_output')} | {count_true(tenm_rows, 'unfinished_sentence')} | {avg(tenm_rows, 'average_sentence_length'):.2f} | {avg(tenm_rows, 'unique_token_ratio'):.4f} | {avg(tenm_rows, 'character_name_consistency'):.4f} |",
        f"| {args.right_label} | {avg(thirtym_rows, 'repetition_score'):.4f} | {count_true(thirtym_rows, 'eos_inside_output')} | {count_true(thirtym_rows, 'unfinished_sentence')} | {avg(thirtym_rows, 'average_sentence_length'):.2f} | {avg(thirtym_rows, 'unique_token_ratio'):.4f} | {avg(thirtym_rows, 'character_name_consistency'):.4f} |",
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
            clip(tenm[(prompt, sample_index)]["text"]),
            "",
            f"**{args.right_label}**",
            "",
            clip(thirtym[(prompt, sample_index)]["text"]),
            "",
        ]

    out = Path(args.out)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
