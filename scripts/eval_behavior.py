import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.fixed_prompts import PROMPT_CATEGORIES
from eval.metrics import opening_diversity, quality_metrics, repetition_score, words
from eval.sampling import generate_text, load_for_sampling
from train.dataset import get_batch, load_split
from train.utils import amp_context, estimate_loss


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def unconditional_section(model, tokenizer, device, args, sampling) -> dict:
    texts, reps = [], []
    for i in range(args.n_unconditional):
        torch.manual_seed(args.seed + i)
        text, gen_ids = generate_text(model, tokenizer, "", device, args.max_new_tokens, **sampling)
        texts.append(text)
        reps.append(repetition_score(words(text)))
    return {
        "n": len(texts),
        "mean_repetition_score": _mean(reps),
        **opening_diversity(texts),
        "examples": [t[:200].replace("\n", " ") for t in texts[:8]],
    }


def fixed_prompt_section(model, tokenizer, device, args, sampling) -> dict:
    by_category = {}
    seed = args.seed + args.n_unconditional
    n = 0
    for category, prompts in PROMPT_CATEGORIES.items():
        metrics = []
        for prompt in prompts:
            for _ in range(args.samples_per_prompt):
                torch.manual_seed(seed + n)
                n += 1
                text, gen_ids = generate_text(model, tokenizer, prompt, device, args.max_new_tokens, **sampling)
                metrics.append(quality_metrics(prompt, text, gen_ids, tokenizer.eos_token_id))
        by_category[category] = {
            "samples": len(metrics),
            "mean_repetition_score": _mean([m["repetition_score"] for m in metrics]),
            "unfinished_rate": _mean([float(m["unfinished_sentence"]) for m in metrics]),
            "mean_sentence_length": _mean([m["average_sentence_length"] for m in metrics]),
            "mean_name_consistency": _mean([m["character_name_consistency"] for m in metrics]),
        }
    return by_category


def val_loss_section(model, ckpt, device, eval_iters: int) -> dict:
    data_dir = ckpt["train_config"]["data_dir"]
    block_size = ckpt["model_config"]["block_size"]
    batch_size = ckpt["train_config"].get("batch_size", 8)
    min_bytes = (block_size + 1) * 2
    val_files = [p for p in sorted(Path(data_dir).glob("val_*.bin")) if p.stat().st_size >= min_bytes]
    splits = {p.stem.removeprefix("val_"): load_split(data_dir, p.stem) for p in val_files}
    if not splits:
        try:
            splits = {"val": load_split(data_dir, "val")}
        except FileNotFoundError:
            return {}

    def batch(split):
        return get_batch(splits[split], block_size, batch_size, device)

    ctx = amp_context(device)
    losses = estimate_loss(model, batch, eval_iters, ctx, tuple(splits))
    weighted = sum(losses[k] * len(splits[k]) for k in splits) / sum(len(d) for d in splits.values())
    return {**{k: round(v, 4) for k, v in losses.items()}, "weighted_val_loss": round(weighted, 4)}


def render_markdown(report: dict) -> str:
    u = report["unconditional"]
    lines = [
        f"# Behavior report — {report['run_name']}",
        "",
        f"checkpoint: `{report['checkpoint']}` · iter {report.get('iter_num', '?')} · params {report.get('params', '?'):,}",
        "",
        "## Per-dataset val loss",
        "",
        "| split | loss |",
        "|---|---|",
    ]
    for k, v in report["val_loss"].items():
        lines.append(f"| {k} | {v} |")
    lines += [
        "",
        "## Opening diversity (unconditional)",
        "",
        f"- samples: **{u['n']}**",
        f"- once-upon-a-time rate: **{u['once_upon_rate']}**  _(lower is better)_",
        f"- unique-opening ratio: **{u['unique_opening_ratio']}**  _(higher is better)_",
        f"- opening entropy (bits): **{u['opening_entropy']}**  _(higher is better)_",
        f"- mean repetition score: **{u['mean_repetition_score']}**  _(lower is better)_",
        "",
        "Top openings:",
        "",
    ]
    for prefix, count in u["top_openings"]:
        lines.append(f"- `{prefix}` × {count}")
    lines += ["", "## Fixed-prompt behavior", "", "| category | rep | unfinished | sent.len | name-consist |", "|---|---|---|---|---|"]
    for cat, m in report["fixed_prompts"].items():
        lines.append(
            f"| {cat} | {m['mean_repetition_score']} | {m['unfinished_rate']} | {m['mean_sentence_length']} | {m['mean_name_consistency']} |"
        )
    lines += ["", "## Example unconditional openings", ""]
    for ex in u["examples"]:
        lines.append(f"- {ex}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/calliope_100m_hf_mix/best.pt")
    parser.add_argument("--adapter", default="", help="LoRA adapter to layer on the base before evaluating")
    parser.add_argument("--n-unconditional", type=int, default=64)
    parser.add_argument("--samples-per-prompt", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--eval-iters", type=int, default=200)
    parser.add_argument("--out", default="")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = "" if args.adapter and args.checkpoint == "checkpoints/calliope_100m_hf_mix/best.pt" else args.checkpoint
    model, tokenizer, ckpt = load_for_sampling(checkpoint, args.adapter, device)
    sampling = dict(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )

    report = {
        "run_name": ckpt.get("run_name", "unknown"),
        "checkpoint": args.checkpoint,
        "iter_num": ckpt.get("iter_num"),
        "params": ckpt.get("params"),
        "sampling": sampling,
        "val_loss": val_loss_section(model, ckpt, device, args.eval_iters),
        "unconditional": unconditional_section(model, tokenizer, device, args, sampling),
        "fixed_prompts": fixed_prompt_section(model, tokenizer, device, args, sampling),
    }

    out_dir = Path(args.out) if args.out else Path("experiments") / report["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "behavior_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "behavior_report.md").write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote behavior_report.json + .md to {out_dir}")
    print(json.dumps({"val_loss": report["val_loss"], "unconditional": {k: report["unconditional"][k] for k in ("once_upon_rate", "unique_opening_ratio", "opening_entropy", "mean_repetition_score")}}, indent=2))


if __name__ == "__main__":
    main()
