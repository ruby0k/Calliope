"""Search LoRA composition weights over the frozen base.

Lays the base + K experts into one model and sweeps a grid of per-expert mixing
weights, scoring each combination on behavioral fitness (opening diversity /
once-upon rate / repetition) measured from unconditional samples. With 2 experts
the grid is the exhaustive version of the evolutionary search — the same fitness
hook generalizes to CMA-ES / GA for more experts.

    uv run python scripts/compose_search.py \
        --base checkpoints/calliope_100m_base_mix_v2/best.pt \
        --adapters checkpoints/lora_wiki/best.pt,checkpoints/lora_fineweb/best.pt \
        --labels wiki,fineweb
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.metrics import opening_diversity, repetition_score, words
from eval.sampling import generate_text, load_composed
from model.lora import set_composition_weights


def evaluate(model, tokenizer, weights, combo, device, n, max_new_tokens, seed, sampling) -> dict:
    set_composition_weights(weights, combo)
    texts = []
    for i in range(n):
        torch.manual_seed(seed + i)
        text, _ = generate_text(model, tokenizer, "", device, max_new_tokens, **sampling)
        texts.append(text)
    od = opening_diversity(texts)
    return {
        "once_upon_rate": od["once_upon_rate"],
        "opening_entropy": od["opening_entropy"],
        "unique_opening_ratio": od["unique_opening_ratio"],
        "mean_repetition": round(sum(repetition_score(words(t)) for t in texts) / len(texts), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="checkpoints/calliope_100m_base_mix_v2/best.pt")
    parser.add_argument("--adapters", required=True, help="comma-separated adapter checkpoint paths")
    parser.add_argument("--labels", default="", help="comma-separated expert labels")
    parser.add_argument("--steps", type=int, default=4, help="grid points per expert weight")
    parser.add_argument("--wmax", type=float, default=1.5, help="max weight per expert")
    parser.add_argument("--n-samples", type=int, default=24)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out", default="experiments/compose_search.json")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    args = parser.parse_args()

    adapter_paths = args.adapters.split(",")
    labels = args.labels.split(",") if args.labels else [Path(p).parent.name for p in adapter_paths]
    k = len(adapter_paths)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, _, weights = load_composed(args.base, adapter_paths, device)
    sampling = dict(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )

    grid = [round(float(v), 3) for v in np.linspace(0.0, args.wmax, args.steps)]
    combos = list(itertools.product(grid, repeat=k))
    print(f"experts: {labels} | grid {grid} | {len(combos)} combinations x {args.n_samples} samples")

    results = []
    for combo in combos:
        metrics = evaluate(model, tokenizer, weights, combo, device, args.n_samples, args.max_new_tokens, args.seed, sampling)
        results.append({"weights": list(combo), **metrics})
        wlabel = ", ".join(f"{lbl}={w}" for lbl, w in zip(labels, combo))
        print(f"  [{wlabel}] once_upon={metrics['once_upon_rate']} entropy={metrics['opening_entropy']} rep={metrics['mean_repetition']}")

    base_row = next(r for r in results if all(w == 0 for w in r["weights"]))
    eligible = [r for r in results if r["once_upon_rate"] <= base_row["once_upon_rate"]]
    best = max(eligible, key=lambda r: r["opening_entropy"])

    report = {"base": args.base, "experts": labels, "grid": grid, "results": results, "base_row": base_row, "best": best}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== base (all-zero weights) ===")
    print(f"  once_upon={base_row['once_upon_rate']} entropy={base_row['opening_entropy']} rep={base_row['mean_repetition']}")
    print("=== best composition (max opening entropy, once_upon <= base) ===")
    print(f"  weights {dict(zip(labels, best['weights']))}")
    print(f"  once_upon={best['once_upon_rate']} entropy={best['opening_entropy']} unique={best['unique_opening_ratio']} rep={best['mean_repetition']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
