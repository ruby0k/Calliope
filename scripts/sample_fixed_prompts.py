import argparse
import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.fixed_prompts import PROMPT_CATEGORIES
from model import ModelConfig, Transformer
from train.checkpoint import load_checkpoint


KNOWN_NAMES = {"Anna", "Ben", "Dad", "Ella", "Lily", "Max", "Mia", "Nora", "Sam", "Timmy", "Tom"}


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text)


def sentence_lengths(text: str) -> list[int]:
    return [len(words(part)) for part in re.split(r"[.!?]+", text) if words(part)]


def repetition_score(tokens: list[str], n: int = 4) -> float:
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return round(1.0 - len(set(grams)) / len(grams), 4)


def character_name_consistency(prompt: str, text: str) -> float:
    prompt_names = set(words(prompt)) & KNOWN_NAMES
    text_names = [name for name in words(text) if name in KNOWN_NAMES]
    if not text_names:
        return 1.0
    if prompt_names:
        return round(sum(name in prompt_names for name in text_names) / len(text_names), 4)
    counts = {name: text_names.count(name) for name in set(text_names)}
    return round(max(counts.values()) / len(text_names), 4)


def quality_metrics(prompt: str, text: str, generated_ids: list[int], eos_id: int) -> dict:
    toks = words(text)
    lengths = sentence_lengths(text)
    return {
        "repetition_score": repetition_score(toks),
        "eos_inside_output": eos_id in generated_ids,
        "unfinished_sentence": not text.rstrip().endswith((".", "!", "?", "\"")),
        "average_sentence_length": round(sum(lengths) / len(lengths), 2) if lengths else 0.0,
        "unique_token_ratio": round(len(set(generated_ids)) / len(generated_ids), 4) if generated_ids else 0.0,
        "character_name_consistency": character_name_consistency(prompt, text),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/calliope_10m/best.pt")
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--out", default="")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, ckpt = load_checkpoint(args.checkpoint, device=device)
    model = Transformer(ModelConfig(**ckpt["model_config"])).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    from transformers import AutoTokenizer

    tokenizer_dir = Path(ckpt["train_config"]["data_dir"]) / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir if tokenizer_dir.exists() else "gpt2")
    out_path = Path(args.out) if args.out else Path("experiments") / ckpt["run_name"] / "fixed_prompt_samples.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with out_path.open("w", encoding="utf-8") as f:
        for category, prompts in PROMPT_CATEGORIES.items():
            for prompt in prompts:
                ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                for i in range(args.samples_per_prompt):
                    torch.manual_seed(args.seed + len(rows))
                    out = model.generate(ids, args.max_new_tokens)[0].tolist()
                    generated_ids = out[ids.shape[1] :]
                    text = tokenizer.decode(out)
                    row = {
                        "category": category,
                        "prompt": prompt,
                        "sample_index": i,
                        "text": text,
                        **quality_metrics(prompt, text, generated_ids, tokenizer.eos_token_id),
                    }
                    rows.append(row)
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} samples to {out_path}")


if __name__ == "__main__":
    main()
