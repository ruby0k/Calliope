import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.sampling import generate_text, load_for_sampling


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/calliope_10m/best.pt")
    parser.add_argument("--adapter", default="", help="LoRA adapter checkpoint to layer on the base")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # When --adapter is set, the base defaults to the adapter's stored base_checkpoint.
    checkpoint = "" if args.adapter and args.checkpoint == "checkpoints/calliope_10m/best.pt" else args.checkpoint
    model, tokenizer, _ = load_for_sampling(checkpoint, args.adapter, device)
    text, _ = generate_text(
        model,
        tokenizer,
        args.prompt,
        device,
        args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
    print(text)


if __name__ == "__main__":
    main()
