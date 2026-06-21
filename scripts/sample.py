import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import ModelConfig, Transformer
from train.checkpoint import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/calliope_10m/best.pt")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, ckpt = load_checkpoint(args.checkpoint, device=device)
    model = Transformer(ModelConfig(**ckpt["model_config"])).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    from transformers import AutoTokenizer

    tokenizer_dir = Path(ckpt["train_config"]["data_dir"]) / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir if tokenizer_dir.exists() else "gpt2")
    ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    out = model.generate(
        ids,
        args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )[0].tolist()
    print(tokenizer.decode(out))


if __name__ == "__main__":
    main()
