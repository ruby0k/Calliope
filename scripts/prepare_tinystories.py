import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def limit_arg(value: int) -> int | None:
    return None if value < 0 else value


def write_split(dataset_name: str, split: str, out_path: Path, tokenizer, max_docs: int | None) -> dict:
    from datasets import load_dataset

    docs = 0
    tokens = 0
    ds = load_dataset(dataset_name, split=split, streaming=True)
    with open(out_path, "wb") as f:
        for row in ds:
            if max_docs is not None and docs >= max_docs:
                break
            ids = tokenizer.encode(row["text"]) + [tokenizer.eos_token_id]
            np.asarray(ids, dtype=np.uint16).tofile(f)
            docs += 1
            tokens += len(ids)
            if docs % 1000 == 0:
                print(f"{split}: {docs} docs, {tokens} tokens")
    return {"split": split, "docs": docs, "tokens": tokens, "path": str(out_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="roneneldan/TinyStories")
    parser.add_argument("--out-dir", default="data/tinystories_gpt2")
    parser.add_argument("--max-train-docs", type=int, default=20000)
    parser.add_argument("--max-val-docs", type=int, default=2000)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.save_pretrained(out_dir / "tokenizer")

    meta = {
        "tokenizer": "gpt2",
        "dataset": args.dataset,
        "train": write_split(args.dataset, "train", out_dir / "train.bin", tokenizer, limit_arg(args.max_train_docs)),
        "val": write_split(args.dataset, "validation", out_dir / "val.bin", tokenizer, limit_arg(args.max_val_docs)),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
