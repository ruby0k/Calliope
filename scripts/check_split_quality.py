import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def docs_from_bin(path: Path, eos_id: int, max_docs: int) -> list[tuple[int, ...]]:
    data = np.memmap(path, dtype=np.uint16, mode="r")
    docs = []
    start = 0
    for i, token in enumerate(data):
        if int(token) == eos_id:
            docs.append(tuple(int(x) for x in data[start : i + 1]))
            start = i + 1
            if len(docs) >= max_docs:
                break
    return docs


def digest(doc: tuple[int, ...]) -> str:
    return hashlib.sha1(np.asarray(doc, dtype=np.uint16).tobytes()).hexdigest()


def summarize(name: str, docs: list[tuple[int, ...]]) -> dict:
    lengths = [len(d) for d in docs]
    hashes = [digest(d) for d in docs]
    return {
        "split": name,
        "docs_checked": len(docs),
        "tokens_checked": sum(lengths),
        "min_doc_tokens": min(lengths) if lengths else 0,
        "avg_doc_tokens": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        "max_doc_tokens": max(lengths) if lengths else 0,
        "duplicate_docs": len(hashes) - len(set(hashes)),
        "very_short_docs": sum(n < 20 for n in lengths),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/tinystories_gpt2")
    parser.add_argument("--max-docs", type=int, default=5000)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    data_dir = Path(args.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(data_dir / "tokenizer" if (data_dir / "tokenizer").exists() else "gpt2")
    train_docs = docs_from_bin(data_dir / "train.bin", tokenizer.eos_token_id, args.max_docs)
    val_docs = docs_from_bin(data_dir / "val.bin", tokenizer.eos_token_id, args.max_docs)
    train_hashes = {digest(d) for d in train_docs}
    val_hashes = {digest(d) for d in val_docs}
    report = {
        "data_dir": str(data_dir),
        "train": summarize("train", train_docs),
        "val": summarize("val", val_docs),
        "train_val_exact_overlap": len(train_hashes & val_hashes),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
