import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def limit_arg(value: int) -> int | None:
    return None if value < 0 else value


def stories(dataset_name: str, split: str, max_docs: int | None):
    from datasets import load_dataset

    for i, row in enumerate(load_dataset(dataset_name, split=split, streaming=True)):
        if max_docs is not None and i >= max_docs:
            break
        yield row["text"]


def write_split(dataset_name: str, split: str, out_path: Path, tokenizer, max_docs: int | None) -> dict:
    docs = 0
    tokens = 0
    eos_id = tokenizer.eos_token_id
    with open(out_path, "wb") as f:
        for text in stories(dataset_name, split, max_docs):
            ids = tokenizer.encode(text) + [eos_id]
            np.asarray(ids, dtype=np.uint16).tofile(f)
            docs += 1
            tokens += len(ids)
            if docs % 1000 == 0:
                print(f"{split}: {docs} docs, {tokens} tokens")
    return {"split": split, "docs": docs, "tokens": tokens, "path": str(out_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="roneneldan/TinyStories")
    parser.add_argument("--out-dir", default="data/tinystories_bpe8192")
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--max-tokenizer-docs", type=int, default=100000)
    parser.add_argument("--max-train-docs", type=int, default=20000)
    parser.add_argument("--max-val-docs", type=int, default=2000)
    args = parser.parse_args()

    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer
    from transformers import PreTrainedTokenizerFast

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = Tokenizer(BPE(unk_token="<unk>"))
    raw.pre_tokenizer = ByteLevel(add_prefix_space=False)
    raw.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(vocab_size=args.vocab_size, special_tokens=["<unk>", "<|endoftext|>"])
    raw.train_from_iterator(stories(args.dataset, "train", limit_arg(args.max_tokenizer_docs)), trainer=trainer)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        unk_token="<unk>",
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
    )
    tokenizer.save_pretrained(out_dir / "tokenizer")

    meta = {
        "tokenizer": f"bytelevel-bpe-{args.vocab_size}",
        "dataset": args.dataset,
        "vocab_size": len(tokenizer),
        "tokenizer_train_docs": args.max_tokenizer_docs,
        "train": write_split(args.dataset, "train", out_dir / "train.bin", tokenizer, limit_arg(args.max_train_docs)),
        "val": write_split(args.dataset, "validation", out_dir / "val.bin", tokenizer, limit_arg(args.max_val_docs)),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
