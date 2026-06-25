import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


SOURCES = [
    ("calliope", "ruby0k/calliope_data", None, "train"),
    ("fineweb_edu", "HuggingFaceFW/fineweb-edu", "sample-10BT", "train"),
    ("simplestories", "SimpleStories/SimpleStories", None, "train"),
    ("wikitext", "Salesforce/wikitext", "wikitext-103-raw-v1", "train"),
    ("tinystories", "roneneldan/TinyStories", None, "train"),
    ("code", "codeparrot/codeparrot-clean", None, "train"),
]


def limit_arg(value: int) -> int | None:
    return None if value < 0 else value


def row_text(row) -> str:
    if isinstance(row, str):
        return row
    for key in ("text", "content", "completion", "story"):
        if isinstance(row.get(key), str):
            return row[key]
    return json.dumps(row, ensure_ascii=False)


def shuffle_iter(items, buffer: int, seed: int):
    if buffer <= 1:
        yield from items
        return
    rng = random.Random(seed)
    pending = []
    for item in items:
        pending.append(item)
        if len(pending) >= buffer:
            yield pending.pop(rng.randrange(len(pending)))
    while pending:
        yield pending.pop(rng.randrange(len(pending)))


def encode_ids(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text) + [tokenizer.eos_token_id]


def write_ids(f, ids: list[int]) -> int:
    np.asarray(ids, dtype=np.uint16).tofile(f)
    return len(ids)


def hf_iter(path: str, name: str | None, split: str, max_docs: int | None, buffer: int, seed: int):
    if path == "ruby0k/calliope_data":
        from huggingface_hub import HfFileSystem

        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        fs = HfFileSystem(token=token)
        raw = fs.open("datasets/ruby0k/calliope_data/calliope_dataset.jsonl", "r", encoding="utf-8")
        rows = (json.loads(line) for line in raw if line.strip())
        source = shuffle_iter(rows, buffer, seed)
        for i, row in enumerate(source):
            if max_docs is not None and i >= max_docs:
                break
            text = row_text(row).strip()
            if text:
                yield text
        raw.close()
        return

    from datasets import load_dataset

    ds = load_dataset(path, name, split=split, streaming=True)
    if buffer > 1:
        ds = ds.shuffle(buffer_size=buffer, seed=seed)
    for i, row in enumerate(ds):
        if max_docs is not None and i >= max_docs:
            break
        text = row_text(row).strip()
        if text:
            yield text


def build_tokenizer(args, out_dir: Path):
    """vocab_size<=0 -> reuse GPT-2 (50257). vocab_size>0 -> train a byte-level BPE of
    that size on a doc sample of the ENABLED sources (so code + general text are covered)."""
    if args.vocab_size <= 0:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("gpt2")
        tok.save_pretrained(out_dir / "tokenizer")
        return tok

    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer
    from transformers import PreTrainedTokenizerFast

    docs_enabled = {
        "calliope": limit_arg(args.max_calliope_docs),
        "fineweb_edu": limit_arg(args.max_fineweb_docs),
        "simplestories": limit_arg(args.max_simplestories_docs),
        "wikitext": limit_arg(args.max_wikitext_docs),
        "tinystories": limit_arg(args.max_tinystories_docs),
        "code": limit_arg(args.max_code_docs),
    }

    def sample_iter():
        for key, path, name, split in SOURCES:
            if docs_enabled[key] == 0:
                continue
            yield from hf_iter(path, name, split, args.max_tokenizer_docs, args.shuffle_buffer, args.seed)

    raw = Tokenizer(BPE(unk_token="<unk>"))
    raw.pre_tokenizer = ByteLevel(add_prefix_space=False)
    raw.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(vocab_size=args.vocab_size, special_tokens=["<unk>", "<|endoftext|>"])
    print(f"training {args.vocab_size}-token byte-level BPE on a sample of enabled sources...", flush=True)
    raw.train_from_iterator(sample_iter(), trainer=trainer)
    tok = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        unk_token="<unk>",
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
    )
    tok.save_pretrained(out_dir / "tokenizer")
    print(f"trained tokenizer: vocab_size={len(tok)}", flush=True)
    return tok


def write_mix(train_f, val_files, tokenizer, args) -> list[dict]:
    limits = {
        "calliope": limit_arg(args.max_calliope_docs),
        "fineweb_edu": limit_arg(args.max_fineweb_docs),
        "simplestories": limit_arg(args.max_simplestories_docs),
        "wikitext": limit_arg(args.max_wikitext_docs),
        "tinystories": limit_arg(args.max_tinystories_docs),
        "code": limit_arg(args.max_code_docs),
    }
    token_limits = {
        "calliope": limit_arg(args.max_calliope_tokens),
        "fineweb_edu": limit_arg(args.max_fineweb_tokens),
        "simplestories": limit_arg(args.max_simplestories_tokens),
        "wikitext": limit_arg(args.max_wikitext_tokens),
        "tinystories": limit_arg(args.max_tinystories_tokens),
        "code": limit_arg(args.max_code_tokens),
    }
    streams = []
    for key, path, name, split in SOURCES:
        if limits[key] == 0:
            continue
        streams.append({
            "key": key,
            "path": path,
            "name": name,
            "iter": hf_iter(path, name, split, limits[key], args.shuffle_buffer, args.seed),
            "train_docs": 0,
            "train_tokens": 0,
            "val_docs": 0,
            "val_tokens": 0,
        })

    all_streams = streams[:]
    rng = random.Random(args.seed)
    doc_num = 0
    last_log = time.time()
    while streams:
        rng.shuffle(streams)
        for stream in streams[:]:
            try:
                text = next(stream["iter"])
            except StopIteration:
                streams.remove(stream)
                continue
            ids = encode_ids(tokenizer, text)
            total_tokens = stream["train_tokens"] + stream["val_tokens"]
            if token_limits[stream["key"]] is not None and total_tokens >= token_limits[stream["key"]]:
                streams.remove(stream)
                continue
            doc_num += 1
            if doc_num % args.val_every == 0:
                stream["val_tokens"] += write_ids(val_files[stream["key"]], ids)
                stream["val_docs"] += 1
            else:
                stream["train_tokens"] += write_ids(train_f, ids)
                stream["train_docs"] += 1
            if doc_num % args.progress_every == 0 or time.time() - last_log >= args.progress_seconds:
                total_tokens = sum(s["train_tokens"] + s["val_tokens"] for s in all_streams)
                parts = [
                    f"{s['key']} {s['train_docs'] + s['val_docs']} docs/{s['train_tokens'] + s['val_tokens']:,} tok"
                    for s in all_streams
                ]
                print(f"progress {doc_num:,} docs/{total_tokens:,} tok | " + " | ".join(parts), flush=True)
                last_log = time.time()

    return [{k: v for k, v in stream.items() if k != "iter"} for stream in all_streams]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/hf_mix_gpt2")
    parser.add_argument("--max-calliope-docs", type=int, default=-1)
    parser.add_argument("--max-fineweb-docs", type=int, default=-1)
    parser.add_argument("--max-simplestories-docs", type=int, default=-1)
    parser.add_argument("--max-wikitext-docs", type=int, default=-1)
    parser.add_argument("--max-tinystories-docs", type=int, default=-1)
    parser.add_argument("--max-tinystories-tokens", type=int, default=237_000_000)
    parser.add_argument("--max-simplestories-tokens", type=int, default=118_500_000)
    parser.add_argument("--max-fineweb-tokens", type=int, default=23_700_000)
    parser.add_argument("--max-wikitext-tokens", type=int, default=71_100_000)
    parser.add_argument("--max-calliope-tokens", type=int, default=23_700_000)
    parser.add_argument("--max-code-docs", type=int, default=-1)
    parser.add_argument("--max-code-tokens", type=int, default=98_000_000)
    parser.add_argument("--vocab-size", type=int, default=0)  # 0 = GPT-2 (50257); >0 = train byte-level BPE
    parser.add_argument("--max-tokenizer-docs", type=int, default=10000)  # per-source docs for BPE training
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument("--shuffle-buffer", type=int, default=1000)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--progress-seconds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    if args.val_every < 2:
        raise SystemExit("--val-every must be >= 2")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = build_tokenizer(args, out_dir)

    val_files = {}
    try:
        for key, *_ in SOURCES:
            val_files[key] = (out_dir / f"val_{key}.bin").open("wb")
        with (out_dir / "train.bin").open("wb") as train_f:
            sources = write_mix(train_f, val_files, tokenizer, args)
    finally:
        for f in val_files.values():
            f.close()

    meta = {
        "tokenizer": "gpt2" if args.vocab_size <= 0 else f"bytelevel-bpe-{args.vocab_size}",
        "vocab_size": len(tokenizer),
        "sources": sources,
        "streaming": True,
        "fineweb_config": "sample-10BT",
        "val_every": args.val_every,
        "shuffle_buffer": args.shuffle_buffer,
        "token_budgets": {
            "calliope": args.max_calliope_tokens,
            "fineweb_edu": args.max_fineweb_tokens,
            "simplestories": args.max_simplestories_tokens,
            "wikitext": args.max_wikitext_tokens,
            "tinystories": args.max_tinystories_tokens,
            "code": args.max_code_tokens,
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
