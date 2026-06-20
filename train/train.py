import argparse
import importlib
import json
import math
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import ModelConfig, Transformer
from train.checkpoint import load_checkpoint, save_checkpoint
from train.dataset import get_batch, load_split
from train.utils import amp_context, append_jsonl, estimate_loss, get_git_commit, pick_device, seed_everything


def get_lr(iter_num: int, cfg) -> float:
    if iter_num < cfg.warmup_iters:
        return cfg.learning_rate * (iter_num + 1) / cfg.warmup_iters
    if iter_num > cfg.max_iters:
        return cfg.min_lr
    ratio = (iter_num - cfg.warmup_iters) / (cfg.max_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


def load_config(name: str):
    module = importlib.import_module(name)
    return module.model_config, module.train_config


def maybe_sample(model, tokenizer, device: str, max_tokens: int) -> str:
    prompt = "Once upon a time"
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    out = model.generate(ids, max_new_tokens=max_tokens)[0].tolist()
    return tokenizer.decode(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs.calliope_10m")
    parser.add_argument("--resume", default="")
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--eval-iters", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    args = parser.parse_args()

    model_cfg, train_cfg = load_config(args.config)
    if args.max_iters is not None:
        train_cfg.max_iters = args.max_iters
    if args.eval_iters is not None:
        train_cfg.eval_iters = args.eval_iters
    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size
    if args.grad_accum_steps is not None:
        train_cfg.grad_accum_steps = args.grad_accum_steps

    seed_everything(train_cfg.seed)
    device = pick_device()
    ctx = amp_context(device)

    train_data = load_split(train_cfg.data_dir, "train")
    val_data = load_split(train_cfg.data_dir, "val")

    run_name = train_cfg.run_name or datetime.now().strftime("calliope_10m_%Y%m%d_%H%M%S")
    out_dir = Path(train_cfg.out_dir)
    exp_dir = Path(train_cfg.experiment_dir) / run_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    model = Transformer(model_cfg).to(device)
    optimizer = model.configure_optimizer(train_cfg)
    if train_cfg.compile:
        model = torch.compile(model)

    iter_num = 0
    best_val_loss = float("inf")
    if args.resume:
        iter_num, best_val_loss, _ = load_checkpoint(args.resume, model, optimizer, device)

    meta = {
        "run_name": run_name,
        "date": datetime.now().isoformat(timespec="seconds"),
        "git_commit": get_git_commit(),
        "model_config": asdict(model_cfg),
        "train_config": asdict(train_cfg),
        "params": model.num_parameters(),
        "dataset": "TinyStories",
        "tokenizer": "gpt2",
    }
    (exp_dir / "config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    tokenizer = None
    tokenizer_dir = Path(train_cfg.data_dir) / "tokenizer"
    if tokenizer_dir.exists():
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    def batch(split: str):
        data = train_data if split == "train" else val_data
        return get_batch(data, model_cfg.block_size, train_cfg.batch_size, device)

    t0 = time.time()
    while iter_num < train_cfg.max_iters:
        lr = get_lr(iter_num, train_cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(train_cfg.grad_accum_steps):
            x, y = batch("train")
            with ctx:
                _, loss = model(x, y)
                loss = loss / train_cfg.grad_accum_steps
            loss.backward()
            loss_accum += loss.item()

        if train_cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        iter_num += 1

        if iter_num % 10 == 0:
            dt = time.time() - t0
            tokens = train_cfg.batch_size * train_cfg.grad_accum_steps * model_cfg.block_size * 10
            print(f"iter {iter_num}: loss {loss_accum:.4f}, lr {lr:.2e}, {tokens / dt:.0f} tok/s")
            t0 = time.time()

        if iter_num % train_cfg.eval_interval == 0 or iter_num == train_cfg.max_iters:
            losses = estimate_loss(model, batch, train_cfg.eval_iters, ctx)
            vram_gb = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
            row = {
                "iter": iter_num,
                "train_loss": losses["train"],
                "val_loss": losses["val"],
                "lr": lr,
                "vram_gb": round(vram_gb, 3),
            }
            if tokenizer and (iter_num % train_cfg.sample_interval == 0 or iter_num == train_cfg.max_iters):
                row["sample"] = maybe_sample(model, tokenizer, device, train_cfg.sample_tokens)
            append_jsonl(exp_dir / "metrics.jsonl", row)
            print(json.dumps(row, ensure_ascii=True))

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_checkpoint(out_dir / "best.pt", model, optimizer, iter_num, best_val_loss, meta)

        if iter_num % train_cfg.ckpt_interval == 0 or iter_num == train_cfg.max_iters:
            save_checkpoint(out_dir / "last.pt", model, optimizer, iter_num, best_val_loss, meta)


if __name__ == "__main__":
    main()
