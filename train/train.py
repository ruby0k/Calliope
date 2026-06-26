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

from eval.sampling import generate_text
from model import ModelConfig, Transformer
from train.checkpoint import load_checkpoint, save_checkpoint
from train.dataset import get_batch, get_sequential_batch, load_split
from train.utils import amp_context, append_jsonl, estimate_loss, get_git_commit, pick_device, seed_everything


def get_lr(iter_num: int, cfg) -> float:
    if iter_num < cfg.warmup_iters:
        return cfg.learning_rate * (iter_num + 1) / cfg.warmup_iters
    if iter_num > cfg.max_iters:
        return cfg.min_lr
    ratio = (iter_num - cfg.warmup_iters) / (cfg.max_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


def lr_fraction(iter_num: int, cfg) -> float:
    """Schedule as a fraction of peak LR (in [min_ratio, 1]). Each optimizer group is
    scaled by this, so Muon and AdamW keep their own peak LRs. Supports cosine and WSD."""
    if iter_num < cfg.warmup_iters:
        return (iter_num + 1) / cfg.warmup_iters
    min_ratio = cfg.min_lr / cfg.learning_rate
    if getattr(cfg, "lr_schedule", "cosine") == "wsd":
        decay_start = int(cfg.max_iters * (1.0 - getattr(cfg, "wsd_decay_frac", 0.2)))
        if iter_num < decay_start:
            return 1.0
        prog = (iter_num - decay_start) / max(1, cfg.max_iters - decay_start)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    if iter_num > cfg.max_iters:
        return min_ratio
    prog = (iter_num - cfg.warmup_iters) / (cfg.max_iters - cfg.warmup_iters)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog))


def load_config(name: str):
    module = importlib.import_module(name)
    return module.model_config, module.train_config


def maybe_sample(model, tokenizer, device: str, max_tokens: int, prompts) -> dict[str, str]:
    """Sample one generation per configured prompt. An empty prompt is unconditional
    (BOS-only), which is the only way to actually observe the model's opening bias."""
    model.eval()
    samples = {}
    for prompt in prompts:
        text, _ = generate_text(model, tokenizer, prompt, device, max_tokens)
        samples["unconditional" if prompt == "" else prompt] = text
    model.train()
    return samples


def is_improved(val_loss: float, best_val_loss: float, min_delta: float) -> bool:
    return val_loss < best_val_loss - min_delta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs.calliope_10m")
    parser.add_argument("--resume", default="")
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--eval-iters", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--ckpt-interval", type=int, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--optimizer", default=None, choices=["adamw", "muon"])
    parser.add_argument("--lr-schedule", default=None, choices=["cosine", "wsd"])
    parser.add_argument("--muon-lr", type=float, default=None)
    parser.add_argument("--warmup-iters", type=int, default=None)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--run-name", default="")
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
    if args.ckpt_interval is not None:
        train_cfg.ckpt_interval = args.ckpt_interval
    if args.early_stop_patience is not None:
        train_cfg.early_stop_patience = args.early_stop_patience
    if args.optimizer is not None:
        train_cfg.optimizer = args.optimizer
    if args.lr_schedule is not None:
        train_cfg.lr_schedule = args.lr_schedule
    if args.muon_lr is not None:
        train_cfg.muon_lr = args.muon_lr
    if args.warmup_iters is not None:
        train_cfg.warmup_iters = args.warmup_iters
    if args.out_dir:
        train_cfg.out_dir = args.out_dir
    if args.data_dir:
        train_cfg.data_dir = args.data_dir
    if args.run_name:
        train_cfg.run_name = args.run_name

    seed_everything(train_cfg.seed)
    device = pick_device()
    ctx = amp_context(device)

    train_data = load_split(train_cfg.data_dir, "train")
    min_val_bytes = (model_cfg.block_size + 1) * 2
    val_files = [p for p in sorted(Path(train_cfg.data_dir).glob("val_*.bin")) if p.stat().st_size >= min_val_bytes]
    val_splits = {p.stem.removeprefix("val_"): load_split(train_cfg.data_dir, p.stem) for p in val_files}
    if not val_splits:
        val_splits = {"val": load_split(train_cfg.data_dir, "val")}

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
    bad_evals = 0
    if args.resume:
        # load_state_dict overwrites param-group hyperparams with the checkpoint's (possibly stale,
        # e.g. missing weight_decay from an older optimizer). Snapshot the config-derived ones and
        # re-apply after load so the current config wins; the optimizer STATE (momentum) is kept.
        intended_groups = [{k: v for k, v in g.items() if k != "params"} for g in optimizer.param_groups]
        iter_num, best_val_loss, _ = load_checkpoint(args.resume, model, optimizer, device)
        for g, snap in zip(optimizer.param_groups, intended_groups):
            g.update(snap)

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

    train_cursor = 0

    def batch(split: str):
        data = train_data if split == "train" else val_splits[split]
        return get_batch(data, model_cfg.block_size, train_cfg.batch_size, device)

    def train_batch():
        nonlocal train_cursor
        if not train_cfg.sequential_train:
            return batch("train")
        x, y, train_cursor = get_sequential_batch(train_data, model_cfg.block_size, train_cfg.batch_size, device, train_cursor)
        return x, y

    t0 = time.time()
    loss_ema = None
    stop_training = False
    while iter_num < train_cfg.max_iters and not stop_training:
        frac = lr_fraction(iter_num, train_cfg)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * frac
        lr = train_cfg.learning_rate * frac  # AdamW-equivalent LR, for logging

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(train_cfg.grad_accum_steps):
            x, y = train_batch()
            with ctx:
                _, loss = model(x, y)
                loss = loss / train_cfg.grad_accum_steps
            loss.backward()
            loss_accum += loss.item()

        if train_cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        iter_num += 1
        loss_ema = loss_accum if loss_ema is None else train_cfg.loss_ema_beta * loss_ema + (1 - train_cfg.loss_ema_beta) * loss_accum

        if iter_num % 10 == 0:
            dt = time.time() - t0
            tokens = train_cfg.batch_size * train_cfg.grad_accum_steps * model_cfg.block_size * 10
            print(f"iter {iter_num}: loss {loss_accum:.4f}, ema {loss_ema:.4f}, lr {lr:.2e}, {tokens / dt:.0f} tok/s")
            append_jsonl(exp_dir / "loss_log.jsonl", {"iter": iter_num, "loss": round(loss_accum, 4), "ema": round(loss_ema, 4), "lr": lr})
            t0 = time.time()

        if iter_num % train_cfg.eval_interval == 0 or iter_num == train_cfg.max_iters:
            losses = estimate_loss(model, batch, train_cfg.eval_iters, ctx, ("train", *val_splits))
            val_loss = losses["val"] if "val" in losses else sum(losses[k] * len(val_splits[k]) for k in val_splits) / sum(
                len(data) for data in val_splits.values()
            )
            vram_gb = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
            improved = is_improved(val_loss, best_val_loss, train_cfg.early_stop_min_delta)
            if improved:
                best_val_loss = val_loss
                bad_evals = 0
            else:
                bad_evals += 1
                stop_training = (
                    train_cfg.early_stop_patience > 0
                    and iter_num >= train_cfg.early_stop_min_iters
                    and bad_evals >= train_cfg.early_stop_patience
                )

            row = {
                "iter": iter_num,
                "train_loss": losses["train"],
                "val_loss": val_loss,
                "weighted_val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "early_stop_bad_evals": bad_evals,
                "loss_ema": loss_ema,
                "lr": lr,
                "vram_gb": round(vram_gb, 3),
            }
            for name in val_splits:
                row[f"{name}_val_loss"] = losses[name]
            if stop_training:
                row["early_stopped"] = True
            if tokenizer and (iter_num % train_cfg.sample_interval == 0 or iter_num == train_cfg.max_iters):
                row["samples"] = maybe_sample(model, tokenizer, device, train_cfg.sample_tokens, train_cfg.sample_prompts)
            append_jsonl(exp_dir / "metrics.jsonl", row)
            print(json.dumps(row, ensure_ascii=True))

            if improved:
                save_checkpoint(out_dir / "best.pt", model, optimizer, iter_num, best_val_loss, meta)

        if iter_num % train_cfg.ckpt_interval == 0 or iter_num == train_cfg.max_iters or stop_training:
            save_checkpoint(out_dir / "last.pt", model, optimizer, iter_num, best_val_loss, meta)


if __name__ == "__main__":
    main()
