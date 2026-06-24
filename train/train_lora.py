import argparse
import importlib
import json
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
from model.lora import inject_lora, mark_only_lora_trainable, num_trainable_parameters, save_lora
from train.checkpoint import load_checkpoint
from train.dataset import get_batch, load_split
from train.train import get_lr
from train.utils import amp_context, append_jsonl, estimate_loss, get_git_commit, pick_device, seed_everything


def load_lora_config(name: str):
    return importlib.import_module(name).lora_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs.lora_demo")
    parser.add_argument("--base-checkpoint", default="")
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--max-iters", type=int, default=None)
    args = parser.parse_args()

    cfg = load_lora_config(args.config)
    if args.base_checkpoint:
        cfg.base_checkpoint = args.base_checkpoint
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.out_dir:
        cfg.out_dir = args.out_dir
    if args.run_name:
        cfg.run_name = args.run_name
    if args.max_iters is not None:
        cfg.max_iters = args.max_iters

    seed_everything(cfg.seed)
    device = pick_device()
    ctx = amp_context(device)

    # Build the base model from its own stored config, load weights, THEN inject LoRA.
    _, _, base_ckpt = load_checkpoint(cfg.base_checkpoint, device=device)
    model_cfg = ModelConfig(**base_ckpt["model_config"])
    model = Transformer(model_cfg).to(device)
    load_checkpoint(cfg.base_checkpoint, model, device=device)

    n_adapted = inject_lora(model, cfg.rank, cfg.alpha, cfg.lora_dropout, cfg.targets)
    mark_only_lora_trainable(model)
    model.to(device)
    trainable = num_trainable_parameters(model)
    total = model.num_parameters()
    print(f"adapted {n_adapted} layers | trainable {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")

    data_dir = cfg.data_dir or base_ckpt["train_config"]["data_dir"]
    train_data = load_split(data_dir, "train")
    min_val_bytes = (model_cfg.block_size + 1) * 2
    val_files = [p for p in sorted(Path(data_dir).glob("val_*.bin")) if p.stat().st_size >= min_val_bytes]
    val_splits = {p.stem.removeprefix("val_"): load_split(data_dir, p.stem) for p in val_files}
    if not val_splits:
        val_splits = {"val": load_split(data_dir, "val")}

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )

    run_name = cfg.run_name or datetime.now().strftime("lora_%Y%m%d_%H%M%S")
    out_dir = Path(cfg.out_dir)
    exp_dir = Path(cfg.experiment_dir) / run_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "run_name": run_name,
        "date": datetime.now().isoformat(timespec="seconds"),
        "git_commit": get_git_commit(),
        "base_checkpoint": cfg.base_checkpoint,
        "model_config": base_ckpt["model_config"],
        "lora_config": asdict(cfg),
        "trainable_params": trainable,
        "data_dir": data_dir,
        "tokenizer": "gpt2",
    }
    (exp_dir / "config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    tokenizer = None
    tokenizer_dir = Path(data_dir) / "tokenizer"
    if tokenizer_dir.exists():
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    def batch(split: str):
        data = train_data if split == "train" else val_splits[split]
        return get_batch(data, model_cfg.block_size, cfg.batch_size, device)

    iter_num = 0
    best_val_loss = float("inf")
    t0 = time.time()
    while iter_num < cfg.max_iters:
        lr = get_lr(iter_num, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(cfg.grad_accum_steps):
            x, y = batch("train")
            with ctx:
                _, loss = model(x, y)
                loss = loss / cfg.grad_accum_steps
            loss.backward()
            loss_accum += loss.item()

        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), cfg.grad_clip)
        optimizer.step()
        iter_num += 1

        if iter_num % 10 == 0:
            dt = time.time() - t0
            tokens = cfg.batch_size * cfg.grad_accum_steps * model_cfg.block_size * 10
            print(f"iter {iter_num}: loss {loss_accum:.4f}, lr {lr:.2e}, {tokens / dt:.0f} tok/s")
            t0 = time.time()

        if iter_num % cfg.eval_interval == 0 or iter_num == cfg.max_iters:
            losses = estimate_loss(model, batch, cfg.eval_iters, ctx, ("train", *val_splits))
            val_loss = sum(losses[k] * len(val_splits[k]) for k in val_splits) / sum(len(d) for d in val_splits.values())
            row = {"iter": iter_num, "train_loss": losses["train"], "val_loss": val_loss, "lr": lr}
            for name in val_splits:
                row[f"{name}_val_loss"] = losses[name]
            if tokenizer and (iter_num % cfg.sample_interval == 0 or iter_num == cfg.max_iters):
                model.eval()
                row["samples"] = {
                    ("unconditional" if p == "" else p): generate_text(model, tokenizer, p, device, cfg.sample_tokens)[0]
                    for p in cfg.sample_prompts
                }
                model.train()
            append_jsonl(exp_dir / "metrics.jsonl", row)
            print(json.dumps({k: row[k] for k in row if k != "samples"}, ensure_ascii=True))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_lora(out_dir / "best.pt", model, {**meta, "iter_num": iter_num, "best_val_loss": best_val_loss})

        if iter_num % cfg.ckpt_interval == 0 or iter_num == cfg.max_iters:
            save_lora(out_dir / "last.pt", model, {**meta, "iter_num": iter_num, "best_val_loss": best_val_loss})


if __name__ == "__main__":
    main()
