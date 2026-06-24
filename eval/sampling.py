from pathlib import Path

import torch

from model import ModelConfig, Transformer
from train.checkpoint import load_checkpoint


def load_model_and_tokenizer(checkpoint: str, device: str):
    _, _, ckpt = load_checkpoint(checkpoint, device=device)
    model = Transformer(ModelConfig(**ckpt["model_config"])).to(device)
    load_checkpoint(checkpoint, model, device=device)
    model.eval()

    from transformers import AutoTokenizer

    tokenizer_dir = Path(ckpt["train_config"]["data_dir"]) / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir if tokenizer_dir.exists() else "gpt2")
    return model, tokenizer, ckpt


def load_for_sampling(checkpoint: str, adapter: str, device: str):
    if not adapter:
        return load_model_and_tokenizer(checkpoint, device)

    from model.lora import inject_lora, load_lora

    meta = torch.load(adapter, map_location="cpu", weights_only=False)
    base = checkpoint or meta["base_checkpoint"]
    model, tokenizer, base_ckpt = load_model_and_tokenizer(base, device)
    lc = meta["lora_config"]
    inject_lora(model, lc["rank"], lc["alpha"], 0.0, tuple(lc["targets"]))
    model.to(device)
    load_lora(adapter, model, device)
    model.eval()
    base_ckpt = {**base_ckpt, "run_name": meta.get("run_name", base_ckpt.get("run_name"))}
    return model, tokenizer, base_ckpt


def generate_text(model, tokenizer, prompt: str, device: str, max_new_tokens: int, **sampling) -> tuple[str, list[int]]:
    if prompt:
        ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    else:
        ids = torch.tensor([[tokenizer.eos_token_id]], device=device)
    out = model.generate(ids, max_new_tokens, **sampling)[0].tolist()
    generated_ids = out[ids.shape[1] :]
    # Drop the leading BOS token from unconditional text so the opening is the model's own words.
    text = tokenizer.decode(out if prompt else out[1:])
    return text, generated_ids
