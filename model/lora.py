import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = (self.lora_dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
        return self.base(x) + delta * self.scaling


def _get_parent(model: nn.Module, qualified_name: str):
    parts = qualified_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    return parent, parts[-1]


def inject_lora(model: nn.Module, rank: int, alpha: int, dropout: float, targets) -> int:
    targets = set(targets)
    to_replace = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name.split(".")[-1] in targets
    ]
    for name in to_replace:
        parent, attr = _get_parent(model, name)
        setattr(parent, attr, LoRALinear(getattr(parent, attr), rank, alpha, dropout))
    return len(to_replace)


def mark_only_lora_trainable(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = "lora_A" in name or "lora_B" in name


def lora_state_dict(model: nn.Module) -> dict:
    return {k: v for k, v in model.state_dict().items() if "lora_A" in k or "lora_B" in k}


def num_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_lora(path, model: nn.Module, meta: dict) -> None:
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"lora": lora_state_dict(model), **meta}, path)


def load_lora(path, model: nn.Module, device: str = "cpu") -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["lora"], strict=False)
    return ckpt
