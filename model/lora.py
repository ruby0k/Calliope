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


class ComposedLoRALinear(nn.Module):
    """base(x) + sum_i w_i * scaling_i * (x A_i^T B_i^T).

    Holds K frozen LoRA experts; the weight vector `weights` is a single shared
    tensor (length K) used by every composed layer, so a search/router sets all
    layers' mixing coefficients in one in-place update. Inference only."""

    def __init__(self, base: nn.Linear, adapters: list[dict], weights: torch.Tensor):
        super().__init__()
        self.base = base
        self.weights = weights  # shared (K,) tensor, set externally
        self.scaling = [a["scaling"] for a in adapters]
        self.k = len(adapters)
        for i, a in enumerate(adapters):
            self.register_buffer(f"A_{i}", a["A"], persistent=False)
            self.register_buffer(f"B_{i}", a["B"], persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        for i in range(self.k):
            delta = (x @ getattr(self, f"A_{i}").t()) @ getattr(self, f"B_{i}").t()
            out = out + (self.weights[i] * self.scaling[i]) * delta
        return out


def inject_composed_lora(model: nn.Module, adapter_paths: list[str], device: str) -> torch.Tensor:
    """Layer base + K LoRA experts into one model for weighted composition.
    All adapters must share the same target layers. Returns the shared (K,)
    weight tensor (init zeros = pure base); set it with set_composition_weights."""
    metas = [torch.load(p, map_location="cpu", weights_only=False) for p in adapter_paths]
    targets = set(metas[0]["lora_config"]["targets"])
    scalings = [m["lora_config"]["alpha"] / m["lora_config"]["rank"] for m in metas]
    sds = [m["lora"] for m in metas]
    base_dtype = next(model.parameters()).dtype
    weights = torch.zeros(len(metas), device=device)

    names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and n.split(".")[-1] in targets]
    for name in names:
        parent, attr = _get_parent(model, name)
        base = getattr(parent, attr)
        adapters = [
            {
                "A": sd[f"{name}.lora_A"].to(device=device, dtype=base_dtype),
                "B": sd[f"{name}.lora_B"].to(device=device, dtype=base_dtype),
                "scaling": sc,
            }
            for sd, sc in zip(sds, scalings)
        ]
        setattr(parent, attr, ComposedLoRALinear(base, adapters, weights))
    return weights


def set_composition_weights(weights: torch.Tensor, values) -> None:
    weights.copy_(torch.tensor(list(values), dtype=weights.dtype, device=weights.device))
