"""Muon optimizer (Keller Jordan, 2024) — orthogonalized-momentum for 2D weight matrices.

Use ONLY for hidden weight matrices (attention/FFN linears). Embeddings, the tied
output head, norms, and biases should use AdamW. Muon is a full-precision algorithmic
speedup (faster convergence to the same/lower loss) — it does not reduce precision.
"""

import torch


def _zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate the orthogonalization (U V^T) of G via a quintic Newton-Schulz iteration."""
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    transpose = G.size(0) > G.size(1)
    if transpose:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transpose:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, nesterov: bool = True, ns_steps: int = 5):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, momentum, nesterov, ns = group["lr"], group["momentum"], group["nesterov"], group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = torch.zeros_like(g)
                    state["momentum_buffer"] = buf
                buf.mul_(momentum).add_(g)
                update = g.add(buf, alpha=momentum) if nesterov else buf
                update = _zeropower_via_newtonschulz5(update, steps=ns)
                # Scale so the update RMS is consistent across non-square matrices.
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.add_(update, alpha=-lr * scale)


class CombinedOptimizer:
    """Steps several optimizers as one (Muon for matrices + AdamW for everything else).
    Exposes a flat `param_groups` so the LR scheduler sets per-group lr uniformly, and a
    state_dict/load_state_dict so checkpoint resume works."""

    def __init__(self, optimizers):
        self.optimizers = optimizers
        self.param_groups = [g for o in optimizers for g in o.param_groups]

    def zero_grad(self, set_to_none: bool = True):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.optimizers:
            o.step()

    def state_dict(self):
        return {"optimizers": [o.state_dict() for o in self.optimizers]}

    def load_state_dict(self, sd):
        for o, s in zip(self.optimizers, sd["optimizers"]):
            o.load_state_dict(s)
