import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import ModelConfig, Transformer


def main() -> None:
    cfg = ModelConfig(vocab_size=128, block_size=16, n_layer=1, n_head=2, n_embd=32)
    model = Transformer(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    logits, loss = model(x, x)
    assert logits.shape == (2, cfg.block_size, cfg.vocab_size)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    out = model.generate(x[:1, :4], max_new_tokens=4)
    assert out.shape == (1, 8)
    print("smoke ok")


if __name__ == "__main__":
    main()
