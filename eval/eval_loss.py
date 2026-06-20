import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import ModelConfig, Transformer
from train.checkpoint import load_checkpoint
from train.dataset import get_batch, load_split
from train.utils import amp_context, estimate_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/calliope_10m/best.pt")
    parser.add_argument("--eval-iters", type=int, default=100)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, ckpt = load_checkpoint(args.checkpoint, device=device)
    model_cfg = ModelConfig(**ckpt["model_config"])
    model = Transformer(model_cfg).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    data_dir = ckpt["train_config"]["data_dir"]
    train_data = load_split(data_dir, "train")
    val_data = load_split(data_dir, "val")

    def batch(split: str):
        data = train_data if split == "train" else val_data
        return get_batch(data, model_cfg.block_size, ckpt["train_config"]["batch_size"], device)

    print(estimate_loss(model, batch, args.eval_iters, amp_context(device)))


if __name__ == "__main__":
    main()
