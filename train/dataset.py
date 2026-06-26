from pathlib import Path

import numpy as np
import torch


def load_split(data_dir: str | Path, split: str) -> np.memmap:
    path = Path(data_dir) / f"{split}.bin"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run scripts/prepare_tinystories.py first")
    return np.memmap(path, dtype=np.uint16, mode="r")


def _slice_batch(data: np.memmap, starts: np.ndarray, block_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    # One vectorised fancy-index gather (B, block+1) instead of a Python per-sample loop.
    offsets = starts[:, None] + np.arange(block_size + 1)[None, :]
    seq = torch.from_numpy(data[offsets].astype(np.int64))
    x = seq[:, :-1].contiguous()
    y = seq[:, 1:].contiguous()
    if device == "cuda":
        # Pinned host buffers make the async (non_blocking) H2D copy actually overlap with compute.
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


def get_batch(
    data: np.memmap,
    block_size: int,
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
    return _slice_batch(data, starts, block_size, device)


def get_sequential_batch(
    data: np.memmap,
    block_size: int,
    batch_size: int,
    device: str,
    cursor: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    needed = batch_size * block_size + 1
    if cursor + needed >= len(data):
        cursor = 0
    starts = cursor + np.arange(batch_size) * block_size
    x, y = _slice_batch(data, starts, block_size, device)
    return x, y, cursor + batch_size * block_size
