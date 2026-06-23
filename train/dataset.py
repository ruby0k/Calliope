from pathlib import Path

import numpy as np
import torch


def load_split(data_dir: str | Path, split: str) -> np.memmap:
    path = Path(data_dir) / f"{split}.bin"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run scripts/prepare_tinystories.py first")
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(
    data: np.memmap,
    block_size: int,
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[int(i) : int(i) + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[int(i) + 1 : int(i) + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


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
    x = torch.stack(
        [torch.from_numpy(data[cursor + i * block_size : cursor + (i + 1) * block_size].astype(np.int64)) for i in range(batch_size)]
    )
    y = torch.stack(
        [torch.from_numpy(data[cursor + i * block_size + 1 : cursor + (i + 1) * block_size + 1].astype(np.int64)) for i in range(batch_size)]
    )
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True), cursor + batch_size * block_size
