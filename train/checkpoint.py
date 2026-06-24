from pathlib import Path

import torch


def save_checkpoint(path: str | Path, model, optimizer, iter_num: int, best_val_loss: float, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file then atomically replace, so a stop/kill mid-write
    # can never corrupt the resumable checkpoint.
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iter_num": iter_num,
            "best_val_loss": best_val_loss,
            **meta,
        },
        tmp,
    )
    tmp.replace(path)


def load_checkpoint(path: str | Path, model=None, optimizer=None, device: str = "cpu") -> tuple[int, float, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if model is not None:
        model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint.get("iter_num", 0), checkpoint.get("best_val_loss", float("inf")), checkpoint
