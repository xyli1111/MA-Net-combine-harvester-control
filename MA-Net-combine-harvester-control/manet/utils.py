from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def ensure_parent(path: str | Path) -> None:
    parent = Path(path).parent
    if str(parent):
        os.makedirs(parent, exist_ok=True)


def regression_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    error = pred - target
    mae = error.abs().mean().item()
    rmse = torch.sqrt((error**2).mean()).item()
    return {"mae": mae, "rmse": rmse}
