from __future__ import annotations

import torch
from torch import nn
from tqdm import tqdm

from .data import build_loaders
from .model import MANet
from .utils import ensure_parent, get_device, regression_metrics, set_seed


def run_epoch(
    model: MANet,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    pred_length: int = 16,
    grad_clip: float | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    total_count = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        with torch.set_grad_enabled(training):
            pred = model(x, pred_length=pred_length, update_memory=training)
            loss = criterion(pred, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        batch_size = x.size(0)
        metrics = regression_metrics(pred.detach(), y)
        total_loss += loss.item() * batch_size
        total_mae += metrics["mae"] * batch_size
        total_rmse += metrics["rmse"] * batch_size
        total_count += batch_size

    return {
        "loss": total_loss / total_count,
        "mae": total_mae / total_count,
        "rmse": total_rmse / total_count,
    }


def train(config: dict, csv_path: str) -> None:
    seed = int(config.get("seed", 42))
    set_seed(seed)
    torch.set_num_threads(int(config.get("num_threads", 1)))
    config["data"]["seed"] = seed
    device = get_device(config.get("device", "auto"))
    train_loader, val_loader, _ = build_loaders(csv_path, config["data"])

    model = MANet(config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )
    criterion = nn.MSELoss()
    pred_length = int(config["data"]["pred_length"])
    grad_clip = config["train"].get("grad_clip")
    grad_clip = None if grad_clip is None else float(grad_clip)
    best_val = float("inf")
    checkpoint = config["train"]["checkpoint"]
    ensure_parent(checkpoint)

    for epoch in tqdm(range(1, int(config["train"]["epochs"]) + 1), desc="training"):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            pred_length=pred_length,
            grad_clip=grad_clip,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            pred_length=pred_length,
        )
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_mae={val_metrics['mae']:.6f} "
            f"val_rmse={val_metrics['rmse']:.6f}"
        )
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config,
                    "best_val_loss": best_val,
                },
                checkpoint,
            )

    print(f"best_val_loss={best_val:.6f}, checkpoint={checkpoint}")
