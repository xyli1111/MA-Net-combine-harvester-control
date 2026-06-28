from __future__ import annotations

import argparse

import torch

from manet.model import MANet
from manet.train import train
from manet.utils import load_config


def smoke(config_path: str) -> None:
    config = load_config(config_path)
    torch.set_num_threads(int(config.get("num_threads", 1)))
    model = MANet(config["model"])
    batch_size = 4
    input_length = int(config["data"]["input_length"])
    pred_length = int(config["data"]["pred_length"])
    input_dim = int(config["model"]["input_dim"])
    x = torch.randn(batch_size, input_length, input_dim)
    y = model(x, pred_length=pred_length, update_memory=True)
    print(f"input shape:  {tuple(x.shape)}")
    print(f"output shape: {tuple(y.shape)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MA-Net for multivariate time-series prediction")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train MA-Net on a CSV dataset")
    train_parser.add_argument("--config", default="configs/default.yaml")
    train_parser.add_argument("--data", required=True, help="path to CSV data")

    smoke_parser = subparsers.add_parser("smoke", help="run a random forward-pass check")
    smoke_parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        config = load_config(args.config)
        train(config, args.data)
    elif args.command == "smoke":
        smoke(args.config)


if __name__ == "__main__":
    main()
