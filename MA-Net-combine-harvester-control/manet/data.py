from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset


@dataclass
class StandardScaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "StandardScaler":
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std[std == 0] = 1.0
        return cls(mean=mean, std=std)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean


class TimeSeriesWindowDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        csv_path: str,
        input_columns: list[str],
        target_columns: list[str],
        input_length: int,
        pred_length: int,
    ) -> None:
        frame = pd.read_csv(csv_path)
        missing = set(input_columns + target_columns) - set(frame.columns)
        if missing:
            raise ValueError(f"CSV is missing columns: {sorted(missing)}")

        self.input_length = input_length
        self.pred_length = pred_length
        self.input_scaler = StandardScaler.fit(frame[input_columns].to_numpy(dtype=np.float32))
        self.target_scaler = StandardScaler.fit(frame[target_columns].to_numpy(dtype=np.float32))
        self.inputs = self.input_scaler.transform(frame[input_columns].to_numpy(dtype=np.float32))
        self.targets = self.target_scaler.transform(frame[target_columns].to_numpy(dtype=np.float32))
        self.window_count = len(frame) - input_length - pred_length + 1
        if self.window_count <= 0:
            raise ValueError("CSV length must be larger than input_length + pred_length")

    def __len__(self) -> int:
        return self.window_count

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        x0 = index
        x1 = index + self.input_length
        y1 = x1 + self.pred_length
        x = torch.tensor(self.inputs[x0:x1], dtype=torch.float32)
        y = torch.tensor(self.targets[x1:y1], dtype=torch.float32)
        return x, y


def build_loaders(csv_path: str, data_config: dict) -> tuple[DataLoader, DataLoader, TimeSeriesWindowDataset]:
    dataset = TimeSeriesWindowDataset(
        csv_path=csv_path,
        input_columns=data_config["input_columns"],
        target_columns=data_config["target_columns"],
        input_length=int(data_config["input_length"]),
        pred_length=int(data_config["pred_length"]),
    )
    val_ratio = float(data_config.get("val_ratio", 0.2))
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("Not enough windows to create train and validation splits")

    split_mode = data_config.get("split_mode", "chronological")
    if split_mode != "chronological":
        raise ValueError("Only chronological split is supported in this public demo to reduce temporal leakage risk")

    train_indices = list(range(train_size))
    val_indices = list(range(train_size, train_size + val_size))
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    batch_size = int(data_config.get("batch_size", 32))
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(val_set, batch_size=batch_size, shuffle=False),
        dataset,
    )
