from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data_provider.loader import compute_stats, standardize_and_normalize


def _time_marks(dates: pd.Series | None, total: int) -> np.ndarray:
    """
    Build simple time features similar to Informer (month/day/weekday/hour).
    Falls back to zeros when date is missing.
    """
    if dates is None:
        return np.zeros((total, 4), dtype=np.float32)

    dt = pd.to_datetime(dates)
    return np.stack(
        [
            (dt.dt.month.values.astype(np.float32) - 1) / 12.0,
            (dt.dt.day.values.astype(np.float32) - 1) / 31.0,
            dt.dt.dayofweek.values.astype(np.float32) / 7.0,
            dt.dt.hour.values.astype(np.float32) / 24.0,
        ],
        axis=1,
    ).astype(np.float32)


def _split_bounds(total: int, train_ratio: float, val_ratio: float) -> dict[str, tuple[int, int]]:
    train_end = max(int(total * train_ratio), 0)
    val_end = min(total, train_end + max(int(total * val_ratio), 0))
    val_end = max(val_end, train_end)
    train_end = min(train_end, total)
    return {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, total),
    }


@dataclass
class InformerDatasetMeta:
    feature_names: list[str]
    stats: dict
    scaling: dict[str, list[float]]
    raw_shape: tuple[int, int]
    normalized_shape: tuple[int, int]


class InformerDataset(Dataset):
    """
    Informer-style dataset producing (seq_x, seq_y, seq_x_mark, seq_y_mark).
    Uses z-score + min-max normalization over the full dataset.
    """

    def __init__(
        self,
        root_path: str,
        data_path: str,
        flag: Literal["train", "val", "test"],
        seq_len: int,
        label_len: int,
        pred_len: int,
        features: str,
        target: str,
        train_ratio: float,
        val_ratio: float,
        window_stride: int,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.features = features
        self.target = target
        self.flag = flag
        self.window_stride = window_stride

        self.values, self.feature_names, dates = self._read_csv(root_path, data_path)
        self.normalized, self.scaling = standardize_and_normalize(self.values)
        self.stats = compute_stats(self.normalized, self.feature_names)
        self.time_marks = _time_marks(dates, total=len(self.normalized))

        self.bounds = _split_bounds(len(self.normalized), train_ratio, val_ratio)
        if flag not in self.bounds:
            raise ValueError(f"Unsupported flag: {flag}")
        self.border1, self.border2 = self.bounds[flag]

        if self.__len__() <= 0:
            raise ValueError(f"Split {flag} has no available windows; check ratios/lengths.")

    def _read_csv(self, root_path: str, data_path: str) -> tuple[np.ndarray, list[str], pd.Series | None]:
        csv_path = Path(root_path) / data_path
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset not found at {csv_path}")

        df = pd.read_csv(csv_path)
        dates = pd.to_datetime(df["date"]) if "date" in df.columns else None

        if self.features == "M":
            cols = [c for c in df.columns if c != "date"]
        elif self.features == "S":
            cols = [self.target]
        else:
            raise ValueError("features must be 'M' or 'S'")

        values = df[cols].to_numpy(dtype=np.float32)
        return values, cols, dates

    def __len__(self) -> int:
        usable = self.border2 - self.border1
        span = self.seq_len + self.pred_len
        return max((usable - span) // self.window_stride + 1, 0)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        start = self.border1 + index * self.window_stride
        end = start + self.seq_len
        label_start = end - self.label_len
        label_end = label_start + self.label_len + self.pred_len

        seq_x = self.normalized[start:end]
        seq_y = self.normalized[label_start:label_end]
        seq_x_mark = self.time_marks[start:end]
        seq_y_mark = self.time_marks[label_start:label_end]

        return (
            torch.from_numpy(seq_x.astype(np.float32)),
            torch.from_numpy(seq_y.astype(np.float32)),
            torch.from_numpy(seq_x_mark.astype(np.float32)),
            torch.from_numpy(seq_y_mark.astype(np.float32)),
        )

    def meta(self) -> InformerDatasetMeta:
        return InformerDatasetMeta(
            feature_names=self.feature_names,
            stats=self.stats,
            scaling=self.scaling,
            raw_shape=self.values.shape,
            normalized_shape=self.normalized.shape,
        )

