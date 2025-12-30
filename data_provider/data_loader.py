from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SplitConfig:
    train_ratio: float = 0.70
    val_ratio: float = 0.10
    test_ratio: float = 0.20


class StandardScalerPerChannel:
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean_ = None
        self.std_ = None

    def fit(self, data: np.ndarray) -> None:
        self.mean_ = data.mean(axis=0, keepdims=True)
        self.std_ = data.std(axis=0, keepdims=True)
        self.std_ = np.maximum(self.std_, self.eps)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean_) / self.std_

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.transform(data)


class ETTh1WindowDataset(Dataset):
    """
    Returns:
      x: [seq_len, C], y: [pred_len, C]
    """
    def __init__(
        self,
        data_norm: np.ndarray,
        seq_len: int,
        pred_len: int,
        split: str,
        split_config: SplitConfig,
    ):
        super().__init__()
        self.data = data_norm
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.split = split
        self.split_config = split_config

        T = data_norm.shape[0]
        self.max_i = T - (seq_len + pred_len) + 1
        if self.max_i <= 0:
            raise ValueError(f"time series too short: T={T}, need >= {seq_len + pred_len}")

        all_idx = np.arange(self.max_i)
        n = len(all_idx)
        n_train = int(n * split_config.train_ratio)
        n_val = int(n * split_config.val_ratio)
        n_test = n - n_train - n_val

        if split == "train":
            self.indices = all_idx[:n_train]
        elif split == "val":
            self.indices = all_idx[n_train:n_train + n_val]
        elif split == "test":
            self.indices = all_idx[n_train + n_val:]
        else:
            raise ValueError("split must be train/val/test")

        self.split_sizes: Tuple[int, int, int] = (n_train, n_val, n_test)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        i = int(self.indices[idx])
        x = self.data[i:i + self.seq_len]
        y = self.data[i + self.seq_len:i + self.seq_len + self.pred_len]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()
