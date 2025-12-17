import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.timefeatures import time_features


class StandardScaler:
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def fit(self, data: np.ndarray) -> None:
        self.mean = data.mean(0)
        self.std = data.std(0)
        self.std = np.where(self.std == 0, 1.0, self.std)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * self.std + self.mean


@dataclass
class DatasetSize:
    seq_len: int
    label_len: int
    pred_len: int


class Dataset_ETT_hour(Dataset):
    """
    Informer2020-style ETTh1 dataset:
    - Uses fixed borders for train/val/test
    - Returns (seq_x, seq_y, seq_x_mark, seq_y_mark)
    """

    def __init__(
        self,
        root_path: str,
        flag: str = "train",
        size: list[int] | tuple[int, int, int] = (96, 48, 96),
        features: str = "M",
        data_path: str = "ETTh1.csv",
        target: str = "OT",
        scale: bool = True,
        timeenc: int = 0,
        freq: str = "h",
    ):
        assert flag in {"train", "val", "test"}
        self.flag = flag
        self.size = DatasetSize(seq_len=size[0], label_len=size[1], pred_len=size[2])
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = self._resolve_data_path(root_path, data_path)

        self.__read_data__()

    @staticmethod
    def _resolve_data_path(root_path: str, data_path: str) -> str:
        p = os.path.join(root_path, data_path)
        if os.path.exists(p):
            return p
        # common confusion: user says etth1.csv but file is ETTh1.csv
        alt = os.path.join(root_path, "ETTh1.csv")
        if os.path.exists(alt):
            return alt
        raise FileNotFoundError(f"Dataset not found: tried {p} and {alt}")

    def __read_data__(self) -> None:
        self.scaler = StandardScaler()

        df_raw = pd.read_csv(self.data_path)
        df_raw["date"] = pd.to_datetime(df_raw["date"])

        # Informer2020 fixed split for ETT hourly data
        seq_len = self.size.seq_len
        border1s = [0, 12 * 30 * 24 - seq_len, 12 * 30 * 24 + 4 * 30 * 24 - seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        split_map = {"train": 0, "val": 1, "test": 2}
        split_id = split_map[self.flag]

        border1 = border1s[split_id]
        border2 = border2s[split_id]

        if self.features == "M":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
            self.feature_names = list(cols_data)
        elif self.features == "S":
            df_data = df_raw[[self.target]]
            self.feature_names = [self.target]
        else:
            raise ValueError(f"Unsupported features={self.features!r}, use 'M' or 'S'")

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]].values
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2].copy()
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index: int):
        s_begin = index
        s_end = s_begin + self.size.seq_len
        r_begin = s_end - self.size.label_len
        r_end = r_begin + self.size.label_len + self.size.pred_len

        # X: encoder input window (shape: [seq_len, n_features])
        seq_x = self.data_x[s_begin:s_end]
        # Y: decoder input + prediction span (shape: [label_len+pred_len, n_features])
        seq_y = self.data_y[r_begin:r_end]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return (
            torch.from_numpy(seq_x).float(),
            torch.from_numpy(seq_y).float(),
            torch.from_numpy(seq_x_mark).float(),
            torch.from_numpy(seq_y_mark).float(),
        )

    def __len__(self) -> int:
        return len(self.data_x) - self.size.seq_len - self.size.pred_len + 1

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(data)
