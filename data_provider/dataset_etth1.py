import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from configs import hparams
from configs.paths import ETTH1_CSV_PATH
from data_provider.scaler import StandardScalerPerChannel
from utils.logging import print_box


class ETTh1WindowDataset(Dataset):
    """
    Returns:
      x: [SEQ_LEN, C], y: [PRED_LEN, C]
    """

    def __init__(self, data_norm: np.ndarray, seq_len: int, pred_len: int, split: str):
        super().__init__()
        self.data = data_norm
        self.seq_len = seq_len
        self.pred_len = pred_len

        T = data_norm.shape[0]
        self.max_i = T - (seq_len + pred_len) + 1
        if self.max_i <= 0:
            raise ValueError(f"Time series too short: T={T}, need >= {seq_len + pred_len}")

        all_idx = np.arange(self.max_i)
        n = len(all_idx)
        n_train = int(n * hparams.TRAIN_RATIO)
        n_val = int(n * hparams.VAL_RATIO)
        n_test = n - n_train - n_val

        if split == "train":
            self.indices = all_idx[:n_train]
        elif split == "val":
            self.indices = all_idx[n_train:n_train + n_val]
        elif split == "test":
            self.indices = all_idx[n_train + n_val:]
        else:
            raise ValueError("split must be train/val/test")

        print(f"[Dataset] split={split} windows={len(self.indices)} (train={n_train}, val={n_val}, test={n_test})")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        i = int(self.indices[idx])
        x = self.data[i: i + self.seq_len]  # [seq_len, C]
        y = self.data[i + self.seq_len: i + self.seq_len + self.pred_len]  # [pred_len, C]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


def load_etth1_csv(path: str = ETTH1_CSV_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find ETTh1 CSV at: {path}")
    df = pd.read_csv(path)
    if "date" in df.columns:
        df_feat = df.drop(columns=["date"])
    else:
        df_feat = df
    data = df_feat.values.astype(np.float32)
    print(f"[Data] raw shape={data.shape} columns={list(df_feat.columns)}")
    if data.shape[1] != hparams.NUM_VARS:
        print(f"[WARN] Expected {hparams.NUM_VARS} vars but got {data.shape[1]}")
    return data


def build_dataloader(data_norm: np.ndarray):
    print_box("2) Build dataset/dataloader")
    ds_train = ETTh1WindowDataset(
        data_norm,
        seq_len=hparams.SEQ_LEN,
        pred_len=hparams.PRED_LEN,
        split="train",
    )
    loader = DataLoader(ds_train, batch_size=hparams.BATCH_SIZE, shuffle=False, drop_last=True)
    return loader


def fit_scaler(data: np.ndarray):
    print_box("1) Fit scaler on train portion")
    T = data.shape[0]
    train_T = int(T * hparams.TRAIN_RATIO)
    scaler = StandardScalerPerChannel()
    scaler.fit(data[:train_T])
    data_norm = scaler.transform(data)
    print(f"[Scaler] mean shape={scaler.mean_.shape}, std shape={scaler.std_.shape}")
    print(f"[Data] norm shape={data_norm.shape}, per-channel mean≈{data_norm[:train_T].mean(axis=0)}, "
          f"std≈{data_norm[:train_T].std(axis=0)}")
    return scaler, data_norm
