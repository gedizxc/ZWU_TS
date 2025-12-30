import os

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from .data_loader import ETTh1WindowDataset, SplitConfig, StandardScalerPerChannel


def data_provider(args, flag: str):
    data_path = os.path.join(args.root_path, args.data_path)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"cannot find data file: {data_path}")

    df = pd.read_csv(data_path)
    if "date" in df.columns:
        df_feat = df.drop(columns=["date"])
    else:
        df_feat = df

    data = df_feat.values.astype(np.float32)
    num_vars = data.shape[1]
    if getattr(args, "num_vars", None) in (None, 0, -1):
        args.num_vars = num_vars

    T = data.shape[0]
    train_T = int(T * args.train_ratio)
    scaler = StandardScalerPerChannel()
    scaler.fit(data[:train_T])
    data_norm = scaler.transform(data)

    split_config = SplitConfig(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    data_set = ETTh1WindowDataset(
        data_norm=data_norm,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        split=flag,
        split_config=split_config,
    )
    data_set.scaler = scaler
    data_set.data_norm = data_norm
    data_set.data_raw = data
    data_set.columns = list(df_feat.columns)

    shuffle_flag = flag == "train"
    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        drop_last=True,
        num_workers=args.num_workers,
    )
    return data_set, data_loader
