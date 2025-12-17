from __future__ import annotations

from torch.utils.data import DataLoader

from data_provider.data_loader import Dataset_ETT_hour


def data_provider(args, flag: str):
    """
    Build dataset + dataloader in the same style as Informer2020.
    """
    data_dict = {
        "ETTh1": Dataset_ETT_hour,
    }

    if args.data not in data_dict:
        raise ValueError(f"Unsupported data={args.data!r}. Supported: {list(data_dict)}")

    Data = data_dict[args.data]
    timeenc = 0 if args.embed != "timeF" else 1

    shuffle_flag = flag == "train"
    drop_last = True
    batch_size = args.batch_size

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=args.freq,
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
    )

    return data_set, data_loader

