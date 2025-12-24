from __future__ import annotations

from torch.utils.data import DataLoader

from data_provider.dataset import InformerDataset


def data_provider(args, flag: str):
    """
    Build Informer-style dataset/dataloader for the given split.
    """
    dataset = InformerDataset(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        features=args.features,
        target=args.target,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        window_stride=args.window_stride or args.seq_len,
    )

    shuffle = flag == "train"
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=getattr(args, "num_workers", 0),
        drop_last=True,
    )
    return dataset, data_loader

