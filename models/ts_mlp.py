import torch.nn as nn

from configs import hparams


def build_ts_mlp():
    model = nn.Sequential(
        nn.Linear(hparams.PATCH_LEN, 2048),
        nn.GELU(),
        nn.Linear(2048, 2048),
    )
    return model
