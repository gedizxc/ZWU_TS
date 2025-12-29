import os
import torch
import numpy as np


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)


def torch_save(obj, path: str):
    ensure_dir(os.path.dirname(path))
    torch.save(obj, path)
