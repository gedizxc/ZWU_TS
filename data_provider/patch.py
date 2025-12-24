
from __future__ import annotations

import numpy as np


def patchify(batch_x: np.ndarray, patch_len: int) -> np.ndarray:
    """
    PatchTST-style non-overlapping patches over time.
    Returns [B, n_patches, patch_len, C]
    """
    bsz, seq_len, channels = batch_x.shape
    stride = patch_len
    n_patches = (seq_len - patch_len) // stride + 1
    if n_patches <= 0:
        raise ValueError("patch_len is longer than seq_len.")
    patches = []
    for sample in batch_x:
        rows = []
        for start in range(0, seq_len - patch_len + 1, stride):
            rows.append(sample[start : start + patch_len])
        patches.append(np.stack(rows, axis=0))
    return np.stack(patches, axis=0)
