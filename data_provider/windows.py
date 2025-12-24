
from __future__ import annotations

from typing import Iterable, Iterator

import numpy as np


def sliding_windows(
    data: np.ndarray,
    seq_len: int,
    pred_len: int,
    stride: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Yield (past, future) windows with step=stride.
    past: [seq_len, C], future: [pred_len, C]
    """
    total = len(data) - seq_len - pred_len + 1
    if total <= 0:
        raise ValueError("Not enough timesteps to create one window.")
    for start in range(0, total, stride):
        past = data[start : start + seq_len]
        future = data[start + seq_len : start + seq_len + pred_len]
        yield past, future


def batchify_windows(
    windows: Iterable[tuple[np.ndarray, np.ndarray]],
    batch_size: int,
    limit_batches: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Group windows into batches.
    """
    batch_x: list[np.ndarray] = []
    batch_y: list[np.ndarray] = []
    emitted = 0
    for past, future in windows:
        batch_x.append(past)
        batch_y.append(future)
        if len(batch_x) == batch_size:
            yield np.stack(batch_x, axis=0), np.stack(batch_y, axis=0)
            emitted += 1
            if limit_batches is not None and emitted >= limit_batches:
                return
            batch_x.clear()
            batch_y.clear()
    if batch_x:
        yield np.stack(batch_x, axis=0), np.stack(batch_y, axis=0)


def compute_all_shapes(
    total_steps: int,
    channels: int,
    seq_len: int,
    pred_len: int,
    patch_len: int,
    stride: int,
) -> dict:
    total_windows = (total_steps - seq_len - pred_len) // stride + 1
    if total_windows <= 0:
        raise ValueError("Not enough timesteps to create one window.")
    n_patches = (seq_len - patch_len) // patch_len + 1
    return {
        "total_windows": total_windows,
        "past_shape": [total_windows, seq_len, channels],
        "future_shape": [total_windows, pred_len, channels],
        "patches_shape": [total_windows, n_patches, patch_len, channels],
        "patch_len": patch_len,
        "stride": stride,
    }
