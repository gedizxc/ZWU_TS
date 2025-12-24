
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_series(root_path: str, data_path: str, features: str = "M", target: str = "OT") -> tuple[np.ndarray, list[str]]:
    """
    Load ETTh1-style CSV, drop `date`, return values + feature names.
    """
    csv_path = Path(root_path) / data_path
    if not csv_path.exists():
        alt = Path(root_path) / "ETTh1.csv"
        if alt.exists():
            csv_path = alt
        else:
            raise FileNotFoundError(f"Dataset not found under {root_path!r}")

    df = pd.read_csv(csv_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    if features == "M":
        cols = [c for c in df.columns if c != "date"]
    elif features == "S":
        cols = [target]
    else:
        raise ValueError("features must be M or S")

    values = df[cols].to_numpy(dtype=np.float32)
    return values, cols


def standardize_and_normalize(values: np.ndarray) -> tuple[np.ndarray, dict[str, list[float]]]:
    """
    Z-score standardization -> min-max normalization (per channel).
    Returns normalized data and scaling metadata.
    """
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    standardized = (values - mean) / std

    vmin = standardized.min(axis=0)
    vmax = standardized.max(axis=0)
    denom = np.where((vmax - vmin) == 0, 1.0, (vmax - vmin))
    normalized = (standardized - vmin) / denom

    meta = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "min": vmin.tolist(),
        "max": vmax.tolist(),
    }
    return normalized.astype(np.float32), meta


def compute_stats(normalized: np.ndarray, feature_names: list[str]) -> dict:
    """
    Compute per-feature and global min/max/mean/variance on normalized data.
    """
    per_feature = {}
    for idx, name in enumerate(feature_names):
        col = normalized[:, idx]
        per_feature[name] = {
            "min": float(col.min()),
            "max": float(col.max()),
            "mean": float(col.mean()),
            "var": float(col.var()),
        }

    flat = normalized.reshape(-1)
    stats = {
        "shape": list(normalized.shape),
        "per_feature": per_feature,
        "global": {
            "min": float(flat.min()),
            "max": float(flat.max()),
            "mean": float(flat.mean()),
            "var": float(flat.var()),
        },
    }
    return stats
