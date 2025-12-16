"""
Generate patch-based videos (PatchTST style) for Qwen3-VL inputs.
- past_len=96, pred_len=96, patch_size=24, stride=12.
- Each past window (96 steps) -> overlapping patches -> each patch is one frame.
- Frame layout matches generate_correlations.py: top 7x7 feature correlations, bottom TxT time correlations (T=patch_size).

Outputs under artifacts/patch_videos/:
- batch_{i}.mp4 (requires imageio.v3); frames are kept in-memory only (no PNGs on disk).
"""
from pathlib import Path
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_CANDIDATES = [
    Path("data/ETTh1.csv"),
    REPO_ROOT / "data/ETTh1.csv",
]

OUTPUT_DIR = REPO_ROOT / "artifacts/patch_videos"

FEATURES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

# PatchTST-style params
PAST_LEN = 96
PRED_LEN = 96  # kept for reference; not used in frame generation
PATCH_SIZE = 24
STRIDE = 12

# To avoid massive output, cap samples; set to None to process all
MAX_SAMPLES = 5

# Require video writer; no frame files will be saved.
try:
    import imageio.v3 as iio  # type: ignore
except Exception as e:
    raise SystemExit("imageio.v3 not available; install with `pip install imageio[ffmpeg]`") from e


def resolve_data_path():
    for p in DATA_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing dataset, tried: {DATA_CANDIDATES}")


def compute_feature_matrices(batch_data: np.ndarray):
    data_mean = batch_data.mean(axis=0)
    data_std = batch_data.std(axis=0)
    data_norm = (batch_data - data_mean) / (data_std + 1e-5)

    fft_data = np.fft.rfft(data_norm, axis=0)
    fft_magnitude = np.abs(fft_data)

    pearson_matrix = np.corrcoef(fft_magnitude, rowvar=False)
    cov_matrix = np.cov(fft_magnitude, rowvar=False)

    num_vars = data_norm.shape[1]
    dtw_matrix = np.zeros((num_vars, num_vars))
    for i in range(num_vars):
        for j in range(num_vars):
            if i == j:
                dtw_matrix[i, j] = 0.0
            elif i > j:
                dtw_matrix[i, j] = dtw_matrix[j, i]
            else:
                dtw_matrix[i, j] = simple_dtw_distance(data_norm[:, i], data_norm[:, j])
    return dtw_matrix, cov_matrix, pearson_matrix


def compute_time_matrices(batch_data: np.ndarray):
    data_mean = batch_data.mean(axis=0)
    data_std = batch_data.std(axis=0)
    data_norm = (batch_data - data_mean) / (data_std + 1e-5)

    pearson_matrix = np.corrcoef(data_norm)
    cov_matrix = np.cov(data_norm)

    num_timesteps = data_norm.shape[0]
    dtw_matrix = np.zeros((num_timesteps, num_timesteps))
    for i in range(num_timesteps):
        for j in range(num_timesteps):
            if i == j:
                dtw_matrix[i, j] = 0.0
            elif i > j:
                dtw_matrix[i, j] = dtw_matrix[j, i]
            else:
                dtw_matrix[i, j] = simple_dtw_distance(data_norm[i, :], data_norm[j, :])
    return dtw_matrix, cov_matrix, pearson_matrix


def simple_dtw_distance(s1: np.ndarray, s2: np.ndarray) -> float:
    n, m = len(s1), len(s2)
    dtw_matrix = np.zeros((n + 1, m + 1))
    dtw_matrix.fill(np.inf)
    dtw_matrix[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s1[i - 1] - s2[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],
                dtw_matrix[i, j - 1],
                dtw_matrix[i - 1, j - 1],
            )
    return float(dtw_matrix[n, m])


def plot_frame(batch_idx: int, patch_idx: int, patch_data: np.ndarray):
    feature_mats = compute_feature_matrices(patch_data)
    time_mats = compute_time_matrices(patch_data)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    feature_dtw, feature_cov, feature_pearson = feature_mats
    time_dtw, time_cov, time_pearson = time_mats

    sns.heatmap(
        feature_dtw,
        annot=True,
        fmt=".1f",
        cmap="viridis",
        xticklabels=FEATURES,
        yticklabels=FEATURES,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title(f"Batch {batch_idx} Patch {patch_idx} Feature DTW (7x7)")

    sns.heatmap(
        feature_cov,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        xticklabels=FEATURES,
        yticklabels=FEATURES,
        ax=axes[0, 1],
    )
    axes[0, 1].set_title(f"Batch {batch_idx} Patch {patch_idx} Feature Covariance (FFT Magnitude)")

    sns.heatmap(
        feature_pearson,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        xticklabels=FEATURES,
        yticklabels=FEATURES,
        ax=axes[0, 2],
    )
    axes[0, 2].set_title(f"Batch {batch_idx} Patch {patch_idx} Feature Pearson (FFT Magnitude)")

    sns.heatmap(
        time_dtw,
        annot=False,
        cmap="viridis_r",
        xticklabels=5,
        yticklabels=5,
        ax=axes[1, 0],
    )
    axes[1, 0].set_title(f"Batch {batch_idx} Patch {patch_idx} Time DTW ({PATCH_SIZE}x{PATCH_SIZE})")
    axes[1, 0].set_xlabel("Time Step")
    axes[1, 0].set_ylabel("Time Step")

    sns.heatmap(
        time_cov,
        annot=False,
        cmap="coolwarm",
        xticklabels=5,
        yticklabels=5,
        ax=axes[1, 1],
    )
    axes[1, 1].set_title(f"Batch {batch_idx} Patch {patch_idx} Time Covariance")

    sns.heatmap(
        time_pearson,
        annot=False,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        xticklabels=5,
        yticklabels=5,
        ax=axes[1, 2],
    )
    axes[1, 2].set_title(f"Batch {batch_idx} Patch {patch_idx} Time Pearson")

    plt.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return img


def make_video(frames, out_path: Path, fps: int = 2):
    iio.imwrite(out_path, frames, fps=fps, codec="h264")
    print(f"Saved video: {out_path}")


def main():
    data_path = resolve_data_path()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    data = df[FEATURES].values
    total_len = len(data)

    # number of usable samples with past+future window
    max_start = total_len - (PAST_LEN + PRED_LEN)
    if max_start <= 0:
        raise ValueError("Dataset too short for past_len + pred_len window.")

    num_samples = max_start + 1
    if MAX_SAMPLES is not None:
        num_samples = min(num_samples, MAX_SAMPLES)

    print(f"Using dataset: {data_path}")
    print(f"Total rows: {total_len}")
    print(f"Past_len={PAST_LEN}, Pred_len={PRED_LEN}, patch_size={PATCH_SIZE}, stride={STRIDE}")
    print(f"Generating samples: {num_samples} (capped by MAX_SAMPLES={MAX_SAMPLES})")

    for sample_idx in range(num_samples):
        start = sample_idx
        past_window = data[start : start + PAST_LEN]  # shape (96,7)

        frames = []
        patch_idx = 0
        for s in range(0, PAST_LEN - PATCH_SIZE + 1, STRIDE):
            patch = past_window[s : s + PATCH_SIZE]  # shape (patch_size, 7)
            img = plot_frame(sample_idx, patch_idx, patch)
            frames.append(img)
            patch_idx += 1

        # write video
        out_video = OUTPUT_DIR / f"batch_{sample_idx}.mp4"
        make_video(frames, out_video)

    print("Done.")


if __name__ == "__main__":
    main()
