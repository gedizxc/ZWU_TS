"""
Merged correlation generator in an Informer-style layout.
- Reads data from data/ETTh1.csv
- Emits combined 3x2 correlation figures per batch under artifacts/correlations
  Top row: 7x7 feature-wise (DTW, Covariance via FFT magnitude, Pearson via FFT magnitude)
  Bottom row: 96x96 time-wise (DTW, Covariance, Pearson)
"""
from pathlib import Path
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


# Resolve dataset path robustly whether run from repo root or elsewhere.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_CANDIDATES = [
    Path("data/ETTh1.csv"),  # relative to CWD
    REPO_ROOT / "data/ETTh1.csv",  # relative to repo root
]
OUTPUT_DIR = Path("artifacts/correlations")
STATS_FILE = OUTPUT_DIR / "batch_stats.txt"
SEQ_LEN = 96
FEATURES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


def simple_dtw_distance(s1: np.ndarray, s2: np.ndarray) -> float:
    """Compute a simple DTW distance between two 1-D sequences."""
    n, m = len(s1), len(s2)
    dtw_matrix = np.zeros((n + 1, m + 1))
    dtw_matrix.fill(np.inf)
    dtw_matrix[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s1[i - 1] - s2[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]
            )
    return float(dtw_matrix[n, m])


def compute_feature_matrices(batch_data: np.ndarray):
    """Return DTW, covariance (FFT magnitude), and Pearson (FFT magnitude) for features."""
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
    """Return DTW, covariance, and Pearson matrices for time steps."""
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


def resolve_data_path():
    for p in DATA_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing dataset, tried: {DATA_CANDIDATES}")


def ensure_paths():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_batch_stats(f, batch_idx: int, batch_data: np.ndarray):
    f.write(f"--- Batch {batch_idx} ---\n")
    stats_list = []
    for i, col in enumerate(FEATURES):
        series = batch_data[:, i]
        stat_str = f"{col}: Mean={series.mean():.2f}, Max={series.max():.2f}, Var={series.var():.2f}"
        stats_list.append(stat_str)
    f.write("\n".join(stats_list))
    f.write("\n\n")


def plot_and_save(
    batch_idx: int,
    feature_mats,
    time_mats,
):
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
    axes[0, 0].set_title(f"Batch {batch_idx} Feature DTW (7x7)")

    sns.heatmap(
        feature_cov,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        xticklabels=FEATURES,
        yticklabels=FEATURES,
        ax=axes[0, 1],
    )
    axes[0, 1].set_title(f"Batch {batch_idx} Feature Covariance (FFT Magnitude)")

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
    axes[0, 2].set_title(f"Batch {batch_idx} Feature Pearson (FFT Magnitude)")

    sns.heatmap(
        time_dtw,
        annot=False,
        cmap="viridis_r",
        xticklabels=10,
        yticklabels=10,
        ax=axes[1, 0],
    )
    axes[1, 0].set_title(f"Batch {batch_idx} Time DTW (96x96)")
    axes[1, 0].set_xlabel("Time Step")
    axes[1, 0].set_ylabel("Time Step")

    sns.heatmap(
        time_cov,
        annot=False,
        cmap="coolwarm",
        xticklabels=10,
        yticklabels=10,
        ax=axes[1, 1],
    )
    axes[1, 1].set_title(f"Batch {batch_idx} Time Covariance")

    sns.heatmap(
        time_pearson,
        annot=False,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        xticklabels=10,
        yticklabels=10,
        ax=axes[1, 2],
    )
    axes[1, 2].set_title(f"Batch {batch_idx} Time Pearson")

    plt.tight_layout()
    save_path = OUTPUT_DIR / f"batch_{batch_idx}.png"
    plt.savefig(save_path)
    plt.close(fig)


def main():
    ensure_paths()
    data_path = resolve_data_path()
    print(f"Using dataset: {data_path}")
    df = pd.read_csv(data_path)
    df_features = df[FEATURES]

    total_len = len(df_features)
    num_batches = total_len // SEQ_LEN

    print(f"Data rows: {total_len}")
    print(f"Sequence length: {SEQ_LEN}")
    print(f"Planned batches: {num_batches}")
    print(f"Images will be saved to: {OUTPUT_DIR.resolve()}")
    print(f"Stats will be saved to: {STATS_FILE.resolve()}")

    with open(STATS_FILE, "w", encoding="utf-8") as f:
        for batch_idx in tqdm(range(num_batches), desc="Generating correlations"):
            start_idx = batch_idx * SEQ_LEN
            end_idx = start_idx + SEQ_LEN

            batch_df = df_features.iloc[start_idx:end_idx]
            batch_data = batch_df.values

            save_batch_stats(f, batch_idx, batch_data)

            feature_mats = compute_feature_matrices(batch_data)
            time_mats = compute_time_matrices(batch_data)

            plot_and_save(batch_idx, feature_mats, time_mats)

    print("All batches processed.")


if __name__ == "__main__":
    main()
