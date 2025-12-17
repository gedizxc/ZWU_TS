from __future__ import annotations

from pathlib import Path

import numpy as np
import subprocess


def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Classic DTW for 1D sequences (L1 cost).
    a: [T], b: [T2]
    """
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    n, m = a.shape[0], b.shape[0]
    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = abs(ai - b[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m])


def _dtw_pairwise_1d(seqs: list[np.ndarray]) -> np.ndarray:
    k = len(seqs)
    out = np.zeros((k, k), dtype=np.float32)
    for i in range(k):
        for j in range(i + 1, k):
            d = _dtw_distance(seqs[i], seqs[j])
            out[i, j] = d
            out[j, i] = d
    return out


def _compute_correlation_panels(x: np.ndarray) -> dict[str, np.ndarray]:
    """
    x: [T, C]

    Returns 6 matrices:
    - var-var: dtw/cov/pearson  -> [C, C]
    - time-time: dtw/cov/pearson -> [T, T]
    """
    x = np.asarray(x, dtype=np.float32)
    T, C = x.shape

    # Top row: variable-variable matrices
    var_series = [x[:, c] for c in range(C)]
    dtw_vars = _dtw_pairwise_1d(var_series)
    cov_vars = np.cov(x, rowvar=False).astype(np.float32)  # [C, C]
    corr_vars = np.corrcoef(x, rowvar=False).astype(np.float32)  # [C, C]

    # Bottom row: time-time matrices (treat each time step as a "sequence" over variables)
    time_series = [x[t, :] for t in range(T)]
    dtw_time = _dtw_pairwise_1d(time_series)
    cov_time = np.cov(x, rowvar=True).astype(np.float32)  # [T, T]
    corr_time = np.corrcoef(x, rowvar=True).astype(np.float32)  # [T, T]

    corr_vars = np.nan_to_num(corr_vars, nan=0.0, posinf=0.0, neginf=0.0)
    corr_time = np.nan_to_num(corr_time, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        "dtw_vars": dtw_vars,
        "cov_vars": cov_vars,
        "corr_vars": corr_vars,
        "dtw_time": dtw_time,
        "cov_time": cov_time,
        "corr_time": corr_time,
    }


def _render_2x3_heatmaps(
    x: np.ndarray,
    title_prefix: str | None = None,
    var_names: list[str] | None = None,
) -> np.ndarray:
    """
    Render a 2x3 heatmap grid as an RGB frame (numpy array).
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    panels = _compute_correlation_panels(x)
    T, C = x.shape
    if var_names is None:
        var_names = [f"v{i}" for i in range(C)]
    else:
        var_names = list(var_names)[:C]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=120)

    def _imshow(ax, mat: np.ndarray, title: str, *, vmin=None, vmax=None, kind: str):
        im = ax.imshow(mat, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        if kind == "var":
            ax.set_xticks(np.arange(C))
            ax.set_yticks(np.arange(C))
            ax.set_xticklabels(var_names, rotation=90, fontsize=6)
            ax.set_yticklabels(var_names, fontsize=6)
            ax.set_xlabel("Variables", fontsize=9)
            ax.set_ylabel("Variables", fontsize=9)
        else:
            # time-time: keep ticks sparse for readability
            ticks = sorted(set([0, T // 4, T // 2, (3 * T) // 4, T - 1]))
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels([str(t) for t in ticks], fontsize=7)
            ax.set_yticklabels([str(t) for t in ticks], fontsize=7)
            ax.set_xlabel("Time index", fontsize=9)
            ax.set_ylabel("Time index", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _imshow(axes[0, 0], panels["dtw_vars"], "DTW (var-var)", kind="var")
    _imshow(axes[0, 1], panels["cov_vars"], "Cov (var-var)", kind="var")
    _imshow(axes[0, 2], panels["corr_vars"], "Pearson (var-var)", vmin=-1, vmax=1, kind="var")
    _imshow(axes[1, 0], panels["dtw_time"], "DTW (time-time)", kind="time")
    _imshow(axes[1, 1], panels["cov_time"], "Cov (time-time)", kind="time")
    _imshow(axes[1, 2], panels["corr_time"], "Pearson (time-time)", vmin=-1, vmax=1, kind="time")

    if title_prefix:
        fig.suptitle(title_prefix, fontsize=12)

    fig.tight_layout()
    FigureCanvas(fig)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    frame = rgba[..., :3].copy()
    plt.close(fig)
    return frame


def export_batch_correlation_images(
    batch_x,
    out_dir: str,
    max_samples: int = 32,
    corr_n_vars: int = -1,
    var_names: list[str] | None = None,
):
    """
    每个 sample 输出 1 张 2x3 相关性热力图:
    - 上排 3 张: DTW/协方差/皮尔逊 (变量-变量) -> [C, C]
    - 下排 3 张: DTW/协方差/皮尔逊 (时间点-时间点) -> [T, T] (T=seq_len)
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    batch_x = batch_x.numpy()
    bsz = min(batch_x.shape[0], max_samples)

    for i in range(bsz):
        x = batch_x[i]
        if corr_n_vars and corr_n_vars > 0:
            x = x[:, :corr_n_vars]
        frame = _render_2x3_heatmaps(x, title_prefix=f"sample={i}", var_names=var_names)
        plt.imsave(out_path / f"sample_{i:02d}.png", frame)


def export_batch_patch_correlation_videos(
    batch_x,
    out_dir: str,
    patch_size: int = 24,
    stride: int = 12,
    fps: int = 4,
    max_samples: int = 32,
    corr_n_vars: int = -1,
    var_names: list[str] | None = None,
    ffmpeg_path: str = "ffmpeg",
):
    """
    视频: 对每个 sample 的 X=[T,C] 做滑窗 patch，patch 内生成 2x3 相关性热力图作为一帧。
    T=seq_len, C=n_vars (可从数据集推断); 你指定 patch_size=24, stride=12。
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    batch_x = batch_x.numpy()
    bsz, seq_len, _ = batch_x.shape
    bsz = min(bsz, max_samples)

    def _resolve_ffmpeg_path(path: str) -> str:
        if path and path.lower() != "auto":
            return path
        try:
            import imageio_ffmpeg  # type: ignore

            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "ffmpeg not found. Install `imageio[ffmpeg]` or pass `--ffmpeg_path` "
                "as 'ffmpeg' (on PATH) or a full path to ffmpeg.exe"
            ) from e

    def _write_mp4_with_ffmpeg(frames: list[np.ndarray], mp4_path: Path) -> None:
        if not frames:
            raise ValueError("No frames to write.")
        h, w, c = frames[0].shape
        if c != 3:
            raise ValueError(f"Expected RGB frames with 3 channels, got shape={frames[0].shape}")

        ffmpeg_exe = _resolve_ffmpeg_path(ffmpeg_path)
        cmd = [
            ffmpeg_exe,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{w}x{h}",
            "-r",
            str(int(fps)),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(mp4_path),
        ]

        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            assert proc.stdin is not None
            for frame in frames:
                proc.stdin.write(frame.astype(np.uint8, copy=False).tobytes())

            proc.stdin.close()
            proc.stdin = None  # 关键：防止 communicate() flush 已关闭的 stdin
            _, err = proc.communicate()
        finally:
            if proc.poll() is None:
                proc.kill()
    for i in range(bsz):
        frames: list[np.ndarray] = []
        x = batch_x[i]
        if corr_n_vars and corr_n_vars > 0:
            x = x[:, :corr_n_vars]

        # number of frames = floor((seq_len - patch_size) / stride) + 1
        # with seq_len=96, patch_size=24, stride=12 -> 7 frames
        for start in range(0, seq_len - patch_size + 1, stride):
            patch = x[start : start + patch_size, :]
            frame = _render_2x3_heatmaps(
                patch,
                title_prefix=f"sample={i} patch=[{start}:{start+patch_size}]",
                var_names=var_names,
            )
            frames.append(frame)

        video_path = out_path / f"sample_{i:02d}.mp4"
        _write_mp4_with_ffmpeg(frames, video_path)
