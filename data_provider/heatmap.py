
from __future__ import annotations

from pathlib import Path
import subprocess

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm


def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
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


def compute_var_correlations(window: np.ndarray) -> dict[str, np.ndarray]:
    """
    window: [T, C] -> three CxC matrices.
    """
    window = np.asarray(window, dtype=np.float32)
    _, C = window.shape
    var_series = [window[:, c] for c in range(C)]
    dtw_vars = _dtw_pairwise_1d(var_series)
    cov_vars = np.cov(window, rowvar=False).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        pearson_vars = np.corrcoef(window, rowvar=False).astype(np.float32)
    pearson_vars = np.nan_to_num(pearson_vars, nan=0.0, posinf=0.0, neginf=0.0)
    return {"dtw": dtw_vars, "cov": cov_vars, "pearson": pearson_vars}


def _normalize_matrix(mat: np.ndarray, symmetric: bool = False) -> np.ndarray:
    mat = np.nan_to_num(mat.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if symmetric:
        mat = (mat + mat.T) / 2
    vmin = float(mat.min())
    vmax = float(mat.max())
    if vmax - vmin < 1e-8:
        return np.zeros_like(mat, dtype=np.float32)
    return (mat - vmin) / (vmax - vmin)


def _enlarge(mat: np.ndarray, scale: int) -> np.ndarray:
    return np.repeat(np.repeat(mat, scale, axis=0), scale, axis=1)


def save_heatmaps(
    batch_x: np.ndarray,
    out_dir: Path,
    mode: str = "both",
    patch_scale: int = 32,
) -> None:
    """
    mode: wide | rgb | both
    wide  -> concat dtw/cov/pearson along width, output size [C, C*3]
    rgb   -> stack into 3 channels at [C, C, 3]
    """
    def _enlarge(mat: np.ndarray) -> np.ndarray:
        return np.repeat(np.repeat(mat, patch_scale, axis=0), patch_scale, axis=1)

    out_dir.mkdir(parents=True, exist_ok=True)
    wide_dir = out_dir / "wide"
    rgb_dir = out_dir / "rgb"
    if mode in {"wide", "both"}:
        wide_dir.mkdir(parents=True, exist_ok=True)
    if mode in {"rgb", "both"}:
        rgb_dir.mkdir(parents=True, exist_ok=True)

    for idx, window in enumerate(batch_x):
        corrs = compute_var_correlations(window)
        dtw = _normalize_matrix(corrs["dtw"], symmetric=True)
        cov = _normalize_matrix(corrs["cov"], symmetric=True)
        pearson = (np.nan_to_num(corrs["pearson"], nan=0.0, posinf=0.0, neginf=0.0) + 1.0) / 2.0
        pearson = np.clip(pearson, 0.0, 1.0).astype(np.float32)

        if mode in {"wide", "both"}:
            wide = np.concatenate([dtw, cov, pearson], axis=1)
            wide_scaled = _enlarge(wide)
            plt.imsave(wide_dir / f"sample_{idx:02d}_wide.png", wide_scaled, cmap="magma")

        if mode in {"rgb", "both"}:
            rgb = np.stack([dtw, cov, pearson], axis=-1)
            rgb_scaled = np.stack([_enlarge(rgb[..., c]) for c in range(3)], axis=-1)
            plt.imsave(rgb_dir / f"sample_{idx:02d}_rgb.png", rgb_scaled)


def save_videos(
    patches: np.ndarray,
    out_dir: Path,
    mode: str = "both",
    patch_scale: int = 32,
    fps: int = 4,
    ffmpeg_path: str = "ffmpeg",
) -> None:
    """
    Create videos over patch sequence; one frame per patch.
    patches: [B, n_patches, patch_len, C] (normalized values)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    wide_dir = out_dir / "wide"
    rgb_dir = out_dir / "rgb"
    if mode in {"wide", "both"}:
        wide_dir.mkdir(parents=True, exist_ok=True)
    if mode in {"rgb", "both"}:
        rgb_dir.mkdir(parents=True, exist_ok=True)

    magma = cm.get_cmap("magma")

    def _resolve_ffmpeg(path: str) -> str:
        # Try explicit path or PATH
        import shutil

        if path and path.lower() != "auto":
            resolved = shutil.which(path) if Path(path).name == path else path
            if resolved and Path(resolved).exists():
                return resolved
        # Try imageio-ffmpeg if available
        try:
            import imageio_ffmpeg  # type: ignore

            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass
        raise FileNotFoundError(
            "ffmpeg executable not found. Install ffmpeg, add to PATH, "
            "or set --ffmpeg_path to the full executable path."
        )

    def _write_mp4(frames: list[np.ndarray], path: Path) -> None:
        if not frames:
            return
        h, w, c = frames[0].shape
        path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg_exe = _resolve_ffmpeg(ffmpeg_path)
        proc = subprocess.Popen(
            [
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
                str(path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert proc.stdin is not None
        for frame in frames:
            if frame.dtype != np.uint8:
                buf = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
            else:
                buf = frame
            proc.stdin.write(buf.tobytes())
        proc.stdin.close()
        proc.communicate()

    for idx, sample in enumerate(patches):
        frames_wide: list[np.ndarray] = []
        frames_rgb: list[np.ndarray] = []
        for patch in sample:
            corrs = compute_var_correlations(patch)
            dtw = _normalize_matrix(corrs["dtw"], symmetric=True)
            cov = _normalize_matrix(corrs["cov"], symmetric=True)
            pearson = (np.nan_to_num(corrs["pearson"], nan=0.0, posinf=0.0, neginf=0.0) + 1.0) / 2.0
            pearson = np.clip(pearson, 0.0, 1.0).astype(np.float32)

            if mode in {"wide", "both"}:
                wide = np.concatenate([dtw, cov, pearson], axis=1)
                wide_scaled = _enlarge(wide, patch_scale)
                wide_rgb = magma(wide_scaled)[..., :3].astype(np.float32)
                frames_wide.append(wide_rgb)

            if mode in {"rgb", "both"}:
                rgb = np.stack([dtw, cov, pearson], axis=-1)
                rgb_scaled = np.stack([_enlarge(rgb[..., c], patch_scale) for c in range(3)], axis=-1)
                frames_rgb.append(rgb_scaled.astype(np.float32))

        if mode in {"wide", "both"}:
            _write_mp4(frames_wide, wide_dir / f"sample_{idx:02d}.mp4")
        if mode in {"rgb", "both"}:
            _write_mp4(frames_rgb, rgb_dir / f"sample_{idx:02d}.mp4")
