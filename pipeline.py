from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from data_provider.data_factory import data_provider
from data_provider.heatmap import save_heatmaps, save_videos
from data_provider.patch import patchify
from qwen.token_stats import QwenTokenCounter
from qwen.vector_export import QwenVectorExporter


@dataclass
class PrepConfig:
    root_path: str = "data"
    data_path: str = "ETTh1.csv"
    features: str = "M"
    target: str = "OT"
    freq: str = "h"

    seq_len: int = 96
    label_len: int = 48
    pred_len: int = 96
    window_stride: int | None = None
    batch_size: int = 32
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    num_workers: int = 0

    patch_len: int = 16
    heatmap_mode: str = "both"
    max_batches: int = 1

    output_root: str = "artifacts/prepared"
    patch_scale: int = 32
    save_videos: bool = True
    video_fps: int = 4
    ffmpeg_path: str = "ffmpeg"

    qwen_model_dir: str = "Qwen3-VL-2B-Instruct"
    max_time_series_tokens: int = 2048


def write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_prompt(path: Path, stats: dict[str, Any]) -> None:
    """
    Generate a compact prompt that stops after the global stats line.
    """
    lines: list[str] = []
    lines.append("这是一个电力时间序列数据集，按小时采样，共有多个变量通道。")
    lines.append("任务：利用过去 96 个时间步去分析/预判未来 96 个时间步的走势或风险。")
    lines.append("")
    lines.append(f"- shape: {stats.get('shape')}")
    glob = stats.get("global", {})
    lines.append(
        f"- 全局 min/max/mean/var: {glob.get('min')}, {glob.get('max')}, {glob.get('mean')}, {glob.get('var')}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _save_numpy_arrays(batch_dir: Path, seq_x: np.ndarray, seq_y: np.ndarray, seq_x_mark: np.ndarray, seq_y_mark: np.ndarray) -> None:
    np.save(batch_dir / "encoder.npy", seq_x)
    np.save(batch_dir / "decoder.npy", seq_y)
    np.save(batch_dir / "encoder_mark.npy", seq_x_mark)
    np.save(batch_dir / "decoder_mark.npy", seq_y_mark)


def _export_one_batch(
    batch_dir: Path,
    batch_x: np.ndarray,
    batch_y: np.ndarray,
    batch_x_mark: np.ndarray,
    batch_y_mark: np.ndarray,
    cfg: PrepConfig,
    exporter: QwenVectorExporter | None,
    prompt_path: Path,
) -> dict[str, Any]:
    batch_dir.mkdir(parents=True, exist_ok=True)
    _save_numpy_arrays(batch_dir, batch_x, batch_y, batch_x_mark, batch_y_mark)

    save_heatmaps(
        batch_x,
        batch_dir / "images",
        mode=cfg.heatmap_mode,
        patch_scale=cfg.patch_scale,
    )

    patches = patchify(batch_x, cfg.patch_len)
    np.save(batch_dir / "patches.npy", patches)

    if cfg.save_videos:
        save_videos(
            patches,
            batch_dir / "videos",
            mode=cfg.heatmap_mode,
            patch_scale=cfg.patch_scale,
            fps=cfg.video_fps,
            ffmpeg_path=cfg.ffmpeg_path,
        )

    shapes = {
        "encoder_shape": list(batch_x.shape),
        "decoder_shape": list(batch_y.shape),
        "encoder_mark_shape": list(batch_x_mark.shape),
        "decoder_mark_shape": list(batch_y_mark.shape),
        "patches_shape": list(patches.shape),
        "patch_len": cfg.patch_len,
    }
    write_json(shapes, batch_dir / "shapes.json")

    vectors: dict[str, Any] | None = None
    if exporter:
        try:
            vectors = exporter.export_batch(
                batch_dir=batch_dir,
                batch_x=batch_x,
                prompt_path=prompt_path,
                images_dir=batch_dir / "images",
                videos_dir=batch_dir / "videos",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] failed to export vectors for {batch_dir}: {exc}")

    return {
        "dir": batch_dir.as_posix(),
        "shapes": shapes,
        "images_dir": (batch_dir / "images").as_posix(),
        "videos_dir": (batch_dir / "videos").as_posix() if cfg.save_videos else None,
        "time_series_path": (batch_dir / "encoder.npy").as_posix(),
        "patches_path": (batch_dir / "patches.npy").as_posix(),
        "vectors_dir": vectors.get("dir") if vectors else None,
    }


def _export_split(split: str, loader, cfg: PrepConfig, prompt_path: Path) -> dict[str, Any]:
    split_root = Path(cfg.output_root) / split
    split_root.mkdir(parents=True, exist_ok=True)

    exporter: QwenVectorExporter | None = None
    try:
        exporter = QwenVectorExporter(cfg.qwen_model_dir, device_map="auto")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] vector exporter init failed: {exc}")
        exporter = None

    saved_batches: list[dict[str, Any]] = []
    for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
        if cfg.max_batches is not None and batch_idx >= cfg.max_batches:
            break
        batch_np = [arr.detach().cpu().numpy() for arr in (batch_x, batch_y, batch_x_mark, batch_y_mark)]
        summary = _export_one_batch(
            split_root / f"batch_{batch_idx:04d}",
            batch_np[0],
            batch_np[1],
            batch_np[2],
            batch_np[3],
            cfg,
            exporter,
            prompt_path,
        )
        saved_batches.append(summary)

    return {
        "split": split,
        "root": split_root.as_posix(),
        "saved_batches": saved_batches,
    }


def _maybe_pick_first_image(split_summary: dict[str, Any]) -> Path | None:
    for batch in split_summary.get("saved_batches", []):
        img_dir = batch.get("images_dir")
        if not img_dir:
            continue
        # prefer wide then rgb
        wide_dir = Path(img_dir) / "wide"
        rgb_dir = Path(img_dir) / "rgb"
        candidates = []
        if wide_dir.exists():
            candidates.extend(sorted(wide_dir.rglob("*.png")))
        if rgb_dir.exists():
            candidates.extend(sorted(rgb_dir.rglob("*.png")))
        if candidates:
            return candidates[0]
    return None


def _maybe_pick_first_video(split_summary: dict[str, Any]) -> Path | None:
    for batch in split_summary.get("saved_batches", []):
        vid_dir = batch.get("videos_dir")
        if not vid_dir:
            continue
        mp4s = sorted(Path(vid_dir).rglob("*.mp4"))
        if mp4s:
            return mp4s[0]
    return None


def _maybe_pick_first_series(split_summary: dict[str, Any]) -> Path | None:
    for batch in split_summary.get("saved_batches", []):
        ts_path = batch.get("time_series_path")
        if ts_path and Path(ts_path).exists():
            return Path(ts_path)
    return None


def compute_token_report(cfg: PrepConfig, prompt_path: Path, split_summaries: dict[str, Any]) -> dict[str, Any]:
    """
    Token counts based on real model outputs (length = hidden_states rows).
    Uses the first saved train batch vectors.
    """
    report: dict[str, Any] = {}
    errors: dict[str, str] = {}

    train_summary = split_summaries.get("train") or {}
    batch0 = (train_summary.get("saved_batches") or [None])[0]
    if not batch0:
        errors["train"] = "no train batches saved"
        report["errors"] = errors
        return report

    vec_dir = batch0.get("vectors_dir")
    if not vec_dir:
        errors["vectors"] = "vector export missing"
        report["errors"] = errors
        return report

    def _load_len(path: Path) -> int | None:
        if not path.exists():
            return None
        arr = np.load(path)
        return int(arr.shape[0])

    try:
        report["text_prompt"] = {"tokens": _load_len(Path(vec_dir) / "prompt_tokens.npy"), "dim": 2048}
    except Exception as exc:  # noqa: BLE001
        errors["text_prompt"] = str(exc)
    try:
        report["time_series"] = {"tokens": _load_len(Path(vec_dir) / "timeseries_tokens.npy"), "dim": 2048}
    except Exception as exc:  # noqa: BLE001
        errors["time_series"] = str(exc)
    try:
        report["images"] = {"tokens": _load_len(Path(vec_dir) / "image_tokens.npy"), "dim": 2048}
    except Exception as exc:  # noqa: BLE001
        errors["images"] = str(exc)
    try:
        report["videos"] = {"tokens": _load_len(Path(vec_dir) / "video_tokens.npy"), "dim": 2048}
    except Exception as exc:  # noqa: BLE001
        errors["videos"] = str(exc)

    if errors:
        report["errors"] = errors

    return report


def run(cfg: PrepConfig) -> dict:
    output_root = Path(cfg.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    stats_path = output_root / "text" / "stats.json"
    prompt_path = output_root / "text" / "prompt.txt"

    split_summaries: dict[str, Any] = {}
    stats_written = False
    for split in ("train", "val"):
        dataset, loader = data_provider(cfg, flag=split)
        meta = dataset.meta()
        if not stats_written:
            write_json(meta.stats, stats_path)
            write_prompt(prompt_path, meta.stats)
            stats_written = True
        split_summaries[split] = _export_split(split, loader, cfg, prompt_path)

    token_report: dict[str, Any] | None = None
    token_report = compute_token_report(cfg, prompt_path, split_summaries)
    write_json(token_report, output_root / "text" / "token_report.json")

    return {
        "output_root": output_root.as_posix(),
        "stats_path": stats_path.as_posix(),
        "prompt_path": prompt_path.as_posix(),
        "splits": split_summaries,
        "token_report": token_report,
    }
