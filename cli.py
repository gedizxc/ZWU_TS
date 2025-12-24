
from __future__ import annotations

import argparse

from pipeline import PrepConfig, run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare ETTh1 data for Qwen3-VL style inputs")
    parser.add_argument("--root_path", type=str, default="data")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv")
    parser.add_argument("--features", type=str, default="M", choices=["M", "S"])
    parser.add_argument("--target", type=str, default="OT")

    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=48, help="decoder context length (Informer style)")
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--window_stride", type=int, default=None, help="stride between windows; default=seq_len")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--output_root", type=str, default="artifacts/prepared")
    parser.add_argument("--heatmap_mode", type=str, default="both", choices=["wide", "rgb", "both"])
    parser.add_argument("--max_batches", type=int, default=1, help="how many batches to save per split")
    parser.add_argument("--patch_scale", type=int, default=32, help="enlarge each correlation block for patch alignment")
    parser.add_argument("--save_videos", action="store_true", default=True, help="save per-sample patch videos")
    parser.add_argument("--no_save_videos", action="store_false", dest="save_videos", help="disable video saving")
    parser.add_argument("--video_fps", type=int, default=4, help="fps for generated videos")
    parser.add_argument("--ffmpeg_path", type=str, default="ffmpeg", help="path to ffmpeg executable")
    parser.add_argument("--qwen_model_dir", type=str, default="Qwen3-VL-2B-Instruct", help="local Qwen3-VL checkpoint dir")
    parser.add_argument(
        "--max_time_series_tokens",
        type=int,
        default=2048,
        help="truncate time-series->text serialization when counting tokens",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> PrepConfig:
    args = build_parser().parse_args(argv)
    return PrepConfig(
        root_path=args.root_path,
        data_path=args.data_path,
        features=args.features,
        target=args.target,
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        window_stride=args.window_stride,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        patch_len=args.patch_len,
        output_root=args.output_root,
        heatmap_mode=args.heatmap_mode,
        max_batches=args.max_batches,
        patch_scale=args.patch_scale,
        save_videos=args.save_videos,
        video_fps=args.video_fps,
        ffmpeg_path=args.ffmpeg_path,
        qwen_model_dir=args.qwen_model_dir,
        max_time_series_tokens=args.max_time_series_tokens,
    )


def main(argv: list[str] | None = None) -> None:
    cfg = parse_args(argv)
    run(cfg)


if __name__ == "__main__":
    main()
