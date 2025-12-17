import argparse
import os

# Windows + (torch/matplotlib) can trigger "libiomp5md.dll already initialized".
# This opt-in makes the script runnable in such environments.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd

from exp.exp_informer import Exp_Informer


def _infer_io_dims(args) -> int:
    data_file = os.path.join(args.root_path, args.data_path)
    if not os.path.exists(data_file):
        # common confusion: user says etth1.csv but file is ETTh1.csv
        alt = os.path.join(args.root_path, "ETTh1.csv")
        if os.path.exists(alt):
            data_file = alt
        else:
            raise FileNotFoundError(f"Dataset not found under {args.root_path!r}")

    df = pd.read_csv(data_file, nrows=1)
    n_vars = len([c for c in df.columns if c != "date"])
    if args.features == "S":
        n_vars = 1
    return n_vars


def get_args():
    parser = argparse.ArgumentParser(description="ZWU_TS - Informer-style pipeline")

    # dataset
    parser.add_argument("--data", type=str, default="ETTh1")
    parser.add_argument("--root_path", type=str, default="data")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv")
    parser.add_argument("--features", type=str, default="M", choices=["M", "S"])
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--freq", type=str, default="h")

    # forecasting length (96 -> 96)
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=96)

    # model (kept in Informer style)
    parser.add_argument("--enc_in", type=int, default=-1)
    parser.add_argument("--dec_in", type=int, default=-1)
    parser.add_argument("--c_out", type=int, default=-1)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embed", type=str, default="timeF", choices=["timeF", "fixed"])

    # training
    parser.add_argument("--is_training", type=int, default=1)
    parser.add_argument("--train_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)  # Informer: batch_size=32
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_step", type=int, default=20)

    # gpu
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)

    # batch -> images/videos (gated)
    parser.add_argument("--visual_root", type=str, default="artifacts/batch_visuals")

    # 默认只生成第一批；如果你把它改大，会生成很多图片/视频，耗时很长
    parser.add_argument("--gen_images_batches", type=int, default=1)
    parser.add_argument("--gen_videos_batches", type=int, default=1)

    parser.add_argument("--patch_size", type=int, default=24)  # patch=24
    parser.add_argument("--patch_stride", type=int, default=12)  # stride=12
    parser.add_argument("--video_fps", type=int, default=4)
    # MP4 encoding uses ffmpeg.
    # - default "auto": try to locate bundled ffmpeg from `imageio-ffmpeg`
    # - or pass "ffmpeg" if it's on PATH
    # - or pass full path to ffmpeg.exe
    parser.add_argument("--ffmpeg_path", type=str, default="auto")

    # correlation heatmap config
    # 默认=-1: 使用数据集全部变量数 (例如 ETTh1 是 7 个变量)
    # 你也可以手动限制只用前 N 个变量来做相关性图（加速/对齐实验）
    parser.add_argument("--corr_n_vars", type=int, default=-1)

    return parser.parse_args()


def main():
    args = get_args()

    # infer ETTh1 variable count and fill Informer-like io dims
    n_vars = _infer_io_dims(args)
    if args.enc_in <= 0:
        args.enc_in = n_vars
    if args.dec_in <= 0:
        args.dec_in = n_vars
    if args.c_out <= 0:
        args.c_out = n_vars
    print(f"[data] inferred n_vars={n_vars} -> enc_in/dec_in/c_out={args.enc_in}")

    setting = f"{args.data}_sl{args.seq_len}_pl{args.pred_len}_bs{args.batch_size}"
    exp = Exp_Informer(args)

    if args.is_training:
        exp.train(setting)
    else:
        exp.test(setting)


if __name__ == "__main__":
    main()
