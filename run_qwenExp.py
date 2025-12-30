import argparse
import os
import torch

from exp.exp_main import Exp_Main
from utils.tools import seed_everything
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

def get_args():
    parser = argparse.ArgumentParser(description="Qwen3-VL time-series multimodal pipeline")

    # basic config
    parser.add_argument("--root_path", type=str, default=".", help="root path of data file")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv", help="data file name")
    parser.add_argument("--output_dir", type=str, default="./first_batch_artifacts", help="output directory")
    parser.add_argument("--qwen_dir", type=str, default="Qwen3-VL-2B-Instruct", help="Qwen3-VL model directory")

    # forecasting task
    parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
    parser.add_argument("--pred_len", type=int, default=96, help="prediction sequence length")
    parser.add_argument("--patch_len", type=int, default=16, help="patch length")
    parser.add_argument("--stride", type=int, default=16, help="patch stride")
    parser.add_argument("--num_vars", type=int, default=-1, help="number of variables (-1 to infer from data)")

    # data loader
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="data loader workers")
    parser.add_argument("--train_ratio", type=float, default=0.70, help="train split ratio")
    parser.add_argument("--val_ratio", type=float, default=0.10, help="val split ratio")
    parser.add_argument("--test_ratio", type=float, default=0.20, help="test split ratio")

    # stats
    parser.add_argument("--dtw_band", type=int, default=8, help="DTW Sakoe-Chiba band")
    parser.add_argument("--dtw_eps", type=float, default=1e-8, help="DTW epsilon for z-norm")
    parser.add_argument("--dtw_tau", type=float, default=1.0, help="DTW similarity temperature")
    parser.add_argument("--cov_clip_lo", type=float, default=1.0, help="covariance clip low percentile")
    parser.add_argument("--cov_clip_hi", type=float, default=99.0, help="covariance clip high percentile")

    # vision
    parser.add_argument("--vision_mb", type=int, default=8, help="vision micro-batch size")
    parser.add_argument("--video_frames", type=int, default=12, help="video frames per sample")
    parser.add_argument("--embed_dim", type=int, default=2048, help="embedding dimension")

    # GPU
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--use_gpu", type=bool, default=True, help="use GPU if available")
    parser.add_argument("--gpu", type=int, default=0, help="gpu device index")

    return parser.parse_args()


def main():
    args = get_args()
    args.root_path = os.path.abspath(args.root_path)
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    seed_everything(args.seed)

    print("Args in experiment:")
    print(args)

    exp = Exp_Main(args)
    exp.run()


if __name__ == "__main__":
    main()
