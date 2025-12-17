# ZWU_TS (Informer2020-style project layout)

This repo is refactored to follow the **Informer2020** project structure and data split style.
Dataset remains `data/ETTh1.csv` (ETTh1 hourly).

```
data/
  ETTh1.csv
data_provider/
  data_loader.py          # Dataset_ETT_hour (Informer split)
  data_factory.py         # build Dataset + DataLoader
exp/
  exp_basic.py
  exp_informer.py         # training/test loop (Informer style)
models/
  model.py                # SimpleInformer (lightweight)
utils/
  timefeatures.py         # time features (Informer style)
  visualize.py            # per-batch image/video exporters
main.py                   # argparse entry (Informer style)
artifacts/
  batch_visuals/          # generated outputs (images/videos)
```

## Requirements
- Python 3.9+
- `numpy`, `pandas`
- `torch`
- For images: `matplotlib`
- For videos: `imageio[ffmpeg]` (or another ffmpeg-backed backend)

## Run (96 -> 96, batch_size=32)
```bash
python run.py --is_training 1 --batch_size 32 --seq_len 96 --pred_len 96
```
If you are using the conda env `qwen3vl`:
```bash
conda run -n qwen3vl python -u run.py --is_training 1 --batch_size 32 --seq_len 96 --pred_len 96
```
At startup it prints dataset split shapes:
- `batch_x (X)` is the **split encoder input** `[B, seq_len, n_features]`
- `batch_y (Y)` is `[B, label_len+pred_len, n_features]`

## Per-batch images / videos (gated)
During training, each batch can export:
- `artifacts/batch_visuals/<setting>/images/batch_0000/` -> **32 PNGs** (每张是 2x3 相关性热力图)
- `artifacts/batch_visuals/<setting>/videos/batch_0000/` -> **32 个 `.mp4` 视频文件** (需要 ffmpeg)

To avoid huge runtime, exporting is gated by these args (default: only first batch):
```bash
python run.py --gen_images_batches 1 --gen_videos_batches 1
```
Patch video params follow your request:
- `--patch_size 24 --patch_stride 12`
MP4 encoding uses ffmpeg:
- Default `--ffmpeg_path auto`: uses bundled ffmpeg from `imageio-ffmpeg` (recommended in conda envs)
- If `ffmpeg` is on PATH: set `--ffmpeg_path ffmpeg`
- If not (common on Windows): pass full path, e.g. `--ffmpeg_path "D:\\ffmpeg\\bin\\ffmpeg.exe"`

Correlation heatmap definition (per sample):
- Top row (var-var, CxC): DTW / Covariance / Pearson; `C` is inferred from dataset (ETTh1 -> 7)
- Bottom row (time-time, TxT): DTW / Covariance / Pearson; `T = seq_len` (default 96)
Axes:
- var-var panels: x/y ticks are **feature names** from the dataset columns (excluding `date`)
- time-time panels: x/y ticks are **time indices** within the encoder window `[0..seq_len-1]`

Optional: limit how many variables participate in correlation plots (uses the first N vars):
```bash
python run.py --corr_n_vars 7
```

## Informer-style scripts
Linux/macOS:
```bash
bash scripts/ETTh1.sh
```
Windows PowerShell:
```powershell
.\scripts\ETTh1.ps1
```
