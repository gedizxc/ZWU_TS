ZWU_TS (Informer-style prep for Qwen3-VL)
=========================================

This project now mirrors Informer naming and data flow: read ETTh-style CSVs, split into train/val/test windows shaped `[32, 96, 7]`, export visuals (images/videos), concise text prompt, and serialize the split time series. The generated assets are then summarized with Qwen3-VL token counts so you know the token×vector matrix size for each modality.

What it produces
- Time series: `artifacts/prepared/{train,val,test}/batch_xxxx/encoder.npy` (encoder windows) plus decoder/mark arrays.
- Visuals: correlation heatmap PNGs (wide+RGB) and MP4s per batch, enlarged by 32× for clearer patches.
- Text: compact prompt in `artifacts/prepared/text/prompt.txt` and stats.json.
- Token report: `artifacts/prepared/text/token_report.json` with `{tokens, dim, matrix}` for prompt, time series, images, videos (based on local Qwen3-VL).

Running
```bash
python main.py \
  --root_path data \
  --data_path ETTh1.csv \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --batch_size 32 \
  --patch_len 16 \
  --max_batches 1 \
  --heatmap_mode both \
  --save_videos \
  --patch_scale 32 \
  --qwen_model_dir Qwen3-VL-2B-Instruct
```

Notes
- Data split ratios: 70% train / 10% val / 20% test by default; stride defaults to `seq_len` for non-overlapping windows.
- Normalization: z-score per channel then min-max per channel across the full dataset; prompt text stops at the global min/max/mean/var line as requested.
- Patch/visuals: patch length 16; correlation blocks are enlarged 32× when saving images/videos.
- Qwen token counting uses the local checkpoint (no network). Time series are serialized to text before tokenization; video tokens are counted via `imageio` + processor grid metadata. If counting fails, a warning is printed and the pipeline still saves assets.
