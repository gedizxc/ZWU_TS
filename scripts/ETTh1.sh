#!/usr/bin/env bash
set -e

# Data-prep only: normalize ETTh1, emit stats + heatmaps + patches for one batch.

python -u run.py \
  --root_path data \
  --data_path ETTh1.csv \
  --features M \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --batch_size 32 \
  --patch_len 16 \
  --max_batches 1 \
  --window_stride 96 \
  --heatmap_mode both \
  --patch_scale 32 \
  --output_root artifacts/prepared
