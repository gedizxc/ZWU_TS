$ErrorActionPreference = "Stop"

# Informer2020-style PowerShell script (ETTh1, 96->96, batch_size=32)
# Default: only export the first batch (32 images + 32 videos). Increase carefully; it is slow.

python -u run.py `
  --is_training 1 `
  --data ETTh1 `
  --root_path data `
  --data_path ETTh1.csv `
  --features M `
  --target OT `
  --freq h `
  --seq_len 96 `
  --label_len 48 `
  --pred_len 96 `
  --batch_size 32 `
  --train_epochs 1 `
  --gen_images_batches 1 `
  --gen_videos_batches 1 `
  --patch_size 24 `
  --patch_stride 12 `
  --ffmpeg_path auto `
  --corr_n_vars -1
