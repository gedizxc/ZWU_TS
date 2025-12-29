import torch

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data/window
BATCH_SIZE = 32
SEQ_LEN = 96
PRED_LEN = 96

# PatchTST
PATCH_LEN = 16
STRIDE = 16
NUM_VARS = 7

# Image/video heatmap grid
CELL_PIX = 32
RENDER_W = 21 * CELL_PIX  # 672
RENDER_H = 7 * CELL_PIX   # 224
VIDEO_FRAMES = 12  # 6 patches duplicated to 12 frames

# DTW
DTW_BAND = 8
DTW_EPS = 1e-8
DTW_TAU = 1.0

# Covariance clipping percentiles
COV_CLIP_LO = 1.0
COV_CLIP_HI = 99.0

# Vision micro-batch
VISION_MB = 8

# Splits
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20
