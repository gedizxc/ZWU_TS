import os

# Base paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), "data")

# Data
ETTH1_CSV_PATH = os.path.join(DATA_DIR, "ETTh1.csv")

# Outputs
OUTPUT_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), "first_batch_artifacts")
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
VIDEOS_DIR = os.path.join(OUTPUT_DIR, "videos")

# Model checkpoint (local)
QWEN_MODEL_ID = "Qwen3-VL-2B-Instruct"
