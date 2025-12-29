import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from pipelines.batch_pipeline import run_batch_pipeline
from utils.seed import seed_everything


def main():
    seed_everything(42)
    run_batch_pipeline()


if __name__ == "__main__":
    main()
