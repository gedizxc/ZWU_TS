
"""
Data-prep entry for converting ETTh1 windows into Qwen3-VL-ready assets.
Sets OpenMP env flags to avoid duplicate runtime errors on Windows (libiomp5md.dll).
"""

import os

# Mitigate "OMP: Error #15: Initializing libiomp5md.dll" when multiple runtimes are present.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from cli import main


if __name__ == "__main__":
    main()
