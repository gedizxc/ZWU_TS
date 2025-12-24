"""
Informer-style data prep + Qwen token report entrypoint.
Sets OpenMP env flags to avoid duplicate runtime errors on Windows (libiomp5md.dll).
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from cli import main

if __name__ == "__main__":
    main()
