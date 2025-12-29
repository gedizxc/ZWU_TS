import numpy as np
from PIL import Image

from configs import hparams


def render_grid_7x21_to_image(grid: np.ndarray, cell_pix: int = hparams.CELL_PIX) -> Image.Image:
    assert grid.shape == (hparams.NUM_VARS, 3 * hparams.NUM_VARS)
    arr = (np.clip(grid, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    img_arr = np.kron(arr, np.ones((cell_pix, cell_pix), dtype=np.uint8))
    img = Image.fromarray(img_arr, mode="L").convert("RGB")
    return img
