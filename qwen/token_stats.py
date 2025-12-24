from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image
from transformers import AutoConfig, AutoProcessor


def _grid_tokens(payload: Any, key: str, patch_size: int, merge_size: int) -> int:
    """
    Extract token count from processor batch output. Falls back to pixel grid size.
    """
    if payload is None:
        return 0
    if hasattr(payload, "data"):  # BatchFeature -> dict-like
        payload = payload.data
    if not isinstance(payload, dict):
        return 0

    grid = payload.get(key)
    if grid is not None:
        arr = np.array(grid)
        # pick first sample if batched
        if arr.ndim > 1:
            arr = arr[0]
        try:
            return int(np.prod(arr) // (merge_size**2))
        except Exception:
            return 0

    pixel_values = payload.get("pixel_values")
    if pixel_values is None:
        return 0

    shape = tuple(pixel_values.shape)
    if len(shape) == 4:
        _, _, h, w = shape
        frames = 1
    elif len(shape) == 5:
        _, frames, _, h, w = shape
    else:
        return 0

    tokens_per_frame = (h // patch_size) * (w // patch_size) // (merge_size**2)
    return int(tokens_per_frame * frames)


def _describe(tokens: int, dim: int) -> dict[str, Any]:
    return {"tokens": tokens, "dim": dim, "matrix": f"{tokens}x{dim}"}


def _fallback_image_tokens(images: Sequence[Image.Image], patch_size: int) -> int:
    tokens = 0
    for img in images:
        w, h = img.size
        tokens += (h // patch_size) * (w // patch_size)
    return int(tokens)


def _fallback_video_tokens(frames: np.ndarray, patch_size: int) -> int:
    """
    frames: [F, H, W, C] (imageio) -> tokens = F * floor(H/patch) * floor(W/patch)
    """
    if frames.ndim == 3:
        # Single frame grayscale or RGB
        frames = np.expand_dims(frames, axis=0)
    if frames.ndim != 4:
        return 0
    f, h, w, _ = frames.shape
    return int(f * (h // patch_size) * (w // patch_size))


@dataclass
class QwenTokenCounter:
    model_dir: str

    def __post_init__(self) -> None:
        self.config = AutoConfig.from_pretrained(
            self.model_dir,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_dir,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.text_dim = int(getattr(self.config, "hidden_size", self.config.text_config.hidden_size))
        self.vision_dim = int(self.config.vision_config.out_hidden_size)
        self.merge_size = int(getattr(self.processor.image_processor, "merge_size", 2))
        self.patch_size = int(getattr(self.processor.image_processor, "patch_size", self.config.vision_config.patch_size))
        self.video_kwargs = {"do_sample_frames": False, "return_metadata": True}

    def count_prompt(self, text: str) -> dict[str, Any]:
        ids = self.processor.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=True,
        )["input_ids"]
        tokens = int(ids.shape[-1])
        return _describe(tokens, self.text_dim)

    def count_time_series(self, series: np.ndarray, max_tokens: int) -> dict[str, Any]:
        """
        Serialize the first sample of a batch into text then count text tokens.
        """
        if series.ndim >= 3:
            sample = series[0]
        else:
            sample = series
        flattened = sample.reshape(-1)
        truncated = flattened[:max_tokens]
        text = np.array2string(truncated, precision=4, separator=",")
        ids = self.processor.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=True,
        )["input_ids"]
        tokens = int(ids.shape[-1])
        return _describe(tokens, self.text_dim)

    def count_images(self, images: Sequence[Image.Image]) -> dict[str, Any]:
        if not images:
            return _describe(0, self.vision_dim)
        try:
            payload = self.processor(images=list(images), return_tensors="pt")
            tokens = _grid_tokens(payload, "image_grid_thw", self.patch_size, self.merge_size)
            if tokens == 0:
                tokens = _fallback_image_tokens(images, self.patch_size) // (self.merge_size**2)
        except Exception:
            tokens = _fallback_image_tokens(images, self.patch_size) // (self.merge_size**2)
        return _describe(tokens, self.vision_dim)

    def count_videos(self, videos: Sequence[np.ndarray]) -> dict[str, Any]:
        if not videos:
            return _describe(0, self.vision_dim)
        # Per-frame token count (aligned with image path) to avoid FPS sampling surprises.
        first = videos[0]
        arr = np.asarray(first)
        if arr.ndim == 3:
            frames = 1
            first_frame = arr
        elif arr.ndim == 4:
            frames = arr.shape[0]
            first_frame = arr[0]
        else:
            frames = 1
            first_frame = arr

        try:
            img = Image.fromarray(first_frame.astype(np.uint8)) if not isinstance(first_frame, Image.Image) else first_frame
            per_frame = self.count_images([img])["tokens"]
            tokens = int(per_frame * frames)
        except Exception:
            tokens = sum(_fallback_video_tokens(v, self.patch_size) for v in videos) // (self.merge_size**2)
        return _describe(tokens, self.vision_dim)

    def count_video_from_file(self, path: str) -> dict[str, Any]:
        """
        Load frames from an mp4 and count tokens. Falls back to image-based counting on the first frame.
        """
        try:
            import imageio.v3 as iio
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("imageio is required for video token counting") from exc

        frames = iio.imread(path, index=None)
        return self.count_videos([frames])
