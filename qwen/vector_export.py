from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def _first_image(images_dir: Path) -> Image.Image | None:
    if not images_dir.exists():
        return None
    wide_dir = images_dir / "wide"
    rgb_dir = images_dir / "rgb"
    candidates = []
    if wide_dir.exists():
        candidates.extend(sorted(wide_dir.rglob("*.png")))
    if rgb_dir.exists():
        candidates.extend(sorted(rgb_dir.rglob("*.png")))
    if not candidates:
        return None
    return Image.open(candidates[0]).convert("RGB")


def _first_video(videos_dir: Path) -> np.ndarray | None:
    if not videos_dir or not videos_dir.exists():
        return None
    wide_dir = videos_dir / "wide"
    rgb_dir = videos_dir / "rgb"
    candidates = []
    if wide_dir.exists():
        candidates.extend(sorted(wide_dir.rglob("*.mp4")))
    if rgb_dir.exists():
        candidates.extend(sorted(rgb_dir.rglob("*.mp4")))
    if not candidates:
        return None
    try:
        import imageio.v3 as iio
    except Exception:
        return None
    return iio.imread(candidates[0], index=None)


def _vision_token_count(grid: Any, merge_size: int) -> int:
    if grid is None:
        return 0
    try:
        import torch
    except Exception:
        torch = None  # type: ignore

    if torch is not None and isinstance(grid, torch.Tensor):
        arr = grid.detach().cpu().numpy()
    else:
        arr = np.array(grid)
    if arr.ndim > 1:
        arr = arr[0]
    try:
        return int(np.prod(arr) // (merge_size**2))
    except Exception:
        return 0


@dataclass
class QwenVectorExporter:
    model_dir: str
    device_map: str = "auto"

    def __post_init__(self) -> None:
        self.processor = AutoProcessor.from_pretrained(
            self.model_dir,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_dir,
            dtype=torch.bfloat16,
            device_map=self.device_map,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _encode_text(self, text: str) -> np.ndarray:
        inputs = self.processor.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden = out.hidden_states[-1]  # [B, T, dim]
        return hidden[0].detach().cpu().float().numpy()

    def _encode_image(self, img: Image.Image) -> np.ndarray:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "."},
                ],
            }
        ]
        payload = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )
        token_cap = _vision_token_count(payload.get("image_grid_thw"), getattr(self.processor.image_processor, "merge_size", 2))
        payload = {k: v.to(self.model.device) for k, v in payload.items()}
        with torch.no_grad():
            out = self.model(
                **payload,
                output_hidden_states=True,
                output_vision_hidden_states=True,
                return_dict=True,
            )
        vision = getattr(out, "vision_hidden_states", None)
        if vision is not None and len(vision) > 0:
            vec = vision[-1][0].detach().cpu().float().numpy()
            return vec[:token_cap] if token_cap and token_cap <= vec.shape[0] else vec
        hidden = out.hidden_states[-1]  # [B, S, dim]
        mm_mask = getattr(out, "mm_token_type_ids", None)
        if mm_mask is not None and (mm_mask[0] == 1).any():
            return hidden[0][mm_mask[0] == 1].detach().cpu().float().numpy()
        return hidden[0].detach().cpu().float().numpy()

    def _encode_video(self, frames: np.ndarray) -> np.ndarray:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames},
                    {"type": "text", "text": "."},
                ],
            }
        ]
        payload = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            videos_kwargs={"do_sample_frames": False, "return_metadata": True},
        )
        token_cap = _vision_token_count(payload.get("video_grid_thw"), getattr(self.processor.video_processor, "merge_size", 2))
        payload = {k: v.to(self.model.device) for k, v in payload.items()}
        with torch.no_grad():
            out = self.model(
                **payload,
                output_hidden_states=True,
                output_vision_hidden_states=True,
                return_dict=True,
            )
        vision = getattr(out, "vision_hidden_states", None)
        if vision is not None and len(vision) > 0:
            vec = vision[-1][0].detach().cpu().float().numpy()
            return vec[:token_cap] if token_cap and token_cap <= vec.shape[0] else vec
        hidden = out.hidden_states[-1]
        mm_mask = getattr(out, "mm_token_type_ids", None)
        if mm_mask is not None and (mm_mask[0] == 2).any():
            return hidden[0][mm_mask[0] == 2].detach().cpu().float().numpy()
        return hidden[0].detach().cpu().float().numpy()

    def export_batch(
        self,
        batch_dir: Path,
        batch_x: np.ndarray,
        prompt_path: Path | None,
        images_dir: Path | None,
        videos_dir: Path | None,
    ) -> dict[str, Any]:
        vec_dir = batch_dir / "vectors"
        vec_dir.mkdir(parents=True, exist_ok=True)

        # Time series -> text -> tokens
        ts_text = np.array2string(batch_x[0], precision=4, separator=",")
        ts_vec = self._encode_text(ts_text)
        np.save(vec_dir / "timeseries_tokens.npy", ts_vec)

        # Prompt text tokens
        if prompt_path and prompt_path.exists():
            prompt = prompt_path.read_text(encoding="utf-8")
            prompt_vec = self._encode_text(prompt)
            np.save(vec_dir / "prompt_tokens.npy", prompt_vec)

        # Image tokens
        img = _first_image(images_dir) if images_dir else None
        if img:
            img_vec = self._encode_image(img)
            np.save(vec_dir / "image_tokens.npy", img_vec)

        # Video tokens
        vid = _first_video(videos_dir) if videos_dir else None
        if vid is not None:
            vid_vec = self._encode_video(vid)
            np.save(vec_dir / "video_tokens.npy", vid_vec)

        return {
            "dir": vec_dir.as_posix(),
            "prompt_tokens": str((vec_dir / "prompt_tokens.npy").as_posix()) if (vec_dir / "prompt_tokens.npy").exists() else None,
            "timeseries_tokens": str((vec_dir / "timeseries_tokens.npy").as_posix()),
            "image_tokens": str((vec_dir / "image_tokens.npy").as_posix()) if (vec_dir / "image_tokens.npy").exists() else None,
            "video_tokens": str((vec_dir / "video_tokens.npy").as_posix()) if (vec_dir / "video_tokens.npy").exists() else None,
        }
