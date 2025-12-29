import os
from typing import List, Tuple

import torch
from PIL import Image

from configs import hparams
from configs.paths import OUTPUT_DIR, IMAGES_DIR, VIDEOS_DIR
from processing.patchify import patchify_ts
from processing.render import render_grid_7x21_to_image
from processing.stats import compute_three_mats, mats_to_7x21
from utils.io import ensure_dir, torch_save
from utils.logging import print_box


def step_ts_embeddings(x, model, ts_mlp) -> torch.Tensor:
    """5.1) Time-series: patchify -> tokens -> MLP -> ts_embeds"""
    print_box("5.1) Time-series: patchify -> tokens -> MLP -> ts_embeds")
    x_dev = x.to(model.device)
    print(f"[TS] moved to device={x_dev.device}; shape={tuple(x_dev.shape)}")

    ts_tokens = patchify_ts(x_dev, patch_len=hparams.PATCH_LEN, stride=hparams.STRIDE)
    print(f"[TS] patchify: ts_tokens shape={tuple(ts_tokens.shape)} (expect B,42,16)")

    with torch.inference_mode():
        ts_embeds = ts_mlp(ts_tokens)
    print(f"[TS] after MLP: ts_embeds shape={tuple(ts_embeds.shape)} dtype={ts_embeds.dtype} device={ts_embeds.device}")

    ts_embeds_cpu = ts_embeds.detach().cpu()
    torch_save({"ts_embeds": ts_embeds_cpu}, os.path.join(OUTPUT_DIR, "first_batch_ts_embeds.pt"))
    print(f"[Save] ts_embeds -> {os.path.join(OUTPUT_DIR, 'first_batch_ts_embeds.pt')}")
    return ts_embeds_cpu


def step_image_modality(x_np_batch) -> List[Image.Image]:
    """5.2) Image modality: compute 7x21 grid -> render 672x224 -> save first batch images"""
    print_box("5.2) Image modality: compute 7x21 grid -> render 672x224 -> save first batch images")
    images = []
    for i, window in enumerate(x_np_batch):
        dtw, cov, pear = compute_three_mats(window, kind=f"Img sample{i} window{hparams.SEQ_LEN}", dtw_band=hparams.DTW_BAND)
        grid = mats_to_7x21(dtw, cov, pear)
        print(f"[Img sample{i}] grid shape={grid.shape} range=({grid.min():.3f},{grid.max():.3f})")
        img = render_grid_7x21_to_image(grid, cell_pix=hparams.CELL_PIX)
        print(f"[Img sample{i}] rendered size={img.size} (W,H) mode={img.mode}")
        images.append(img)
        img_path = os.path.join(IMAGES_DIR, f"batch0_sample{i:02d}_heatmap.png")
        img.save(img_path)
    print(f"[Save] images saved to {IMAGES_DIR} count={len(images)}")
    return images


def step_video_modality(x_np_batch) -> List[List[Image.Image]]:
    """5.3) Video modality: per patch(16) 7x21 grid -> 6 frames -> duplicate to 12 -> save frames"""
    print_box("5.3) Video modality: per patch(16) 7x21 grid -> 6 frames -> duplicate to 12 -> save frames")
    videos = []
    num_patches = (hparams.SEQ_LEN - hparams.PATCH_LEN) // hparams.STRIDE + 1
    assert num_patches == 6, f"Expected 6 patches, got {num_patches}"

    for i, window in enumerate(x_np_batch):
        frames = []
        for p in range(num_patches):
            start = p * hparams.STRIDE
            patch = window[start:start + hparams.PATCH_LEN]
            dtw, cov, pear = compute_three_mats(
                patch,
                kind=f"Vid sample{i} patch{p} len{hparams.PATCH_LEN}",
                dtw_band=min(hparams.DTW_BAND, hparams.PATCH_LEN),
            )
            grid = mats_to_7x21(dtw, cov, pear)
            frame = render_grid_7x21_to_image(grid, cell_pix=hparams.CELL_PIX)
            frames.append(frame)

        frames12 = []
        for f in frames:
            frames12.append(f)
            frames12.append(f.copy())
        frames12 = frames12[:hparams.VIDEO_FRAMES]
        print(f"[Vid sample{i}] frames raw={len(frames)} duplicated={len(frames12)} frame_size={frames12[0].size}")

        vid_dir = os.path.join(VIDEOS_DIR, f"batch0_sample{i:02d}")
        ensure_dir(vid_dir)
        for fi, fr in enumerate(frames12):
            fr_path = os.path.join(vid_dir, f"frame_{fi:02d}.png")
            fr.save(fr_path)

        videos.append(frames12)

    print(f"[Save] video frames saved to {VIDEOS_DIR} videos={len(videos)}")
    return videos


def step_encode_vision(model, processor, images: List[Image.Image], videos: List[List[Image.Image]]):
    """5.4) Qwen3-VL Vision encoding: images -> img_embeds, videos -> vid_embeds"""
    print_box("5.4) Qwen3-VL Vision encoding: images -> img_embeds, videos -> vid_embeds")
    from models.qwen_encoders import encode_images_qwen3vl, encode_videos_qwen3vl

    img_token_seqs, img_pooled = encode_images_qwen3vl(
        model=model,
        processor=processor,
        images=images,
        device=str(model.device),
        micro_bs=hparams.VISION_MB,
    )
    vid_token_seqs, vid_pooled = encode_videos_qwen3vl(
        model=model,
        processor=processor,
        videos=videos,
        device=str(model.device),
        micro_bs=max(1, hparams.VISION_MB // 4),
    )

    torch_save(
        {
            "img_token_seqs": img_token_seqs,
            "img_pooled": img_pooled,
            "vid_token_seqs": vid_token_seqs,
            "vid_pooled": vid_pooled,
        },
        os.path.join(OUTPUT_DIR, "first_batch_vision_embeds.pt"),
    )
    print(f"[Save] vision embeddings -> {os.path.join(OUTPUT_DIR, 'first_batch_vision_embeds.pt')}")
    return img_token_seqs, img_pooled, vid_token_seqs, vid_pooled


def step_summary(ts_embeds_cpu, img_pooled, vid_pooled, txt_token_embeds, txt_pooled):
    """5.5) Summary (to embeddings only)"""
    print_box("5.5) SUMMARY (to embeddings only)")
    print(f"[Summary] ts_embeds: {tuple(ts_embeds_cpu.shape)}  (B,42,2048)")
    print(f"[Summary] img_pooled: {tuple(img_pooled.shape)}  (B,2048)")
    print(f"[Summary] vid_pooled: {tuple(vid_pooled.shape)}  (B,2048)")
    print(f"[Summary] txt_token_embeds: {tuple(txt_token_embeds.shape)}  (1,L,2048)")
    print(f"[Summary] txt_pooled: {tuple(txt_pooled.shape)}  (2048,)")
