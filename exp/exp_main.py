import json
import os
import time

import numpy as np
import torch

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import (
    load_qwen3_vl,
    build_ts_mlp,
    encode_images_qwen3vl,
    encode_videos_qwen3vl,
    encode_text_global,
)
from utils import print_box, ensure_dir
from utils.render import mats_to_rgb_grid, render_rgb_grid_to_image
from utils.stats import compute_three_mats
from utils.ts import patchify_ts


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        self.model = None
        self.processor = None
        self.cell_pix = None
        self.video_cell_pix = None

    def _prepare_output(self):
        ensure_dir(self.args.output_dir)
        ensure_dir(os.path.join(self.args.output_dir, "images"))
        ensure_dir(os.path.join(self.args.output_dir, "videos"))

    def _load_vision_patch_size(self) -> int:
        config_path = os.path.join(self.args.qwen_dir, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return int(cfg["vision_config"]["patch_size"])

    def _load_data(self):
        print_box("0) Load ETTh1.csv")
        data_set, data_loader = data_provider(self.args, flag="train")

        data = data_set.data_raw
        columns = data_set.columns
        print(f"[Data] raw shape={data.shape} columns={columns}")
        if data.shape[1] != self.args.num_vars:
            print(f"[WARN] Expected {self.args.num_vars} vars but got {data.shape[1]}")

        train_T = int(data.shape[0] * self.args.train_ratio)
        data_norm = data_set.data_norm
        print(f"[Scaler] mean shape={data_set.scaler.mean_.shape}, std shape={data_set.scaler.std_.shape}")
        print(
            f"[Data] norm shape={data_norm.shape}, per-channel mean~{data_norm[:train_T].mean(axis=0)}, "
            f"std~{data_norm[:train_T].std(axis=0)}"
        )
        print_box("1) Build dataset/dataloader")
        print(f"[Dataset] split=train windows={len(data_set)} "
              f"(train={data_set.split_sizes[0]}, val={data_set.split_sizes[1]}, test={data_set.split_sizes[2]})")
        return data_set, data_loader

    def _load_model(self):
        print_box("2) Load Qwen3-VL")
        model, processor = load_qwen3_vl(self.args.qwen_dir)
        print(f"[Model] loaded on device={model.device} dtype={next(model.parameters()).dtype}")
        return model, processor

    def _encode_ts_embeddings(self, x, ts_mlp, expected_tokens):
        print_box("4.1) Time-series: patchify -> tokens -> MLP -> ts_embeds")
        x_dev = x.to(self.model.device)
        print(f"[TS] moved to device={x_dev.device}; shape={tuple(x_dev.shape)}")

        ts_tokens = patchify_ts(x_dev, patch_len=self.args.patch_len, stride=self.args.stride)
        print(
            f"[TS] patchify: ts_tokens shape={tuple(ts_tokens.shape)} "
            f"(expect B,{expected_tokens},{self.args.patch_len})"
        )

        with torch.inference_mode():
            ts_embeds = ts_mlp(ts_tokens)
        print(f"[TS] after MLP: ts_embeds shape={tuple(ts_embeds.shape)} dtype={ts_embeds.dtype} device={ts_embeds.device}")

        ts_embeds_cpu = ts_embeds.detach().cpu()
        torch.save({"ts_embeds": ts_embeds_cpu}, os.path.join(self.args.output_dir, "first_batch_ts_embeds.pt"))
        print(f"[Save] ts_embeds -> {os.path.join(self.args.output_dir, 'first_batch_ts_embeds.pt')}")
        return ts_embeds_cpu

    def _build_image_modality(self, x_np, batch_size):
        print_box(f"4.2) Image modality: compute {self.args.num_vars}x{self.args.num_vars} grid -> render -> save")
        images = []
        for i in range(batch_size):
            window = x_np[i]
            dtw, cov, pear = compute_three_mats(
                window,
                kind=f"Img sample{i} window{self.args.seq_len}",
                dtw_band=self.args.dtw_band,
                dtw_eps=self.args.dtw_eps,
                verbose=(i == 0),
            )
            grid = mats_to_rgb_grid(
                dtw,
                cov,
                pear,
                cov_clip_lo=self.args.cov_clip_lo,
                cov_clip_hi=self.args.cov_clip_hi,
                dtw_tau=self.args.dtw_tau,
            )
            if i == 0:
                print(f"[Img sample{i}] grid shape={grid.shape} range=({grid.min():.3f},{grid.max():.3f})")
            img = render_rgb_grid_to_image(grid, cell_pix=self.cell_pix)
            if i == 0:
                print(f"[Img sample{i}] rendered size={img.size} (W,H) mode={img.mode}")
            images.append(img)

            img_path = os.path.join(self.args.output_dir, "images", f"batch0_sample{i:02d}_heatmap.png")
            img.save(img_path)
        print(f"[Save] images saved to {os.path.join(self.args.output_dir, 'images')} count={len(images)}")
        return images

    def _build_video_modality(self, x_np, batch_size, num_patches):
        print_box(
            f"4.3) Video modality: per patch({self.args.patch_len}) {self.args.num_vars}x{self.args.num_vars} "
            f"grid -> {num_patches} frames -> duplicate to {self.args.video_frames} -> save frames"
        )
        if num_patches <= 0:
            raise ValueError(
                f"invalid patch config: seq_len={self.args.seq_len}, patch_len={self.args.patch_len}, stride={self.args.stride}"
            )

        videos = []
        for i in range(batch_size):
            window = x_np[i]
            frames = []
            for p in range(num_patches):
                start = p * self.args.stride
                patch = window[start:start + self.args.patch_len]
                dtw, cov, pear = compute_three_mats(
                    patch,
                    kind=f"Vid sample{i} patch{p} len{self.args.patch_len}",
                    dtw_band=min(self.args.dtw_band, self.args.patch_len),
                    dtw_eps=self.args.dtw_eps,
                    verbose=(i == 0 and p == 0),
                )
                grid = mats_to_rgb_grid(
                    dtw,
                    cov,
                    pear,
                    cov_clip_lo=self.args.cov_clip_lo,
                    cov_clip_hi=self.args.cov_clip_hi,
                    dtw_tau=self.args.dtw_tau,
                )
                frame = render_rgb_grid_to_image(grid, cell_pix=self.video_cell_pix)
                frames.append(frame)

            frames_out = []
            for f in frames:
                frames_out.append(f)
                frames_out.append(f.copy())
            frames_out = frames_out[:self.args.video_frames]
            if i == 0:
                print(
                    f"[Vid sample{i}] frames raw={len(frames)} duplicated={len(frames_out)} "
                    f"frame_size={frames_out[0].size}"
                )

            vid_dir = os.path.join(self.args.output_dir, "videos", f"batch0_sample{i:02d}")
            ensure_dir(vid_dir)
            for fi, fr in enumerate(frames_out):
                fr_path = os.path.join(vid_dir, f"frame_{fi:02d}.png")
                fr.save(fr_path)

            videos.append(frames_out)

        print(f"[Save] video frames saved to {os.path.join(self.args.output_dir, 'videos')} videos={len(videos)}")
        return videos

    def _encode_vision_modalities(self, images, videos):
        print_box("4.4) Qwen3-VL Vision encoding: images -> img_embeds, videos -> vid_embeds")
        img_token_seqs, img_pooled = encode_images_qwen3vl(
            model=self.model,
            processor=self.processor,
            images=images,
            device=str(self.model.device),
            micro_bs=self.args.vision_mb,
        )
        vid_token_seqs, vid_pooled = encode_videos_qwen3vl(
            model=self.model,
            processor=self.processor,
            videos=videos,
            device=str(self.model.device),
            micro_bs=max(1, self.args.vision_mb // 4),
        )

        save_path = os.path.join(self.args.output_dir, "first_batch_vision_embeds.pt")
        torch.save(
            {
                "img_token_seqs": img_token_seqs,
                "img_pooled": img_pooled,
                "vid_token_seqs": vid_token_seqs,
                "vid_pooled": vid_pooled,
            },
            save_path,
        )
        print(f"[Save] vision embeddings -> {save_path}")
        return img_token_seqs, img_pooled, vid_token_seqs, vid_pooled

    def _print_summary(self, ts_embeds_cpu, img_pooled, vid_pooled, txt_token_embeds, txt_pooled):
        print_box("4.5) SUMMARY (to embeddings only)")
        print(f"[Summary] ts_embeds: {tuple(ts_embeds_cpu.shape)}  (B,42,2048)")
        print(f"[Summary] img_pooled: {tuple(img_pooled.shape)}  (B,2048)")
        print(f"[Summary] vid_pooled: {tuple(vid_pooled.shape)}  (B,2048)")
        print(f"[Summary] txt_token_embeds: {tuple(txt_token_embeds.shape)}  (1,L,2048)")
        print(f"[Summary] txt_pooled: {tuple(txt_pooled.shape)}  (2048,)")

    def _process_batch(self, batch_idx, x, y, ts_mlp, expected_tokens, num_patches, txt_token_embeds, txt_pooled):
        t0 = time.time()
        print_box(f"Batch {batch_idx} start")
        print(f"[Batch] x shape={tuple(x.shape)} y shape={tuple(y.shape)} dtype={x.dtype}")
        batch_size = x.shape[0]

        ts_embeds_cpu = self._encode_ts_embeddings(x, ts_mlp, expected_tokens)

        x_np = x.numpy()
        images = self._build_image_modality(x_np, batch_size)
        videos = self._build_video_modality(x_np, batch_size, num_patches)

        _, img_pooled, _, vid_pooled = self._encode_vision_modalities(images, videos)
        self._print_summary(ts_embeds_cpu, img_pooled, vid_pooled, txt_token_embeds, txt_pooled)

        print(f"[Time] batch0 done in {time.time() - t0:.2f}s")
        print("[NOTE] processed first batch only; remove the break to process all batches.")

    def run(self):
        self._prepare_output()
        data_set, loader = self._load_data()

        self.model, self.processor = self._load_model()

        patch_size = self._load_vision_patch_size()
        self.cell_pix = patch_size * 2
        self.video_cell_pix = patch_size

        print_box("3) Build TS MLP")
        ts_mlp = build_ts_mlp(self.args.patch_len, embed_dim=self.args.embed_dim).to(self.model.device)
        ts_mlp.eval()
        print(f"[TS_MLP] {ts_mlp}")

        global_prompt = (
            "Task: multivariate time-series forecasting on ETTh1.\n"
            f"Input window length={self.args.seq_len}, forecast horizon={self.args.pred_len}, variables={self.args.num_vars}.\n"
            f"We provide four modalities per sample: time-series patch tokens, a {self.args.num_vars}x{self.args.num_vars} "
            f"heatmap image (DTW|Cov|Pearson), and a {self.args.video_frames}-frame video of patch-wise heatmaps.\n"
            "Encode modalities into embeddings for fusion."
        )
        txt_token_embeds, txt_pooled = encode_text_global(
            self.model,
            self.processor,
            global_prompt,
            device=str(self.model.device),
        )
        torch.save(
            {"text_token_embeds": txt_token_embeds, "text_pooled": txt_pooled, "prompt": global_prompt},
            os.path.join(self.args.output_dir, "global_text_embeddings.pt"),
        )
        print(f"[Save] global text embeddings -> {os.path.join(self.args.output_dir, 'global_text_embeddings.pt')}")

        print_box("4) Iterate dataloader (FIRST BATCH ONLY)")
        num_patches = (self.args.seq_len - self.args.patch_len) // self.args.stride + 1
        expected_tokens = num_patches * self.args.num_vars

        for batch_idx, (x, y) in enumerate(loader):
            self._process_batch(
                batch_idx,
                x,
                y,
                ts_mlp,
                expected_tokens,
                num_patches,
                txt_token_embeds,
                txt_pooled,
            )
            break
