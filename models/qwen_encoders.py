from typing import List

import torch
from PIL import Image

from utils.logging import print_box


def encode_images_qwen3vl(model, processor, images: List[Image.Image], device: str, micro_bs: int = 8):
    print_box("Encode IMAGES with Qwen3-VL vision encoder")
    model.eval()

    all_token_seqs = []
    pooled = []

    with torch.inference_mode():
        for start in range(0, len(images), micro_bs):
            end = min(len(images), start + micro_bs)
            chunk = images[start:end]
            print(f"[Vision][Image] chunk {start}:{end} size={len(chunk)}")

            try:
                enc = processor(images=chunk, return_tensors="pt")
            except TypeError:
                enc = processor.image_processor(images=chunk, return_tensors="pt")

            pixel_values = enc.get("pixel_values", None)
            image_grid_thw = enc.get("image_grid_thw", None)
            if pixel_values is None or image_grid_thw is None:
                raise RuntimeError(f"Processor did not return pixel_values/image_grid_thw. keys={list(enc.keys())}")

            print(f"[Vision][Image] pixel_values shape={tuple(pixel_values.shape)} dtype={pixel_values.dtype}")
            print(f"[Vision][Image] image_grid_thw shape={tuple(image_grid_thw.shape)} value={image_grid_thw}")

            pixel_values = pixel_values.to(device)
            image_grid_thw = image_grid_thw.to(device)

            img_feats_list, _ = model.model.get_image_features(pixel_values, image_grid_thw)
            for i, t in enumerate(img_feats_list):
                print(f"[Vision][Image]   item{i}: tokens={t.shape[0]} dim={t.shape[1]} dtype={t.dtype} device={t.device}")
                all_token_seqs.append(t.detach().cpu())
                pooled.append(t.mean(dim=0, keepdim=False).detach().cpu())

    pooled = torch.stack(pooled, dim=0)
    print(f"[Vision][Image] pooled embeds shape={tuple(pooled.shape)}")
    return all_token_seqs, pooled


def encode_videos_qwen3vl(model, processor, videos: List[List[Image.Image]], device: str, micro_bs: int = 2):
    print_box("Encode VIDEOS with Qwen3-VL vision encoder")
    model.eval()

    all_token_seqs = []
    pooled = []

    with torch.inference_mode():
        for start in range(0, len(videos), micro_bs):
            end = min(len(videos), start + micro_bs)
            chunk = videos[start:end]
            print(f"[Vision][Video] chunk {start}:{end} size={len(chunk)} frames_each={len(chunk[0]) if len(chunk)>0 else 'NA'}")

            try:
                enc = processor(videos=chunk, return_tensors="pt")
            except TypeError:
                enc = processor.video_processor(videos=chunk, return_tensors="pt",do_sample_frames=False)

            pixel_values_videos = enc.get("pixel_values_videos", None)
            video_grid_thw = enc.get("video_grid_thw", None)
            if pixel_values_videos is None or video_grid_thw is None:
                raise RuntimeError(f"Processor did not return pixel_values_videos/video_grid_thw. keys={list(enc.keys())}")

            print(f"[Vision][Video] pixel_values_videos shape={tuple(pixel_values_videos.shape)} dtype={pixel_values_videos.dtype}")
            print(f"[Vision][Video] video_grid_thw shape={tuple(video_grid_thw.shape)} value={video_grid_thw}")

            pixel_values_videos = pixel_values_videos.to(device)
            video_grid_thw = video_grid_thw.to(device)

            if hasattr(model.model, "get_video_features"):
                vid_feats_list, _ = model.model.get_video_features(pixel_values_videos, video_grid_thw)
            else:
                raise RuntimeError("model.model has no get_video_features; check transformers version/model class.")

            for i, t in enumerate(vid_feats_list):
                print(f"[Vision][Video]   item{i}: tokens={t.shape[0]} dim={t.shape[1]} dtype={t.dtype} device={t.device}")
                all_token_seqs.append(t.detach().cpu())
                pooled.append(t.mean(dim=0, keepdim=False).detach().cpu())

    pooled = torch.stack(pooled, dim=0)
    print(f"[Vision][Video] pooled embeds shape={tuple(pooled.shape)}")
    return all_token_seqs, pooled


def encode_text_global(model, processor, prompt: str, device: str):
    print_box("Encode TEXT (global prompt) with Qwen3-VL text embedding layer")
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("processor has no tokenizer attribute; check AutoProcessor.")

    toks = tokenizer(prompt, return_tensors="pt")
    input_ids = toks["input_ids"].to(device)
    attn_mask = toks.get("attention_mask", None)
    if attn_mask is not None:
        attn_mask = attn_mask.to(device)

    print(f"[Text] prompt chars={len(prompt)}")
    print(f"[Text] input_ids shape={tuple(input_ids.shape)}")
    if attn_mask is not None:
        print(f"[Text] attention_mask shape={tuple(attn_mask.shape)}")

    with torch.inference_mode():
        emb_layer = model.model.get_input_embeddings()
        token_embeds = emb_layer(input_ids)
        print(f"[Text] token_embeds shape={tuple(token_embeds.shape)} dtype={token_embeds.dtype} device={token_embeds.device}")

        if attn_mask is None:
            pooled = token_embeds.mean(dim=1).squeeze(0)
        else:
            m = attn_mask.float().unsqueeze(-1)
            pooled = (token_embeds * m).sum(dim=1) / (m.sum(dim=1).clamp_min(1.0))
            pooled = pooled.squeeze(0)

    print(f"[Text] pooled shape={tuple(pooled.shape)}")
    return token_embeds.detach().cpu(), pooled.detach().cpu()
