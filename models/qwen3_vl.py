from typing import List, Tuple

import torch
from PIL import Image


def load_qwen3_vl(model_dir: str):
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir,
        dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_dir)
    return model, processor


def encode_images_qwen3vl(
    model,
    processor,
    images: List[Image.Image],
    device: str,
    micro_bs: int = 8,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    print("Encode IMAGES with Qwen3-VL vision encoder")
    model.eval()

    all_token_seqs = []
    pooled = []

    with torch.inference_mode():
        for start in range(0, len(images), micro_bs):
            end = min(len(images), start + micro_bs)
            chunk = images[start:end]
            if start == 0:
                print(f"[Vision][Image] chunk {start}:{end} size={len(chunk)}")

            try:
                enc = processor.image_processor(
                    images=chunk,
                    return_tensors="pt",
                    do_resize=False,
                    do_center_crop=False,
                )
            except TypeError:
                enc = processor.image_processor(images=chunk, return_tensors="pt")

            keys = list(enc.keys())
            if start == 0:
                print(f"[Vision][Image] processor outputs keys={keys}")
            pixel_values = enc.get("pixel_values", None)
            image_grid_thw = enc.get("image_grid_thw", None)

            if pixel_values is None or image_grid_thw is None:
                raise RuntimeError(f"processor did not return pixel_values/image_grid_thw. got keys={keys}")

            if start == 0:
                print(f"[Vision][Image] pixel_values shape={tuple(pixel_values.shape)} dtype={pixel_values.dtype}")
                print(f"[Vision][Image] image_grid_thw shape={tuple(image_grid_thw.shape)} value={image_grid_thw}")

            pixel_values = pixel_values.to(device)
            image_grid_thw = image_grid_thw.to(device)

            if hasattr(model.model, "get_image_features"):
                img_feats_list, _deep = model.model.get_image_features(pixel_values, image_grid_thw)
            else:
                raise RuntimeError("model.model has no get_image_features; check transformers/model class")

            for feats in img_feats_list:
                all_token_seqs.append(feats.detach().cpu())

            pooled.append(torch.stack([feats.mean(dim=0) for feats in img_feats_list], dim=0).cpu())

    pooled = torch.cat(pooled, dim=0)
    print(f"[Vision][Image] pooled shape={tuple(pooled.shape)}")
    return all_token_seqs, pooled


def encode_videos_qwen3vl(
    model,
    processor,
    videos: List[List[Image.Image]],
    device: str,
    micro_bs: int = 2,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    print("Encode VIDEOS with Qwen3-VL vision encoder")
    model.eval()

    all_token_seqs = []
    pooled = []

    with torch.inference_mode():
        for start in range(0, len(videos), micro_bs):
            end = min(len(videos), start + micro_bs)
            chunk = videos[start:end]
            if start == 0:
                print(f"[Vision][Video] chunk {start}:{end} size={len(chunk)}")

            try:
                enc = processor(videos=chunk, return_tensors="pt")
            except Exception:
                enc = processor.video_processor(videos=chunk, return_tensors="pt", do_sample_frames=False)

            keys = list(enc.keys())
            if start == 0:
                print(f"[Vision][Video] processor outputs keys={keys}")
            pixel_values_videos = enc.get("pixel_values_videos", None)
            video_grid_thw = enc.get("video_grid_thw", None)

            if pixel_values_videos is None or video_grid_thw is None:
                raise RuntimeError(f"processor did not return pixel_values_videos/video_grid_thw. got keys={keys}")

            if start == 0:
                print(f"[Vision][Video] pixel_values_videos shape={tuple(pixel_values_videos.shape)} dtype={pixel_values_videos.dtype}")
                print(f"[Vision][Video] video_grid_thw shape={tuple(video_grid_thw.shape)} value={video_grid_thw}")

            pixel_values_videos = pixel_values_videos.to(device)
            video_grid_thw = video_grid_thw.to(device)

            if hasattr(model.model, "get_video_features"):
                vid_feats_list, _deep = model.model.get_video_features(pixel_values_videos, video_grid_thw)
            else:
                raise RuntimeError("model.model has no get_video_features; check transformers/model class")

            for feats in vid_feats_list:
                all_token_seqs.append(feats.detach().cpu())

            pooled.append(torch.stack([feats.mean(dim=0) for feats in vid_feats_list], dim=0).cpu())

    pooled = torch.cat(pooled, dim=0)
    print(f"[Vision][Video] pooled shape={tuple(pooled.shape)}")
    return all_token_seqs, pooled


def encode_text_global(model, processor, prompt: str, device: str):
    enc = processor.tokenizer(prompt, return_tensors="pt")
    input_ids = enc.get("input_ids", None)
    attn_mask = enc.get("attention_mask", None)
    if input_ids is None:
        raise RuntimeError("tokenizer did not return input_ids")

    input_ids = input_ids.to(device)
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
