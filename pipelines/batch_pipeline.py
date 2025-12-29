import os
import time

from configs import hparams
from configs.paths import OUTPUT_DIR, IMAGES_DIR, VIDEOS_DIR, QWEN_MODEL_ID
from data_provider.dataset_etth1 import load_etth1_csv, fit_scaler, build_dataloader
from models.qwen_encoders import encode_text_global
from models.ts_mlp import build_ts_mlp
from pipelines.batch_steps import (
    step_ts_embeddings,
    step_image_modality,
    step_video_modality,
    step_encode_vision,
    step_summary,
)
from utils.io import ensure_dir, torch_save
from utils.logging import print_box


def build_global_prompt() -> str:
    return (
        "Task: multivariate time-series forecasting on ETTh1.\n"
        f"Input window length={hparams.SEQ_LEN}, forecast horizon={hparams.PRED_LEN}, variables={hparams.NUM_VARS}.\n"
        "We provide four modalities per sample: time-series patch tokens, a 7x21 heatmap image "
        "(DTW|Cov|Pearson), and a 12-frame video of patch-wise heatmaps.\n"
        "Encode modalities into embeddings for fusion."
    )


def run_batch_pipeline():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(IMAGES_DIR)
    ensure_dir(VIDEOS_DIR)

    print_box("0) Load ETTh1.csv")
    data = load_etth1_csv()

    scaler, data_norm = fit_scaler(data)
    loader = build_dataloader(data_norm)

    print_box("3) Load Qwen3-VL")
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_ID,
        dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
    print(f"[Model] loaded on device={model.device} dtype={next(model.parameters()).dtype}")

    print_box("4) Build TS MLP (16 -> 2048)")
    ts_mlp = build_ts_mlp().to(model.device)
    ts_mlp.eval()
    print(f"[TS_MLP] {ts_mlp}")

    global_prompt = build_global_prompt()
    txt_token_embeds, txt_pooled = encode_text_global(model, processor, global_prompt, device=str(model.device))
    torch_save(
        {"text_token_embeds": txt_token_embeds, "text_pooled": txt_pooled, "prompt": global_prompt},
        os.path.join(OUTPUT_DIR, "global_text_embeddings.pt"),
    )
    print(f"[Save] global text embeddings -> {os.path.join(OUTPUT_DIR, 'global_text_embeddings.pt')}")

    print_box("5) Iterate dataloader (FIRST BATCH ONLY)")
    for batch_idx, (x, y) in enumerate(loader):
        t0 = time.time()
        print_box(f"Batch {batch_idx} start")
        print(f"[Batch] x shape={tuple(x.shape)} y shape={tuple(y.shape)} dtype={x.dtype}")

        # 5.1 TS patchify -> MLP -> ts_embeds
        ts_embeds_cpu = step_ts_embeddings(x, model, ts_mlp)

        # 5.2 Image modality: DTW|Cov|Pearson -> heatmap
        images = step_image_modality(x.numpy())

        # 5.3 Video modality: patch-wise heatmaps -> duplicated frames
        videos = step_video_modality(x.numpy())

        # 5.4 Vision encoder: images/videos -> embeddings
        img_token_seqs, img_pooled, vid_token_seqs, vid_pooled = step_encode_vision(
            model=model,
            processor=processor,
            images=images,
            videos=videos,
        )

        # 5.5 Summary logs
        step_summary(ts_embeds_cpu, img_pooled, vid_pooled, txt_token_embeds, txt_pooled)

        print(f"[Time] batch0 done in {time.time() - t0:.2f}s")
        print("\n[NOTE] 已处理第一批并 break。后续要处理全部 batch，删除下面的 break 即可。\n")
        break
