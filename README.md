# ZWU_TS Informer-Style Layout

```
data/
  ETTh1.csv
src/
  generate_correlations.py      # merged 3x2 correlation generator
  tokenize_stats.py             # turn batch_stats.txt into tokens/ids
  legacy/                       # previous standalone scripts kept for reference
configs/
models/
utils/
exp/
scripts/
logs/
checkpoints/
results/
artifacts/
  correlations/                 # combined correlation figures + stats/tokens
```

## Dependencies
Python 3 with `numpy`, `pandas`, `seaborn`, `matplotlib`, `tqdm`.

## Generate Correlation Images + Stats
```bash
python src/generate_correlations.py
```
Reads `data/ETTh1.csv`, splits into 96-step batches, and writes 3x2 images plus `batch_stats.txt` to `artifacts/correlations/`.
Note: If you run from another working directory, the script will try both `./data/ETTh1.csv` and `<repo>/data/ETTh1.csv`. Ensure the dataset stays under `data/`.

## Tokenize Stats for Text+Image Pairing (e.g., Qwen3-VL inputs)
After stats exist:
```bash
python src/tokenize_stats.py
```
Outputs:
- `artifacts/correlations/vocab.json` (token -> id mapping with specials `<pad>=0,<unk>=1,<bos>=2,<eos>=3`)
- `artifacts/correlations/batch_stats_tokens.jsonl` (per batch: tokens + ids)

You can then pair each `batch_{i}.png` image with the corresponding JSONL row (matching `batch` index) to feed into text+image models. If you need a different tokenizer, swap `tokenize_line` in `src/tokenize_stats.py` or replace with your model’s tokenizer before vectorization.

## Tokenize with Qwen3 Tokenizer (preferred for Qwen/Qwen3-VL-2B-Instruct)
Requires `transformers` and access to the model weights (local cache or network):
```bash
python src/tokenize_stats_qwen.py
```
Outputs:
- `artifacts/correlations/vocab_qwen.json` (Qwen tokenizer vocab)
- `artifacts/correlations/batch_stats_qwen.jsonl` (per batch: text, tokens, ids)

Notes:
- Model ID used: `Qwen/Qwen3-VL-2B-Instruct` (loads with `AutoTokenizer.from_pretrained(..., trust_remote_code=True)`).
- Ensure the model/tokenizer is available locally or that your environment can download it once.

## Patch-based Frames/Videos (PatchTST style) for Qwen Video Input
Generate patch videos (patch_size=24, stride=12) over past 96 steps (future 96 kept as target span):
```bash
python src/patch_video_generator.py
```
Outputs under `artifacts/patch_videos/`:
- `batch_{i}.mp4` (requires `imageio.v3` + codec, install with `pip install imageio[ffmpeg]`)

Frames are kept in-memory (no PNG files). Default caps to `MAX_SAMPLES=5` to avoid huge outputs. Adjust inside `src/patch_video_generator.py` if you want all samples or different patch params.
