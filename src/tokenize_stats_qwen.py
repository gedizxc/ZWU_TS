"""
Tokenize batch_stats.txt with the Qwen3 tokenizer for text+image pairing.
Outputs under artifacts/correlations/:
- vocab_qwen.json        (token -> id from Qwen tokenizer vocab)
- batch_stats_qwen.jsonl (per-batch text + tokens + ids)

Defaults to local model dir `Qwen3-VL-2B-Instruct` at repo root; falls back to hub ID if not found.
Requires: transformers>=4.38. No network needed if the local dir exists.
"""
from pathlib import Path
import json

try:
    from transformers import AutoTokenizer
except ImportError as e:
    raise SystemExit("transformers not installed; please install transformers>=4.38") from e


LOCAL_MODEL_DIR = Path("Qwen3-VL-2B-Instruct")
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

REPO_ROOT = Path(__file__).resolve().parent.parent

# Search for stats in both repo-root and src-relative artifacts (covers different run CWDs).
STATS_CANDIDATES = [
    Path("artifacts/correlations/batch_stats.txt"),
    REPO_ROOT / "artifacts/correlations/batch_stats.txt",
    Path("src/artifacts/correlations/batch_stats.txt"),
    REPO_ROOT / "src/artifacts/correlations/batch_stats.txt",
]

# Output under repo-root artifacts by default.
OUTPUT_DIR = REPO_ROOT / "artifacts/correlations"
VOCAB_PATH = OUTPUT_DIR / "vocab_qwen.json"
TOKENS_PATH = OUTPUT_DIR / "batch_stats_qwen.jsonl"


def resolve_stats_path():
    for p in STATS_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing stats file; tried: {STATS_CANDIDATES}")


def resolve_tokenizer_path():
    local_path = LOCAL_MODEL_DIR
    if local_path.exists():
        return local_path
    # Fallback: hub ID (requires network/cache)
    return MODEL_ID


def parse_batches(text: str):
    batches = []
    raw_blocks = text.strip().split("\n\n")
    for block in raw_blocks:
        lines = block.strip().split("\n")
        if not lines or not lines[0].startswith("--- Batch"):
            continue
        header = lines[0]
        batch_idx = int(header.split()[2])
        content_lines = lines[1:]
        batches.append((batch_idx, content_lines))
    return batches


def main():
    stats_path = resolve_stats_path()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer_path = resolve_tokenizer_path()
    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    vocab = tokenizer.get_vocab()
    VOCAB_PATH.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")

    text = stats_path.read_text(encoding="utf-8")
    batches = parse_batches(text)

    with TOKENS_PATH.open("w", encoding="utf-8") as f:
        for batch_idx, lines in batches:
            batch_text = " ".join(lines)
            ids = tokenizer.encode(batch_text, add_special_tokens=True)
            tokens = tokenizer.convert_ids_to_tokens(ids)
            record = {
                "batch": batch_idx,
                "text": batch_text,
                "tokens": tokens,
                "ids": ids,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote Qwen vocab to: {VOCAB_PATH.resolve()}")
    print(f"Wrote Qwen-tokenized batches to: {TOKENS_PATH.resolve()}")


if __name__ == "__main__":
    main()
