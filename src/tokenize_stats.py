"""
Tokenize batch_stats.txt into token/id sequences for text+image pairing.
Outputs:
- artifacts/correlations/vocab.json  (token -> id mapping)
- artifacts/correlations/batch_stats_tokens.jsonl (per-batch tokens + ids)

Special tokens: <pad>=0, <unk>=1, <bos>=2, <eos>=3
"""
from pathlib import Path
import json
import re


STATS_PATH = Path("artifacts/correlations/batch_stats.txt")
OUTPUT_DIR = Path("artifacts/correlations")
VOCAB_PATH = OUTPUT_DIR / "vocab.json"
TOKENS_PATH = OUTPUT_DIR / "batch_stats_tokens.jsonl"

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def ensure_paths():
    if not STATS_PATH.exists():
        raise FileNotFoundError(f"Missing stats file: {STATS_PATH}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def tokenize_line(line: str):
    # Split into alphanumerics, numbers, and punctuation markers
    return re.findall(r"[A-Za-z0-9\\.\\-]+|[:=,]|\\n", line.strip())


def build_vocab(token_sequences):
    vocab = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
    for seq in token_sequences:
        for tok in seq:
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def encode(seq, vocab):
    bos, eos, unk = vocab["<bos>"], vocab["<eos>"], vocab["<unk>"]
    ids = [bos]
    for tok in seq:
        ids.append(vocab.get(tok, unk))
    ids.append(eos)
    return ids


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
    ensure_paths()
    text = STATS_PATH.read_text(encoding="utf-8")
    batches = parse_batches(text)
    token_sequences = []
    per_batch_tokens = []

    for batch_idx, lines in batches:
        joined = " ".join(lines)
        tokens = tokenize_line(joined)
        token_sequences.append(tokens)
        per_batch_tokens.append((batch_idx, tokens))

    vocab = build_vocab(token_sequences)
    VOCAB_PATH.write_text(json.dumps(vocab, ensure_ascii=True, indent=2), encoding="utf-8")

    with TOKENS_PATH.open("w", encoding="utf-8") as f:
        for batch_idx, tokens in per_batch_tokens:
            ids = encode(tokens, vocab)
            record = {
                "batch": batch_idx,
                "tokens": tokens,
                "ids": ids,
            }
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    print(f"Wrote vocab to: {VOCAB_PATH.resolve()}")
    print(f"Wrote tokenized batches to: {TOKENS_PATH.resolve()}")


if __name__ == "__main__":
    main()
