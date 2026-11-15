"""Embed every chunk in data/chunks/fixed_v1.jsonl with one candidate model.

Run from backend/ after `uv run python scripts/build_chunks.py`:
    uv run python scripts/embed_chunks.py --model bge-base
    uv run python scripts/embed_chunks.py --model pubmedbert
    uv run python scripts/embed_chunks.py --model openai-small   # uses the OpenAI API

Writes data/embeddings/<model>.npy (one unit-length row per chunk, in .jsonl order; OpenAI
at 512 numbers, local models at 768) and <model>.json (embed time and token stats).
Skips a model that is done.
"""

import argparse
import json
import statistics
import time

import numpy as np

from rxcite.config import DATA_DIR
from rxcite.embeddings import MODELS, embed, token_counts

CHUNKS = DATA_DIR / "chunks" / "fixed_v1.jsonl"
OUT_DIR = DATA_DIR / "embeddings"
LOCAL_TOKEN_LIMIT = 512  # local models cut off anything longer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    args = parser.parse_args()

    out = OUT_DIR / f"{args.model}.npy"
    if out.exists():
        print(f"{out} already exists; delete it to re-embed")
        raise SystemExit

    texts = [json.loads(line)["text"] for line in CHUNKS.open()]
    print(f"Embedding {len(texts)} chunks with {MODELS[args.model].name}")
    start = time.perf_counter()
    vectors = embed(texts, args.model, kind="passage", show_progress=True)
    seconds = time.perf_counter() - start

    stats = {
        "model": MODELS[args.model].name,
        "chunks": len(texts),
        "dim": vectors.shape[1],
        "embed_seconds": round(seconds, 1),
    }
    if not MODELS[args.model].api:
        counts = token_counts(texts, args.model)
        stats["tokens"] = {
            "mean": round(statistics.mean(counts), 1),
            "max": max(counts),
            "over_limit": sum(c > LOCAL_TOKEN_LIMIT for c in counts),
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(out, vectors)
    out.with_suffix(".json").write_text(json.dumps(stats, indent=2) + "\n")
    norms = np.linalg.norm(vectors, axis=1)
    print(f"\n{stats}")
    print(f"rows {vectors.shape[0]} (chunks {len(texts)})", end=", ")
    print(f"norms {norms.min():.4f}-{norms.max():.4f}")
    print(f"Wrote {out}")
