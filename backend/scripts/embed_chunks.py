"""Embed every chunk of a chunk set with one candidate model.

Run from backend/ after `uv run python scripts/build_chunks.py [--set section]`:
    uv run python scripts/embed_chunks.py --model bge-base
    uv run python scripts/embed_chunks.py --model pubmedbert
    uv run python scripts/embed_chunks.py --model openai-small   # uses the OpenAI API
    uv run python scripts/embed_chunks.py --set section --model openai-small

--set fixed (default) embeds the text of data/chunks/fixed_v1.jsonl and writes
data/embeddings/<model>.npy (one unit-length row per chunk, in .jsonl order; OpenAI at 512
numbers, local models at 768) and <model>.json (embed time and token stats).
--set section embeds the search_text (drug | section header + text) of
data/chunks/section_v1.jsonl, openai-small only, into data/embeddings/section_v1-openai-small.npy.
Skips a model that is done.
"""

import argparse
import json
import statistics
import time

import numpy as np

from rxcite.config import DATA_DIR, EMBEDDING_MODEL
from rxcite.retrieval.embeddings import MODELS, embed, token_counts

CHUNKS = {
    "fixed": DATA_DIR / "chunks" / "fixed_v1.jsonl",
    "section": DATA_DIR / "chunks" / "section_v1.jsonl",
}
OUT_DIR = DATA_DIR / "embeddings"
SECTION_MODEL = EMBEDDING_MODEL  # the section set is only embedded with the chosen embedder
LOCAL_TOKEN_LIMIT = 512  # local models cut off anything longer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--set", default="fixed", choices=sorted(CHUNKS), dest="chunk_set")
    args = parser.parse_args()
    if args.chunk_set == "section" and args.model != SECTION_MODEL:
        parser.error(f"--set section is only embedded with --model {SECTION_MODEL}")

    if args.chunk_set == "section":
        out, field = OUT_DIR / f"section_v1-{args.model}.npy", "search_text"
    else:
        out, field = OUT_DIR / f"{args.model}.npy", "text"
    if out.exists():
        print(f"{out} already exists; delete it to re-embed")
        raise SystemExit

    texts = [json.loads(line)[field] for line in CHUNKS[args.chunk_set].open()]
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
