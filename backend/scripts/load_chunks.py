"""Load the fixed-size chunks and the chosen embedder's vectors into Supabase.

Run from backend/ after build_chunks.py and embed_chunks.py --model <EMBEDDING_MODEL>:
    uv run python scripts/load_chunks.py

Safe to rerun: empties the chunks table, reloads every row, then builds the HNSW index.
"""

import json

import numpy as np

from rxcite import db
from rxcite.config import DATA_DIR, EMBEDDING_DIM, EMBEDDING_MODEL

CHUNKS = DATA_DIR / "chunks" / "fixed_v1.jsonl"
VECTORS = DATA_DIR / "embeddings" / f"{EMBEDDING_MODEL}.npy"
FREE_TIER_MB = 500

if __name__ == "__main__":
    chunks = [json.loads(line) for line in CHUNKS.open()]
    vectors = np.load(VECTORS)
    if vectors.shape != (len(chunks), EMBEDDING_DIM):
        expected = (len(chunks), EMBEDDING_DIM)
        raise SystemExit(f"{VECTORS.name} is {vectors.shape}, expected {expected}")

    db.create_tables()
    with db.connect() as conn:
        conn.execute("set statement_timeout = 0")
        conn.execute("drop index if exists chunks_embedding_idx")
        conn.execute("truncate chunks")
        with conn.cursor().copy(
            "copy chunks (chunk_id, set_id, section, text, embedding) from stdin"
        ) as copy:
            for c, v in zip(chunks, vectors, strict=True):
                vec = "[" + ",".join(f"{x:.6g}" for x in v) + "]"
                copy.write_row((c["chunk_id"], c["set_id"], c["section"], c["text"], vec))
        print(f"Loaded {len(chunks)} chunks; building the HNSW index...")
        # Free-tier instance has little shared memory: build the index in one process
        conn.execute("set max_parallel_maintenance_workers = 0")
        conn.execute("set maintenance_work_mem = '64MB'")
        conn.execute(db.CHUNKS_INDEX)
        conn.commit()

        rows = conn.execute("select count(*) from chunks").fetchone()[0]
        table, total = conn.execute(
            """select pg_total_relation_size('chunks') / 1048576.0,
                      pg_database_size(current_database()) / 1048576.0"""
        ).fetchone()

    print(f"chunks table: {rows} rows (expected {len(chunks)}), {table:.0f} MB with index")
    print(f"database: {total:.0f} MB of {FREE_TIER_MB} MB free tier")
