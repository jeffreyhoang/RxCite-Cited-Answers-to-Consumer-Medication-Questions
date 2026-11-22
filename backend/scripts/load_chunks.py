"""Load a chunk set and the chosen embedder's vectors into Supabase.

Run from backend/ after build_chunks.py and embed_chunks.py --model <EMBEDDING_MODEL>:
    uv run python scripts/load_chunks.py                 # fixed-size chunks -> chunks
    uv run python scripts/load_chunks.py --set section   # section-based -> section_chunks

--set fixed loads data/chunks/fixed_v1.jsonl with data/embeddings/<EMBEDDING_MODEL>.npy;
--set section loads data/chunks/section_v1.jsonl with
data/embeddings/section_v1-<EMBEDDING_MODEL>.npy (the stored text is the raw chunk text,
not its search_text).
Safe to rerun: empties the set's table, reloads every row, then builds its HNSW index.
"""

import argparse
import json

import numpy as np

from rxcite import db
from rxcite.config import DATA_DIR, EMBEDDING_DIM, EMBEDDING_MODEL
from rxcite.retrieval.search import chunk_table

CHUNKS = {
    "fixed": DATA_DIR / "chunks" / "fixed_v1.jsonl",
    "section": DATA_DIR / "chunks" / "section_v1.jsonl",
}
VECTORS = {
    "fixed": DATA_DIR / "embeddings" / f"{EMBEDDING_MODEL}.npy",
    "section": DATA_DIR / "embeddings" / f"section_v1-{EMBEDDING_MODEL}.npy",
}
FREE_TIER_MB = 500

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", default="fixed", choices=sorted(CHUNKS), dest="chunk_set")
    args = parser.parse_args()
    # Table and index names come from fixed dicts (CHUNK_TABLES, db.CHUNK_INDEXES), never input
    table = chunk_table(args.chunk_set)
    index_name, index_sql = db.CHUNK_INDEXES[args.chunk_set]
    vectors_path = VECTORS[args.chunk_set]

    chunks = [json.loads(line) for line in CHUNKS[args.chunk_set].open()]
    vectors = np.load(vectors_path)
    if vectors.shape != (len(chunks), EMBEDDING_DIM):
        expected = (len(chunks), EMBEDDING_DIM)
        raise SystemExit(f"{vectors_path.name} is {vectors.shape}, expected {expected}")

    db.create_tables()
    with db.connect() as conn:
        conn.execute("set statement_timeout = 0")
        conn.execute(f"drop index if exists {index_name}")
        conn.execute(f"truncate {table}")
        with conn.cursor().copy(
            f"copy {table} (chunk_id, set_id, section, text, embedding) from stdin"
        ) as copy:
            for c, v in zip(chunks, vectors, strict=True):
                vec = "[" + ",".join(f"{x:.6g}" for x in v) + "]"
                copy.write_row((c["chunk_id"], c["set_id"], c["section"], c["text"], vec))
        print(f"Loaded {len(chunks)} chunks into {table}; building the HNSW index...")
        # Free-tier instance has little shared memory: build the index in one process
        conn.execute("set max_parallel_maintenance_workers = 0")
        conn.execute("set maintenance_work_mem = '64MB'")
        conn.execute(index_sql)
        conn.commit()

        rows = conn.execute(f"select count(*) from {table}").fetchone()[0]
        size, total = conn.execute(
            """select pg_total_relation_size(%s::regclass) / 1048576.0,
                      pg_database_size(current_database()) / 1048576.0""",
            (table,),
        ).fetchone()

    print(f"{table} table: {rows} rows (expected {len(chunks)}), {size:.0f} MB with index")
    print(f"database: {total:.0f} MB of {FREE_TIER_MB} MB free tier")
