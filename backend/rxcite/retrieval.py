# Retrieval over the Supabase chunks table (pgvector)
#       dense_search is the shared dense retriever for the RAG architectures (Naive RAG in T6, ...)
#       One connection is opened once and reused (opening one costs ~1 s); it reconnects if dropped.

import psycopg

from rxcite import db
from rxcite.config import EMBEDDING_DIM, EMBEDDING_MODEL, TOP_K
from rxcite.embeddings import embed
from rxcite.pipelines.base import Chunk

EF_SEARCH = 100  # HNSW search breadth; higher = closer to exact search, a bit slower

_conn: psycopg.Connection | None = None


# The shared connection, opened on first use or after it was closed or broken
def _connection() -> psycopg.Connection:
    global _conn
    if _conn is None or _conn.closed or _conn.broken:
        _conn = db.connect()
        _conn.autocommit = True  # read-only queries; no open transaction left idle
        _conn.execute(f"set hnsw.ef_search = {EF_SEARCH}")
    return _conn


SEARCH_SQL = f"""
select c.chunk_id, c.set_id, l.drug_name, c.section, c.text,
       1 - (c.embedding <=> %(q)s::halfvec({EMBEDDING_DIM})) as score
from chunks c join labels l using (set_id)
order by c.embedding <=> %(q)s::halfvec({EMBEDDING_DIM})
limit %(k)s
"""


# Top-k chunks by cosine similarity to the query, best first
def dense_search(query: str, k: int = TOP_K) -> list[Chunk]:
    vector = embed([query], EMBEDDING_MODEL, kind="query")[0]
    params = {"q": "[" + ",".join(f"{x:.6g}" for x in vector) + "]", "k": k}
    try:
        rows = _connection().execute(SEARCH_SQL, params).fetchall()
    except psycopg.OperationalError:  # connection dropped (e.g. idle timeout): retry once
        global _conn
        _conn = None
        rows = _connection().execute(SEARCH_SQL, params).fetchall()
    return [Chunk(*row) for row in rows]
