# Retrieval over the Supabase chunk tables (pgvector), plus Reciprocal Rank Fusion
#       dense_search is the shared dense retriever for the RAG architectures (Naive RAG in T6, ...)
#       Queries run on db.query's shared connection (opened once, reconnects if dropped).
#       Two chunk sets: "fixed" (T5 fixed-size chunks) and "section" (T7 section-based chunks).

from dataclasses import replace

from rxcite import db
from rxcite.config import EMBEDDING_DIM, EMBEDDING_MODEL, RERANK_CANDIDATES, RRF_K, TOP_K
from rxcite.embeddings import embed
from rxcite.pipelines.base import Chunk

# Chunk set name -> table. Table names are only ever formatted into SQL from this dict.
CHUNK_TABLES = {"fixed": "chunks", "section": "section_chunks"}

SEARCH_SQL = """
select c.chunk_id, c.set_id, l.drug_name, c.section, c.text,
       1 - (c.embedding <=> %(q)s::halfvec({dim})) as score
from {table} c join labels l using (set_id)
order by c.embedding <=> %(q)s::halfvec({dim})
limit %(k)s
"""

# Only the given labels' chunks. "materialized" makes Postgres check every one of them
# (exact search) instead of using the HNSW index and filtering afterwards, which could
# return fewer than k chunks. Filtered sets are small (a few labels), so this is fast.
FILTERED_SQL = """
with c as materialized (
    select chunk_id, set_id, section, text, embedding from {table}
    where set_id = any(%(set_ids)s)
)
select c.chunk_id, c.set_id, l.drug_name, c.section, c.text,
       1 - (c.embedding <=> %(q)s::halfvec({dim})) as score
from c join labels l using (set_id)
order by c.embedding <=> %(q)s::halfvec({dim})
limit %(k)s
"""


def chunk_table(chunk_set: str) -> str:
    if chunk_set not in CHUNK_TABLES:
        raise ValueError(f"unknown chunk set {chunk_set!r}; expected one of {list(CHUNK_TABLES)}")
    return CHUNK_TABLES[chunk_set]


def _vector_literal(vector) -> str:
    return "[" + ",".join(f"{x:.6g}" for x in vector) + "]"


def _search(vector, k: int, chunk_set: str, set_ids: list[str] | None) -> list[Chunk]:
    table = chunk_table(chunk_set)
    params = {"q": _vector_literal(vector), "k": k}
    sql = SEARCH_SQL
    if set_ids:
        sql, params["set_ids"] = FILTERED_SQL, list(set_ids)
    rows = db.query(sql.format(table=table, dim=EMBEDDING_DIM), params)
    return [Chunk(*row) for row in rows]


# Top-k chunks by cosine similarity to the query, best first.
# With set_ids, only those labels are searched; an empty list means search everything.
def dense_search(
    query: str, k: int = TOP_K, set_ids: list[str] | None = None, chunk_set: str = "fixed"
) -> list[Chunk]:
    chunk_table(chunk_set)  # fail before paying for an embedding
    vector = embed([query], EMBEDDING_MODEL, kind="query")[0]
    return _search(vector, k, chunk_set, set_ids)


# Reciprocal Rank Fusion: each chunk scores sum(1 / (k + rank)) over the lists (rank from 1).
# Deduplicated by chunk_id; ties keep first-seen order. Returns copies with score = fused score.
def rrf(lists: list[list[Chunk]], k: int = RRF_K, top_n: int = RERANK_CANDIDATES) -> list[Chunk]:
    scores: dict[str, float] = {}
    first: dict[str, Chunk] = {}
    for ranked in lists:
        for rank, c in enumerate(ranked, 1):
            first.setdefault(c.chunk_id, c)
            scores[c.chunk_id] = scores.get(c.chunk_id, 0.0) + 1.0 / (k + rank)
    order = sorted(first, key=lambda cid: -scores[cid])
    return [replace(first[cid], score=scores[cid]) for cid in order[:top_n]]
