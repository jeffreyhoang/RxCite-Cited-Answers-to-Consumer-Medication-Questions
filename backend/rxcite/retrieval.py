# Retrieval over the Supabase chunks table (pgvector)
#       dense_search is the shared dense retriever for the RAG architectures (Naive RAG in T6, ...)
#       Queries run on db.query's shared connection (opened once, reconnects if dropped).

from rxcite import db
from rxcite.config import EMBEDDING_DIM, EMBEDDING_MODEL, TOP_K
from rxcite.embeddings import embed
from rxcite.pipelines.base import Chunk

SEARCH_SQL = f"""
select c.chunk_id, c.set_id, l.drug_name, c.section, c.text,
       1 - (c.embedding <=> %(q)s::halfvec({EMBEDDING_DIM})) as score
from chunks c join labels l using (set_id)
order by c.embedding <=> %(q)s::halfvec({EMBEDDING_DIM})
limit %(k)s
"""

# Only the given labels' chunks. "materialized" makes Postgres check every one of them
# (exact search) instead of using the HNSW index and filtering afterwards, which could
# return fewer than k chunks. Filtered sets are small (a few labels), so this is fast.
FILTERED_SQL = f"""
with c as materialized (
    select chunk_id, set_id, section, text, embedding from chunks
    where set_id = any(%(set_ids)s)
)
select c.chunk_id, c.set_id, l.drug_name, c.section, c.text,
       1 - (c.embedding <=> %(q)s::halfvec({EMBEDDING_DIM})) as score
from c join labels l using (set_id)
order by c.embedding <=> %(q)s::halfvec({EMBEDDING_DIM})
limit %(k)s
"""


# Top-k chunks by cosine similarity to the query, best first.
# With set_ids, only those labels are searched; an empty list means search everything.
def dense_search(query: str, k: int = TOP_K, set_ids: list[str] | None = None) -> list[Chunk]:
    vector = embed([query], EMBEDDING_MODEL, kind="query")[0]
    params = {"q": "[" + ",".join(f"{x:.6g}" for x in vector) + "]", "k": k}
    sql = SEARCH_SQL
    if set_ids:
        sql, params["set_ids"] = FILTERED_SQL, list(set_ids)
    rows = db.query(sql, params)
    return [Chunk(*row) for row in rows]
