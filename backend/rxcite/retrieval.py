# Retrieval over the Supabase chunks table (pgvector)
#       dense_search is the shared dense retriever for the RAG architectures (Naive RAG in T6, ...)

from rxcite import db
from rxcite.config import EMBEDDING_DIM, EMBEDDING_MODEL, TOP_K
from rxcite.embeddings import embed
from rxcite.pipelines.base import Chunk

EF_SEARCH = 100  # HNSW search breadth; higher = closer to exact search, a bit slower


# Top-k chunks by cosine similarity to the query, best first
def dense_search(query: str, k: int = TOP_K) -> list[Chunk]:
    vector = embed([query], EMBEDDING_MODEL, kind="query")[0]
    literal = "[" + ",".join(f"{x:.6g}" for x in vector) + "]"
    with db.connect() as conn:
        conn.execute(f"set hnsw.ef_search = {EF_SEARCH}")
        rows = conn.execute(
            f"""select c.chunk_id, c.set_id, l.drug_name, c.section, c.text,
                       1 - (c.embedding <=> %(q)s::halfvec({EMBEDDING_DIM})) as score
                from chunks c join labels l using (set_id)
                order by c.embedding <=> %(q)s::halfvec({EMBEDDING_DIM})
                limit %(k)s""",
            {"q": literal, "k": k},
        ).fetchall()
    return [Chunk(*row) for row in rows]
