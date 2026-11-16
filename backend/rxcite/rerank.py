# Cross-encoder reranking for Advanced RAG (T7): bge-reranker-base scores (question, chunk)
#       pairs locally. The model loads once, on first use.

from dataclasses import replace
from functools import cache

from rxcite.config import RERANKER_MAX_LENGTH, RERANKER_MODEL, TOP_K
from rxcite.pipelines.base import Chunk, search_text


@cache
def reranker():
    from sentence_transformers import CrossEncoder  # slow import, only when needed

    return CrossEncoder(RERANKER_MODEL, max_length=RERANKER_MAX_LENGTH)


# Top-k chunks by reranker score, best first; returns copies with score = reranker score
def rerank(question: str, chunks: list[Chunk], k: int = TOP_K) -> list[Chunk]:
    if not chunks:
        return []
    pairs = [(question, search_text(c.drug_name, c.section, c.text)) for c in chunks]
    scores = [float(s) for s in reranker().predict(pairs)]
    order = sorted(range(len(chunks)), key=lambda i: -scores[i])
    return [replace(chunks[i], score=scores[i]) for i in order[:k]]
