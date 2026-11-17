# Advanced RAG (T7): query rewrite -> dense + BM25 over section-based chunks -> RRF fusion
#       -> top 50 -> cross-encoder rerank -> top 10 -> shared generator.
#       Each step can be switched off for the ablation. Search and the reranker see the rewritten
#       query (reranking against the original lowered dev MRR 0.81 -> 0.57); the generator
#       gets the original question.

import time

from rxcite.bm25 import bm25_search
from rxcite.config import RERANK_CANDIDATES
from rxcite.generation import generate
from rxcite.pipelines.base import Chunk, Generator, RAGPipeline
from rxcite.rerank import rerank as rerank_chunks  # alias: the `rerank` flag would shadow it
from rxcite.retrieval import dense_search, rrf
from rxcite.rewrite import rewrite_query


class AdvancedPipeline(RAGPipeline):
    name = "advanced"

    # rewrite / hybrid / rerank switch steps off for the ablation (registry adv-* variants)
    def __init__(
        self,
        generator: Generator = generate,
        name: str | None = None,
        chunk_set: str = "section",
        rewrite: bool = True,
        hybrid: bool = True,
        rerank: bool = True,
    ):
        super().__init__(generator)
        self.name = name or self.name
        self.chunk_set = chunk_set
        self.rewrite = rewrite
        self.hybrid = hybrid
        self.rerank = rerank

    def retrieve(self, query: str) -> list[Chunk]:
        return self.retrieve_with_info(query)[0]

    # info: the search query used, plus the reranker's time in seconds when it runs
    def retrieve_with_info(self, query: str) -> tuple[list[Chunk], dict]:
        search_q = rewrite_query(query) if self.rewrite else query
        info: dict = {"search_query": search_q}
        n = RERANK_CANDIDATES if (self.hybrid or self.rerank) else self.top_k
        dense = dense_search(search_q, n, chunk_set=self.chunk_set)
        if self.hybrid:
            cands = rrf([dense, bm25_search(search_q, n, chunk_set=self.chunk_set)])
        else:
            cands = dense
        if self.rerank:
            start = time.perf_counter()
            top = rerank_chunks(search_q, cands, self.top_k)
            info["rerank_s"] = time.perf_counter() - start
        else:
            top = cands[: self.top_k]
        return top, info
