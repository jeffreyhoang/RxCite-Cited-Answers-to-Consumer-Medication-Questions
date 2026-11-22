# Naive RAG (baseline architecture): embed the question -> top-k dense chunks -> generate

from rxcite.generation import generate
from rxcite.pipelines.base import Chunk, RAGPipeline
from rxcite.retrieval.search import dense_search


class NaivePipeline(RAGPipeline):
    name = "naive"

    def __init__(self):
        super().__init__(generate)

    def retrieve(self, query: str) -> list[Chunk]:
        return dense_search(query, self.top_k)
