# Decomposition RAG: split the question into sub-questions -> Naive's dense search for the
#       original question and each sub-question -> merge by taking turns -> top 10 -> generate.
#       Same chunks, search and generator as Naive; only the splitting and merging differ.

from rxcite.generation import generate
from rxcite.pipelines.base import Chunk, Generator, RAGPipeline
from rxcite.query.decomposition import decompose, merge_round_robin, search_queries
from rxcite.retrieval.search import dense_search


class DecompositionPipeline(RAGPipeline):
    name = "decomposition"

    def __init__(self, generator: Generator = generate, name: str | None = None):
        super().__init__(generator)
        self.name = name or self.name

    def retrieve(self, query: str) -> list[Chunk]:
        return self.retrieve_with_info(query)[0]

    # info: sub-questions searched besides the original
    def retrieve_with_info(self, query: str) -> tuple[list[Chunk], dict]:
        queries = search_queries(query, decompose(query))
        chunks = merge_round_robin([dense_search(q, self.top_k) for q in queries], self.top_k)
        return chunks, {"sub_questions": queries[1:]}
