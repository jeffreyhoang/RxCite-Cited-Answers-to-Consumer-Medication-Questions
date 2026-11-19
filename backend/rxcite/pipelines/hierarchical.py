# Hierarchical RAG: detect drug(s) and question type -> search child chunks only in those
#       drugs' labels (all labels if none found) -> return their parent sections -> generate.
#       Children are the T5 fixed-size chunks; parents are label sections (rxcite/hierarchy.py).

from rxcite.generation import generate
from rxcite.hierarchy import N_PARENTS, expand_to_parents
from rxcite.pipelines.base import Chunk, Generator, RAGPipeline
from rxcite.query_analysis import analyze, labels_for
from rxcite.retrieval import dense_search

N_CHILDREN = 50  # child chunks searched before grouping into parents


class HierarchicalPipeline(RAGPipeline):
    name = "hierarchical"

    def __init__(self, generator: Generator = generate, name: str | None = None):
        super().__init__(generator, top_k=N_PARENTS)
        self.name = name or self.name

    def retrieve(self, query: str) -> list[Chunk]:
        return self.retrieve_with_info(query)[0]

    # info: detected drugs and question type
    def retrieve_with_info(self, query: str) -> tuple[list[Chunk], dict]:
        info = analyze(query)
        children = dense_search(query, N_CHILDREN, labels_for(info.drugs))
        parents = expand_to_parents(children, info.section_type, self.top_k)
        return parents, {"drugs": info.drugs, "section_type": info.section_type}
