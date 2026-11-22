# Plain dense retrieval over the fixed-size chunks; both use dummy_generator (retrieval only)

import json

import numpy as np

from rxcite.config import DATA_DIR
from rxcite.pipelines.base import Chunk, RAGPipeline
from rxcite.pipelines.baselines import dummy_generator
from rxcite.retrieval.embeddings import embed
from rxcite.retrieval.search import dense_search

CHUNKS = DATA_DIR / "chunks" / "fixed_v1.jsonl"
EMBEDDINGS = DATA_DIR / "embeddings"


class DensePipeline(RAGPipeline):
    def __init__(self, model: str):
        super().__init__(dummy_generator)
        self.model = model
        self.name = f"dense-{model}"
        self.chunks = [json.loads(line) for line in CHUNKS.open()]
        self.vectors = np.load(EMBEDDINGS / f"{model}.npy")
        if len(self.chunks) != len(self.vectors):
            raise ValueError(f"{model}.npy doesn't match {CHUNKS.name}; re-run embed_chunks.py")

    def retrieve(self, query: str) -> list[Chunk]:
        scores = self.vectors @ embed([query], self.model, kind="query")[0]
        top = np.argsort(-scores)[: self.top_k]
        return [
            Chunk(
                c["chunk_id"], c["set_id"], c["drug_name"], c["section"], c["text"], float(s)
            )
            for c, s in ((self.chunks[i], scores[i]) for i in top)
        ]


class PgvectorDensePipeline(RAGPipeline):
    name = "dense"

    def __init__(self):
        super().__init__(dummy_generator)

    def retrieve(self, query: str) -> list[Chunk]:
        return dense_search(query, self.top_k)
