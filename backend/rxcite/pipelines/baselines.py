# Baseline pipelines that test the eval harness (not real RAG systems)
#       oracle: retrieves exactly the gold sections -> recall@10 should be 1.0
#       dummy:  retrieves 10 random sections        -> recall@10 should be near 0
#       Both use dummy_generator. Until T5 adds chunks, a "chunk" is a whole label section.

import hashlib
import random

from rxcite import db
from rxcite.eval.datasets import load_eval_sets
from rxcite.pipelines.base import Answer, Chunk, Citation, RAGPipeline


# Fixed answer citing the top chunk; refuses when nothing was retrieved
def dummy_generator(query: str, chunks: list[Chunk]) -> Answer:
    if not chunks:
        return Answer(text="I can't answer this from the FDA labels I have.", refused=True)
    top = chunks[0]
    return Answer(
        text=f"See the {top.section.replace('_', ' ')} section of the {top.drug_name} label.",
        citations=[Citation(top.set_id, top.section)],
    )


# Label sections as Chunks, in the order given
def load_sections(pairs: list[tuple[str, str]]) -> list[Chunk]:
    if not pairs:
        return []
    with db.connect() as conn:
        rows = conn.execute(
            """select s.set_id, s.section, l.drug_name, s.text
               from sections s join labels l using (set_id)
               where (s.set_id, s.section) in (select * from unnest(%s::text[], %s::text[]))""",
            ([s for s, _ in pairs], [sec for _, sec in pairs]),
        ).fetchall()
    found = {(s, sec): Chunk(f"{s}:{sec}", s, drug, sec, text) for s, sec, drug, text in rows}
    return [found[p] for p in pairs if p in found]


class OraclePipeline(RAGPipeline):
    name = "oracle"

    def __init__(self):
        super().__init__(dummy_generator)
        answerable, _ = load_eval_sets()
        self.gold = {q.question: q.gold for q in answerable}

    def retrieve(self, query: str) -> list[Chunk]:
        return load_sections(self.gold.get(query, [])[: self.top_k])


class DummyPipeline(RAGPipeline):
    name = "dummy"

    def __init__(self):
        super().__init__(dummy_generator)
        with db.connect() as conn:
            self.all_sections = conn.execute(
                "select set_id, section from sections order by set_id, section"
            ).fetchall()

    def retrieve(self, query: str) -> list[Chunk]:
        seed = int(hashlib.sha256(query.encode()).hexdigest(), 16)  # same query -> same picks
        return load_sections(random.Random(seed).sample(self.all_sections, self.top_k))
