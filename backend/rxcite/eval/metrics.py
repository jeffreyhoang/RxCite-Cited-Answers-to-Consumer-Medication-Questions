# Evaluation metrics for RxCite

from collections.abc import Collection, Sequence
from rxcite.config import TOP_K #set in config.py
from rxcite.pipelines.base import Chunk, RAGResult
 
GoldSection = tuple[str, str]  #(set_id, section)

# True if chunk comes from one of the gold label sections
def is_hit(chunk: Chunk, gold: Collection[GoldSection]) -> bool:
    return (chunk.set_id, chunk.section) in gold

# Fraction of gold sections covered by the top-k chunks
def recall_at_k(chunks: Sequence[Chunk], gold: Collection[GoldSection], k: int = TOP_K) -> float:
    if not gold:
        raise ValueError("recall_at_k needs at least one gold section")
    gold_set = set(gold)
    found = {(c.set_id, c.section) for c in chunks[:k]} & gold_set
    return len(found) / len(gold_set)

# Reciprocal rank of the first hit
def reciprocal_rank(chunks: Sequence[Chunk], gold: Collection[GoldSection]) -> float:
    if not gold:
        raise ValueError("reciprocal_rank needs at least one gold section")
    for rank, chunk in enumerate(chunks, start=1):
        if is_hit(chunk, gold):
            return 1.0 / rank
    return 0.0

# TODO: evaluate_retrieval() - compute recall_at_k (top_k)and mean reciprocal rank (full list, not top_k)for a retrieved list
def evaluate_retrieval(runs: Sequence[tuple[Sequence[Chunk], Collection[GoldSection]]], k: int = TOP_K) -> dict[str, float | int]:
    #Implement here
    return

# TODO: refusal_accuracy() - fraction of questions where model refused (use on refusal eval set)
def refusal_accuracy(results: Sequence[RAGResult]) -> float:
    #Implement here
    return

# TODO: false_refusal_rate() - fraction of results refused (whether correct or not)
def false_refusal_rate(results: Sequence[RAGResult]) -> float:
    #Implement here
    return