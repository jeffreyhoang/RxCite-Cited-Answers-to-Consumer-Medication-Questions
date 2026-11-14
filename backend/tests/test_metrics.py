import pytest

from rxcite.eval.datasets import load_eval_sets
from rxcite.eval.metrics import (
    evaluate_retrieval,
    false_refusal_rate,
    recall_at_k,
    reciprocal_rank,
    refusal_accuracy,
)
from rxcite.pipelines.base import Answer, Chunk, RAGPipeline, RAGResult

GOLD = [("A", "dose"), ("B", "dose")]


def chunk(set_id: str, section: str) -> Chunk:
    return Chunk(
        chunk_id=f"{set_id}-{section}", set_id=set_id, drug_name="x", section=section, text=""
    )


def result(refused: bool) -> RAGResult:
    return RAGResult(
        question="q", architecture="t", answer="", citations=[], chunks=[], refused=refused
    )


def test_recall_at_k_counts_gold_found_in_top_k():
    chunks = [chunk("A", "dose"), chunk("C", "dose"), chunk("B", "dose")]
    assert recall_at_k(chunks, GOLD, k=3) == 1.0
    assert recall_at_k(chunks, GOLD, k=2) == 0.5  # B is at rank 3
    assert recall_at_k([chunk("A", "dose"), chunk("A", "dose")], GOLD) == 0.5  # repeats count once


def test_reciprocal_rank_uses_first_hit():
    assert reciprocal_rank([chunk("C", "x"), chunk("B", "dose")], GOLD) == 0.5
    assert reciprocal_rank([chunk("C", "x")], GOLD) == 0.0
    assert reciprocal_rank([chunk("A", "other")], GOLD) == 0.0  # right label, wrong section


def test_evaluate_retrieval_averages_runs():
    runs = [([chunk("A", "dose")], GOLD), ([chunk("C", "x"), chunk("B", "dose")], [("B", "dose")])]
    out = evaluate_retrieval(runs, k=10)
    assert out == {"recall@10": (0.5 + 1.0) / 2, "mrr": (1.0 + 0.5) / 2, "n": 2}


def test_refusal_metrics():
    assert refusal_accuracy([result(True), result(True), result(False), result(True)]) == 0.75
    assert false_refusal_rate([result(False), result(True)]) == 0.5


def test_metrics_reject_empty_input():
    with pytest.raises(ValueError):
        recall_at_k([chunk("A", "dose")], [])
    with pytest.raises(ValueError):
        evaluate_retrieval([])
    with pytest.raises(ValueError):
        refusal_accuracy([])


def test_pipeline_answer_times_and_passes_through():
    class Fixed(RAGPipeline):
        name = "fixed"

        def retrieve(self, query):
            return [chunk("A", "dose")]

    out = Fixed(lambda q, chunks: Answer(text="ok", refused=True)).answer("q")
    assert out.architecture == "fixed" and out.answer == "ok" and out.refused
    assert out.chunks[0].set_id == "A" and out.latency_s >= 0


def test_load_eval_sets_splits_questions():
    answerable, refusal = load_eval_sets()
    assert len(answerable) == 160 and len(refusal) == 95 + 41
    assert all(q.gold and q.reason is None for q in answerable)
    assert all(not q.gold and q.reason for q in refusal)
    ids = [q.id for q in answerable + refusal]
    assert len(ids) == len(set(ids))
