import itertools

import pytest

from rxcite.config import RERANK_CANDIDATES, TOP_K
from rxcite.generation import generate
from rxcite.pipelines import PIPELINES
from rxcite.pipelines import advanced as adv
from rxcite.pipelines.advanced import AdvancedPipeline
from rxcite.pipelines.base import Answer, Chunk
from rxcite.pipelines.baselines import dummy_generator

QUESTION = "can i drink while on advil?"
REWRITTEN = "ibuprofen Advil alcohol drug interactions"


def fake_chunks(prefix, n):
    return [Chunk(f"{prefix}{i}", "s", "drug", "warnings", "text", 1 / (i + 1)) for i in range(n)]


# Replaces every retrieval step in the advanced module and records how each was called
@pytest.fixture
def calls(monkeypatch):
    log = {"rewrite": [], "dense": [], "bm25": [], "rerank": []}

    def rewrite_query(q):
        log["rewrite"].append(q)
        return REWRITTEN

    def dense_search(q, k, set_ids=None, chunk_set="fixed"):
        log["dense"].append((q, k, chunk_set))
        return fake_chunks("d", k)

    def bm25_search(q, k, chunk_set="fixed"):
        log["bm25"].append((q, k, chunk_set))
        return fake_chunks("b", k)

    def rerank(q, chunks, k):
        log["rerank"].append((q, [c.chunk_id for c in chunks], k))
        return list(reversed(chunks))[:k]

    monkeypatch.setattr(adv, "rewrite_query", rewrite_query)
    monkeypatch.setattr(adv, "dense_search", dense_search)
    monkeypatch.setattr(adv, "bm25_search", bm25_search)
    monkeypatch.setattr(adv, "rerank_chunks", rerank)
    return log


FLAGS = list(itertools.product(["fixed", "section"], [False, True], [False, True], [False, True]))


@pytest.mark.parametrize("chunk_set,rewrite,hybrid,rerank", FLAGS)
def test_flags_call_the_right_steps(calls, chunk_set, rewrite, hybrid, rerank):
    p = AdvancedPipeline(dummy_generator, "t", chunk_set, rewrite, hybrid, rerank)
    top, info = p.retrieve_with_info(QUESTION)

    search_q = REWRITTEN if rewrite else QUESTION
    n = RERANK_CANDIDATES if (hybrid or rerank) else TOP_K
    assert calls["rewrite"] == ([QUESTION] if rewrite else [])
    assert calls["dense"] == [(search_q, n, chunk_set)]
    assert calls["bm25"] == ([(search_q, n, chunk_set)] if hybrid else [])
    assert info["search_query"] == search_q
    assert len(top) <= TOP_K

    ids = [c.chunk_id for c in top]
    if rerank:
        [(q, cand_ids, k)] = calls["rerank"]
        assert q == search_q and k == TOP_K  # the reranker sees the search query
        assert len(cand_ids) == RERANK_CANDIDATES
        assert any(c.startswith("b") for c in cand_ids) == hybrid  # fused candidates
        assert ids == list(reversed(cand_ids))[:TOP_K]
        assert info["rerank_s"] >= 0
    else:
        assert calls["rerank"] == [] and "rerank_s" not in info
        if hybrid:  # RRF: d0 and b0 tie at rank 1, dense list first
            assert ids[:2] == ["d0", "b0"] and len(ids) == TOP_K
        else:
            assert ids == [f"d{i}" for i in range(TOP_K)]
    assert p.retrieve(QUESTION) == top


def test_answer_records_search_query_and_uses_original_question(calls):
    seen = []

    def generator(q, chunks):
        seen.append((q, len(chunks)))
        return Answer(text="ok")

    result = AdvancedPipeline(generator=generator).answer(QUESTION)
    assert seen == [(QUESTION, TOP_K)]
    assert result.question == QUESTION and result.architecture == "advanced"
    assert result.info["search_query"] == REWRITTEN and "rerank_s" in result.info
    assert calls["rerank"][0][0] == REWRITTEN


def test_registry_variants():
    expected = {  # name: (chunk_set, rewrite, hybrid, rerank), cumulative ablation
        "adv-section": ("section", False, False, False),
        "adv-rewrite": ("section", True, False, False),
        "adv-hybrid": ("section", True, True, False),
        "advanced-retrieval": ("section", True, True, True),
        "advanced": ("section", True, True, True),
    }
    for name, flags in expected.items():
        p = PIPELINES[name]()  # builds without touching the network
        assert isinstance(p, AdvancedPipeline) and p.name == name  # own cache folder each
        assert (p.chunk_set, p.rewrite, p.hybrid, p.rerank) == flags
        assert p.generator is (generate if name == "advanced" else dummy_generator)
