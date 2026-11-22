import copy

from pytest import approx

from rxcite.pipelines.base import Chunk
from rxcite.retrieval.search import rrf


def chunks(*ids):
    return [Chunk(i, "s", "drug", "warnings", f"text {i}", score=0.5) for i in ids]


def ids(result):
    return [c.chunk_id for c in result]


def test_deduplicates_by_chunk_id():
    out = rrf([chunks("a", "b", "c"), chunks("b", "c", "d")])
    assert sorted(ids(out)) == ["a", "b", "c", "d"]


def test_found_in_both_lists_ranks_higher():
    # "x" is rank 2 in both lists; "a" and "b" are rank 1 in only one
    out = rrf([chunks("a", "x", "c"), chunks("b", "x", "d")])
    assert ids(out)[0] == "x"
    assert set(ids(out)[1:3]) == {"a", "b"}


def test_fused_scores():
    out = {c.chunk_id: c.score for c in rrf([chunks("a", "b"), chunks("b", "c")], k=60)}
    assert out["a"] == approx(1 / 61)
    assert out["b"] == approx(1 / 62 + 1 / 61)
    assert out["c"] == approx(1 / 62)


def test_top_n_cap():
    out = rrf([chunks(*"abcdef"), chunks(*"ghij")], top_n=3)
    assert len(out) == 3
    assert len(rrf([chunks(*"abc")], top_n=50)) == 3


def test_ties_keep_first_seen_order():
    # Same rank in different lists -> same score; the first list's chunk comes first
    out = rrf([chunks("a", "c"), chunks("b", "d")])
    assert ids(out) == ["a", "b", "c", "d"]


def test_inputs_not_mutated():
    lists = [chunks("a", "b"), chunks("b", "c")]
    before = copy.deepcopy(lists)
    out = rrf(lists)
    assert lists == before
    assert all(c.score == 0.5 for x in lists for c in x)
    assert out[0] is not lists[1][0]  # copies, not the input chunks


def test_empty():
    assert rrf([]) == [] and rrf([[], []]) == []
