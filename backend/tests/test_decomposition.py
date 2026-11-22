from rxcite.pipelines.base import Chunk
from rxcite.query import decomposition
from rxcite.query.decomposition import decompose, merge_round_robin, search_queries


class FakeResponse:
    def __init__(self, text):
        self.text = text


def fake_reply(monkeypatch, text):
    monkeypatch.setattr(decomposition.llm, "call", lambda *a, **k: FakeResponse(text))


def test_split_is_capped_at_three(monkeypatch):
    fake_reply(monkeypatch, '{"sub_questions": ["a?", " b? ", "", "c?", "d?"]}')
    assert decompose("q") == ["a?", "b?", "c?"]  # blanks dropped, trimmed, max 3


def test_cut_off_or_empty_reply_falls_back_to_question(monkeypatch):
    fake_reply(monkeypatch, "")
    assert decompose("original?") == ["original?"]
    fake_reply(monkeypatch, '{"sub_questions": []}')
    assert decompose("original?") == ["original?"]


def chunks(*ids):
    return [Chunk(i, "s", "d", "sec", "") for i in ids]


def test_merge_takes_turns_and_skips_duplicates():
    a, b, c = chunks("a1", "a2", "a3"), chunks("b1", "a1", "b3"), chunks("c1", "c2")
    merged = merge_round_robin([a, b, c], k=10)
    assert [x.chunk_id for x in merged] == ["a1", "b1", "c1", "a2", "c2", "a3", "b3"]


def test_merge_stops_at_k_and_handles_one_list():
    a, b = chunks("a1", "a2", "a3"), chunks("b1", "b2", "b3")
    assert [x.chunk_id for x in merge_round_robin([a, b], k=3)] == ["a1", "b1", "a2"]
    assert merge_round_robin([a], k=10) == a  # single-part question: same as Naive
    assert merge_round_robin([]) == []


def test_identical_sub_question_is_searched_once():
    assert search_queries("How to stop X?", ["how to  stop x?"]) == ["How to stop X?"]
    assert search_queries("q", ["a", "b"]) == ["q", "a", "b"]


def test_single_sub_question_keeps_original_words(monkeypatch):
    fake_reply(monkeypatch, '{"sub_questions": ["active ingredients in balsalazide"]}')
    assert decompose("active incredience in balsalazide") == ["active incredience in balsalazide"]
