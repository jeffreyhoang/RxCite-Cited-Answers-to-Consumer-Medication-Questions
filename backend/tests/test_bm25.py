import pytest

from rxcite import bm25
from rxcite.bm25 import bm25_search, index, tokenize

ROWS = [
    ("m:adverse_reactions:0", "m", "metformin", "adverse_reactions", "Nausea, lactic acidosis."),
    (
        "l:drug_interactions:0",
        "l",
        "lisinopril",
        "drug_interactions",
        "Potassium raises potassium.",
    ),
    ("i:warnings:0", "i", "ibuprofen", "warnings", "Alcohol raises the risk of stomach bleeding."),
]


# Fake Supabase: answers the count and row queries, and records which ones ran
@pytest.fixture
def toy_db(tmp_path, monkeypatch):
    calls = []

    def fake_query(sql, params=None):
        if "count(*)" in sql:
            calls.append("count")
            return [(len(ROWS),)]
        calls.append("rows")
        return list(ROWS)

    monkeypatch.setattr(bm25.db, "query", fake_query)
    monkeypatch.setattr(bm25, "INDEX_DIR", tmp_path)
    index.cache_clear()
    yield calls
    index.cache_clear()


def test_tokenize():
    assert tokenize("Can I take Advil with ALCOHOL?") == ["take", "advil", "alcohol"]
    assert tokenize("metformin-HCl 500mg, 2.5 mg") == ["metformin", "hcl", "500mg", "2", "5", "mg"]
    assert tokenize("") == [] and tokenize("the of and") == []


def test_ranking_on_toy_index(toy_db):
    out = bm25_search("lisinopril potassium", k=3)
    assert [c.chunk_id for c in out] == ["l:drug_interactions:0"]  # no-match chunks left out
    assert out[0].drug_name == "lisinopril" and out[0].score > 0

    out = bm25_search("metformin alcohol bleeding", k=3)
    assert [c.chunk_id for c in out] == ["i:warnings:0", "m:adverse_reactions:0"]
    assert out[0].score > out[1].score


def test_header_is_searched(toy_db):
    # "interactions" is only in the section name, which search_text adds to the chunk
    assert [c.chunk_id for c in bm25_search("interactions")] == ["l:drug_interactions:0"]


def test_k_and_empty_query(toy_db):
    assert len(bm25_search("metformin lisinopril ibuprofen", k=2)) == 2
    assert bm25_search("the and of") == []


def test_pickle_reused_until_row_count_changes(toy_db, tmp_path, monkeypatch):
    bm25_search("alcohol")
    assert (tmp_path / "bm25_fixed.pkl").exists() and toy_db == ["count", "rows"]

    index.cache_clear()
    toy_db.clear()
    assert bm25_search("alcohol")[0].chunk_id == "i:warnings:0"
    assert toy_db == ["count"]  # loaded from the pickle

    index.cache_clear()
    toy_db.clear()
    fewer = ROWS[:2]
    monkeypatch.setattr(
        bm25.db, "query", lambda sql, p=None: [(2,)] if "count(*)" in sql else list(fewer)
    )
    assert bm25_search("alcohol") == []  # rebuilt without the ibuprofen chunk


def test_unknown_chunk_set(toy_db):
    with pytest.raises(ValueError):
        bm25_search("alcohol", chunk_set="nope")
