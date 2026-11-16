# BM25 (Okapi) keyword search over a chunk table, the sparse half of Advanced RAG's hybrid search
#       Built from Supabase over search_text, then pickled to data/indexes/bm25_<set>.pkl
#       (rebuilt when the table's row count changes).

import pickle
import re
from functools import cache

import numpy as np

from rxcite import db
from rxcite.config import DATA_DIR, TOP_K
from rxcite.pipelines.base import Chunk, search_text
from rxcite.retrieval import chunk_table

INDEX_DIR = DATA_DIR / "indexes"

# Short list of common English words that carry no meaning for search. No stemming: drug names
# matter more than word endings, and it avoids another dependency.
STOPWORDS = frozenset(
    """a about after an and any are as at be been before being but by can could did do does
    doing for from had has have having he her his how i if in into is it its may me might
    must my no not of on or our she should so than that the their them then there these they
    this those to too was we were what when where which while who why will with would you
    your""".split()
)

# Every chunk of a table, with its drug name (for search_text and for the returned Chunks)
ROWS_SQL = """
select c.chunk_id, c.set_id, l.drug_name, c.section, c.text
from {table} c join labels l using (set_id)
order by c.chunk_id
"""
COUNT_SQL = "select count(*) from {table}"


# Lowercase, split on anything that isn't a letter or digit, drop stopwords
def tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t and t not in STOPWORDS]


def index_path(chunk_set: str):
    return INDEX_DIR / f"bm25_{chunk_set}.pkl"


# BM25 index of one chunk set: (rows, BM25Okapi), rows as (chunk_id, set_id, drug, section, text)
@cache
def index(chunk_set: str = "fixed"):
    table = chunk_table(chunk_set)
    count = db.query(COUNT_SQL.format(table=table))[0][0]
    path = index_path(chunk_set)
    if path.exists():
        with path.open("rb") as f:
            saved = pickle.load(f)
        if saved["count"] == count:
            return saved["rows"], saved["bm25"]

    from rank_bm25 import BM25Okapi  # only needed when (re)building

    rows = [tuple(r) for r in db.query(ROWS_SQL.format(table=table))]
    bm25 = BM25Okapi([tokenize(search_text(drug, sec, text)) for _, _, drug, sec, text in rows])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump({"count": count, "rows": rows, "bm25": bm25}, f)
    tmp.replace(path)
    return rows, bm25


# Top-k chunks by BM25 score, best first (ties keep table order). Chunks with no matching
# terms (score 0) are left out, so fewer than k can come back.
def bm25_search(query: str, k: int = TOP_K, chunk_set: str = "fixed") -> list[Chunk]:
    rows, bm25 = index(chunk_set)
    tokens = tokenize(query)
    if not tokens or not rows:
        return []
    scores = np.asarray(bm25.get_scores(tokens))
    top = np.argsort(-scores, kind="stable")[:k]
    return [Chunk(*rows[i], score=float(scores[i])) for i in top if scores[i] > 0]
