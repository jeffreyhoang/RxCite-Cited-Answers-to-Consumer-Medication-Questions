# Query rewriting for Advanced RAG (T7): consumer question -> keyword-style label search string
#       Only BM25 and dense search see the rewrite; the reranker and generator get the original.
#       Check it with: uv run python -m rxcite.rewrite

import json

from rxcite import llm
from rxcite.config import GENERATOR_MODEL

SYSTEM = """You rewrite a consumer's medication question into a short search query for \
FDA drug label text.

Rules:
- Keep every drug name from the question. If the question uses a brand name, keep it and add \
the generic name (for example, "Rapaflo" -> "Rapaflo silodosin").
- Translate everyday words into the terms drug labels use, for example: "drink" -> "alcohol", \
"side effects" -> "adverse reactions", "can I take it with" -> "drug interactions", \
"pregnant" -> "pregnancy", "how much should I take" -> "dosage and administration".
- Do not add facts, answers, or drugs that are not in the question.
- Write keywords, not a sentence. At most 25 words."""

SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
    "additionalProperties": False,
}


# Search string for the question; the original question if the reply is unusable
def rewrite_query(question: str, model: str = GENERATOR_MODEL) -> str:
    resp = llm.call(question, model, system=SYSTEM, max_tokens=500, schema=SCHEMA)
    try:
        query = json.loads(resp.text)["query"].strip()
    except (json.JSONDecodeError, KeyError):  # cut off or malformed: search the question as is
        return question
    return query or question


# Check: a few examples, then every answerable question's rewrite
if __name__ == "__main__":
    from rxcite.eval.datasets import load_eval_sets

    examples = [
        "can i drink while taking zestril?",
        "what are the side effects of metformin?",
        "how much tylenol can i take in a day",
    ]
    for q in examples:
        print(f"{q}\n  -> {rewrite_query(q)}")

    answerable, _ = load_eval_sets()
    print()
    for q in answerable:
        print(f"  {q.id}: {q.question[:60]!r}\n     -> {rewrite_query(q.question)}")
