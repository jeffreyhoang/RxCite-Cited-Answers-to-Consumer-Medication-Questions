# Question splitting for Decomposition RAG: one question -> 1-3 self-contained sub-questions

import json

from rxcite import llm
from rxcite.config import GENERATOR_MODEL, TOP_K
from rxcite.pipelines.base import Chunk

MAX_SUB_QUESTIONS = 3

SYSTEM = """You prepare consumer medication questions for searching FDA drug labels.

Split the question into separate sub-questions only when it asks about more than one thing \
that would be found in different places, for example two different drugs, or two different \
kinds of information such as an interaction and when to take a dose. Return at most 3.

Rules:
- If the question asks one thing, return it unchanged as the only sub-question. Don't split \
just to rephrase.
- Make each sub-question self-contained: repeat the drug name instead of "it" or "this".
- Keep the user's meaning. Don't add facts, drugs or topics that aren't in the question.
- If the question isn't about medicine, return it unchanged."""

SCHEMA = {
    "type": "object",
    "properties": {"sub_questions": {"type": "array", "items": {"type": "string"}}},
    "required": ["sub_questions"],
    "additionalProperties": False,
}


def decompose(query: str, model: str = GENERATOR_MODEL) -> list[str]:
    resp = llm.call(query, model, system=SYSTEM, max_tokens=1000, schema=SCHEMA)
    try:
        subs = json.loads(resp.text)["sub_questions"]
    except (json.JSONDecodeError, KeyError):  # cut off or malformed: treat as one part
        return [query]
    subs = [s.strip() for s in subs if s.strip()][:MAX_SUB_QUESTIONS]
    # Not split: keep the user's exact words. Haiku often fixes typos here, which would turn
    # this step into query rewriting instead of decomposition.
    return subs if len(subs) > 1 else [query]


# Texts to search: the original question first, then each sub-question, each text once
# (a single-part split equals the original, so it is searched once, exactly like Naive)
def search_queries(query: str, sub_questions: list[str]) -> list[str]:
    seen, out = set(), []
    for text in [query, *sub_questions]:
        key = " ".join(text.lower().split())
        if key not in seen:
            seen.add(key)
            out.append(text)
    return out


# Take turns: #1 of each list, then #2 of each, ...; skip chunks already taken; stop at k
def merge_round_robin(lists: list[list[Chunk]], k: int = TOP_K) -> list[Chunk]:
    merged, taken = [], set()
    for rank in range(max((len(lst) for lst in lists), default=0)):
        for lst in lists:
            if rank < len(lst) and lst[rank].chunk_id not in taken:
                taken.add(lst[rank].chunk_id)
                merged.append(lst[rank])
                if len(merged) == k:
                    return merged
    return merged


# Check: a few examples, then how often the 160 answerable questions are split
if __name__ == "__main__":
    from collections import Counter

    from rxcite.eval.datasets import load_eval_sets

    examples = [
        "What are the side effects of metformin?",
        "how to stop taking bisoprolol",
        "can i take doxepin and prexum together?",
        "what is the difference between gabbentibe lyrica for pain",
        "find the nearest mcdonalds",
    ]
    for q in examples:
        print(f"{q}\n  -> {decompose(q)}")

    answerable, _ = load_eval_sets()
    splits = {q.id: decompose(q.question) for q in answerable}
    counts = Counter(len(s) for s in splits.values())
    print(f"\n160 answerable questions by number of sub-questions: {dict(sorted(counts.items()))}")
    for q in answerable:
        if len(splits[q.id]) > 1:
            print(f"  {q.id} ({q.type}): {q.question[:60]!r}\n     -> {splits[q.id]}")
