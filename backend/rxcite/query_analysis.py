# Query analysis for Hierarchical RAG: which drug(s) and what kind of question?
#       Haiku picks drug names from the corpus list (or none) and one section type.
#       The JSON schema restricts drugs to that list, so it can't invent a name.
#       Check it with: uv run python -m rxcite.query_analysis

import json
from dataclasses import dataclass
from functools import cache

from rxcite import db, llm
from rxcite.config import GENERATOR_MODEL

# Section type -> label sections that usually answer it (used as a score bonus, not a filter)
SECTION_GROUPS = {
    "dosage": ["dosage_and_administration", "dosage_forms_and_strengths", "how_supplied"],
    "side_effects": ["adverse_reactions", "warnings", "warnings_and_cautions", "spl_medguide"],
    "interactions": ["drug_interactions", "drug_and_or_laboratory_test_interactions"],
    "warnings": ["boxed_warning", "warnings_and_cautions", "warnings", "precautions"],
    "contraindications": ["contraindications", "do_not_use"],
    "usage": ["indications_and_usage", "purpose", "clinical_studies"],
    "how_it_works": ["mechanism_of_action", "clinical_pharmacology", "pharmacodynamics"],
    "pregnancy_breastfeeding": [
        "pregnancy",
        "nursing_mothers",
        "use_in_specific_populations",
        "pregnancy_or_breast_feeding",
    ],
    "ingredients_appearance": [
        "description",
        "how_supplied",
        "active_ingredient",
        "inactive_ingredient",
    ],
    "storage": ["storage_and_handling", "how_supplied"],
    "overdose": ["overdosage"],
    "stopping": ["warnings", "warnings_and_cautions", "dosage_and_administration"],
    "other": [],
}

SYSTEM = """You read a consumer's medication question and identify:

1. drugs: every name from the corpus list below that is the drug the question asks about. \
Map brand names, misspellings and abbreviations to the generic name (for example, Rapaflo -> \
silodosin, Kadian -> morphine). Include every list entry that is the same drug in a different \
salt or form (for example, both "metformin hcl" and "metformin hydrochloride"). Include a \
combination product only if the question is about that combination. If the question names \
no drug, or its drug is not on the list, return an empty list. Never pick a different drug \
just because it is similar.

2. section_type: the kind of information asked for.

Corpus drug names:
"""


@dataclass
class QueryInfo:
    drugs: list[str]  # corpus drug names (labels.drug_name); empty if none found
    section_type: str  # a key of SECTION_GROUPS


@cache
def drug_names() -> list[str]:
    with db.connect() as conn:
        return [r[0] for r in conn.execute("select distinct drug_name from labels order by 1")]


def _schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "drugs": {"type": "array", "items": {"type": "string", "enum": drug_names()}},
            "section_type": {"type": "string", "enum": list(SECTION_GROUPS)},
        },
        "required": ["drugs", "section_type"],
        "additionalProperties": False,
    }


def analyze(query: str, model: str = GENERATOR_MODEL) -> QueryInfo:
    system = SYSTEM + "\n".join(drug_names())
    resp = llm.call(query, model, system=system, max_tokens=1000, schema=_schema())
    try:
        out = json.loads(resp.text)
    except json.JSONDecodeError:  # cut off (stop_reason max_tokens): treat as "no drug found"
        return QueryInfo([], "other")
    return QueryInfo(sorted(set(out["drugs"])), out["section_type"])


# set_ids of every label for these corpus drug names
def labels_for(drugs: list[str]) -> list[str]:
    if not drugs:
        return []
    rows = db.query("select set_id from labels where drug_name = any(%s) order by set_id", (drugs,))
    return [r[0] for r in rows]


# Check: how often the detected labels include the gold labels (160 answerable questions)
if __name__ == "__main__":
    from rxcite.eval.datasets import load_eval_sets

    answerable, _ = load_eval_sets()
    with db.connect() as conn:
        label_of = dict(conn.execute("select set_id, drug_name from labels").fetchall())
    any_hit = all_hit = none_found = 0
    misses = []
    for q in answerable:
        info = analyze(q.question)
        found = set(labels_for(info.drugs))
        gold = {set_id for set_id, _ in q.gold}
        any_hit += bool(found & gold)
        all_hit += gold <= found
        none_found += not found
        if not gold <= found:
            want = sorted({label_of[s] for s in gold})
            misses.append(f"  {q.id}: {q.question[:60]!r} -> {info.drugs} (gold: {want})")
    n = len(answerable)
    print(f"{n} answerable questions")
    print(f"  some gold label found: {any_hit} ({any_hit / n:.0%})")
    print(f"  all gold labels found: {all_hit} ({all_hit / n:.0%})")
    print(f"  no drug found:         {none_found} ({none_found / n:.0%})")
    print("Misses:\n" + "\n".join(misses))
