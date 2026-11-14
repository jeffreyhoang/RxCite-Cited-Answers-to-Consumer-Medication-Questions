"""Draft the gold (label, section) mapping for the MedicationQA questions, for team review.

Run from backend/ after ingest_openfda.py, with the question set version:
    uv run python scripts/build_gold.py v2

Writes datasets/gold_<version>.csv. Once reviewed, that CSV is the source of truth, so the
script refuses to overwrite a CSV that has reviewed rows.
"""

import csv
import json
import re
import sys

from build_eval_sets import load_medicationqa

from rxcite import db
from rxcite.config import DATASETS_DIR

MAX_SECTIONS = 3

# Each entry is a list of slots; a slot is a list of section names, and the first one the label
# has is used (labels name some sections differently, e.g. warnings vs. warnings_and_cautions)
WARNINGS = ["warnings_and_cautions", "warnings", "precautions"]
PATIENT_INFO = ["information_for_patients", "spl_patient_package_insert", "spl_medguide"]

# MedicationQA "Section Title" (lowercased, ending punctuation removed) -> sections
TITLE_SECTIONS = {
    "dosage and administration": [["dosage_and_administration"]],
    "dosage & administration": [["dosage_and_administration"]],
    "how should this medicine be used": [["dosage_and_administration"]],
    "directions": [["dosage_and_administration"]],
    "use as directed": [["dosage_and_administration"]],
    "why is this medication prescribed": [["indications_and_usage"]],
    "indications and usage": [["indications_and_usage"]],
    "uses": [["indications_and_usage"]],
    "ingredients and appearance": [["description"], ["how_supplied"]],
    "product characteristics": [["description"], ["how_supplied"]],
    "pharmaceutical form": [["dosage_forms_and_strengths", "how_supplied"]],
    "dosage forms and strengths": [["dosage_forms_and_strengths"]],
    "description": [["description"]],
    "how supplied": [["how_supplied"]],
    "storage and handling": [["storage_and_handling", "how_supplied"]],
    "clinical pharmacology": [["clinical_pharmacology"]],
    "warnings": [WARNINGS],
    "warnings and precautions": [WARNINGS],
    "boxed warning": [["boxed_warning"]],
    "important warning": [["boxed_warning"]],
    "adverse reactions": [["adverse_reactions"]],
    "what side effects can this medication cause": [["adverse_reactions"]],
    "drug interactions": [["drug_interactions"]],
    "contraindications": [["contraindications"]],
    "overdosage": [["overdosage"]],
    "patient information": [PATIENT_INFO],
    "consumer information": [PATIENT_INFO],
}

# Question type -> sections, used when the title isn't a label section (blogs, MedlinePlus, ...)
TYPE_SECTIONS = {
    "Action": [["mechanism_of_action", "clinical_pharmacology"]],
    "Action/time": [["pharmacokinetics", "clinical_pharmacology"]],
    "Alternatives": [["indications_and_usage"]],
    "Appearance": [["how_supplied"], ["dosage_forms_and_strengths", "description"]],
    "Comparison": [["indications_and_usage"], ["clinical_pharmacology"]],
    "Contraindication": [["contraindications"], WARNINGS],
    "Dose": [["dosage_and_administration"]],
    "Indication": [["indications_and_usage", "purpose"]],
    "Information": [["indications_and_usage"], ["description"]],
    "Ingredient": [["description", "active_ingredient"], ["inactive_ingredient"]],
    "Interaction": [["drug_interactions"]],
    "Overdose": [["overdosage"]],
    "Side effects": [["adverse_reactions"], WARNINGS],
    "Stopping/tapering": [["dosage_and_administration"], WARNINGS],
    "Storage and disposal": [["storage_and_handling", "how_supplied"]],
    "Usage": [["dosage_and_administration"], PATIENT_INFO],
    "Usage/duration": [["dosage_and_administration"]],
    "Usage/time": [["dosage_and_administration"]],
}


# Slots from the Section Title (a title can name several, e.g. "A; B" or "A/B"), else None
def title_slots(title: str | None) -> list[list[str]] | None:
    if not title:
        return None
    title = title.lower().strip().rstrip("?:").strip()
    parts = [title] if title in TITLE_SECTIONS else re.split(r";|/| & ", title)
    slots = [s for p in parts for s in TITLE_SECTIONS.get(p.strip().rstrip("?:").strip(), [])]
    return slots or None


# First section of each slot that one of the labels has, up to MAX_SECTIONS
def pick_sections(slots: list[list[str]], label_sections: list[set[str]]) -> list[str]:
    have = set().union(*label_sections)
    picked = []
    for slot in slots:
        name = next((s for s in slot if s in have), None)
        if name and name not in picked:
            picked.append(name)
    return picked[:MAX_SECTIONS]


# {drug_name: (set_id, {section names})} for the drug-list labels in Supabase
def corpus() -> dict[str, tuple[str, set[str]]]:
    with db.connect() as conn:
        rows = conn.execute(
            """select l.drug_name, l.set_id, array_agg(s.section)
               from labels l join sections s using (set_id)
               where not l.is_distractor group by 1, 2"""
        ).fetchall()
    return {drug: (set_id, set(secs)) for drug, set_id, secs in rows}


def gold_rows(version: str) -> list[dict]:
    qa = json.loads((DATASETS_DIR / f"medicationqa_{version}.json").read_text())
    questions = qa["questions"]
    aliases = json.loads((DATASETS_DIR / "drug_list_v1.json").read_text())["aliases"]
    aliases |= qa.get("aliases", {})  # v2 brings its own drug-name cleanup
    titles = load_medicationqa().set_index("question")["Section Title"]
    labels = corpus()
    rows = []
    for q in questions:
        found = [labels[d] for d in aliases[q["drug"].lower()] if d in labels]
        title = titles.get(q["question"])
        title = None if not isinstance(title, str) else title
        label_secs = [secs for _, secs in found]
        from_title = title_slots(title)
        sections = pick_sections(from_title, label_secs) if from_title and found else []
        if found and not sections:  # title isn't a label section, or the label lacks it
            from_title = None
            sections = pick_sections(TYPE_SECTIONS[q["type"]], label_secs)
        rows.append(
            {
                "id": q["id"],
                "question": q["question"],
                "type": q["type"],
                "drug": q["drug"],
                "set_ids": ";".join(set_id for set_id, _ in found),
                "sections": ";".join(sections),
                "no_label": "yes" if not found else "",
                "reviewed": "",
                "note": f"from title: {title}" if from_title else "from type",
            }
        )
    return rows


if __name__ == "__main__":
    version = sys.argv[1]
    gold_csv = DATASETS_DIR / f"gold_{version}.csv"
    if gold_csv.exists() and any(r["reviewed"] for r in csv.DictReader(gold_csv.open())):
        sys.exit(f"{gold_csv} has reviewed rows; not overwriting it")
    rows = gold_rows(version)
    with gold_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    flagged = sum(r["no_label"] == "yes" for r in rows)
    from_title = sum(r["note"].startswith("from title") for r in rows)
    print(f"Wrote {gold_csv}: {len(rows)} rows, {flagged} flagged no_label")
    print(f"Sections from title: {from_title}, from type: {len(rows) - from_title}")
