import csv
import json
from pathlib import Path

DATASETS_DIR = Path(__file__).resolve().parents[2] / "datasets"


QUESTION_SETS = ["medicationqa_v1.json", "medicationqa_v2.json", "refusal_v1.json"]


def load(name: str) -> dict:
    return json.loads((DATASETS_DIR / name).read_text())


def test_question_ids_unique_and_typed():
    for name in QUESTION_SETS:
        questions = load(name)["questions"]
        ids = [q["id"] for q in questions]
        assert len(ids) == len(set(ids)), f"duplicate IDs in {name}"
        assert all(q["type"] for q in questions), f"missing type in {name}"


def test_ids_unique_across_sets():
    ids = [q["id"] for name in QUESTION_SETS for q in load(name)["questions"]]
    assert len(ids) == len(set(ids))


def test_excluded_drugs_not_in_drug_list():
    drug_list = load("drug_list_v1.json")
    assert not set(drug_list["excluded_drugs"]) & set(drug_list["drugs"])


def test_gold_matches_questions():
    for v in ["v1", "v2"]:
        rows = list(csv.DictReader((DATASETS_DIR / f"gold_{v}.csv").open()))
        assert [r["id"] for r in rows] == [
            q["id"] for q in load(f"medicationqa_{v}.json")["questions"]
        ]
        for r in rows:
            assert r["reviewed"] == "yes", r["id"]
            assert bool(r["sections"]) != (r["no_label"] == "yes"), r["id"]  # gold xor no_label
