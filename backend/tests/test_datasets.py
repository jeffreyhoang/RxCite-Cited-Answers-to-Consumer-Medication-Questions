import json
from pathlib import Path

DATASETS_DIR = Path(__file__).resolve().parents[2] / "datasets"


def load(name: str) -> dict:
    return json.loads((DATASETS_DIR / name).read_text())


def test_question_ids_unique_and_typed():
    for name in ["medicationqa_v1.json", "refusal_v1.json"]:
        questions = load(name)["questions"]
        ids = [q["id"] for q in questions]
        assert len(ids) == len(set(ids)), f"duplicate IDs in {name}"
        assert all(q["type"] for q in questions), f"missing type in {name}"


def test_ids_unique_across_sets():
    ids = [
        q["id"]
        for name in ["medicationqa_v1.json", "refusal_v1.json"]
        for q in load(name)["questions"]
    ]
    assert len(ids) == len(set(ids))


def test_excluded_drugs_not_in_drug_list():
    drug_list = load("drug_list_v1.json")
    assert not set(drug_list["excluded_drugs"]) & set(drug_list["drugs"])
