# Load the evaluation sets from datasets/ into two lists the harness can use
#       answerable: MedicationQA questions with reviewed gold (retrieval, judge, false refusals)
#       refusal: MedicationQA questions with no supporting label + the handwritten refusal set

import csv
import json
from dataclasses import dataclass, field

from rxcite.config import DATASETS_DIR

GoldSection = tuple[str, str]  # (set_id, section)
VERSIONS = ["v1", "v2"]  # medicationqa_<v>.json + gold_<v>.csv; v2 is the top-up


@dataclass
class EvalQuestion:
    id: str
    question: str
    type: str  # MedicationQA question type, or refusal group for the handwritten set
    gold: list[GoldSection] = field(default_factory=list)  # empty for refusal questions
    reference: str | None = None  # MedicationQA reference answer (for the judge)
    reason: str | None = None  # why it should be refused (refusal questions only)


# Every (label, section) pair of a gold row; each listed section is in each listed label
def gold_pairs(row: dict) -> list[GoldSection]:
    return [(s, sec) for s in row["set_ids"].split(";") for sec in row["sections"].split(";")]


# (answerable, refusal) questions from all versions plus the handwritten refusal set
def load_eval_sets() -> tuple[list[EvalQuestion], list[EvalQuestion]]:
    answerable, refusal = [], []
    for v in VERSIONS:
        questions = json.loads((DATASETS_DIR / f"medicationqa_{v}.json").read_text())["questions"]
        gold = {r["id"]: r for r in csv.DictReader((DATASETS_DIR / f"gold_{v}.csv").open())}
        for q in questions:
            row = gold[q["id"]]
            item = EvalQuestion(q["id"], q["question"], q["type"], reference=q["answer"])
            if row["no_label"] == "yes":
                item.reason = "no_supporting_label"
                refusal.append(item)
            else:
                item.gold = gold_pairs(row)
                answerable.append(item)
    handwritten = json.loads((DATASETS_DIR / "refusal_v1.json").read_text())["questions"]
    refusal += [
        EvalQuestion(q["id"], q["question"], q["type"], reason=q["type"]) for q in handwritten
    ]
    return answerable, refusal
