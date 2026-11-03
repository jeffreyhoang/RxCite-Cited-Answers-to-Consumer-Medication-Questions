"""Download one openFDA label per drug-list drug and per distractor (cached in data/raw/openfda/).

Run from backend/: uv run python scripts/fetch_openfda.py
"""

import json

from rxcite.config import DATASETS_DIR
from rxcite.data.openfda import fetch_label, pick_distractors

if __name__ == "__main__":
    drug_list = json.loads((DATASETS_DIR / "drug_list_v1.json").read_text())
    excluded = set(drug_list["excluded_drugs"])
    drugs = [d for d in drug_list["drugs"] if d not in excluded]
    missing = [d for d in drugs if fetch_label(d) is None]
    print(f"Found {len(drugs) - len(missing)} of {len(drugs)} drugs")
    print("Not found:", ", ".join(missing))

    distractors = pick_distractors(drug_list["drugs"] + drug_list["excluded_drugs"])
    missing = [d for d in distractors if fetch_label(d) is None]
    print(f"Found {len(distractors) - len(missing)} of {len(distractors)} distractors")
    print("Not found:", ", ".join(missing))
