"""Load the saved openFDA labels and their sections into Supabase.

Run from backend/ after `uv run python scripts/fetch_openfda.py`:
    uv run python scripts/ingest_openfda.py
"""

import json
from collections import Counter
from datetime import datetime

from rxcite import db
from rxcite.config import DATASETS_DIR, RESULTS_DIR
from rxcite.data.openfda import DISTRACTORS_FILE, fetch_label

# Label keys that aren't sections (metadata and packaging); *_table keys are dropped too
NOT_SECTIONS = {
    "openfda",
    "id",
    "set_id",
    "version",
    "effective_time",
    "spl_product_data_elements",
    "package_label_principal_display_panel",
}


# (drug_name, label, is_distractor) for every label found, read from the cache
def saved_labels() -> list[tuple[str, dict, bool]]:
    drug_list = json.loads((DATASETS_DIR / "drug_list_v1.json").read_text())
    excluded = set(drug_list["excluded_drugs"])
    drugs = [(d, False) for d in drug_list["drugs"] if d not in excluded]
    drugs += [(d, True) for d in json.loads(DISTRACTORS_FILE.read_text())]
    return [(d, label, dis) for d, dis in drugs if (label := fetch_label(d))]


# Each text field of a label becomes one section: {section name: text}
def sections(label: dict) -> dict[str, str]:
    out = {}
    for key, value in label.items():
        if key in NOT_SECTIONS or key.endswith("_table") or not isinstance(value, list):
            continue
        text = "\n\n".join(value).replace("\x00", "").strip()  # Postgres rejects NUL bytes
        if text:
            out[key] = text
    return out


def label_row(drug_name: str, label: dict, is_distractor: bool) -> tuple:
    info = label.get("openfda", {})
    return (
        label["set_id"],
        drug_name,
        info.get("generic_name", [None])[0],
        info.get("brand_name", [None])[0],
        datetime.strptime(label["effective_time"], "%Y%m%d").date(),
        is_distractor,
    )


# Corpus summary for the report: what was found, and how big the corpus is
def corpus_stats(labels: list[tuple[str, dict, bool]]) -> dict:
    drug_list = json.loads((DATASETS_DIR / "drug_list_v1.json").read_text())
    excluded = set(drug_list["excluded_drugs"])
    drugs = [d for d in drug_list["drugs"] if d not in excluded]
    distractors = json.loads(DISTRACTORS_FILE.read_text())
    all_sections = [sections(label) for _, label, _ in labels]
    product_types = Counter(
        label.get("openfda", {}).get("product_type", ["?"])[0] for _, label, _ in labels
    )
    section_counts = Counter(name for secs in all_sections for name in secs)
    return {
        "drug_list": {
            "drugs": len(drugs),
            "found": sum(fetch_label(d) is not None for d in drugs),
            "not_found": [d for d in drugs if fetch_label(d) is None],
        },
        "distractors": {
            "drugs": len(distractors),
            "found": sum(fetch_label(d) is not None for d in distractors),
            "not_found": [d for d in distractors if fetch_label(d) is None],
        },
        "labels": len(labels),
        "labels_by_product_type": dict(product_types.most_common()),
        "sections": sum(section_counts.values()),
        "sections_per_label": round(sum(section_counts.values()) / len(labels), 1),
        "total_text_chars": sum(len(t) for secs in all_sections for t in secs.values()),
        "most_common_sections": dict(section_counts.most_common(15)),
    }


if __name__ == "__main__":
    labels = saved_labels()
    label_rows = [label_row(*item) for item in labels]
    section_rows = [
        (label["set_id"], name, text)
        for _, label, _ in labels
        for name, text in sections(label).items()
    ]
    with db.connect() as conn, conn.cursor() as cur:
        # upserts, so running twice updates rows instead of duplicating them
        cur.executemany(
            """insert into labels (set_id, drug_name, generic_name, brand_name,
                                   effective_date, is_distractor)
               values (%s, %s, %s, %s, %s, %s)
               on conflict (set_id) do update set
                   drug_name = excluded.drug_name, generic_name = excluded.generic_name,
                   brand_name = excluded.brand_name, effective_date = excluded.effective_date,
                   is_distractor = excluded.is_distractor""",
            label_rows,
        )
        cur.executemany(
            """insert into sections (set_id, section, text) values (%s, %s, %s)
               on conflict (set_id, section) do update set text = excluded.text""",
            section_rows,
        )
        n_labels, n_sections = cur.execute(
            "select (select count(*) from labels), (select count(*) from sections)"
        ).fetchone()
    print(f"Loaded {len(label_rows)} labels and {len(section_rows)} sections")
    print(f"Supabase now has {n_labels} labels and {n_sections} sections")

    RESULTS_DIR.mkdir(exist_ok=True)
    stats_path = RESULTS_DIR / "corpus_stats.json"
    stats_path.write_text(json.dumps(corpus_stats(labels), indent=2) + "\n")
    print(f"Wrote {stats_path}")
