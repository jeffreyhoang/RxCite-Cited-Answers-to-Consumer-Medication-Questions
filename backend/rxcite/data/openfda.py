# Downloads openFDA labels into data/raw/openfda/

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from rxcite.config import DATA_DIR

API_URL = "https://api.fda.gov/drug/label.json"
CACHE = DATA_DIR / "raw" / "openfda"
SEARCH_FIELDS = ["generic_name", "brand_name", "substance_name"]  # tried in this order
PRODUCT_TYPES = ["HUMAN PRESCRIPTION DRUG", "HUMAN OTC DRUG"]  # prescription first
NOT_IN_OPENFDA = {"vitamin a"}  # only wrong-product matches (e.g. "Vitamin A D" skin ointment)


# Up to 100 labels whose openfda.<field> has every component of name, newest first ([] if none)
def search(field: str, name: str) -> list[dict]:
    terms = " AND ".join(f'openfda.{field}:"{c}"' for c in components(name))
    query = urllib.parse.urlencode({"search": terms, "sort": "effective_time:desc", "limit": 100})
    time.sleep(0.3)  # stay under 240 requests per minute
    try:
        with urllib.request.urlopen(f"{API_URL}?{query}", timeout=30) as resp:
            return json.load(resp)["results"]
    except urllib.error.HTTPError as e:
        if e.code == 404:  # openFDA answers 404 when nothing matches
            return []
        raise


# Split "hydrocodone bitartrate and acetaminophen" into ["hydrocodone bitartrate", "acetaminophen"]
def components(name: str) -> list[str]:
    return [c.strip() for c in re.split(r",| and ", name.lower()) if c.strip()]


# Same drug(s) as ours, allowing salt suffixes ("atorvastatin calcium"); combinations don't match
def same_drug(label_name: str, name: str) -> bool:
    ours, theirs = components(name), components(label_name)
    return len(ours) == len(theirs) and all(
        any(t == o or t.startswith(o + " ") for t in theirs) for o in ours
    )


# Newest matching label, prescription before OTC; None if nothing matches
def pick_label(labels: list[dict], name: str) -> dict | None:
    for product_type in PRODUCT_TYPES:
        for label in labels:  # already newest first
            info = label.get("openfda", {})
            # not substance_name: it lists each ingredient of a combination on its own
            names = info.get("generic_name", []) + info.get("brand_name", [])
            if product_type in info.get("product_type", []) and any(
                same_drug(n, name) for n in names
            ):
                return label
    return None


# One label for a drug, from the cache or the API (not-found results are cached too)
def fetch_label(name: str) -> dict | None:
    path = CACHE / f"{re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')}.json"
    if path.exists():
        return json.loads(path.read_text())["label"]
    label = None
    for field in SEARCH_FIELDS if name not in NOT_IN_OPENFDA else []:
        label = pick_label(search(field, name), name)
        if label:
            break
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"drug": name, "label": label}))
    return label


NUM_DISTRACTORS = 300
DISTRACTORS_FILE = DATA_DIR / "raw" / "openfda_distractors.json"
SKIP_DISTRACTORS = {"oxygen", "nitrogen", "nitrous oxide", "carbon dioxide", "helium", "air"}


# Most common prescription generic names (by label count) that aren't already taken, cached
def pick_distractors(taken: list[str]) -> list[str]:
    if DISTRACTORS_FILE.exists():
        return json.loads(DISTRACTORS_FILE.read_text())
    query = urllib.parse.urlencode(
        {
            "search": 'openfda.product_type:"HUMAN PRESCRIPTION DRUG"',
            "count": "openfda.generic_name.exact",
            "limit": 1000,
        }
    )
    with urllib.request.urlopen(f"{API_URL}?{query}", timeout=30) as resp:
        terms = [r["term"].lower() for r in json.load(resp)["results"]]  # most labels first
    chosen: list[str] = []
    for term in terms:
        # skip gases and salt forms of drugs we already have ("atorvastatin calcium")
        if term in SKIP_DISTRACTORS or any(
            same_drug(term, t) or same_drug(t, term) for t in taken + chosen
        ):
            continue
        chosen.append(term)
        if len(chosen) == NUM_DISTRACTORS:
            break
    DISTRACTORS_FILE.write_text(json.dumps(chosen, indent=2))
    return chosen
