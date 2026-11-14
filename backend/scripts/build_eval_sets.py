"""Build the evaluation question sets.

MedicationQA source: https://github.com/abachaa/Medication_QA_MedInfo2019 (downloaded 2026-11-02)

Run from backend/: uv run python scripts/build_eval_sets.py
"""

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
MQA_XLSX = REPO_ROOT / "data" / "raw" / "medicationqa" / "MedInfo2019-QA-Medications.xlsx"
DATASETS_DIR = REPO_ROOT / "datasets"

VERSION = "v1"
SEED = 42  # same as RANDOM_SEED in rxcite/config.py
SAMPLE_SIZE = 200
MIN_TYPE_SIZE = 7  # drop question types with fewer questions than this
MIN_PER_TYPE = 5  # every kept type gets at least this many questions


# Load MedicationQA, drop unusable rows, keep only the main question types
def load_medicationqa() -> pd.DataFrame:
    df = pd.read_excel(MQA_XLSX)
    df = df.rename(
        columns={
            "Question": "question",
            "Focus (Drug)": "drug",
            "Question Type": "type",
            "Answer": "answer",
            "URL": "url",
        }
    )
    for col in ["question", "drug", "type"]:
        df[col] = df[col].str.strip()
    df = df.dropna(subset=["question", "drug", "type"])
    counts = df["type"].value_counts()
    df = df[df["type"].isin(counts[counts >= MIN_TYPE_SIZE].index)]
    return df.drop_duplicates(subset="question")  # same question can appear under several types


# How many questions to take per type: MIN_PER_TYPE each, the rest in proportion to type size
def allocate(counts: pd.Series, total: int) -> dict[str, int]:
    alloc = {t: min(MIN_PER_TYPE, n) for t, n in counts.items()}
    spare = {t: counts[t] - alloc[t] for t in alloc}
    remaining = total - sum(alloc.values())
    shares = {t: remaining * s / sum(spare.values()) for t, s in spare.items()}
    for t in alloc:
        alloc[t] += int(shares[t])
    # hand out slots lost to rounding, largest remainder first (ties broken by name)
    leftover = total - sum(alloc.values())
    for t in sorted(shares, key=lambda t: (-(shares[t] % 1), t))[:leftover]:
        alloc[t] += 1
    return alloc


# Stratified sample with stable IDs, in original dataset order
def sample_questions(df: pd.DataFrame) -> list[dict]:
    alloc = allocate(df["type"].value_counts(), SAMPLE_SIZE)
    sample = pd.concat(
        df[df["type"] == t].sample(n=n, random_state=SEED) for t, n in sorted(alloc.items())
    ).sort_index()
    return [
        {
            "id": f"mqa-{i:03d}",
            "question": row.question,
            "type": row.type,
            "drug": row.drug,
            "answer": None if pd.isna(row.answer) else row.answer,
            "url": None if pd.isna(row.url) else row.url,
        }
        for i, row in enumerate(sample.itertuples(), start=1)
    ]


# Hand cleanup of MedicationQA drug names (lowercased) -> drug names to fetch from openFDA.
# Brands map to generics, doses and typos are removed, and pairs of separate drugs are split.
# Combination products stay as one name (e.g. "hydrocodone and acetaminophen").
# An empty list means no single drug to fetch: a drug class, an illicit drug, or a supplement.
# Names not listed here are used as is.
DRUG_OVERRIDES: dict[str, list[str]] = {
    '"bupropion patch" (probably buprenorphine [butrans])': ["buprenorphine"],
    "adacel tdap": ["adacel"],
    "aldactazide": ["spironolactone and hydrochlorothiazide"],
    "amitriptyline, alprazolam": ["amitriptyline", "alprazolam"],
    "amlodipine; metoprolol": ["amlodipine", "metoprolol"],
    "amoxicillin & hydrocodone": ["amoxicillin", "hydrocodone"],
    "aspirin 81 mg": ["aspirin"],
    "ativan": ["lorazepam"],
    "atorvastatin; lipitor": ["atorvastatin"],
    "bimatoprost ophthalmic": ["bimatoprost"],
    "boric acid powder": ["boric acid"],
    "bsg vaccine": ["bcg vaccine"],
    "bupropion hcl 300mg": ["bupropion"],
    "calan sr": ["verapamil"],
    "calcium and alendronate": ["alendronate"],
    "ceftriaxone injection": ["ceftriaxone"],
    "ciprodex": ["ciprofloxacin and dexamethasone"],
    "cobalamin": ["cyanocobalamin"],
    "cortisone shots": ["cortisone"],
    "cymbalta and effexor": ["duloxetine", "venlafaxine"],
    "cymbalta, muscle relaxants": ["duloxetine"],
    "diazepam gel": ["diazepam"],
    "diclofenac sodium": ["diclofenac"],
    "doxepin, prexum": ["doxepin", "perindopril"],
    "dulera": ["mometasone and formoterol"],
    "entocort ec": ["budesonide"],
    "estradiol transdermal patch": ["estradiol"],
    "estradiol transdermal patch; estrogen": ["estradiol"],
    "famotidine 10 mg": ["famotidine"],
    "fosamax": ["alendronate"],
    "gabapentin and lyrica": ["gabapentin", "pregabalin"],
    "hydrocodone + acetaminophen": ["hydrocodone and acetaminophen"],
    "kadian": ["morphine"],
    "klonopin": ["clonazepam"],
    "latanoprost ophthalmic": ["latanoprost"],
    "levaquin": ["levofloxacin"],
    "lexapro": ["escitalopram"],
    "lidocaine rinse": ["lidocaine"],
    "magnesium and dilantin": ["phenytoin"],
    "metoprolol tartrate": ["metoprolol"],
    "neupro and ropinirole": ["rotigotine", "ropinirole"],
    "normal saline": ["sodium chloride"],
    "norvasc": ["amlodipine"],
    "oxycodone er": ["oxycodone"],
    'oxycodone hcl 5 mg; oxycodone "10-325"': ["oxycodone", "oxycodone and acetaminophen"],
    "oxycodone, fentanyl": ["oxycodone", "fentanyl"],
    "pneumococcal vaccines": ["pneumococcal vaccine"],
    "premarin cream": ["conjugated estrogens"],
    "proair hfa": ["albuterol"],
    "prozac": ["fluoxetine"],
    "qvar": ["beclomethasone"],
    "ramipril, cabergoline": ["ramipril", "cabergoline"],
    "rapaflo": ["silodosin"],
    "riva-clonazepam (clonazepam)": ["clonazepam"],
    "singulair": ["montelukast"],
    "spiriva": ["tiotropium"],
    "spironolactone + furosemide": ["spironolactone", "furosemide"],
    "tegretol": ["carbamazepine"],
    "testosterone injection": ["testosterone"],
    "topamax": ["topiramate"],
    "valium": ["diazepam"],
    "vicodin": ["hydrocodone and acetaminophen"],
    "vitamin b-12": ["cyanocobalamin"],
    "vitamin b12": ["cyanocobalamin"],
    "warfarin; cabbage": ["warfarin"],
    "xanax": ["alprazolam"],
    # no single drug to fetch
    "anti inflammatory gel": [],
    "anticholinergic drugs": [],
    "barbiturates": [],
    "bioflavonoid": [],
    "diabetes medicine": [],
    "drugs that cause erectile dysfunction": [],
    "drugs that cause hirsutism": [],
    "drugs that reduce cholesterol": [],
    "ecstacy": [],
    "heroin": [],
    "high cholesterol medication": [],
    "marijuana": [],
    "n/a (exploration)": [],
    "pain medications": [],
    "peppermint oil": [],
    "statin": [],
    "statins": [],
    "sulfonylureas": [],
    "thera-tabs": [],
    "vaccination": [],
}


# v2 top-up only: more MedicationQA drug names -> drug-list drugs we already have labels for.
# Kept apart from DRUG_OVERRIDES so v1 doesn't change.
TOPUP_OVERRIDES: dict[str, list[str]] = {
    "(voltaren or cataflam) and medrol": ["diclofenac", "methylprednisolone"],
    "albuterol nebulizers": ["albuterol"],
    "albuterol sulfate inhalation solution": ["albuterol"],
    "amoxiciliin": ["amoxicillin"],
    "asa ec": ["aspirin"],
    "b12": ["cyanocobalamin"],
    "carbidopa and levodopa, extended release": ["carbidopa and levodopa"],
    "cataflam, medrol": ["diclofenac", "methylprednisolone"],
    "cipro": ["ciprofloxacin"],
    "concerta and lexapro": ["methylphenidate", "escitalopram"],
    "coreg": ["carvedilol"],
    "coumadin": ["warfarin"],
    "coumadine": ["warfarin"],
    "diclofenac sodium topical gel": ["diclofenac"],
    "duragesic patch and oxycontin": ["fentanyl", "oxycodone"],
    "efudex": ["fluorouracil"],
    "gabapentin, pregabalin": ["gabapentin", "pregabalin"],
    "insulin glargine": ["insulin"],
    "lantus solostar": ["insulin"],
    "lasix": ["furosemide"],
    "llevodopa and carbidopa; alcohol": ["carbidopa and levodopa"],
    "lorcet 7.5 liquid": ["hydrocodone and acetaminophen"],
    "metoprolol + lexapro": ["metoprolol", "escitalopram"],
    "metoprolol succinate": ["metoprolol"],
    "metroprolol succinate": ["metoprolol"],
    "miralax": ["polyethylene glycol"],
    "norco": ["hydrocodone and acetaminophen"],
    "percocet": ["oxycodone and acetaminophen"],
    "premarin": ["conjugated estrogens"],
    "seroquel": ["quetiapine"],
    "trospium chloride": ["trospium"],
    "warfarin; spices": ["warfarin"],
}
TOPUP_SIZE = 55
# types where half or fewer of the v1 questions (with a label) got a gold section
TOPUP_SKIP_TYPES = {"Alternatives", "Storage and disposal", "Stopping/tapering"}


# Drugs used by the missing_from_corpus refusal questions (datasets/refusal_v1.json).
# None had any openFDA label hit on 2026-11-02; T3 must not fetch them.
EXCLUDED_DRUGS = [
    "betahistine",
    "bilastine",
    "drotaverine",
    "flupentixol",
    "levosulpiride",
    "meldonium",
    "metamizole",
    "nifuroxazide",
    "nimesulide",
    "piracetam",
    "rupatadine",
    "thiocolchicoside",
    "tianeptine",
    "trimetazidine",
]


# Unique cleaned drug names, plus how each MedicationQA drug name maps to them
def build_drug_list(questions: list[dict]) -> dict:
    raw_names = sorted({q["drug"].lower() for q in questions})
    aliases = {raw: DRUG_OVERRIDES.get(raw, [raw]) for raw in raw_names}
    drugs = sorted({name for names in aliases.values() for name in names})
    return {
        "version": VERSION,
        "drugs": drugs,
        "aliases": aliases,
        "excluded_drugs": EXCLUDED_DRUGS,
    }


# v2 top-up: unused questions about drugs that have a label (from T3's corpus_stats.json),
# skipping weak types. IDs continue after v1 (mqa-201...).
def sample_topup(df: pd.DataFrame, v1: list[dict]) -> tuple[list[dict], dict]:
    stats = json.loads((REPO_ROOT / "results" / "corpus_stats.json").read_text())
    have = set(build_drug_list(v1)["drugs"]) - set(stats["drug_list"]["not_found"])
    overrides = DRUG_OVERRIDES | TOPUP_OVERRIDES
    names = df["drug"].str.lower().map(lambda raw: overrides.get(raw, [raw]))
    pool = df[
        ~df["question"].isin({q["question"] for q in v1})
        & ~df["type"].isin(TOPUP_SKIP_TYPES)
        & names.map(lambda n: bool(n) and all(d in have for d in n))
    ]
    sample = pool.sample(n=TOPUP_SIZE, random_state=SEED).sort_index()
    questions = [
        {
            "id": f"mqa-{i:03d}",
            "question": row.question,
            "type": row.type,
            "drug": row.drug,
            "answer": None if pd.isna(row.answer) else row.answer,
            "url": None if pd.isna(row.url) else row.url,
        }
        for i, row in enumerate(sample.itertuples(), start=len(v1) + 1)
    ]
    aliases = {
        raw: overrides.get(raw, [raw]) for raw in sorted({q["drug"].lower() for q in questions})
    }
    return questions, aliases


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    df = load_medicationqa()
    questions = sample_questions(df)
    write_json(
        DATASETS_DIR / f"medicationqa_{VERSION}.json",
        {"version": VERSION, "source": "MedicationQA", "seed": SEED, "questions": questions},
    )
    print(f"Wrote {len(questions)} questions")
    print(pd.Series([q["type"] for q in questions]).value_counts().to_string())

    drug_list = build_drug_list(questions)
    write_json(DATASETS_DIR / f"drug_list_{VERSION}.json", drug_list)
    no_drug = [raw for raw, names in drug_list["aliases"].items() if not names]
    print(f"\nWrote {len(drug_list['drugs'])} drugs ({len(no_drug)} drug names map to none)")

    topup, topup_aliases = sample_topup(df, questions)
    write_json(
        DATASETS_DIR / "medicationqa_v2.json",
        {
            "version": "v2",
            "source": "MedicationQA top-up: drugs with a label in the T3 corpus",
            "seed": SEED,
            "aliases": topup_aliases,
            "questions": topup,
        },
    )
    print(f"\nWrote {len(topup)} top-up questions (v2)")
    print(pd.Series([q["type"] for q in topup]).value_counts().to_string())
