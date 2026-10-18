# RxCite: Cited Answers to Consumer Medication Questions

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env #fill in API keys
```

## How the project works
### 1. Build the dataset (run once)
```bash
python -m src.data_prep.ingest       #openFDA labels -> sections -> chunks -> embeddings -> indexes
python -m src.data_prep.build_eval   #MedicationQA questions -> matched label sections -> eval file
```
- `ingest.py` fetches FDA drug labels from openFDA, splits them into sections, chunks them, and builds the vector and BM25 indexes. Output goes to `data/labels/` and `data/indexes/` (both gitignored)
- `build_eval.py` samples ~200 MedicationQA questions, matches each to the label section containing the answer, and writes `data/eval/medicationqa.jsonl`.
- `data/eval/refusal.jsonl` (~40 questions that should be refused) written by hand.

### 2. Run the experiments
```bash
python run_experiments.py
```
For each of the four architectures (Naive, Advanced, Hierarchical, Decomposition), the script retrieves chunks for every eval question, generates answers, and scores them with the functions in `src/core/evaluate.py` (recall@10, MRR, etc). It saves the results to `results/`.

### 3. Run the demo
```bash
python demo.py
```
Ask a medication question and get an answer with citations to the FDA label sections it came from, using the architecture you choose.

## Code layout
| Folder | Purpose |
|---|---|
| `src/core/` | Shared pieces: constants, dataclasses, model wrappers (`shared.py`), answer generation (`generate_answer.py`), and metrics (`evaluate.py`) |
| `src/architectures/` | One file per RAG architecture, all implementing `retrieve(query, k)` from `base.py`; `__init__.py` holds the name-to-class registry |
| `src/data_prep/` | One-time dataset scripts |
| `data/` | `labels/` and `indexes/` (generated, gitignored); `eval/` (committed) |
| `results/` | Output tables from `run_experiments.py` |