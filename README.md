# RxCite: Cited Answers to Consumer Medication Questions

A retrieval-augmented generation (RAG) assistant that answers consumer medication questions using only FDA-approved drug labels from openFDA, with citations to the label sections used. We compare four RAG architectures: Naive, Advanced, Hierarchical, and Decomposition.

## Project structure

```
├── backend/             # Python project (uv)
│   ├── pyproject.toml   # dependencies, ruff, pytest config
│   ├── rxcite/          # all RAG logic
│   │   ├── config.py    # env vars, model names, constants
│   │   ├── db.py        # Supabase (Postgres + pgvector) access
│   │   ├── llm.py       # Claude client wrapper
│   │   ├── data/        # openFDA ingestion, section parsing, chunking, embedding
│   │   ├── retrieval/   # dense, BM25, hybrid search, reranking
│   │   ├── pipelines/   # one file per RAG architecture + shared base interface
│   │   └── eval/        # recall@10, MRR, LLM judge, experiment runner
│   ├── api/             # FastAPI app (imports rxcite)
│   ├── scripts/         # command-line entry points (ingest, run eval, smoke test)
│   └── tests/           # pytest tests
├── frontend/            # React + Vite + TypeScript app
├── datasets/            # committed eval sets: MedicationQA, refusal, gold mapping
├── results/             # experiment outputs (tables, figures)
├── notebooks/           # exploration and report figures
└── data/                # gitignored: raw openFDA downloads and caches
```

Folders are added as the work that needs them begins.

## Setup

1. Install [uv](https://docs.astral.sh/uv/): `brew install uv`
2. Install dependencies (uv installs Python 3.12 automatically):
   ```bash
   cd backend
   uv sync
   ```
3. Create your env file and fill in the values (ask a teammate for the Supabase connection string):
   ```bash
   cp .env.example .env
   ```
4. Check your setup:
   ```bash
   uv run python scripts/smoke_test.py   # should print: Supabase: OK
   uv run ruff check . && uv run pytest
   ```

Use `uv add <package>` to add dependencies (not `pip install`), and commit `pyproject.toml` and `uv.lock`. Never commit `.env`.
