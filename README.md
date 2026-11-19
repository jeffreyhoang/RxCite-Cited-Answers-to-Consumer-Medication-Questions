# RxCite: Cited Answers to Consumer Medication Questions

A retrieval-augmented generation (RAG) assistant that answers consumer medication questions using only FDA-approved drug labels from openFDA, with citations to the label sections used. We compare four RAG architectures: Naive, Advanced, Hierarchical, and Decomposition.

## Project structure

```
├── backend/                # Python code
│   ├── rxcite/             # the RAG system
│   │   ├── config.py       # settings: paths, model names, numbers
│   │   ├── db.py           # talks to the Supabase database
│   │   ├── llm.py          # talks to Claude
│   │   ├── data/           # downloads drug labels and splits them into chunks
│   │   ├── retrieval/      # finds the chunks that match a question
│   │   ├── pipelines/      # the four RAG types, one file each
│   │   └── eval/           # scores how well each RAG type does
│   ├── api/                # web server the website calls
│   ├── scripts/            # commands you run by hand
│   ├── tests/              # automated tests
│   └── pyproject.toml      # list of Python packages
├── frontend/               # the website
├── datasets/               # test questions and drug list (committed)
├── results/                # experiment tables and charts
├── notebooks/              # exploring data, making report figures
└── data/                   # downloaded files (not committed)
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

## Run the API

From `backend/`, with `.env` filled in:

```bash
uv run uvicorn api.main:app --reload --port 8000
```

Interactive docs: http://localhost:8000/docs. Startup preloads the BM25 index and the reranker for Advanced RAG (a few seconds); set `RXCITE_WARMUP=0` to skip that.

| Endpoint | What it does |
|---|---|
| `GET /health` | `status` (`ok`, or `degraded` if a setting is missing or a pipeline failed to load), the loaded architectures, and what is missing |
| `GET /architectures` | The four architectures: `id`, `name`, `description` |
| `POST /ask` | Body `{"question": "...", "architecture": "naive" \| "advanced" \| "hierarchical" \| "decomposition"}`. Returns the answer (with `[n]` markers), `refused`, `citations` (numbered, with DailyMed links), `sources` (retrieved chunks, best first), `info` (intermediate steps, e.g. the rewritten query or sub-questions) and `latency_s` |

Errors: `422` for an invalid request, `503` when an architecture or a setting is unavailable, `502` when Claude, OpenAI or Supabase fails. CORS allows the Vite dev server (`http://localhost:5173`); set `RXCITE_CORS_ORIGINS` (comma-separated) to change it.
