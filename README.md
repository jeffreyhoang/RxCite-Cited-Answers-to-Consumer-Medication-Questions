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
