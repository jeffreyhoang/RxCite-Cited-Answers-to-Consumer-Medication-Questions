# RxCite: Cited Answers to Consumer Medication Questions

A retrieval-augmented generation (RAG) assistant that answers consumer medication questions using only FDA-approved drug labels from openFDA, with citations to the label sections used. We compare four RAG architectures: Naive, Advanced, Hierarchical, and Decomposition.

## Project structure

```
├── backend/                  # Python backend (uv project: pyproject.toml, uv.lock, .env)
│   ├── api/                  # FastAPI server: /health, /architectures, /ask; frontend calls it
│   ├── rxcite/               # Rag library
│   │   ├── config.py         # config settings
│   │   ├── db.py             # talks to supabase
│   │   ├── generation.py     # shared generator for every RAG architecture
│   │   ├── llm.py            # talks to Claude
│   │   ├── data/             # downloads openFDA labels and splits them into chunks
│   │   ├── query/            # rewrites, analyzes, or splits the question before search
│   │   ├── retrieval/        # finds chunks: embedders, BM25, dense, RRF, reranker, parent expansion
│   │   ├── pipelines/        # the four RAG types + baselines
│   │   └── eval/             # scoring: loads eval sets, retrieval metrics, LLM judge, eval runner
│   ├── scripts/              # commands run manually: fetch, ingest, chunk, embed, load, build eval sets, run evals
│   ├── tests/                # pytest tests
│   └── pyproject.toml        # Python packages
├── frontend/                 # React + Vite + TypeScript
├── datasets/                 # eval sets and drug list (committed)
├── data/                     # downloaded openFDA labels and MedicationQA files (not committed)
└── results/                  # eval results
```

## Experimental results
This section describes the breakdown of experimental results, located in the results/ directory:

#### 1. Corpus description
| File | What it shows |
|--------------|-----------|
| corpus_stats.json | Statistics about the document corpus |

#### 2. Embedding model comparison (retrieval only)
| File | Recall@10 | MRR |
|------|-----------|-----|
| dense-openai-small_full.json | **0.659** | **0.498** |
| dense-bge-base_full.json | 0.581 | 0.410 |
| dense-pubmedbert_full.json  | 0.250 | 0.183 |

#### 3. Advanced RAG ablation study (retrieval only)
| File | Step added | Recall@10 | MRR |
|------|------------|-----------|-----|
| dense_full.json | baseline (Naive) | 0.659 | 0.497 |
| adv-section_full.json | + section-based chunks | 0.700 | 0.535 |
| adv-rewrite_full.json | + query rewrite | **0.809** | **0.766** | 
| adv-hybrid_full.json | + hybrid search (BM25 + dense) | 0.784 | 0.716 |
| advanced-retrieval_full.json | + reranker (full Advanced) | 0.781 | 0.718 |

#### 4. Retrieval only runs for other architectures
| File | Recall@10 | MRR |
|------|-----------|-----|
| hierarchical-retrieval_full.json | 0.791 | 0.680 |
| decomposition-retrieval_full.json | 0.659 | 0.498 |

#### 5. Final comparison (retrieval and generation)
| File | Recall@10 | MRR | Correctness | Groundedness | Citation Accuracy | Refusal Accuracy | FRR | 
|------|-----------|-----|-------------|--------------|-------------------|------------------|-----|
| naive_full.json | 0.659 | 0.497 | 1.74 | 1.82 | 1.83 | 0.654 | 0.099 |
| advanced_full.json | 0.781 | 0.718 | 1.79 | 1.85 | 1.86 | 0.588 | 0.044 | 
| hierarchical_full.json |0.791 | 0.680 | 1.80 | 1.85 | 1.85 | 0.574 | 0.019 |
| decomposition_full.json | 0.659 | 0.498 | 1.74 | 1.82 | 1.82 | 0.669 | 0.088 |

The LLM judge scores correctness, groundedness, and citation accuracy from 0 to 2.

#### 6. Dev runs and sanity checks
naive_dev.json, hierarchical_dev.json, and decomposition_dev.json contain dev run results scored on 20 answerable and 10 refusal questions. oracle_dev.json and dummy_dev.json are evaluation sanity checks that expect a recall@10 of 1.000 and 0.000, respectively. 

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

## Running the app

Needs Node 20.19+ (or 22.12+) for the frontend. Use two terminals:

```bash
# Terminal 1: API on http://localhost:8000
cd backend
uv run uvicorn api.main:app --reload --port 8000

# Terminal 2: website on http://localhost:5173
cd frontend
npm install        # first time only
npm run dev
```

Open http://localhost:5173. To point the site at a different API, copy
`frontend/.env.example` to `frontend/.env.local` and change `VITE_API_URL`.
The site must run on port 5173, because that's the only origin the API's CORS allows.

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

Errors: `422` for an invalid request, `503` when an architecture or a setting is unavailable, `502` when Claude, OpenAI or Supabase fails, `500` for any other error while answering (details in the server log). CORS allows the Vite dev server (`http://localhost:5173`); set `RXCITE_CORS_ORIGINS` (comma-separated) to change it.
