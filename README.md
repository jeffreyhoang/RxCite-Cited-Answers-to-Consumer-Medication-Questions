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

**Requirements:** [uv](https://docs.astral.sh/uv/) (`brew install uv`) and Node 20.19+ (or 22.12+).

1. Clone the repo and install the backend (uv install Python 3.12 automatically):
   ```bash
   git clone https://github.com/jeffreyhoang/RxCite-Cited-Answers-to-Consumer-Medication-Questions.git
   cd RxCite-Cited-Answers-to-Consumer-Medication-Questions/backend
   uv sync
   ```
2. Create your env file:
   ```bash
   cp .env.example .env
   ```
   `SUPABASE_DB_URL` is already filled in with a read-only key.
   Add your own keys:
   ```
   ANTHROPIC_API_KEY=...
   OPENAI_API_KEY=...
   ```
3. Check that everything connects:
   ```bash
   uv run python scripts/smoke_test.py   # should print: Supabase: OK
   ```
4. Install the frontend:
   ```bash
   cd ../frontend
   npm install
   ```

Use `uv add <package>` to add dependencies (not `pip install`), and commit `pyproject.toml` and `uv.lock`. Never commit `.env`.

## Run the app

Use two terminals, both starting from the repo root.

**Terminal 1: backend** (API on http://localhost:8000)
```bash
cd backend
uv run uvicorn api.main:app --reload --port 8000
```
Startup preloads the BM25 index and the reranker (a few seconds). Set `RXCITE_WARMUP=0` to skip that.

**Terminal 2: frontend** (website on http://localhost:5173)
```bash
cd frontend
npm run dev
```

Open http://localhost:5173, pick an architecture, and ask a question.

The site must run on port 5173, because that's the only origin the API's CORS allows.