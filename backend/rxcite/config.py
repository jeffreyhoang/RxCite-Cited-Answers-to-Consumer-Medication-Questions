# Configuration settings for the RxCite application
#       -paths, env vars, model names, constants, etc.

import os
from pathlib import Path
from dotenv import load_dotenv

# Paths
BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
 
DATA_DIR = REPO_ROOT / "data"  #(gitignored) raw openFDA downloads, caches
DATASETS_DIR = REPO_ROOT / "datasets"  #eval sets, gold mapping
RESULTS_DIR = REPO_ROOT / "results"
CACHE_DIR = DATA_DIR / "cache"  #cached LLM calls, eval runs

 # Env Vars
load_dotenv(BACKEND_DIR / ".env")
def require_env(name: str) -> str:
    """
    Read a required env var, failing with a clear message if it is missing
    """
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set. Make sure you have the correct .env file in backend")
    return value

def anthropic_api_key() -> str:
    return require_env("ANTHROPIC_API_KEY")
def supabase_db_url() -> str:
    return require_env("SUPABASE_DB_URL")

# Model names (e.g. "claude-haiku-5-5")
GENERATOR_MODEL = ""
JUDGE_MODEL = ""
FINAL_RUN_GENERATOR_MODEL = ""

EMBEDDING_MODEL = ""
EMBEDDING_DIM =   # must match the pgvector column; update with EMBEDDING_MODEL
RERANKER_MODEL = ""


# Retrieval Constants
TOP_K = 10 # final number of chunks scored
RERANK_CANDIDATES = 50 # how many chunks reranker sees before picking TOP_K
RRF_K = 60  # Reciprocal Rank Fusion parameter

# Generation Constants
MAX_TOKENS = 2000  # max tokens for generation
TEMPERATURE = 0.0  # temperature for generation

# Evaluation / Budget Constants
RANDOM_SEED = 42
BUDGET_USD = 20.0 # total API budget
