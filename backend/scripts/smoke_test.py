"""Check that the environment can reach Supabase (with pgvector) and Claude.

Run from backend/: uv run python scripts/smoke_test.py
"""

import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def check_supabase() -> bool:
    url = os.environ.get("SUPABASE_DB_URL")
    if not url:
        print("Supabase: FAIL (SUPABASE_DB_URL not set in backend/.env)")
        return False
    try:
        with psycopg.connect(url, connect_timeout=10) as conn:
            row = conn.execute(
                "select extversion from pg_extension where extname = 'vector'"
            ).fetchone()
    except psycopg.Error as e:
        print(f"Supabase: FAIL ({type(e).__name__})")
        return False
    if row is None:
        print("Supabase: FAIL (pgvector extension not enabled)")
        return False
    print(f"Supabase: OK (pgvector {row[0]})")
    return True


def check_claude() -> bool:
    import anthropic

    from rxcite import llm
    from rxcite.config import GENERATOR_MODEL

    try:
        first = llm.call("Reply with exactly: OK", GENERATOR_MODEL, max_tokens=50)
        again = llm.call("Reply with exactly: OK", GENERATOR_MODEL, max_tokens=50)
    except (anthropic.APIError, RuntimeError) as e:
        print(f"Claude: FAIL ({type(e).__name__})")
        return False
    print(f"Claude: OK ({GENERATOR_MODEL} said {first.text.strip()!r})")
    print(f"Claude repeat: cache hit = {again.cached}")
    return again.cached


if __name__ == "__main__":
    results = [check_supabase(), check_claude()]
    sys.exit(0 if all(results) else 1)
