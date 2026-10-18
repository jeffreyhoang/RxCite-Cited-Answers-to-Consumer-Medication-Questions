"""Check that the environment can reach Supabase with pgvector enabled.

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


if __name__ == "__main__":
    sys.exit(0 if check_supabase() else 1)
