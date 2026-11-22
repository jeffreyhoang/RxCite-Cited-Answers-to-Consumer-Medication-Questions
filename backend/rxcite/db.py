# Supabase (Postgres + pgvector) access

import psycopg

from rxcite.config import EMBEDDING_DIM, supabase_db_url

HNSW_EF_SEARCH = 100  # HNSW search breadth; higher = closer to exact search, a bit slower

_shared: psycopg.Connection | None = None


# Open a connection to the Supabase database
def connect() -> psycopg.Connection:
    return psycopg.connect(supabase_db_url(), connect_timeout=10)


# One reused connection for read-only queries at answer time
def _shared_connection() -> psycopg.Connection:
    global _shared
    if _shared is None or _shared.closed or _shared.broken:
        _shared = connect()
        _shared.autocommit = True  # read-only queries; no open transaction left idle
        _shared.execute(f"set hnsw.ef_search = {HNSW_EF_SEARCH}")
    return _shared


# Run a read-only query on the shared connection; reconnects once if it was dropped
def query(sql: str, params=None) -> list[tuple]:
    global _shared
    try:
        return _shared_connection().execute(sql, params).fetchall()
    except psycopg.OperationalError:
        _shared = None
        return _shared_connection().execute(sql, params).fetchall()


# One row per FDA label; one row per named section of a label
SCHEMA = """
create table if not exists labels (
    set_id         text primary key,  -- openFDA label id (used in citations)
    drug_name      text not null,     -- our name from datasets/drug_list_v1.json
    generic_name   text,
    brand_name     text,
    effective_date date,
    is_distractor  boolean not null default false
);

create table if not exists sections (
    set_id  text not null references labels (set_id) on delete cascade,
    section text not null,  -- openFDA field name, e.g. "dosage_and_administration"
    text    text not null,
    primary key (set_id, section)
);
"""


# Fixed-size chunks
CHUNKS_SCHEMA = f"""
create extension if not exists vector;

create table if not exists chunks (
    chunk_id  text primary key,  -- "<set_id>:<section>:<n>"
    set_id    text not null,
    section   text not null,
    text      text not null,
    embedding halfvec({EMBEDDING_DIM}) not null,
    foreign key (set_id, section) references sections (set_id, section) on delete cascade
);
"""

# Built after a bulk load
CHUNKS_INDEX = """
create index if not exists chunks_embedding_idx
    on chunks using hnsw (embedding halfvec_cosine_ops);
"""


# Section-based chunks
SECTION_CHUNKS_SCHEMA = f"""
create table if not exists section_chunks (
    chunk_id  text primary key,  -- "<set_id>:<section>:<n>"
    set_id    text not null,
    section   text not null,
    text      text not null,
    embedding halfvec({EMBEDDING_DIM}) not null,
    foreign key (set_id, section) references sections (set_id, section) on delete cascade
);
"""

SECTION_CHUNKS_INDEX = """
create index if not exists section_chunks_embedding_idx
    on section_chunks using hnsw (embedding halfvec_cosine_ops);
"""

# Chunk set -> (HNSW index name, its create statement), for load_chunks.py
CHUNK_INDEXES = {
    "fixed": ("chunks_embedding_idx", CHUNKS_INDEX),
    "section": ("section_chunks_embedding_idx", SECTION_CHUNKS_INDEX),
}


# Create the tables if they don't exist yet (safe to run more than once)
def create_tables() -> None:
    with connect() as conn:
        conn.execute(SCHEMA)
        conn.execute(CHUNKS_SCHEMA)
        conn.execute(SECTION_CHUNKS_SCHEMA)
