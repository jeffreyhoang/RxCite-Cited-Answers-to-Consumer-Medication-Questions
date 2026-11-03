# Supabase (Postgres + pgvector) access

import psycopg

from rxcite.config import supabase_db_url


# Open a connection to the Supabase database (use as `with connect() as conn:`)
def connect() -> psycopg.Connection:
    return psycopg.connect(supabase_db_url(), connect_timeout=10)


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


# Create the tables if they don't exist yet (safe to run more than once)
def create_tables() -> None:
    with connect() as conn:
        conn.execute(SCHEMA)
