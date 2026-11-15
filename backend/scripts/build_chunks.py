"""Split every label section in Supabase into fixed-size chunks.

Run from backend/:
    uv run python scripts/build_chunks.py

Writes data/chunks/fixed_v1.jsonl, one chunk per line:
    chunk_id, set_id, section, drug_name, text
Reads Supabase only; nothing is written to the database.
"""

import json
import statistics

from rxcite import db
from rxcite.chunking import MAX_CHARS, OVERLAP_CHARS, chunk_section
from rxcite.config import DATA_DIR

OUT = DATA_DIR / "chunks" / "fixed_v1.jsonl"

if __name__ == "__main__":
    with db.connect() as conn:
        rows = conn.execute(
            """select s.set_id, s.section, l.drug_name, s.text
               from sections s join labels l using (set_id)
               order by s.set_id, s.section"""
        ).fetchall()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    lengths, per_section = [], []
    with OUT.open("w") as f:
        for set_id, section, drug, text in rows:
            chunks = chunk_section(text)
            per_section.append(len(chunks))
            for i, chunk in enumerate(chunks):
                lengths.append(len(chunk))
                record = {
                    "chunk_id": f"{set_id}:{section}:{i}",
                    "set_id": set_id,
                    "section": section,
                    "drug_name": drug,
                    "text": chunk,
                }
                f.write(json.dumps(record) + "\n")

    print(f"Chunk size {MAX_CHARS} chars, overlap {OVERLAP_CHARS} chars")
    print(f"{len(rows)} sections -> {len(lengths)} chunks")
    print(f"  sections with no chunk: {per_section.count(0)}")
    print(f"  sections kept whole:    {per_section.count(1)}")
    print(f"  chunks per section: mean {statistics.mean(per_section):.1f}, max {max(per_section)}")
    print(
        f"  chunk chars: min {min(lengths)}, median {statistics.median(lengths):.0f}, "
        f"max {max(lengths)}"
    )
    print(f"Wrote {OUT}")
