"""Split every label section in Supabase into chunks: fixed-size (T5) or section-based (T7).

Run from backend/:
    uv run python scripts/build_chunks.py                 # fixed-size chunks (default)
    uv run python scripts/build_chunks.py --set section   # split at subheadings / paragraphs

--set fixed writes data/chunks/fixed_v1.jsonl, one chunk per line:
    chunk_id, set_id, section, drug_name, text
--set section writes data/chunks/section_v1.jsonl with the same fields plus
    search_text  (the "drug | section" header + text, which is what gets embedded and searched)
and also prints how the long sections were split (subheadings, paragraphs, sentences).
Reads Supabase only; nothing is written to the database.
"""

import argparse
import json
import statistics
from collections import Counter

from rxcite import db
from rxcite.chunking import (
    MAX_CHARS,
    OVERLAP_CHARS,
    SECTION_MAX_CHARS,
    chunk_section,
    section_pieces,
    split_section,
)
from rxcite.config import DATA_DIR
from rxcite.pipelines.base import search_text

OUT = {
    "fixed": DATA_DIR / "chunks" / "fixed_v1.jsonl",
    "section": DATA_DIR / "chunks" / "section_v1.jsonl",
}


# How the long sections (over the cap) were split, for the report
def print_strategies(texts: list[str]) -> None:
    strategies, oversize = Counter(), Counter()
    for text in texts:
        pieces, strategy = section_pieces(text)
        strategies[strategy] += 1
        if any(len(p) > SECTION_MAX_CHARS for p in pieces):
            oversize[strategy] += 1
    long = sum(n for s, n in strategies.items() if s != "whole")
    print(f"  long sections (> {SECTION_MAX_CHARS} chars): {long}")
    for strategy, label in [
        ("headings", "numbered subheadings"),
        ("paragraphs", "paragraph breaks only"),
        ("sentences", "neither (sentences)"),
    ]:
        n = strategies[strategy]
        share = f"{n / long:.0%}" if long else "-"
        # Pieces over the cap are split again by sentences (always the case for "sentences")
        extra = "" if strategy == "sentences" else f", {oversize[strategy]} with a piece over cap"
        print(f"    split by {label}: {n} ({share}){extra}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", default="fixed", choices=sorted(OUT), dest="chunk_set")
    args = parser.parse_args()
    out = OUT[args.chunk_set]
    split = split_section if args.chunk_set == "section" else chunk_section

    with db.connect() as conn:
        rows = conn.execute(
            """select s.set_id, s.section, l.drug_name, s.text
               from sections s join labels l using (set_id)
               order by s.set_id, s.section"""
        ).fetchall()

    out.parent.mkdir(parents=True, exist_ok=True)
    lengths, per_section = [], []
    with out.open("w") as f:
        for set_id, section, drug, text in rows:
            chunks = split(text)
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
                if args.chunk_set == "section":
                    record["search_text"] = search_text(drug, section, chunk)
                f.write(json.dumps(record) + "\n")

    if args.chunk_set == "section":
        print(
            f"Section-based chunks: cap {SECTION_MAX_CHARS} chars "
            f"(sentence fallback overlap {OVERLAP_CHARS} chars)"
        )
    else:
        print(f"Chunk size {MAX_CHARS} chars, overlap {OVERLAP_CHARS} chars")
    print(f"{len(rows)} sections -> {len(lengths)} chunks")
    print(f"  sections with no chunk: {per_section.count(0)}")
    print(f"  sections kept whole:    {per_section.count(1)}")
    print(f"  chunks per section: mean {statistics.mean(per_section):.1f}, max {max(per_section)}")
    print(
        f"  chunk chars: min {min(lengths)}, median {statistics.median(lengths):.0f}, "
        f"max {max(lengths)}"
    )
    if args.chunk_set == "section":
        print_strategies([text for *_, text in rows])
    print(f"Wrote {out}")
