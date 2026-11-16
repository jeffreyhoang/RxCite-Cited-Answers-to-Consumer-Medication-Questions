"""Build the Advanced RAG ablation table (T7) from saved eval results.

Run the rows first (from backend/), then this script:
    uv run python scripts/run_eval.py --arch <row> --full --no-judge   # for each row below
    uv run python scripts/advanced_ablation.py

Reads results/<row>_full.json and the cached advanced-retrieval answers
(data/cache/runs/advanced-retrieval/*.json, for the reranker time per question). Writes
results/advanced_ablation.md. Missing rows are skipped with a note. No database or API calls.
"""

import json
import statistics

from rxcite.config import RERANK_CANDIDATES, RESULTS_DIR
from rxcite.eval.runner import RUNS_CACHE

# Cumulative ablation: each row adds one step to the row above it
ROWS = [
    ("dense", "Fixed chunks, dense top-10 (Naive retrieval)"),
    ("adv-section", "+ section-based chunks with drug | section header"),
    ("adv-rewrite", "+ query rewriting"),
    ("adv-hybrid", "+ BM25 + dense, RRF fusion (k = 60)"),
    ("advanced-retrieval", "+ bge-reranker-base over the top 50 (full Advanced)"),
]
RERANK_RUN = "advanced-retrieval"


# Results JSON for one row, or None if it hasn't been run on the full set yet
def load_row(name: str) -> dict | None:
    path = RESULTS_DIR / f"{name}_full.json"
    return json.loads(path.read_text()) if path.exists() else None


# Reranker seconds per question from the cached answers (older answers may lack "rerank_s")
def rerank_times(name: str = RERANK_RUN) -> list[float]:
    times = []
    for path in sorted((RUNS_CACHE / name).glob("*.json")):
        info = json.loads(path.read_text()).get("info") or {}
        if "rerank_s" in info:
            times.append(info["rerank_s"])
    return times


def build_report() -> tuple[str, list[str]]:
    lines = [
        "# Advanced RAG ablation (T7)",
        "",
        "Retrieval only (no generation), all answerable questions. Each row adds one step to",
        "the row above it. A hit is a retrieved chunk from a gold (set_id, section).",
        "",
        "| Row | Pipeline | recall@10 | MRR | Mean latency |",
        "|---|---|---|---|---|",
    ]
    missing, n = [], None
    for name, desc in ROWS:
        out = load_row(name)
        if out is None:
            missing.append(name)
            continue
        n = n or out["n_answerable"]
        m = out["metrics"]
        r = m["retrieval"]
        lines.append(
            f"| `{name}` | {desc} | {r['recall@10']:.3f} | {r['mrr']:.3f} "
            f"| {m['latency_s']['mean']:.2f} s |"
        )
    lines += ["", f"n = {n} answerable questions." if n else "No rows found.", ""]
    lines.append(
        "Latency is the mean over every question run (answerable + refusal). Rewrites are "
        "cached, so only the first row run with rewriting pays for the Haiku calls; later rows "
        "reuse them and understate that cost."
    )
    if missing:
        lines += ["", "Not run yet: " + ", ".join(f"`{x}`" for x in missing) + "."]

    times = rerank_times()
    lines += ["", "## Reranker time per question", ""]
    if times:
        lines += [
            f"`{RERANK_RUN}`, {len(times)} questions, up to {RERANK_CANDIDATES} (question, chunk) "
            f"pairs each: "
            f"mean {statistics.mean(times):.2f} s, median {statistics.median(times):.2f} s, "
            f"max {max(times):.2f} s.",
        ]
    else:
        lines.append(f"No cached `{RERANK_RUN}` answers with `rerank_s` found.")

    lines += [
        "",
        "## Which step helped most",
        "",
        "TODO: a few lines on which step moved recall@10 / MRR the most, and at what latency.",
        "",
    ]
    return "\n".join(lines), missing


if __name__ == "__main__":
    report, missing = build_report()
    for name in missing:
        print(f"Skipping {name}: results/{name}_full.json not found")
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / "advanced_ablation.md"
    path.write_text(report)
    print(f"Wrote {path}")
