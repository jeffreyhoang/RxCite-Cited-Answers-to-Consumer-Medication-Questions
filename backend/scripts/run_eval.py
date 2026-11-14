"""Score one RAG architecture on the eval sets and save the results.

Run from backend/:
    uv run python scripts/run_eval.py --arch <name>          # dev subset (20 + 10 questions)
    uv run python scripts/run_eval.py --arch <name> --full   # all 160 + 136 questions

Writes results/<arch>_<dev|full>.json. Answers and judge scores are cached, so reruns are fast
and make no new API calls.
"""

import argparse
import json

from rxcite.config import RESULTS_DIR
from rxcite.eval.runner import run
from rxcite.pipelines import PIPELINES

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True, choices=sorted(PIPELINES) or None)
    parser.add_argument(
        "--full", action="store_true", help="run every question, not the dev subset"
    )
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM judge")
    args = parser.parse_args()

    out = run(PIPELINES[args.arch](), full=args.full, use_judge=not args.no_judge)
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{out['arch']}_{out['mode']}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")

    m = out["metrics"]
    r = m["retrieval"]
    print(f"{out['arch']} ({out['mode']}): {out['n_answerable']} answerable,", end=" ")
    print(f"{out['n_refusal']} refusal")
    print(f"  recall@10 {r['recall@10']:.3f} | MRR {r['mrr']:.3f}")
    if "judge" in m:
        j = m["judge"]
        print(
            f"  judge (0-2, n={j['n']}): correctness {j['correctness']:.2f} | "
            f"groundedness {j['groundedness']:.2f} | citations {j['citation_accuracy']:.2f}"
        )
    print(f"  refusal accuracy {m['refusal_accuracy']:.2f}", end=" | ")
    print(f"false refusals {m['false_refusal_rate']:.2f}")
    print(f"  latency mean {m['latency_s']['mean']:.2f}s")
    print(f"  new API runs: {out['new_api_runs']}")
    print(f"Wrote {path}")
