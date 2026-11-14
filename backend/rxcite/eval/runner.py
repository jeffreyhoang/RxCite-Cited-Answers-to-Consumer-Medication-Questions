# Run one RAG pipeline over the eval sets, judge the answers, and summarize the metrics
#       Each pipeline answer is cached in data/cache/runs/<arch>/<question id>.json,
#       and judge calls are cached by llm.py, so a rerun makes no new API calls.

import json
import random
import statistics
from collections import defaultdict
from dataclasses import asdict

from rxcite.config import CACHE_DIR, RANDOM_SEED
from rxcite.eval.datasets import EvalQuestion, load_eval_sets
from rxcite.eval.judge import judge
from rxcite.eval.metrics import evaluate_retrieval, false_refusal_rate, refusal_accuracy
from rxcite.pipelines.base import Chunk, Citation, RAGPipeline, RAGResult

RUNS_CACHE = CACHE_DIR / "runs"
DEV_ANSWERABLE = 20
DEV_REFUSAL = 10


# Fixed dev subset (seed 42) or everything
def eval_sets(full: bool) -> tuple[list[EvalQuestion], list[EvalQuestion]]:
    answerable, refusal = load_eval_sets()
    if full:
        return answerable, refusal
    rng = random.Random(RANDOM_SEED)
    return rng.sample(answerable, DEV_ANSWERABLE), rng.sample(refusal, DEV_REFUSAL)


def _load_result(data: dict) -> RAGResult:
    data["chunks"] = [Chunk(**c) for c in data["chunks"]]
    data["citations"] = [Citation(**c) for c in data["citations"]]
    return RAGResult(**data)


# Pipeline answer for one question, from the cache if this arch already answered it
def answer(pipeline: RAGPipeline, q: EvalQuestion) -> tuple[RAGResult, bool]:
    path = RUNS_CACHE / pipeline.name / f"{q.id}.json"
    if path.exists():
        return _load_result(json.loads(path.read_text())), True
    result = pipeline.answer(q.question)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(result)))
    return result, False


def run(pipeline: RAGPipeline, full: bool = False, use_judge: bool = True) -> dict:
    answerable, refusal = eval_sets(full)
    new_answers = new_judgments = 0
    rows = []

    # Answerable: retrieval, judge (only answers that weren't refused), false refusals
    answered_results, runs, scores = [], [], []
    for q in answerable:
        result, hit = answer(pipeline, q)
        new_answers += not hit
        answered_results.append(result)
        runs.append((result.chunks, q.gold))
        row = {
            "id": q.id,
            "set": "answerable",
            "refused": result.refused,
            "latency_s": result.latency_s,
        }
        if use_judge and not result.refused:
            s = judge(q.question, result.chunks, result.answer, result.citations, q.reference)
            new_judgments += not s.cached
            scores.append(s)
            row["judge"] = {k: v for k, v in asdict(s).items() if k != "cached"}
        rows.append(row)

    # Refusal: did the pipeline refuse? Also broken down by why it should refuse
    refusal_results, by_reason = [], defaultdict(list)
    for q in refusal:
        result, hit = answer(pipeline, q)
        new_answers += not hit
        refusal_results.append(result)
        by_reason[q.reason].append(result)
        rows.append(
            {
                "id": q.id,
                "set": "refusal",
                "reason": q.reason,
                "refused": result.refused,
                "latency_s": result.latency_s,
            }
        )

    latencies = [r.latency_s for r in answered_results + refusal_results]
    metrics = {
        "retrieval": evaluate_retrieval(runs),
        "false_refusal_rate": false_refusal_rate(answered_results),
        "refusal_accuracy": refusal_accuracy(refusal_results),
        "refusal_accuracy_by_reason": {
            reason: refusal_accuracy(results) for reason, results in sorted(by_reason.items())
        },
        "latency_s": {"mean": statistics.mean(latencies), "median": statistics.median(latencies)},
    }
    if scores:
        metrics["judge"] = {
            name: statistics.mean(getattr(s, name) for s in scores)
            for name in ["correctness", "groundedness", "citation_accuracy"]
        } | {"n": len(scores)}
    return {
        "arch": pipeline.name,
        "mode": "full" if full else "dev",
        "n_answerable": len(answerable),
        "n_refusal": len(refusal),
        "new_api_runs": {"pipeline_answers": new_answers, "judge_calls": new_judgments},
        "metrics": metrics,
        "questions": rows,
    }
