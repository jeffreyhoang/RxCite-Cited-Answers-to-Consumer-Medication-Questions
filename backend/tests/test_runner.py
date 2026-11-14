from rxcite.eval import runner
from rxcite.eval.judge import JudgeScore
from rxcite.pipelines.base import Answer, Chunk, Citation, RAGPipeline


# Retrieves each question's gold sections, answers answerable ones, refuses the rest
class Perfect(RAGPipeline):
    name = "perfect_test"

    def __init__(self, gold_by_question):
        self.gold = gold_by_question
        super().__init__(self.generate)

    def retrieve(self, query):
        return [Chunk(f"{s}-{sec}", s, "x", sec, "") for s, sec in self.gold.get(query, [])]

    def generate(self, query, chunks):
        if not chunks:
            return Answer(text="I can't answer that from the labels.", refused=True)
        return Answer(text="answer", citations=[Citation(chunks[0].set_id, chunks[0].section)])


def test_run_scores_and_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RUNS_CACHE", tmp_path)
    calls = []

    def fake_judge(question, chunks, answer, citations, reference):
        calls.append(question)
        return JudgeScore(2, 2, 1, "fine", cached=len(calls) > runner.DEV_ANSWERABLE)

    monkeypatch.setattr(runner, "judge", fake_judge)
    answerable, refusal = runner.eval_sets(full=False)
    assert (len(answerable), len(refusal)) == (20, 10)
    pipeline = Perfect({q.question: q.gold for q in answerable})

    first = runner.run(pipeline)
    m = first["metrics"]
    assert m["retrieval"]["recall@10"] == 1.0 and m["retrieval"]["mrr"] == 1.0
    assert m["refusal_accuracy"] == 1.0 and m["false_refusal_rate"] == 0.0
    assert m["judge"] == {"correctness": 2, "groundedness": 2, "citation_accuracy": 1, "n": 20}
    assert first["new_api_runs"] == {"pipeline_answers": 30, "judge_calls": 20}

    second = runner.run(pipeline)  # answers come from the cache; fake judge reports cache hits
    assert second["new_api_runs"] == {"pipeline_answers": 0, "judge_calls": 0}
    assert second["metrics"] == m


def test_dev_subset_is_fixed():
    a1, r1 = runner.eval_sets(full=False)
    a2, r2 = runner.eval_sets(full=False)
    assert [q.id for q in a1 + r1] == [q.id for q in a2 + r2]
    full_a, full_r = runner.eval_sets(full=True)
    assert (len(full_a), len(full_r)) == (160, 136)
