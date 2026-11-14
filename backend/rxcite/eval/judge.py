# LLM judge: scores one answer 0-2 on correctness, groundedness, and citation accuracy
#       Check it with: uv run python -m rxcite.eval.judge

import json
from dataclasses import dataclass

from rxcite import llm
from rxcite.config import JUDGE_MODEL
from rxcite.pipelines.base import Chunk, Citation

SYSTEM = """You grade answers from a medication question-answering assistant that must answer \
only from FDA drug label excerpts and cite the label sections it used.

Score each criterion 0, 1, or 2:

correctness - does the answer give the right information for the question?
  2 = correct and answers the question; 1 = partly correct or incomplete; 0 = wrong, or does not \
answer. Use the reference answer as a guide, but the label excerpts take priority if they differ \
(for example, a different manufacturer's pill color).

groundedness - is every claim supported by the label excerpts?
  2 = all claims supported; 1 = minor unsupported details; 0 = key claims unsupported or \
contradicted by the excerpts.

citation_accuracy - do the citations point to excerpts that support the answer?
  2 = every key claim is cited and each citation supports it; 1 = some claims uncited or some \
citations off; 0 = no citations, or the citations don't support the answer.

Judge only what is written. Give a one-sentence reason."""

SCORE = {"type": "integer", "enum": [0, 1, 2]}
SCHEMA = {
    "type": "object",
    "properties": {
        "correctness": SCORE,
        "groundedness": SCORE,
        "citation_accuracy": SCORE,
        "reason": {"type": "string"},
    },
    "required": ["correctness", "groundedness", "citation_accuracy", "reason"],
    "additionalProperties": False,
}


@dataclass
class JudgeScore:
    correctness: int
    groundedness: int
    citation_accuracy: int
    reason: str
    cached: bool = False  # True if the score came from the disk cache (no API call)

    @property
    def total(self) -> int:
        return self.correctness + self.groundedness + self.citation_accuracy


def build_prompt(
    question: str, chunks: list[Chunk], answer: str, citations: list[Citation], reference: str
) -> str:
    excerpts = "\n\n".join(f"[{c.set_id} | {c.section}] ({c.drug_name})\n{c.text}" for c in chunks)
    cited = "\n".join(f"[{c.set_id} | {c.section}]" for c in citations) or "(none)"
    return (
        f"<question>\n{question}\n</question>\n\n"
        f"<label_excerpts>\n{excerpts}\n</label_excerpts>\n\n"
        f"<answer>\n{answer}\n</answer>\n\n"
        f"<citations>\n{cited}\n</citations>\n\n"
        f"<reference_answer>\n{reference or '(none)'}\n</reference_answer>"
    )


def judge(
    question: str,
    chunks: list[Chunk],
    answer: str,
    citations: list[Citation],
    reference: str,
    model: str = JUDGE_MODEL,
) -> JudgeScore:
    resp = llm.call(
        build_prompt(question, chunks, answer, citations, reference),
        model,
        system=SYSTEM,
        max_tokens=1000,
        effort="low",
        schema=SCHEMA,
    )
    return JudgeScore(**json.loads(resp.text), cached=resp.cached)


# Check: three hand-written answers (correct, wrong, uncited) to a real question
if __name__ == "__main__":
    from rxcite import db

    question = 'do i take over the counter calcium with "alendronate?"'  # mqa-121
    reference = (
        "Co-administration of alendronate sodium and calcium, antacids, or oral medications "
        "containing multivalent cations will interfere with absorption of alendronate."
    )
    with db.connect() as conn:
        rows = conn.execute(
            """select s.set_id, l.drug_name, s.section, s.text from sections s
               join labels l using (set_id)
               where l.drug_name = 'alendronate'
                 and s.section in ('drug_interactions', 'indications_and_usage')"""
        ).fetchall()
    chunks = [Chunk(f"{sid}-{sec}", sid, drug, sec, text) for sid, drug, sec, text in rows]
    gold = next(c for c in chunks if c.section == "drug_interactions")
    cite = [Citation(gold.set_id, gold.section)]
    answers = {
        "correct": (
            "Don't take them at the same time. Calcium supplements interfere with how "
            "alendronate is absorbed, so take alendronate first and wait before taking calcium.",
            cite,
        ),
        "wrong": (
            "Yes, take them together. Calcium helps your body absorb alendronate better.",
            cite,
        ),
        "uncited": (
            "Don't take them at the same time. Calcium supplements interfere with how "
            "alendronate is absorbed, so take alendronate first and wait before taking calcium.",
            [],
        ),
    }
    scores = {}
    for name, (text, citations) in answers.items():
        scores[name] = judge(question, chunks, text, citations, reference)
        s = scores[name]
        print(
            f"{name:8} correct={s.correctness} grounded={s.groundedness} "
            f"cited={s.citation_accuracy} total={s.total} | {s.reason}"
        )
    c, w, u = scores["correct"], scores["wrong"], scores["uncited"]
    ok = c.total > w.total and c.total > u.total and w.correctness < u.correctness
    ok = ok and u.citation_accuracy < c.citation_accuracy
    print("PASS" if ok else "FAIL")
