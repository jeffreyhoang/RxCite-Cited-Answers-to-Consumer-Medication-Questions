# Shared generator for every RAG architecture: answers only from the retrieved chunks
#       The model sees numbered excerpts and cites them inline as [1], [2], ...
#       Code turns each marker into a (set_id, section) citation, then renumbers the markers so
#       [k] in the final answer is the k-th citation (chunks from the same section share one).
#       Check it with: uv run python -m rxcite.generation

import json
import re

from rxcite import llm
from rxcite.config import GENERATOR_MODEL
from rxcite.pipelines.base import Answer, Chunk, Citation

NO_CONTEXT = "I can't answer this from the FDA drug labels I have."

SYSTEM = """You answer consumer questions about medications using only the FDA drug label \
excerpts provided. The excerpts are numbered [1], [2], and so on.

Rules:
- Use only facts stated in the excerpts. Don't add outside knowledge, even if you know it.
- Cite every claim inline with the excerpt number(s) it came from, like [2] or [1][3].
- Write in plain, simple language for a patient, in a few sentences or a short list.
- If the excerpts contain information that addresses the question, even partly or \
indirectly, answer with it (refused = false) and say briefly what the labels don't cover. \
For example, "Is X used for stroke?" is answered by what the label says X is used for.
- Set refused to true only when the excerpts contain nothing relevant to the question, \
then briefly say the labels don't cover it. Don't guess or answer from memory.
- Also refuse personal medical decisions a label can't make for someone (for example, \
"should I stop my medicine?"), and suggest asking a doctor or pharmacist. General questions \
such as "how do I stop taking X?" are not personal: answer them from the excerpts.
- Excerpts may come from other drugs. Only use excerpts about the drug the question asks about."""

SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "refused": {"type": "boolean"},
    },
    "required": ["answer", "refused"],
    "additionalProperties": False,
}

_MARKER = re.compile(r"\[(\d+)\]")


def build_prompt(query: str, chunks: list[Chunk]) -> str:
    excerpts = "\n\n".join(
        f"[{i}] {c.drug_name} | {c.section.replace('_', ' ')}\n{c.text}"
        for i, c in enumerate(chunks, 1)
    )
    return f"<excerpts>\n{excerpts}\n</excerpts>\n\n<question>\n{query}\n</question>"


# Turn [n] markers (n = excerpt number) into citations; renumber so [k] = k-th citation.
# Markers that point to no excerpt are removed.
def resolve_citations(text: str, chunks: list[Chunk]) -> tuple[str, list[Citation]]:
    citations: list[Citation] = []

    def replace(m: re.Match) -> str:
        n = int(m.group(1))
        if not 1 <= n <= len(chunks):
            return ""
        c = chunks[n - 1]
        cite = Citation(c.set_id, c.section, c.drug_name)
        if cite not in citations:
            citations.append(cite)
        return f"[{citations.index(cite) + 1}]"

    text = _MARKER.sub(replace, text)
    text = re.sub(r"(\[\d+\])\1+", r"\1", text)  # [1][1] -> [1] when two chunks share a section
    return _tidy(text), citations


# Remove spaces left before punctuation after markers are dropped
def _tidy(text: str) -> str:
    return re.sub(r" +([.,;:])", r"\1", text).strip()


def generate(query: str, chunks: list[Chunk], model: str = GENERATOR_MODEL) -> Answer:
    if not chunks:
        return Answer(text=NO_CONTEXT, refused=True)
    resp = llm.call(build_prompt(query, chunks), model, system=SYSTEM, schema=SCHEMA)
    out = json.loads(resp.text)
    if out["refused"]:
        return Answer(text=_tidy(_MARKER.sub("", out["answer"])), refused=True)
    text, citations = resolve_citations(out["answer"], chunks)
    return Answer(text=text, citations=citations)


# Check: one answerable and one should-refuse question, retrieved with dense_search
if __name__ == "__main__":
    from rxcite.retrieval import dense_search

    questions = [
        ("answerable", 'do i take over the counter calcium with "alendronate?"'),  # mqa-121
        ("refusal", "I've been on lisinopril for a year. Can I just stop taking it now?"),
    ]
    results = {}
    for kind, q in questions:
        a = generate(q, dense_search(q))
        results[kind] = a
        print(f"--- {kind}: {q}\nrefused={a.refused}\n{a.text}")
        for i, c in enumerate(a.citations, 1):
            print(f"  [{i}] {c.set_id} | {c.section}")
    ok = not results["answerable"].refused and results["answerable"].citations
    ok = ok and results["refusal"].refused
    print("PASS" if ok else "FAIL")
