# Parent expansion for Hierarchical RAG: matched child chunks -> their parent label sections
#       Parents are ranked by their best child's score, plus TOPIC_BONUS when the section fits
#       the question type. Each parent's text is the whole section, or, if the section is longer
#       than MAX_PARENT_CHARS, only the matched chunks and their neighbors.

from dataclasses import dataclass, field

from rxcite import db
from rxcite.pipelines.base import Chunk
from rxcite.query.analysis import SECTION_GROUPS

N_PARENTS = 5
MAX_PARENT_CHARS = 6000  # ~1,500 tokens; 5 parents ≈ Naive's 10 chunks
TOPIC_BONUS = 0.05  # added to a parent's score when its section fits the question type
GAP = "\n[...]\n"  # marks text skipped between non-adjacent chunks


@dataclass
class Parent:
    set_id: str
    section: str
    drug_name: str
    score: float
    matched: dict[int, float] = field(default_factory=dict)  # chunk number -> child score


def _chunk_number(chunk_id: str) -> int:
    return int(chunk_id.rsplit(":", 1)[1])  # "<set_id>:<section>:<n>"


# Group children by (set_id, section) and keep the n best parents
def rank_parents(children: list[Chunk], section_type: str, n: int = N_PARENTS) -> list[Parent]:
    preferred = set(SECTION_GROUPS.get(section_type, []))
    parents: dict[tuple[str, str], Parent] = {}
    for c in children:
        p = parents.setdefault(
            (c.set_id, c.section), Parent(c.set_id, c.section, c.drug_name, float("-inf"))
        )
        p.matched[_chunk_number(c.chunk_id)] = c.score
        p.score = max(p.score, c.score)
    for p in parents.values():
        p.score += TOPIC_BONUS if p.section in preferred else 0.0
    return sorted(parents.values(), key=lambda p: p.score, reverse=True)[:n]


# Whole section if short enough; otherwise matched chunks 1 neighbors, best matches first,
# until the cap. The best match is always kept.
def parent_text(
    section_text: str,
    chunk_texts: dict[int, str],
    matched: dict[int, float],
    max_chars: int = MAX_PARENT_CHARS,
) -> str:
    if len(section_text) <= max_chars:
        return section_text
    keep: set[int] = set()

    def size(numbers: set[int]) -> int:
        return sum(len(chunk_texts[i]) for i in numbers) + len(GAP) * len(numbers)

    for i in sorted(matched, key=matched.get, reverse=True):
        window = {j for j in (i - 1, i, i + 1) if j in chunk_texts}
        if size(keep | window) <= max_chars:
            keep |= window
        elif size(keep | {i}) <= max_chars or not keep:
            keep.add(i)
    parts, prev = [], None
    for i in sorted(keep):
        if prev is not None and i != prev + 1:
            parts.append(GAP)
        elif prev is not None:
            parts.append(" ")
        parts.append(chunk_texts[i])
        prev = i
    return "".join(parts)


# The n best parents as Chunks (chunk_id "<set_id>:<section>"), best first
def expand_to_parents(children: list[Chunk], section_type: str, n: int = N_PARENTS) -> list[Chunk]:
    parents = rank_parents(children, section_type, n)
    if not parents:
        return []
    keys = ([p.set_id for p in parents], [p.section for p in parents])
    sections = {
        (s, sec): text
        for s, sec, text in db.query(
            """select set_id, section, text from sections
               where (set_id, section) in (select * from unnest(%s::text[], %s::text[]))""",
            keys,
        )
    }
    long_keys = [
        (p.set_id, p.section)
        for p in parents
        if len(sections[(p.set_id, p.section)]) > MAX_PARENT_CHARS
    ]
    chunk_texts: dict[tuple[str, str], dict[int, str]] = {k: {} for k in long_keys}
    if long_keys:
        rows = db.query(
            """select set_id, section, chunk_id, text from chunks
               where (set_id, section) in (select * from unnest(%s::text[], %s::text[]))""",
            ([s for s, _ in long_keys], [sec for _, sec in long_keys]),
        )
        for s, sec, chunk_id, text in rows:
            chunk_texts[(s, sec)][_chunk_number(chunk_id)] = text
    return [
        Chunk(
            f"{p.set_id}:{p.section}",
            p.set_id,
            p.drug_name,
            p.section,
            parent_text(
                sections[(p.set_id, p.section)],
                chunk_texts.get((p.set_id, p.section), {}),
                p.matched,
            ),
            p.score,
        )
        for p in parents
    ]
