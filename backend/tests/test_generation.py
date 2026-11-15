from rxcite import generation
from rxcite.generation import NO_CONTEXT, generate, resolve_citations
from rxcite.pipelines.base import Chunk, Citation

CHUNKS = [
    Chunk("a:warnings:0", "a", "drugA", "warnings", "..."),
    Chunk("a:warnings:1", "a", "drugA", "warnings", "..."),  # same section as [1]
    Chunk("b:dosage:0", "b", "drugB", "dosage", "..."),
]


def test_markers_become_renumbered_citations():
    text, cites = resolve_citations("Take with food [3]. May cause dizziness [1][2].", CHUNKS)
    assert cites == [Citation("b", "dosage"), Citation("a", "warnings")]
    assert text == "Take with food [1]. May cause dizziness [2]."


def test_out_of_range_markers_are_dropped():
    text, cites = resolve_citations("Claim [7]. Other claim [0].", CHUNKS)
    assert cites == [] and text == "Claim. Other claim."


def test_no_chunks_refuses_without_api(monkeypatch):
    monkeypatch.setattr(generation.llm, "call", lambda *a, **k: 1 / 0)  # fails if called
    a = generate("anything", [])
    assert a.refused and a.text == NO_CONTEXT and a.citations == []


def test_citation_url_and_label():
    c = Citation("abc-123", "drug_interactions", "lisinopril")
    assert c.url == "https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid=abc-123"
    assert c.label == "Lisinopril — Drug interactions"
    assert Citation("x", "spl_medguide").label == "Medication guide"
    assert c == Citation("abc-123", "drug_interactions")  # drug name doesn't affect matching
