from rxcite.data.chunking import (
    MAX_CHARS,
    MIN_CHARS,
    SECTION_MAX_CHARS,
    chunk_section,
    section_pieces,
    split_section,
)


def test_short_section_stays_whole():
    assert chunk_section("Take one tablet daily.") == ["Take one tablet daily."]
    assert chunk_section("   ") == []


def test_long_section_respects_limit_and_overlaps():
    text = " ".join(f"Sentence number {i} about dosing and side effects." for i in range(200))
    chunks = chunk_section(text)
    assert len(chunks) > 1
    assert all(len(c) <= MAX_CHARS for c in chunks)
    # Each chunk starts with a sentence from the end of the previous one
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert nxt.split(". ")[0] in prev
    # Nothing is lost: every sentence appears in some chunk
    assert all(f"Sentence number {i} " in " ".join(chunks) for i in range(200))


def test_very_long_sentence_is_cut_at_words():
    text = "word " * 1000  # one 5000-char "sentence" with no period
    chunks = chunk_section(text)
    assert all(len(c) <= MAX_CHARS for c in chunks)
    assert all(not c.startswith(" ") and "wor d" not in c for c in chunks)
    assert sum(c.count("word") for c in chunks) >= 1000


def test_chunks_stay_inside_one_section():
    # Chunking works one section at a time, so text from two sections never mixes
    a, b = "A" * 10 + ". " + "Alpha text. " * 300, "Beta text. " * 300
    assert not any("Beta" in c for c in chunk_section(a))
    assert not any("Alpha" in c for c in chunk_section(b))


# n sentences of filler (~60 chars each), tagged so each one can be found again
def _body(tag: str, n: int) -> str:
    return " ".join(f"Patients in group {tag}-{i} should be monitored closely." for i in range(n))


# A flattened PLR section like openFDA's: "5 WARNINGS AND PRECAUTIONS 5.1 ... 5.2 ... 5.3 ..."
def _warnings(*subsections: tuple[str, int]) -> str:
    parts = ["5 WARNINGS AND PRECAUTIONS"]
    for i, (title, n) in enumerate(subsections, 1):
        parts.append(f"5.{i} {title} {_body(f'5.{i}', n)}")
    return " ".join(parts)


def _words(text: str) -> set[str]:
    return set(text.split())


def test_short_section_kept_whole():
    text = "5 WARNINGS AND PRECAUTIONS 5.1 Lactic Acidosis Stop the drug.\n\n5.2 Hypoglycemia Eat."
    assert split_section(text) == [" ".join(text.split())]
    assert section_pieces(text)[1] == "whole"
    assert split_section("  \n ") == []


def test_splits_at_numbered_subheadings():
    text = _warnings(("Lactic Acidosis", 10), ("Hypoglycemia", 12), ("Vitamin B12 Levels", 8))
    assert len(text) > SECTION_MAX_CHARS
    chunks = split_section(text)
    assert section_pieces(text)[1] == "headings"
    assert len(chunks) == 3
    # The bare section title is merged into the first subsection; each chunk starts at a heading
    assert chunks[0].startswith("5 WARNINGS AND PRECAUTIONS 5.1 Lactic Acidosis")
    assert chunks[1].startswith("5.2 Hypoglycemia")
    assert chunks[2].startswith("5.3 Vitamin B12 Levels")
    assert " ".join(chunks) == " ".join(text.split())  # nothing lost or repeated


def test_dose_is_not_a_heading():
    text = (
        "2 DOSAGE AND ADMINISTRATION 2.1 Recommended Dosage "
        + _body("a", 12)
        + " Start with 2.5 mg once daily. Increase to 2.5 Mg if needed. "
        + _body("b", 12)
        + " 2.2 Renal Impairment "
        + _body("c", 6)
    )
    chunks = split_section(text)
    assert len(chunks) == 2
    assert chunks[1].startswith("2.2 Renal Impairment")
    assert not any(c.startswith("2.5") for c in chunks)


def test_out_of_order_numbers_ignored():
    # 5.2 before 5.1, 5.3 skipped to from 5.1, 15.1 and (5.2) cross-references: none split
    text = (
        "5 WARNINGS AND PRECAUTIONS See 5.2 Hypoglycemia below. "
        + "5.1 Lactic Acidosis "
        + _body("a", 14)
        + " Table 15.1 Results. Also see (5.2) and 5.3 Other Risks here. "
        + _body("b", 14)
        + " 5.2 Hypoglycemia "
        + _body("c", 14)
    )
    chunks = split_section(text)
    starts = [c.split()[0] for c in chunks]
    assert starts[0] == "5"
    assert all(s in {"5", "5.1", "5.2"} for s in starts)
    assert sum(c.startswith("5.2 Hypoglycemia " + "Patients") for c in chunks) == 1
    assert not any(c.startswith(("5.3", "15.1", "See")) for c in chunks)


def test_paragraph_fallback():
    # Older non-PLR label: no numbered headings, but openFDA list entries joined with \n\n
    paras = [f"WARNING {i}. " + _body(str(i), 9) for i in range(5)]
    text = "\n\n".join(paras)
    assert len(text) > SECTION_MAX_CHARS
    pieces, strategy = section_pieces(text)
    assert strategy == "paragraphs"
    chunks = split_section(text)
    assert all(len(c) <= SECTION_MAX_CHARS for c in chunks)
    # Paragraphs are packed whole: every chunk starts at a paragraph
    assert all(c.startswith("WARNING ") for c in chunks)
    assert len(chunks) < len(paras)
    assert " ".join(chunks) == " ".join(text.split())


def test_oversize_piece_falls_back_to_sentences():
    text = _warnings(("Lactic Acidosis", 70), ("Hypoglycemia", 5))  # 5.1 is ~4,000 chars
    chunks = split_section(text)
    assert len(chunks) > 3
    assert all(len(c) <= SECTION_MAX_CHARS for c in chunks)
    assert chunks[-1].startswith("5.2 Hypoglycemia")
    # One long run-on paragraph with neither headings nor breaks: plain sentence packing
    flat = _body("x", 80)
    assert section_pieces(flat)[1] == "sentences"
    assert split_section(flat) == chunk_section(flat)


def test_tiny_pieces_are_merged():
    # Bare headings and a short last subsection must not become chunks of their own
    text = (
        "5 WARNINGS AND PRECAUTIONS 5.1 Lactic Acidosis 5.2 Hypoglycemia "
        + _body("a", 20)
        + " 5.3 Hypersensitivity "
        + _body("b", 20)
        + " 5.4 Rash Stop use."
    )
    chunks = split_section(text)
    assert all(len(c) >= MIN_CHARS for c in chunks)
    assert chunks[0].startswith("5 WARNINGS AND PRECAUTIONS 5.1 Lactic Acidosis 5.2")
    assert chunks[-1].endswith("5.4 Rash Stop use.")
    # Tiny paragraphs too
    paras = ["CAUTION."] + [_body(str(i), 12) for i in range(3)] + ["Keep out of reach."]
    chunks = split_section("\n\n".join(paras))
    assert all(len(c) >= MIN_CHARS for c in chunks)
    # A tiny last paragraph after an oversize one joins it before the sentence fallback
    pieces, _ = section_pieces("\n\n".join([_body("a", 30), "Keep out of reach."]))
    assert len(pieces) == 1
    assert pieces[0].endswith("a-29 should be monitored closely. Keep out of reach.")


def test_split_section_loses_no_text():
    texts = [
        _warnings(("Lactic Acidosis", 70), ("Hypoglycemia", 3), ("Rash", 25)),
        "\n\n".join([_body("p", 40), "Short.", _body("q", 10)]),
        _body("s", 90),
    ]
    for text in texts:
        chunks = split_section(text)
        assert _words(text) <= _words(" ".join(chunks))
        assert all(c == " ".join(c.split()) and c for c in chunks)  # normalized, none empty
        assert all(len(c) <= SECTION_MAX_CHARS for c in chunks)
