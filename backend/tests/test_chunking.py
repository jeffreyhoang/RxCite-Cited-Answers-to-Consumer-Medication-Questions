from rxcite.chunking import MAX_CHARS, chunk_section


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
