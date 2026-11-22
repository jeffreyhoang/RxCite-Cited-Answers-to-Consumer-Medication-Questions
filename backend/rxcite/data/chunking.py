# Chunking: split labels into fixed-size and section-based chunks

import re

MAX_CHARS = 1600  # ~400 tokens
OVERLAP_CHARS = 200  # ~50 tokens repeated from the end of the previous chunk

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


# Sentences, with any sentence longer than max_chars cut at word breaks
def _units(text: str, max_chars: int) -> list[str]:
    units = []
    for sentence in _SENTENCE_END.split(text):
        while len(sentence) > max_chars:
            cut = sentence.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            units.append(sentence[:cut])
            sentence = sentence[cut:].lstrip()
        if sentence:
            units.append(sentence)
    return units


# Split one section's text into chunks of up to max_chars, ending at sentence breaks
def chunk_section(
    text: str, max_chars: int = MAX_CHARS, overlap_chars: int = OVERLAP_CHARS
) -> list[str]:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return [text] if text else []

    chunks, current = [], []
    for unit in _units(text, max_chars):
        if current and len(" ".join(current + [unit])) > max_chars:
            chunks.append(" ".join(current))
            # Start the next chunk with the last few sentences, up to overlap_chars
            overlap = []
            for prev in reversed(current):
                if len(" ".join([prev] + overlap + [unit])) > max_chars:
                    break
                if len(" ".join([prev] + overlap)) > overlap_chars:
                    break
                overlap.insert(0, prev)
            current = overlap
        current.append(unit)
    chunks.append(" ".join(current))
    return chunks


SECTION_MAX_CHARS = MAX_CHARS  # with the drug | section header, still fits bge-reranker's 512
MIN_CHARS = 200  # a smaller piece (e.g. a bare heading) is merged into the next one

_FIRST_NUMBER = re.compile(r"\s*(\d+)(?:\.\d+)?\s")  # "5 WARNINGS ..." or "5.1 Lactic ..."
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")  # ingest_openfda.py joins openFDA's entries with \n\n


# Start offsets of numbered subheadings ("5.1 Lactic Acidosis"): X must be the section's own
#       number, Y the next in sequence, then a capital letter. Rejects "2.5 mg", "(5.2)", "15.1".
def _heading_starts(text: str) -> list[int]:
    first = _FIRST_NUMBER.match(text)
    if not first:
        return []
    heading = re.compile(rf"(?<![\w.]){first.group(1)}\.(\d+)(?=\s+[A-Z])")
    starts, expected = [], 1
    for match in heading.finditer(text):
        if int(match.group(1)) == expected:
            starts.append(match.start())
            expected += 1
    return starts


# Pack consecutive paragraphs together up to max_chars
def _pack(paragraphs: list[str], max_chars: int) -> list[str]:
    pieces: list[str] = []
    for para in paragraphs:
        if pieces and len(pieces[-1]) + 1 + len(para) <= max_chars:
            pieces[-1] += " " + para
        else:
            pieces.append(para)
    return pieces


# Merge each piece under MIN_CHARS into the next one (the last into the one before it)
def _merge_small(pieces: list[str]) -> list[str]:
    merged: list[str] = []
    carry = ""
    for piece in pieces:
        piece = f"{carry} {piece}" if carry else piece
        if len(piece) < MIN_CHARS:
            carry = piece
        else:
            merged.append(piece)
            carry = ""
    if carry:
        if merged:
            merged[-1] += " " + carry
        else:
            merged.append(carry)
    return merged


# A section's pieces before the size fallback (may still be over the cap), and the strategy:
#       "whole", "headings", "paragraphs", or "sentences" (neither found)
def section_pieces(text: str, max_chars: int = SECTION_MAX_CHARS) -> tuple[list[str], str]:
    flat = " ".join(text.split())
    if len(flat) <= max_chars:
        return ([flat] if flat else []), "whole"

    starts = _heading_starts(flat)
    if starts:
        bounds = [0, *starts, len(flat)] if starts[0] > 0 else [*starts, len(flat)]
        pieces = [flat[a:b].strip() for a, b in zip(bounds, bounds[1:], strict=False)]
        return _merge_small(pieces), "headings"

    # Paragraph breaks have to be found in the raw text: normalizing removes them
    paragraphs = [" ".join(p.split()) for p in _PARAGRAPH_BREAK.split(text)]
    paragraphs = [p for p in paragraphs if p]
    if len(paragraphs) > 1:
        return _merge_small(_pack(paragraphs, max_chars)), "paragraphs"
    return [flat], "sentences"


# Split one section at its subheadings (or paragraphs); any piece still over the cap falls back
# to chunk_section (sentence packing with overlap). A short section stays one chunk.
def split_section(text: str, max_chars: int = SECTION_MAX_CHARS) -> list[str]:
    chunks: list[str] = []
    for piece in section_pieces(text, max_chars)[0]:
        chunks += [piece] if len(piece) <= max_chars else chunk_section(piece, max_chars)
    return chunks
