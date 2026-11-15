# Fixed-size chunking (baseline for T5 and Naive RAG)
#       Each section is split on its own, so a chunk never crosses a section and can be cited
#       as (set_id, section). Sizes are in characters (~4 characters per token):
#       1600 characters ≈ 400 tokens, which fits every candidate embedder's 512-token limit.

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
