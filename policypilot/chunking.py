import hashlib
import re
from typing import Literal

from policypilot.models import Chunk, Document, Section

Strategy = Literal["section", "fixed"]

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n+")
_WORD = re.compile(r"\S+")


def split_sentences(text: str) -> list[str]:
    parts = (p.strip() for p in _SENTENCE_END.split(text))
    return [p for p in parts if p]


def word_count(text: str) -> int:
    return len(_WORD.findall(text))


def _split_long_sentence(sentence: str, size: int) -> list[str]:
    words = sentence.split()
    return [" ".join(words[i : i + size]) for i in range(0, len(words), size)]


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Pack whole sentences into windows of roughly `size` words.

    Consecutive windows share up to `overlap` words of trailing sentences, so
    a fact that straddles a boundary is fully present in at least one chunk.
    """
    if size <= 0:
        raise ValueError("chunk size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be in [0, size)")

    sentences: list[str] = []
    for sentence in split_sentences(text):
        if word_count(sentence) > size:
            sentences.extend(_split_long_sentence(sentence, size))
        else:
            sentences.append(sentence)

    chunks: list[str] = []
    window: list[str] = []
    window_words = 0
    fresh = 0
    for sentence in sentences:
        n = word_count(sentence)
        if window and window_words + n > size:
            chunks.append(" ".join(window))
            carried: list[str] = []
            carried_words = 0
            for prev in reversed(window):
                w = word_count(prev)
                if carried_words + w > overlap:
                    break
                carried.insert(0, prev)
                carried_words += w
            window, window_words, fresh = carried, carried_words, 0
        window.append(sentence)
        window_words += n
        fresh += 1
    if window and fresh:
        chunks.append(" ".join(window))
    return chunks


def _chunk_id(doc_id: str, position: int, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"{doc_id}-{position:04d}-{digest}"


def _merge_sections(document: Document) -> list[Section]:
    """Collapse sections into one span per page (PDF) or per document, keeping headings inline."""
    merged: dict[int | None, list[str]] = {}
    for section in document.sections:
        text = f"{section.heading}\n{section.text}" if section.heading else section.text
        merged.setdefault(section.page, []).append(text)
    return [Section(text="\n".join(parts), page=page) for page, parts in merged.items()]


def chunk_document(document: Document, size: int, overlap: int, strategy: Strategy = "section") -> list[Chunk]:
    """Split a document into chunks.

    "section" chunks each heading-delimited section independently, so every chunk maps to
    exactly one page and heading. "fixed" ignores section boundaries and slides a window
    over each page (or the whole document for formats without pages).
    """
    sections = document.sections if strategy == "section" else _merge_sections(document)
    chunks: list[Chunk] = []
    for section in sections:
        body = section.text
        for piece in chunk_text(body, size, overlap):
            position = len(chunks)
            embedded_text = f"{section.heading}: {piece}" if section.heading and not piece.startswith(section.heading) else piece
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(document.doc_id, position, piece),
                    doc_id=document.doc_id,
                    source=document.source,
                    title=document.title,
                    text=embedded_text,
                    position=position,
                    page=section.page,
                    heading=section.heading,
                )
            )
    return chunks
