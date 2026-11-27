import pytest

from policypilot.chunking import chunk_document, chunk_text, split_sentences, word_count
from policypilot.models import Document, Section


def _sentences(n: int, words: int = 10) -> str:
    return " ".join(f"Sentence {i} " + " ".join(["word"] * (words - 3)) + " end." for i in range(n))


def test_split_sentences():
    assert split_sentences("One fact. Two facts! Is it three? 4 is next.") == [
        "One fact.", "Two facts!", "Is it three?", "4 is next."
    ]


def test_does_not_split_on_decimal_or_currency():
    assert split_sentences("The week is 37.5 hours. The rate is £0.45.") == ["The week is 37.5 hours.", "The rate is £0.45."]


def test_chunks_respect_size():
    chunks = chunk_text(_sentences(20), size=50, overlap=0)
    assert len(chunks) == 4
    assert all(word_count(c) <= 50 for c in chunks)


def test_overlap_repeats_trailing_sentences():
    chunks = chunk_text(_sentences(20), size=50, overlap=10)
    assert len(chunks) > 4
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev.split(". ")[-1].rstrip(".") in nxt


def test_long_sentence_is_split():
    chunks = chunk_text(" ".join(["word"] * 250), size=100, overlap=0)
    assert [word_count(c) for c in chunks] == [100, 100, 50]


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, -1)])
def test_invalid_parameters(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("text", size, overlap)


def _doc() -> Document:
    return Document(
        doc_id="d1", source="h.pdf", title="Handbook", file_type="pdf",
        sections=[
            Section("Leave is 25 days. Carry over five days.", page=1, heading="Annual leave"),
            Section("Call in sick by 09:30.", page=2, heading="Sickness"),
        ],
    )


def test_section_strategy_keeps_location_and_heading():
    chunks = chunk_document(_doc(), size=100, overlap=0, strategy="section")
    assert [(c.page, c.heading) for c in chunks] == [(1, "Annual leave"), (2, "Sickness")]
    assert chunks[0].text.startswith("Annual leave: ")
    assert chunks[1].location == "p. 2, Sickness"


def test_fixed_strategy_merges_sections_per_page():
    doc = _doc()
    doc.sections.insert(1, Section("Book through PeopleHub.", page=1, heading="Requests"))
    chunks = chunk_document(doc, size=100, overlap=0, strategy="fixed")
    assert [c.page for c in chunks] == [1, 2]
    assert "Requests. Book through PeopleHub." in chunks[0].text


def test_chunk_ids_are_unique_and_stable():
    first = chunk_document(_doc(), 100, 0)
    second = chunk_document(_doc(), 100, 0)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert len({c.chunk_id for c in first}) == len(first)
