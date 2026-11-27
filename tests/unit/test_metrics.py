import pytest

from evaluation.metrics import fact_recall, first_relevant_rank, groundedness, is_relevant, mean_reciprocal_rank, recall_at_k
from policypilot.models import Chunk


def _chunk(text: str, source: str = "a.pdf") -> Chunk:
    return Chunk(chunk_id=text[:5], doc_id="d", source=source, title="T", text=text, position=0)


EXAMPLE = {"source": "a.pdf", "evidence": "45p per   mile"}


def test_relevance_requires_source_and_evidence():
    assert is_relevant(_chunk("Claim 45p per\nmile here."), EXAMPLE)
    assert not is_relevant(_chunk("Claim 45p per mile.", source="b.pdf"), EXAMPLE)
    assert not is_relevant(_chunk("Claim 25p per mile."), EXAMPLE)


def test_first_relevant_rank():
    chunks = [_chunk("nothing"), _chunk("pay 45p per mile")]
    assert first_relevant_rank(chunks, EXAMPLE) == 2
    assert first_relevant_rank(chunks[:1], EXAMPLE) is None


def test_recall_and_mrr():
    ranks = [1, 3, None, 2]
    assert recall_at_k(ranks, 1) == 0.25
    assert recall_at_k(ranks, 3) == 0.75
    assert mean_reciprocal_rank(ranks) == pytest.approx((1 + 1 / 3 + 0 + 0.5) / 4)
    assert recall_at_k([], 5) == 0.0


def test_fact_recall_is_case_and_space_insensitive():
    assert fact_recall("You get 25  Days of leave", ["25 days", "31 March"]) == 0.5
    assert fact_recall("anything", []) == 1.0


def test_groundedness_verbatim_answer_is_fully_supported(embedder):
    ctx = ["Mileage is paid at 45p per mile. Hotels are capped at £150."]
    g = groundedness("Mileage is paid at 45p per mile [1].", ctx, embedder)
    assert g.sentence_support == 1.0
    assert g.token_precision == 1.0


def test_groundedness_penalises_unsupported_content(embedder):
    ctx = ["Mileage is paid at 45p per mile."]
    g = groundedness("Employees receive a company car and private healthcare.", ctx, embedder)
    assert g.sentence_support == 0.0
    assert g.token_precision < 0.5
