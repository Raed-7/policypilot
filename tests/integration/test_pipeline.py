import pytest

from policypilot.llm import LLMError
from policypilot.pipeline import PolicyPilot
from policypilot.prompts import REFUSAL
from tests.conftest import SAMPLE_DOCS


def test_ingest_directory_indexes_all_documents(indexed_pipeline):
    docs = indexed_pipeline.documents()
    assert len(docs) == 7
    assert all(d["chunks"] > 0 for d in docs)
    handbook = next(d for d in docs if d["source"] == "employee-handbook.pdf")
    assert handbook["pages"] == 5


def test_reingesting_unchanged_document_is_a_no_op(indexed_pipeline):
    before = indexed_pipeline.store.count()
    result = indexed_pipeline.ingest_bytes((SAMPLE_DOCS / "code-of-conduct.md").read_bytes(), "code-of-conduct.md")
    assert result.status == "unchanged"
    assert indexed_pipeline.store.count() == before


def test_updated_document_replaces_previous_version(indexed_pipeline):
    original = indexed_pipeline.store.count()
    result = indexed_pipeline.ingest_bytes(b"# Code of Conduct\n\nGifts up to \xc2\xa3100 are fine.", "code-of-conduct.md")
    assert result.status == "indexed"
    sources = [d["source"] for d in indexed_pipeline.documents()]
    assert sources.count("code-of-conduct.md") == 1
    assert indexed_pipeline.store.count() < original


def test_answer_cites_correct_page(indexed_pipeline):
    answer = indexed_pipeline.ask("What is the mileage rate per mile for business miles?")
    assert answer.answerable
    assert "45p" in answer.answer
    assert any(c.source == "expenses-and-travel-policy.pdf" and c.page == 2 for c in answer.citations)
    assert set(answer.latency_ms) == {"retrieval", "generation", "total"}


def test_unrelated_question_is_refused(indexed_pipeline):
    answer = indexed_pipeline.ask("zebra quantum spaceship telescope", evidence_threshold=0.5)
    assert not answer.answerable
    assert answer.answer == REFUSAL
    assert answer.citations == []


def test_empty_index_refuses(pipeline):
    answer = pipeline.ask("How much annual leave do I get?")
    assert not answer.answerable
    assert answer.contexts == []


def test_delete_document(indexed_pipeline):
    doc = indexed_pipeline.documents()[0]
    assert indexed_pipeline.delete_document(doc["doc_id"]) == doc["chunks"]
    assert doc["doc_id"] not in {d["doc_id"] for d in indexed_pipeline.documents()}


class FailingProvider:
    name, model = "groq", "llama"

    def complete(self, system, prompt):
        raise LLMError("HTTP 503")


class EchoProvider:
    name, model = "fake", "fake-1"

    def __init__(self, reply):
        self.reply = reply

    def complete(self, system, prompt):
        return self.reply


@pytest.fixture
def llm_pipeline(settings, embedder, store):
    def make(provider):
        pp = PolicyPilot(settings, embedder=embedder, store=store, llm=provider)
        pp.ingest_directory(SAMPLE_DOCS)
        return pp

    return make


def test_llm_failure_falls_back_to_extractive(llm_pipeline):
    answer = llm_pipeline(FailingProvider()).ask("What is the mileage rate per mile?")
    assert answer.provider == "extractive"
    assert answer.answerable
    assert any("groq unavailable" in w for w in answer.warnings)


def test_llm_refusal_is_respected(llm_pipeline):
    answer = llm_pipeline(EchoProvider(REFUSAL)).ask("What is the mileage rate per mile?")
    assert not answer.answerable
    assert answer.citations == []


def test_llm_answer_without_citations_is_flagged(llm_pipeline):
    answer = llm_pipeline(EchoProvider("It is 45p per mile.")).ask("What is the mileage rate per mile?")
    assert answer.answerable
    assert answer.provider == "fake"
    assert answer.warnings
