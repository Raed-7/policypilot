"""End-to-end checks with the real embedding model and the evaluated default settings."""

import os

import pytest

from policypilot.config import Settings
from policypilot.pipeline import PolicyPilot
from policypilot.vectorstore import VectorStore
from tests.conftest import SAMPLE_DOCS

pytestmark = pytest.mark.slow

if os.environ.get("SKIP_SLOW") == "1":
    pytest.skip("slow tests disabled", allow_module_level=True)


@pytest.fixture(scope="module")
def pp(tmp_path_factory):
    settings = Settings(_env_file=None, llm_provider="extractive", chroma_dir=tmp_path_factory.mktemp("chroma"))
    pipeline = PolicyPilot(settings, store=VectorStore(None, collection_name="real-model-test"))
    pipeline.ingest_directory(SAMPLE_DOCS)
    return pipeline


@pytest.mark.parametrize(
    "question,fact,source",
    [
        ("How much can I claim for a home desk and chair?", "£300", "remote-work-policy.md"),
        ("How quickly must I report a stolen laptop?", "one hour", "information-security-policy.md"),
        ("How long is paternity leave?", "four weeks", "parental-and-family-leave.txt"),
    ],
)
def test_answers_with_grounded_citation(pp, question, fact, source):
    answer = pp.ask(question)
    assert answer.answerable
    assert fact in answer.answer
    assert answer.citations[0].source == source


@pytest.mark.parametrize("question", ["Can I bring my dog to the office?", "Who is the CEO of Halden Analytics?"])
def test_refuses_questions_the_documents_do_not_cover(pp, question):
    assert not pp.ask(question).answerable
