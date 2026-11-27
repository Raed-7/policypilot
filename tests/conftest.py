import uuid
from pathlib import Path

import pytest

from policypilot.config import Settings
from policypilot.embeddings import HashingEmbedder
from policypilot.pipeline import PolicyPilot
from policypilot.vectorstore import VectorStore

SAMPLE_DOCS = Path(__file__).resolve().parent.parent / "data" / "sample_policies"


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        llm_provider="extractive",
        chroma_dir=tmp_path / "chroma",
        results_dir=tmp_path / "results",
        chunk_size=120,
        chunk_overlap=20,
        score_threshold=0.05,
        evidence_threshold=0.2,
        seed_sample_docs=False,
    )


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dimension=1024)


@pytest.fixture
def store() -> VectorStore:
    return VectorStore(None, collection_name=f"test-{uuid.uuid4().hex}")


@pytest.fixture
def pipeline(settings, embedder, store) -> PolicyPilot:
    return PolicyPilot(settings, embedder=embedder, store=store, llm=None)


@pytest.fixture
def indexed_pipeline(pipeline) -> PolicyPilot:
    pipeline.ingest_directory(SAMPLE_DOCS)
    return pipeline
