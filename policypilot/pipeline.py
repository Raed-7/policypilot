import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from policypilot.chunking import chunk_document
from policypilot.config import Settings, get_settings
from policypilot.embeddings import Embedder, SentenceTransformerEmbedder
from policypilot.generation import ExtractiveGenerator, LLMGenerator, extract_citations, is_refusal
from policypilot.ingestion import load_directory
from policypilot.ingestion.loaders import load_bytes
from policypilot.llm import LLMError, LLMProvider, build_provider
from policypilot.models import Document, RetrievedChunk
from policypilot.prompts import REFUSAL
from policypilot.retrieval import Mode, Retriever
from policypilot.vectorstore import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    index: int
    chunk_id: str
    doc_id: str
    source: str
    title: str
    page: int | None
    heading: str
    score: float
    text: str


@dataclass
class Answer:
    question: str
    answer: str
    answerable: bool
    citations: list[Citation]
    contexts: list[RetrievedChunk]
    provider: str
    model: str
    confidence: float
    latency_ms: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class IngestResult:
    doc_id: str
    source: str
    title: str
    chunks: int
    status: str


class PolicyPilot:
    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        llm: LLMProvider | None | str = "from_settings",
    ):
        self.settings = settings or get_settings()
        self.embedder = embedder or SentenceTransformerEmbedder(
            self.settings.embedding_model, batch_size=self.settings.embedding_batch_size
        )
        self.store = store or VectorStore(self.settings.chroma_dir, self.settings.collection_name)
        self.retriever = Retriever(self.store, self.embedder)
        provider = build_provider(self.settings) if llm == "from_settings" else llm
        self.llm_generator = LLMGenerator(provider) if provider else None
        self.extractive = ExtractiveGenerator(self.embedder)

    @property
    def provider_name(self) -> str:
        return self.llm_generator.provider.name if self.llm_generator else self.extractive.name

    @property
    def model_name(self) -> str:
        return self.llm_generator.provider.model if self.llm_generator else self.extractive.model

    def ingest(self, document: Document, refresh: bool = True) -> IngestResult:
        if self.store.has_document(document.doc_id):
            existing = next(d for d in self.store.documents() if d["doc_id"] == document.doc_id)
            return IngestResult(document.doc_id, document.source, document.title, existing["chunks"], "unchanged")
        for existing in self.store.documents():
            if existing["source"] == document.source:
                self.store.delete_document(existing["doc_id"])
        chunks = chunk_document(document, self.settings.chunk_size, self.settings.chunk_overlap)
        embeddings = self.embedder.embed_documents([c.text for c in chunks])
        self.store.add(chunks, embeddings)
        if refresh:
            self.retriever.refresh()
        return IngestResult(document.doc_id, document.source, document.title, len(chunks), "indexed")

    def ingest_bytes(self, data: bytes, filename: str) -> IngestResult:
        return self.ingest(load_bytes(data, filename))

    def ingest_directory(self, directory: str | Path) -> list[IngestResult]:
        results = [self.ingest(doc, refresh=False) for doc in load_directory(directory)]
        self.retriever.refresh()
        return results

    def delete_document(self, doc_id: str) -> int:
        removed = self.store.delete_document(doc_id)
        self.retriever.refresh()
        return removed

    def documents(self) -> list[dict]:
        return self.store.documents()

    def search(self, query: str, top_k: int | None = None, threshold: float | None = None, mode: Mode | None = None):
        return self.retriever.retrieve(
            query,
            k=top_k or self.settings.top_k,
            threshold=self.settings.score_threshold if threshold is None else threshold,
            mode=mode or self.settings.retrieval_mode,
        )

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        threshold: float | None = None,
        mode: Mode | None = None,
    ) -> Answer:
        started = time.perf_counter()
        contexts = self.search(question, top_k, threshold, mode)
        retrieved_at = time.perf_counter()
        warnings: list[str] = []

        if not contexts:
            text, provider, model = REFUSAL, self.provider_name, self.model_name
        elif self.llm_generator:
            try:
                result = self.llm_generator.generate(question, contexts)
            except LLMError as exc:
                logger.warning("LLM generation failed, using extractive answer: %s", exc)
                warnings.append(f"{self.llm_generator.provider.name} unavailable ({exc}); showing extractive answer.")
                result = self.extractive.generate(question, contexts)
            text, provider, model = result.text, result.provider, result.model
        else:
            result = self.extractive.generate(question, contexts)
            text, provider, model = result.text, result.provider, result.model

        answerable = bool(contexts) and not is_refusal(text)
        citations = []
        if answerable:
            for number in extract_citations(text, len(contexts)):
                item = contexts[number - 1]
                c = item.chunk
                citations.append(
                    Citation(number, c.chunk_id, c.doc_id, c.source, c.title, c.page, c.heading, round(item.dense_score, 4), c.text)
                )
            if not citations:
                warnings.append("The answer did not cite any source; verify it against the excerpts below.")
        else:
            text = REFUSAL

        finished = time.perf_counter()
        return Answer(
            question=question,
            answer=text,
            answerable=answerable,
            citations=citations,
            contexts=contexts,
            provider=provider,
            model=model,
            confidence=round(max((c.dense_score for c in contexts), default=0.0), 4),
            latency_ms={
                "retrieval": round((retrieved_at - started) * 1000, 1),
                "generation": round((finished - retrieved_at) * 1000, 1),
                "total": round((finished - started) * 1000, 1),
            },
            warnings=warnings,
        )
