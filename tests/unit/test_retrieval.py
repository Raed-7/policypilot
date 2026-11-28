import numpy as np

from policypilot.lexical import BM25Index, tokenize
from policypilot.models import Chunk
from policypilot.retrieval import Retriever, reciprocal_rank_fusion


def _chunk(i: int, text: str) -> Chunk:
    return Chunk(chunk_id=f"c{i}", doc_id="d", source="p.md", title="P", text=text, position=i)


TEXTS = [
    "Mileage is reimbursed at 45p per mile for business journeys.",
    "Annual leave is 25 days plus bank holidays.",
    "Passwords must be at least 14 characters long.",
    "Hotels in London are capped at £150 per night.",
]


def test_tokenize_drops_stopwords_and_keeps_currency():
    assert tokenize("What is the hotel cap in £150?") == ["hotel", "cap", "£150"]


def test_bm25_ranks_matching_chunk_first():
    index = BM25Index()
    index.build([(f"c{i}", t) for i, t in enumerate(TEXTS)])
    results = index.search("password length characters", k=2)
    assert results[0][0] == "c2"
    assert all(score > 0 for _, score in results)


def test_bm25_handles_plurals_and_empty_index():
    index = BM25Index()
    assert index.search("anything", 3) == []
    index.build([("a", "hotels are booked"), ("b", "flights are booked")])
    assert index.search("hotel", 1)[0][0] == "a"


def test_reciprocal_rank_fusion_rewards_agreement():
    scores = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]])
    assert max(scores, key=scores.get) in {"a", "b"}
    assert scores["a"] == scores["b"] > scores["c"] > 0
    assert scores["d"] == scores["c"]


def _retriever(store, embedder) -> Retriever:
    chunks = [_chunk(i, t) for i, t in enumerate(TEXTS)]
    store.add(chunks, embedder.embed_documents(TEXTS))
    return Retriever(store, embedder)


def test_dense_retrieval_returns_scored_chunks(store, embedder):
    retriever = _retriever(store, embedder)
    hits = retriever.retrieve("How long must passwords be?", k=2, mode="dense")
    assert hits[0].chunk.chunk_id == "c2"
    assert hits[0].dense_score >= hits[1].dense_score


def test_hybrid_retrieval_uses_lexical_matches(store, embedder):
    retriever = _retriever(store, embedder)
    hits = retriever.retrieve("hotel London £150", k=3, mode="hybrid")
    assert hits[0].chunk.chunk_id == "c3"
    assert hits[0].lexical_score > 0


def test_threshold_filters_weak_matches(store, embedder):
    retriever = _retriever(store, embedder)
    assert retriever.retrieve("quantum chromodynamics", k=3, threshold=0.3) == []


def test_empty_store_and_blank_query(store, embedder):
    retriever = Retriever(store, embedder)
    assert retriever.retrieve("anything") == []
    retriever = _retriever(store, embedder)
    assert retriever.retrieve("   ") == []


def test_vector_store_document_management(store, embedder):
    chunks = [_chunk(i, t) for i, t in enumerate(TEXTS)]
    store.add(chunks, embedder.embed_documents(TEXTS))
    assert store.count() == 4
    assert store.documents() == [{"doc_id": "d", "source": "p.md", "title": "P", "chunks": 4, "pages": None}]
    assert store.has_document("d")
    assert set(store.get_embeddings(["c0", "c1"])) == {"c0", "c1"}
    assert np.isclose(np.linalg.norm(store.get_embeddings(["c0"])["c0"]), 1.0)
    assert store.delete_document("d") == 4
    assert store.count() == 0


def test_hashing_embedder_is_deterministic_and_normalised(embedder):
    a = embedder.embed_query("annual leave")
    b = embedder.embed_documents(["annual leave"])[0]
    assert np.allclose(a, b)
    assert np.isclose(np.linalg.norm(a), 1.0)
    assert embedder.embed_documents([]).shape == (0, embedder.dimension)


class KeywordReranker:
    name = "keyword"

    def __init__(self, keyword: str):
        self.keyword = keyword
        self.calls = 0

    def score(self, query, texts):
        self.calls += 1
        return [1.0 if self.keyword in t else 0.0 for t in texts]


def test_reranker_reorders_candidates(store, embedder):
    retriever = _retriever(store, embedder)
    retriever.reranker = KeywordReranker("Passwords")
    hits = retriever.retrieve("hotel London £150", k=2, mode="hybrid")
    assert hits[0].chunk.chunk_id == "c2"
    assert hits[0].rerank_score == 1.0
    assert retriever.reranker.calls == 1


def test_reranking_can_be_disabled_per_call(store, embedder):
    retriever = _retriever(store, embedder)
    retriever.reranker = KeywordReranker("Passwords")
    hits = retriever.retrieve("hotel London £150", k=2, mode="hybrid", rerank=False)
    assert hits[0].chunk.chunk_id == "c3"
    assert hits[0].rerank_score is None
    assert retriever.reranker.calls == 0
