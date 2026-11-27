from typing import Literal

import numpy as np

from policypilot.embeddings import Embedder
from policypilot.lexical import BM25Index
from policypilot.models import Chunk, RetrievedChunk
from policypilot.vectorstore import VectorStore

Mode = Literal["dense", "hybrid"]


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return scores


class Retriever:
    """Dense retrieval over ChromaDB, optionally fused with BM25 via reciprocal rank fusion.

    The relevance threshold is always applied to the dense cosine similarity, which is
    calibrated across queries, unlike BM25 or RRF scores.
    """

    def __init__(self, store: VectorStore, embedder: Embedder, candidate_multiplier: int = 4):
        self.store = store
        self.embedder = embedder
        self.candidate_multiplier = candidate_multiplier
        self._bm25 = BM25Index()
        self._chunks: dict[str, Chunk] = {}
        self.refresh()

    def refresh(self) -> None:
        chunks = self.store.all_chunks()
        self._chunks = {c.chunk_id: c for c in chunks}
        self._bm25.build([(c.chunk_id, c.text) for c in chunks])

    def retrieve(
        self, query: str, k: int = 5, threshold: float = 0.0, mode: Mode = "hybrid"
    ) -> list[RetrievedChunk]:
        if not query.strip() or self.store.count() == 0:
            return []
        query_vec = self.embedder.embed_query(query)
        n_candidates = max(k * self.candidate_multiplier, 20)
        dense_hits = self.store.query(query_vec, n_candidates)
        dense_scores = {chunk.chunk_id: score for chunk, score in dense_hits}
        chunks = {chunk.chunk_id: chunk for chunk, _ in dense_hits}

        if mode == "dense":
            results = [RetrievedChunk(chunk=c, score=s, dense_score=s) for c, s in dense_hits]
        else:
            lexical_hits = self._bm25.search(query, n_candidates)
            lexical_scores = dict(lexical_hits)
            missing = [cid for cid, _ in lexical_hits if cid not in dense_scores and cid in self._chunks]
            for cid, vec in self.store.get_embeddings(missing).items():
                dense_scores[cid] = float(np.dot(query_vec, vec))
                chunks[cid] = self._chunks[cid]
            fused = reciprocal_rank_fusion([[c.chunk_id for c, _ in dense_hits], [cid for cid, _ in lexical_hits]])
            results = [
                RetrievedChunk(
                    chunk=chunks[cid],
                    score=score,
                    dense_score=dense_scores[cid],
                    lexical_score=lexical_scores.get(cid, 0.0),
                )
                for cid, score in sorted(fused.items(), key=lambda x: x[1], reverse=True)
                if cid in chunks
            ]

        return [r for r in results if r.dense_score >= threshold][:k]
