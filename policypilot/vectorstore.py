from collections import defaultdict
from pathlib import Path

import chromadb
import numpy as np

from policypilot.models import Chunk


class VectorStore:
    """ChromaDB collection holding chunk embeddings with cosine distance."""

    def __init__(self, persist_dir: str | Path | None, collection_name: str = "policies"):
        if persist_dir is None:
            self._client = chromadb.EphemeralClient()
        else:
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._name = collection_name
        self._collection = self._get_collection()

    def _get_collection(self):
        return self._client.get_or_create_collection(
            name=self._name, metadata={"hnsw:space": "cosine"}, embedding_function=None
        )

    def add(self, chunks: list[Chunk], embeddings: np.ndarray, batch_size: int = 512) -> None:
        if not chunks:
            return
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            self._collection.upsert(
                ids=[c.chunk_id for c in batch],
                embeddings=embeddings[start : start + batch_size].tolist(),
                documents=[c.text for c in batch],
                metadatas=[c.to_metadata() for c in batch],
            )

    def query(self, embedding: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        count = self.count()
        if count == 0:
            return []
        result = self._collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=min(k, count),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for chunk_id, text, meta, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            hits.append((Chunk.from_metadata(chunk_id, text, meta), 1.0 - float(distance)))
        return hits

    def get_embeddings(self, chunk_ids: list[str]) -> dict[str, np.ndarray]:
        if not chunk_ids:
            return {}
        result = self._collection.get(ids=chunk_ids, include=["embeddings"])
        return {cid: np.asarray(vec, dtype=np.float32) for cid, vec in zip(result["ids"], result["embeddings"])}

    def all_chunks(self) -> list[Chunk]:
        result = self._collection.get(include=["documents", "metadatas"])
        chunks = [Chunk.from_metadata(i, d, m) for i, d, m in zip(result["ids"], result["documents"], result["metadatas"])]
        return sorted(chunks, key=lambda c: (c.source, c.position))

    def has_document(self, doc_id: str) -> bool:
        return bool(self._collection.get(where={"doc_id": doc_id}, limit=1)["ids"])

    def delete_document(self, doc_id: str) -> int:
        ids = self._collection.get(where={"doc_id": doc_id})["ids"]
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def documents(self) -> list[dict]:
        result = self._collection.get(include=["metadatas"])
        grouped: dict[str, dict] = {}
        pages: dict[str, set] = defaultdict(set)
        for meta in result["metadatas"]:
            doc_id = meta["doc_id"]
            entry = grouped.setdefault(
                doc_id, {"doc_id": doc_id, "source": meta["source"], "title": meta.get("title", ""), "chunks": 0}
            )
            entry["chunks"] += 1
            if meta.get("page", -1) >= 0:
                pages[doc_id].add(meta["page"])
        for doc_id, entry in grouped.items():
            entry["pages"] = len(pages[doc_id]) or None
        return sorted(grouped.values(), key=lambda d: d["source"])

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        self._client.delete_collection(self._name)
        self._collection = self._get_collection()
