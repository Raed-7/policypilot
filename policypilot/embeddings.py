import hashlib
import re
from typing import Protocol

import numpy as np

_TOKEN = re.compile(r"[a-z0-9£$€%]+")

# Instruction prefixes recommended by the model authors for asymmetric retrieval.
_QUERY_PREFIXES = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
}


class Embedder(Protocol):
    name: str
    dimension: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, batch_size: int = 32, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        self._batch_size = batch_size
        self._query_prefix = _QUERY_PREFIXES.get(model_name, "")
        self.dimension = int(self._model.get_embedding_dimension())

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self._model.encode(
            texts, batch_size=self._batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([self._query_prefix + text])[0]


class HashingEmbedder:
    """Deterministic bag-of-words embedder used in tests and offline environments."""

    def __init__(self, dimension: int = 512):
        self.name = f"hashing-{dimension}"
        self.dimension = dimension

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        tokens = _TOKEN.findall(text.lower())
        features = tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
        for feature in features:
            digest = hashlib.md5(feature.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[index] += sign
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.stack([self._vector(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)
