import re
from dataclasses import dataclass

import numpy as np

from policypilot.chunking import split_sentences
from policypilot.generation import strip_citations
from policypilot.lexical import tokenize
from policypilot.models import Chunk


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def is_relevant(chunk: Chunk, example: dict) -> bool:
    """A chunk is relevant if it comes from the labelled source and contains the labelled evidence span."""
    return chunk.source == example["source"] and normalise(example["evidence"]) in normalise(chunk.text)


def first_relevant_rank(chunks: list[Chunk], example: dict) -> int | None:
    for rank, chunk in enumerate(chunks, start=1):
        if is_relevant(chunk, example):
            return rank
    return None


def recall_at_k(ranks: list[int | None], k: int) -> float:
    return float(np.mean([r is not None and r <= k for r in ranks])) if ranks else 0.0


def mean_reciprocal_rank(ranks: list[int | None]) -> float:
    return float(np.mean([1.0 / r if r else 0.0 for r in ranks])) if ranks else 0.0


def fact_recall(answer: str, facts: list[str]) -> float:
    if not facts:
        return 1.0
    text = normalise(answer)
    return sum(normalise(f) in text for f in facts) / len(facts)


@dataclass
class Groundedness:
    sentence_support: float
    token_precision: float


def groundedness(answer: str, contexts: list[str], embedder, support_threshold: float = 0.75) -> Groundedness:
    """Score how well an answer is supported by its source excerpts.

    sentence_support: share of answer sentences whose closest source sentence has cosine
    similarity >= support_threshold (or that appear verbatim in a source).
    token_precision: share of answer content words that occur in the sources.
    """
    sentences = [s for s in split_sentences(strip_citations(answer)) if len(s.split()) >= 3]
    if not sentences or not contexts:
        return Groundedness(0.0, 0.0)
    source_text = normalise(" ".join(contexts))
    source_sentences = [s for c in contexts for s in split_sentences(c)]

    answer_vecs = embedder.embed_documents(sentences)
    source_vecs = embedder.embed_documents(source_sentences)
    best = (answer_vecs @ source_vecs.T).max(axis=1)
    supported = [normalise(s).rstrip(".") in source_text or b >= support_threshold for s, b in zip(sentences, best)]

    source_tokens = set(tokenize(source_text))
    answer_tokens = [t for s in sentences for t in tokenize(s)]
    precision = sum(t in source_tokens for t in answer_tokens) / len(answer_tokens) if answer_tokens else 0.0
    return Groundedness(float(np.mean(supported)), float(precision))


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0
