import re
from dataclasses import dataclass

import numpy as np

from policypilot.chunking import split_sentences
from policypilot.embeddings import Embedder
from policypilot.lexical import tokenize
from policypilot.llm.base import LLMProvider
from policypilot.models import RetrievedChunk
from policypilot.prompts import REFUSAL, SYSTEM_PROMPT, build_user_prompt

_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass
class Generation:
    text: str
    provider: str
    model: str


def extract_citations(text: str, n_contexts: int) -> list[int]:
    """Return the distinct, valid 1-based context numbers cited in `text`, in order of first use."""
    seen: list[int] = []
    for group in _CITATION.findall(text):
        for raw in group.split(","):
            number = int(raw)
            if 1 <= number <= n_contexts and number not in seen:
                seen.append(number)
    return seen


def strip_citations(text: str) -> str:
    return re.sub(r"\s+([.,;:])", r"\1", _CITATION.sub("", text)).strip()


def is_refusal(text: str) -> bool:
    normalised = text.strip().strip('"').lower()
    return normalised.startswith(REFUSAL.lower()[:30]) or normalised.startswith("i don't have enough information")


class LLMGenerator:
    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def generate(self, question: str, contexts: list[RetrievedChunk]) -> Generation:
        if not contexts:
            return Generation(REFUSAL, self.provider.name, self.provider.model)
        text = self.provider.complete(SYSTEM_PROMPT, build_user_prompt(question, contexts))
        return Generation(text, self.provider.name, self.provider.model)


class ExtractiveGenerator:
    """Answers by selecting the retrieved sentences most similar to the question.

    Deterministic and key-free: every sentence it returns is verbatim from a source,
    so the output is grounded by construction.
    """

    name = "extractive"
    model = "sentence-selection"

    def __init__(self, embedder: Embedder, max_sentences: int = 3, margin: float = 0.06, lexical_weight: float = 0.25):
        self.embedder = embedder
        self.max_sentences = max_sentences
        self.margin = margin
        self.lexical_weight = lexical_weight

    def _candidates(self, contexts: list[RetrievedChunk]) -> list[tuple[int, int, str]]:
        candidates = []
        for ctx_index, item in enumerate(contexts, start=1):
            text = item.chunk.text
            if item.chunk.heading and text.startswith(item.chunk.heading + ": "):
                text = text[len(item.chunk.heading) + 2 :]
            for pos, sentence in enumerate(split_sentences(text)):
                if len(sentence.split()) >= 4:
                    candidates.append((ctx_index, pos, sentence))
        return candidates

    def generate(self, question: str, contexts: list[RetrievedChunk]) -> Generation:
        candidates = self._candidates(contexts)
        if not candidates:
            return Generation(REFUSAL, self.name, self.model)

        query_vec = self.embedder.embed_query(question)
        sentence_vecs = self.embedder.embed_documents([s for _, _, s in candidates])
        semantic = sentence_vecs @ query_vec
        query_terms = set(tokenize(question))
        lexical = np.array(
            [len(query_terms & set(tokenize(s))) / len(query_terms) if query_terms else 0.0 for _, _, s in candidates]
        )
        rank_prior = np.array([0.02 * (len(contexts) - ctx) / len(contexts) for ctx, _, _ in candidates])
        scores = semantic + self.lexical_weight * lexical + rank_prior

        order = np.argsort(-scores)
        best = scores[order[0]]
        chosen: list[int] = []
        for idx in order:
            if len(chosen) >= self.max_sentences or scores[idx] < best - self.margin:
                break
            if any(float(sentence_vecs[idx] @ sentence_vecs[j]) > 0.92 for j in chosen):
                continue
            chosen.append(int(idx))

        chosen.sort(key=lambda i: (candidates[i][0], candidates[i][1]))
        parts = []
        for i in chosen:
            ctx_index, _, sentence = candidates[i]
            sentence = sentence.rstrip()
            if sentence[-1] not in ".!?":
                sentence += "."
            parts.append(f"{sentence} [{ctx_index}]")
        return Generation(" ".join(parts), self.name, self.model)
