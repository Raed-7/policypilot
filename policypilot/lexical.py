import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9£$€%]+(?:\.[0-9]+)?")
STOPWORDS = frozenset(
    """a an and are as at be but by can do does for from has have how i if in into is it its may me must my
    no not of on or our should so than that the their them then there these they this to up us was we were
    what when where which who why will with would you your yours""".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


def _stem(token: str) -> str:
    for suffix in ("ies", "es", "s", "ing", "ed"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return token


class BM25Index:
    """Okapi BM25 over chunk texts."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self._ids: list[str] = []
        self._tfs: list[Counter] = []
        self._lengths: list[int] = []
        self._df: Counter = Counter()
        self._avg_len = 0.0

    @staticmethod
    def _terms(text: str) -> list[str]:
        return [_stem(t) for t in tokenize(text)]

    def build(self, items: list[tuple[str, str]]) -> None:
        self._ids, self._tfs, self._lengths, self._df = [], [], [], Counter()
        for chunk_id, text in items:
            terms = self._terms(text)
            tf = Counter(terms)
            self._ids.append(chunk_id)
            self._tfs.append(tf)
            self._lengths.append(len(terms))
            self._df.update(tf.keys())
        self._avg_len = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0

    def __len__(self) -> int:
        return len(self._ids)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if not self._ids:
            return []
        n = len(self._ids)
        terms = set(self._terms(query))
        scores = []
        for idx, tf in enumerate(self._tfs):
            score = 0.0
            norm = self.k1 * (1 - self.b + self.b * self._lengths[idx] / (self._avg_len or 1))
            for term in terms:
                freq = tf.get(term)
                if not freq:
                    continue
                idf = math.log(1 + (n - self._df[term] + 0.5) / (self._df[term] + 0.5))
                score += idf * freq * (self.k1 + 1) / (freq + norm)
            if score > 0:
                scores.append((self._ids[idx], score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
