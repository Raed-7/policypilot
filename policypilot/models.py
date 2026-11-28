from dataclasses import dataclass, field


@dataclass(frozen=True)
class Section:
    """A contiguous span of a source document with a stable location."""

    text: str
    page: int | None = None
    heading: str = ""


@dataclass
class Document:
    doc_id: str
    source: str
    title: str
    sections: list[Section]
    file_type: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    title: str
    text: str
    position: int
    page: int | None = None
    heading: str = ""

    @property
    def location(self) -> str:
        parts = []
        if self.page is not None:
            parts.append(f"p. {self.page}")
        if self.heading:
            parts.append(self.heading)
        return ", ".join(parts)

    def to_metadata(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "title": self.title,
            "position": self.position,
            "page": self.page if self.page is not None else -1,
            "heading": self.heading,
        }

    @classmethod
    def from_metadata(cls, chunk_id: str, text: str, meta: dict) -> "Chunk":
        page = meta.get("page", -1)
        return cls(
            chunk_id=chunk_id,
            doc_id=meta["doc_id"],
            source=meta["source"],
            title=meta.get("title", meta["source"]),
            text=text,
            position=int(meta.get("position", 0)),
            page=None if page is None or page < 0 else int(page),
            heading=meta.get("heading", ""),
        )


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float
    dense_score: float
    lexical_score: float = 0.0
    rerank_score: float | None = None
